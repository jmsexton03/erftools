"""HRRR land-surface initial state for an LSM (Noah-MP) wrfinput, following WRF real.exe.

The atmosphere path (NativeHRRR, terrain_remap.remap_snapshot) carries no land state. This
module reads it from an HRRR GRIB2 file on the HRRR grid and maps it to the output mass
points:

1. read_hrrr_land: the surface fields (land mask, terrain, skin temperature, sea ice) and the
   RUC soil levels (temperature, volumetric and liquid moisture), named by GRIB name, from
   one or more GRIB2 files. In 2021 HRRR the full RUC soil column (TSOIL / SOILW, 0 to 300 cm)
   is in wrfprsf00; wrfsfcf00 has only MSTAV, and wrfnatf00 has soil moisture at 0 and 1 cm
   and no soil temperature (see soil_state_from_hrrr, `fallback`).
2. LandRegridder: bilinear in HRRR's projected plane, restricted to valid (land) source
   points so that land and water never mix; a point with no valid corner takes the nearest
   valid source point.
3. soil_state_from_hrrr: soil levels -> LSM layer centres (linear in depth); optionally
   (adjust_soil_temp=True) the soil temperature adjusted to the new terrain with a constant
   lapse rate, as WRF real.exe adjust_soil_temp_new (per WRF documentation, not verified
   against source here).

real.exe behaviour quoted here is as documented for module_initialize_real.F and
share/module_soil_pre.F; there is no WRF source in this repository to check it against.
"""
import os

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
from scipy.spatial import cKDTree

from .hrrr import NativeHRRR, hrrr_projection
from .terrain_remap import STD_LAPSE

NOAHMP_DZS = (0.10, 0.30, 0.60, 1.00)  # m: Noah / Noah-MP default layers, 0-10-40-100-200 cm
NOAHMP_ZBOT = 8.0                      # m: depth of TMN (NoahmpTable.TBL ZBOT_DATA = -8.0)

# typeOfLevel=depthBelowLandLayer variables by GRIB name (lower case); cfgrib shortName fallback
SOIL_GRIB_NAMES = {
    'soil temperature': 'SOIL_T',
    'volumetric soil moisture content': 'SOIL_M',
    'liquid volumetric soil moisture (non-frozen)': 'SOIL_L',
}
SOIL_SHORT_NAMES = {'st': 'SOIL_T', 'tsoil': 'SOIL_T', 'soilw': 'SOIL_M', 'soill': 'SOIL_L'}
# typeOfLevel=surface, stepType=instant: cfgrib shortName -> (name, scale to WRF units)
SURFACE_GRIB_NAMES = {
    'lsm': ('LANDMASK', 1.0),
    'orog': ('HGT', 1.0),
    't': ('SKINT', 1.0),
    'ci': ('SEAICE', 1.0),
}


def _open_cfgrib(path, keys, indexpath):
    import cfgrib
    try:
        return cfgrib.open_datasets(path, backend_kwargs={'filter_by_keys': keys, 'indexpath': indexpath})
    except KeyError:  # no message matches the filter (cfgrib finds no 'paramId')
        return []


def read_hrrr_land(path, indexpath=''):
    """Land-surface fields of HRRR GRIB2 files on the HRRR grid, dims (y, x), with HRRR
    projected x / y coordinates [m] (as NativeHRRR builds them) and 2-D latitude / longitude.

    path: one file, or a list of files valid at the same time on the same grid, e.g. the
    wrfnat (or wrfsfc) file for the surface and snow fields plus a file with the RUC soil
    levels (wrfprs in 2021 HRRR). A surface field is taken from the first file that has it;
    a soil variable from the file with the most levels (wrfnat's 0 / 1 cm SOILW never masks
    a full column). Soil variables have dims (soil_level, y, x) and the coordinate
    SOIL_DEPTH [m, positive down, increasing]; absent ones are simply not in the Dataset.
    indexpath='' keeps cfgrib from writing an index file next to the GRIB file."""
    paths = [path] if isinstance(path, (str, os.PathLike)) else list(path)
    out, soil = {}, {}
    for p in paths:
        for ds in _open_cfgrib(p, {'stepType': 'instant', 'typeOfLevel': 'surface'}, indexpath):
            for short, (name, scale) in SURFACE_GRIB_NAMES.items():
                if short in ds and name not in out:
                    out[name] = ds[short].drop_vars([c for c in ds[short].coords if c not in
                                                     ('latitude', 'longitude')]) * scale
        for ds in _open_cfgrib(p, {'typeOfLevel': 'depthBelowLandLayer'}, indexpath):
            for short, da in ds.data_vars.items():
                name = SOIL_GRIB_NAMES.get(str(da.attrs.get('GRIB_name', '')).lower(),
                                           SOIL_SHORT_NAMES.get(short))
                if name is None:
                    continue
                if 'depthBelowLandLayer' not in da.dims:
                    da = da.expand_dims('depthBelowLandLayer')
                da = da.sortby('depthBelowLandLayer')
                if name not in soil or da.sizes['depthBelowLandLayer'] > soil[name].sizes['depthBelowLandLayer']:
                    soil[name] = da
    missing = [n for n in ('LANDMASK', 'HGT') if n not in out]
    if missing:
        raise KeyError(f'{paths}: no surface {missing} in the GRIB file(s)')
    land = xr.Dataset(out)
    shape = out['LANDMASK'].shape
    bad = [n for n, da in soil.items() if da.shape[-2:] != shape]
    if bad:
        raise ValueError(f'soil variables {bad} are not on the grid of the surface fields {shape}')
    depths = {name: tuple(np.round(da['depthBelowLandLayer'].values.astype(np.float64), 6))
              for name, da in soil.items()}
    if len(set(depths.values())) > 1:
        # keep the soil variables on the levels of the moisture (or the first) variable
        ref = depths.get('SOIL_M', next(iter(depths.values())))
        soil = {n: da for n, da in soil.items() if depths[n] == ref}
    for name, da in soil.items():
        depth = da['depthBelowLandLayer'].values.astype(np.float64)
        land[name] = (('soil_level', 'y', 'x'), da.values.astype(np.float64))
        land = land.assign_coords(SOIL_DEPTH=('soil_level', depth))
    shell = NativeHRRR.__new__(NativeHRRR)
    shell.ds = land
    shell.verbose = False
    shell._setup_hrrr_grid()  # x / y in hrrr_projection, the plane NativeHRRR interpolates in
    land = shell.ds
    land.attrs['source'] = ', '.join(str(p) for p in paths)
    return land


def hrrr_xy(lat, lon):
    """Output points (lat, lon arrays, degrees) in HRRR projected coordinates [m]."""
    pts = hrrr_projection.transform_points(ccrs.Geodetic(), np.asarray(lon, np.float64),
                                           np.asarray(lat, np.float64))
    return pts[..., 0], pts[..., 1]


def subset_to_points(land, xo, yo, margin=8):
    """The source Dataset cut to the output points' bounding box plus `margin` cells."""
    x1, y1 = land['x'].values, land['y'].values
    i = np.searchsorted(x1, [np.min(xo), np.max(xo)])
    j = np.searchsorted(y1, [np.min(yo), np.max(yo)])
    return land.isel(x=slice(max(i[0] - margin, 0), min(i[1] + margin, x1.size)),
                     y=slice(max(j[0] - margin, 0), min(j[1] + margin, y1.size)))


class LandRegridder:
    """Bilinear interpolation from a source grid with 1-D increasing x, y (HRRR projection,
    metres) to output points xo, yo (any shape), restricted to valid source points: invalid
    corners are dropped and the remaining weights renormalized, and an output point with no
    valid corner takes the nearest valid source point (cf. WPS's search fallback)."""

    def __init__(self, x1, y1, xo, yo):
        self.x1 = np.asarray(x1, np.float64)
        self.y1 = np.asarray(y1, np.float64)
        self.xo = np.asarray(xo, np.float64)
        self.yo = np.asarray(yo, np.float64)
        if not (np.all(np.diff(self.x1) > 0) and np.all(np.diff(self.y1) > 0)):
            raise ValueError('source x and y must increase')
        if (self.xo.min() < self.x1[0] or self.xo.max() > self.x1[-1]
                or self.yo.min() < self.y1[0] or self.yo.max() > self.y1[-1]):
            raise ValueError('output points outside the source grid')
        fi = np.interp(self.xo, self.x1, np.arange(self.x1.size))
        fj = np.interp(self.yo, self.y1, np.arange(self.y1.size))
        i0 = np.clip(np.floor(fi).astype(int), 0, self.x1.size - 2)
        j0 = np.clip(np.floor(fj).astype(int), 0, self.y1.size - 2)
        wx, wy = fi - i0, fj - j0
        self.corners = [(j0, i0), (j0, i0 + 1), (j0 + 1, i0), (j0 + 1, i0 + 1)]
        self.weights = [(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx]

    @classmethod
    def from_dataset(cls, land, xo, yo):
        return cls(land['x'].values, land['y'].values, xo, yo)

    def __call__(self, field, valid):
        """field (..., ny_src, nx_src), valid (ny_src, nx_src) bool -> (values with shape
        (..., *xo.shape), number of points that took the nearest valid source point)."""
        field = np.asarray(field, np.float64)
        valid = np.asarray(valid, bool) & np.all(np.isfinite(field), axis=tuple(range(field.ndim - 2)))
        if not valid.any():
            raise ValueError('no valid source points')
        f0 = np.where(valid, field, 0.0)
        num, den = 0.0, 0.0
        for (jj, ii), w in zip(self.corners, self.weights):
            m = valid[jj, ii] * w
            num = num + f0[..., jj, ii] * m
            den = den + m
        good = den > 1e-12
        out = num / np.where(good, den, 1.0)
        nfall = int((~good).sum())
        if nfall:
            jv, iv = np.nonzero(valid)
            tree = cKDTree(np.column_stack([self.x1[iv], self.y1[jv]]))
            _, k = tree.query(np.column_stack([self.xo[~good], self.yo[~good]]))
            out[..., ~good] = f0[..., jv[k], iv[k]]
        return out, nfall


def layer_centres(dzs):
    """ZS: depths [m] of the layer centres for layer thicknesses DZS."""
    dzs = np.asarray(dzs, np.float64)
    return np.cumsum(dzs) - 0.5 * dzs


def soil_levels_to_layers(depth_src, f_src, zs):
    """Linear interpolation in depth from source levels (depth_src [m], increasing; f_src
    (nlev, ...)) to the depths zs, constant beyond the source column."""
    d = np.asarray(depth_src, np.float64)
    f = np.asarray(f_src, np.float64)
    if d.size == 1:
        return np.repeat(f, len(zs), axis=0)
    k = np.clip(np.searchsorted(d, zs) - 1, 0, d.size - 2)
    w = np.clip((np.asarray(zs) - d[k]) / (d[k + 1] - d[k]), 0.0, 1.0)
    w = w.reshape((-1,) + (1,) * (f.ndim - 1))
    return f[k] * (1.0 - w) + f[k + 1] * w


def soil_state_from_hrrr(land, regrid, hgt_new, landmask, tmn=None, tsk=None, sst=None,
                         dzs=NOAHMP_DZS, lapse_rate=STD_LAPSE, adjust_soil_temp=False,
                         fallback=None):
    """Soil state at the output mass points from read_hrrr_land fields.

    land: source Dataset (read_hrrr_land, optionally subset_to_points); regrid: LandRegridder
    to the output points; hgt_new, landmask: output terrain [m] and land mask (shape of the
    output points). Land points use land HRRR points only:
      TSLB  = SOIL_T at ZS (layer centres of dzs, linear in depth); with adjust_soil_temp=True
              also + lapse_rate (HGT_hrrr - hgt_new), HGT_hrrr being HRRR's terrain
              interpolated with the same land-only weights, as WRF real.exe
              adjust_soil_temp_new with the source terrain as TOPOSOIL (per WRF
              documentation, not verified against source here). Off by default.
      SMOIS = SOIL_M at ZS, in [0, 1]
      SH2O  = SOIL_L at ZS if the file has it, else SMOIS (Noah-MP recomputes SH2O from SMOIS
              and TSLB for frozen soil at its initialization either way)
    Water points: TSLB = sst (else tsk, else tmn), SMOIS = SH2O = 1 (real.exe's water values).

    fallback=None raises when the file has no soil temperature or its moisture stops above
    the deepest layer centre (a wrfnat file). fallback='skin_tmn' then sets, on land,
    TSLB(z) = TSK' + (TMN - TSK') z / NOAHMP_ZBOT with TSK' = HRRR skin temperature (SKINT,
    land-only weights; else tsk), + lapse_rate (HGT_hrrr - hgt_new) with adjust_soil_temp,
    and holds the deepest
    moisture level below its depth. That is NOT what real.exe does; it is for testing
    without a soil file, and stats records it.

    Returns (fields: TSLB/SMOIS/SH2O (nsoil, ...), ZS/DZS (nsoil,); stats dict)."""
    hgt_new = np.asarray(hgt_new, np.float64)
    land_out = np.asarray(landmask) > 0.5
    zs = layer_centres(dzs)
    valid = land['LANDMASK'].values > 0.5
    hgt_src, nfall = regrid(land['HGT'].values, valid)
    dt = lapse_rate * (hgt_src - hgt_new) if adjust_soil_temp else np.zeros(hgt_new.shape)
    depth = land['SOIL_DEPTH'].values if 'SOIL_DEPTH' in land.coords else np.zeros(0)
    dh = (hgt_src - hgt_new)[land_out]
    stats = {'soil_source_levels_m': [float(d) for d in depth], 'zs_m': [float(z) for z in zs],
             'soil_points_nearest_fallback': nfall, 'adjust_soil_temp': bool(adjust_soil_temp),
             'lapse_rate_K_per_m': lapse_rate if adjust_soil_temp else None,
             'dhgt_hrrr_minus_new_m': [float(dh.min()), float(dh.max())] if dh.size else [],
             'fallback': None}
    if 'SOIL_M' not in land:
        raise KeyError(f"{land.attrs.get('source', 'HRRR file')}: no soil moisture")
    shallow = depth.size == 0 or depth.max() < zs[-1]
    if ('SOIL_T' not in land or shallow) and fallback is None:
        raise ValueError(
            f"{land.attrs.get('source', 'HRRR file')}: soil temperature "
            f"{'present' if 'SOIL_T' in land else 'absent'}, soil levels {list(depth)} m; a full "
            "soil column needs a file with HRRR's RUC soil levels (wrfprsf00 in 2021 HRRR; "
            "or pass fallback='skin_tmn')")
    if fallback not in (None, 'skin_tmn'):
        raise ValueError(f'unknown fallback {fallback!r}')

    smois_src, _ = regrid(land['SOIL_M'].values, valid)
    smois = np.clip(soil_levels_to_layers(depth, smois_src, zs), 0.0, 1.0)
    if 'SOIL_L' in land:
        sh2o_src, _ = regrid(land['SOIL_L'].values, valid)
        sh2o = np.minimum(np.clip(soil_levels_to_layers(depth, sh2o_src, zs), 0.0, 1.0), smois)
    else:
        sh2o = smois.copy()
    if 'SOIL_T' in land and not shallow:
        tsrc, _ = regrid(land['SOIL_T'].values, valid)
        tslb = soil_levels_to_layers(depth, tsrc, zs) + dt[None]
    else:
        if tmn is None or ('SKINT' not in land and tsk is None):
            raise ValueError("fallback='skin_tmn' needs tmn, and SKINT in the file or tsk")
        if 'SKINT' in land:
            tsk_src, _ = regrid(land['SKINT'].values, valid)  # HRRR skin, land points only
        else:
            tsk_src = np.asarray(tsk, np.float64)
        tsk_adj = tsk_src + dt
        frac = np.minimum(zs / NOAHMP_ZBOT, 1.0).reshape((-1,) + (1,) * hgt_new.ndim)
        tslb = tsk_adj[None] + (np.asarray(tmn, np.float64)[None] - tsk_adj[None]) * frac
        stats['fallback'] = ('skin_tmn: TSLB linear between the skin temperature '
                             f'and TMN at {NOAHMP_ZBOT} m; SMOIS held below {depth.max():g} m')
    t_water = next((np.asarray(v, np.float64) for v in (sst, tsk, tmn) if v is not None), None)
    if t_water is None and not land_out.all():
        raise ValueError('water points need sst, tsk or tmn')
    if t_water is not None:
        tslb = np.where(land_out[None], tslb, np.broadcast_to(t_water, hgt_new.shape)[None])
    smois = np.where(land_out[None], smois, 1.0)
    sh2o = np.where(land_out[None], sh2o, 1.0)
    fields = {'TSLB': tslb, 'SMOIS': smois, 'SH2O': sh2o,
              'ZS': zs, 'DZS': np.asarray(dzs, np.float64)}
    for k in ('TSLB', 'SMOIS'):
        v = fields[k][:, land_out]
        stats[f'{k}_land_range'] = [float(v.min()), float(v.max())] if v.size else []
    return fields, stats
