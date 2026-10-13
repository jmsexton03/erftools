"""python -m erftools.geog --preset sail_default --landuse modis --out FILE  (see cli.py)"""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
