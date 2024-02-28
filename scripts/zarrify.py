#!/usr/bin/env python
"""Converts a directory of netcdf files into a single zarr store.

Usage:
    zarrify.py path/to/input/dir outname.zarr

path/to/input/dir must contain one or more .nc files, and outname.zarr
must not already exist.


On Casper, zarrifying 1 year of hourly conus404 data (1 variable,
1015x1367 = 1.39e6 gridpoints, 8760 timesteps, total data volume ~
25GB) takes about 10 minutes wallclock and requires about 300MB of
memory.


Author: Seth McGinnis
Contact: mcginnis@ucar.edu

# Dependencies:
# - xarray
# - sys
# - os.path
# - glob

"""
import xarray
import sys
import os.path
import glob

if(len(sys.argv) != 3):
    print("error: not enough arguments")
    print("Usage: python zarrify.py path/to/input/dir outname.zarr")
    print("       creates outname.zarr from all .nc files in indir")
    quit()

indir = sys.argv[1]
inglob = indir + "/*.nc"
outzarr = sys.argv[2]

if(os.path.exists(os.path.expanduser(outzarr))):
    print("error: outname.zarr must not already exist")
    quit()

if(len(glob.glob(os.path.expanduser(inglob))) < 1):
    print("error: input directory contains no .nc files")
    quit()


ds = xarray.open_mfdataset(inglob)

## delete all global attributes
ds.attrs = {}

## manual dataset.chunk()-ing goes here if needed
## Default is 1 day, all space, which seems like a good start

ds.to_zarr(outzarr)

