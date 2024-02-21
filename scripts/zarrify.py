"""
Author: Seth McGinnis
Contact: mcginnis@ucar.edu

This script converts a directory of netcdf files into a single zarr store.

# Dependencies:
# - xarray
# - sys
# 
# Example use:
# python zarrify.py data/T2/2020 zarr/T2/T2.2020.zarr
# 
"""
import xarray
import sys

if(len(sys.argv) != 3):
    print("Usage: python zarrify.py path/to/input/dir outname.zarr")
    print("creates outname.zarr from all .nc files in indir")
    quit()

indir = sys.argv[1]
inglob = indir + "/*.nc"
outzarr = sys.argv[2]

ds = xarray.open_mfdataset(inglob)

## delete all global attributes
ds.attrs = {}

## manual dataset.chunk()-ing goes here if needed
## Default is 1 day, all space, which seems like a good start

ds.to_zarr(outzarr)

