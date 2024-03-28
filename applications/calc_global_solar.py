from credit.solar import get_solar_radiation_loc
from mpi4py import MPI
import argparse
import xarray as xr
import numpy as np
import os
import pandas as pd


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    parser = argparse.ArgumentParser(description='Calculate global solar radiation')
    parser.add_argument('-s', '--start', type=str, default="2000-01-01", help="Start date (inclusive)")
    parser.add_argument('-e', '--end', type=str, default="2000-12-31 23:00", help="End date (inclusive")
    parser.add_argument('-t', '--step', type=str, default="1h", help="Step frequency")
    parser.add_argument('-u', '--sub', type=str, default="5Min", help="Sub Frequency")
    parser.add_argument('-i', '--input', type=str,
                        default="/glade/u/home/wchapman/MLWPS/DataLoader/static_variables_ERA5_zhght.nc",
                        help="File containing longitudes, latitudes, and geopotential height.")
    parser.add_argument("-g", "--geo", type=str, default="Z_GDS4_SFC",
                        help="Geopotential height variable.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    grid_points = None
    if rank == 0:
        with xr.open_dataset(args.input) as static_ds:
            lons = static_ds["longitude"].values
            lats = static_ds["latitude"].values
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            heights = static_ds[args.geo].values / 9.81
            grid_points = np.vstack([lon_grid.ravel(), lat_grid.ravel(), heights.ravel()]).T.tolist()
    rank_points = comm.scatter(grid_points, root=0)
    solar_data = []
    for r, rank_point in enumerate(rank_points):
        print(rank, rank_point, r, len(rank_points))
        solar_data.append(get_solar_radiation_loc(rank_point[0], rank_point[1], rank_point[2],
                                args.start, args.end, step_freq=args.step, sub_freq=args.sub))
    all_data = comm.gather(solar_data, root=0)
    if rank == 0:
        combined = xr.combine_by_coords(all_data)
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        out_time = pd.Timestamp.utcnow().strftime("%Y-%m-%d_%H%M")
        filename = f"solar_radiation_{out_time}.nc"
        combined.to_netcdf(os.path.join(args.output, filename), encoding={"tsi": {"zlib": True, "complevel": 4}})
    return


if __name__ == "__main__":
    main()