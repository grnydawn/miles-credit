from haversine import haversine_vector, Unit
import xarray as xr
import argparse
import numpy as np
from multiprocessing import Pool
from functools import partial
from os.path import join, exists
from os import makedirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--coord", help="Path to xarray file containing coordinates")
    parser.add_argument("-o", "--out", help="Path to output directory")
    parser.add_argument("-d", "--dist", type=float, help="Max distance for adjacency (km)")
    parser.add_argument("-p", "--procs", type=int, help="Number of processes")
    args = parser.parse_args()
    coords = xr.open_dataset(args.coord)
    lon = coords["longitude"].values
    lon[lon > 180] = lon[lon > 180] - 360.0
    lat = coords["latitude"].values
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    lon_flat = lon_grid.ravel()
    lat_flat = lat_grid.ravel()
    coord_set = [(i, lon_flat[i], lat_flat[i]) for i in range(lon_flat.size)]
    calc_edge_p = partial(calc_edges, all_coords=coord_set, max_dist=args.dist)
    with Pool(processes=args.procs) as p:
        edge_indices_list = p.map(calc_edge_p, coord_set)
    edge_indices_arr = np.concatenate(edge_indices_list)
    output_ds = xr.Dataset({"edges": (("index", "pair"), edge_indices_arr),
                "longitude": (("index", ), lon_flat),
                "latitude": (("index", ), lat_flat),
                }, coords={"index": list(range(lon_flat.size))},
                attrs=dict(coord_file=args.coord, max_distance=args.dist))
    if not exists(args.out):
        makedirs(args.out)
    output_ds.to_netcdf(join(args.out, f"grid_edge_pairs_{args.dist:0.1f}.nc"))
    return


def calc_edges(coord, all_coords=None, max_dist=25.0):
    coord_distances = haversine_vector(coord[1:], all_coords[:, 1:], Unit.KILOMETERS)
    close_coords = np.where(coord_distances < max_dist)[0]
    edge_indices = [(int(coord[0]), c) for c in close_coords]
    return edge_indices

if __name__ == "__main__":
    main()