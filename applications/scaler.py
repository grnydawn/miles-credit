import os
import numpy as np
import xarray as xr
import pandas as pd
import yaml
import argparse
from glob import glob
from bridgescaler.distributed import DQuantileScaler
from bridgescaler.backend import print_scaler
from os.path import exists, join
from mpi4py import MPI
from multiprocessing import Pool
from functools import partial
from time import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config file")
    parser.add_argument("-o", "--out", help="Path to save scaler files.")
    parser.add_argument("-t", "--time", type=int, default=1752, help="Number of hours sampled/year")

    parser.add_argument("-p", "--procs", type=int, default=1, help="Number of processors")
    args = parser.parse_args()
    args_dict = vars(args)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        config = args_dict.pop("config")
        with open(config) as cf:
            conf = yaml.load(cf, Loader=yaml.FullLoader)
        all_era5_files = sorted(glob(conf["data"]["save_loc"]))
        for e5 in all_era5_files:
            if "_small_" in e5:
                all_era5_files.remove(e5)
        all_era5_filenames = np.array([f.split("/")[-1] for f in all_era5_files])
    else:
        all_era5_files = None
        all_era5_filenames = None
    era5_subset = comm.scatter(all_era5_files, root=0)
    print(f"Rank {rank:d}", era5_subset, type(era5_subset))
    if type(era5_subset) == list:
        scalers = []
        for era5_file in era5_subset:
            scalers.append(fit_era5_scaler_year(era5_file, n_times=args.time, n_jobs=args.procs))
    else:
        scalers = fit_era5_scaler_year(era5_subset, n_times=args.time, n_jobs=args.procs)
    all_scalers = comm.gather(scalers, root=0)
    if rank == 0:
        all_scalers_df = pd.DataFrame(all_scalers, columns=["scaler_3d", "scaler_surface"],
                                      index=all_era5_filenames)
        if not exists(args.out):
            os.makedirs(args.out)
        now = pd.Timestamp.utcnow().strftime("%Y-%m-%d_%H:%M")
        all_scalers_df.to_parquet(join(args.out, f"era5_quantile_scalers_{now}.parquet"))
    return


def fit_era5_scaler_year(era5_file, n_times=60, n_jobs=60):
    pool = Pool(n_jobs, maxtasksperchild=1) 
    eds = xr.open_zarr(era5_file)
    vars_3d = ['U', 'V', 'T', 'Q']
    vars_surf = ['SP', 't2m', 'V500', 'U500', 'T500', 'Z500', 'Q500']
    levels = eds.level.values
    var_levels = []
    for var in vars_3d:
        for level in levels:
            var_levels.append(f"{var}_{level:d}")
    rand_times = np.sort(np.random.choice(eds["time"].values, size=n_times, replace=False))
    fit_time_func = partial(fit_era5_scaler_time, era5_file=era5_file, vars_3d=vars_3d, vars_surf=vars_surf, 
                            levels=levels, var_levels=var_levels, times=rand_times)
    results = np.array(pool.map(fit_time_func, rand_times))
    dqs_3d = np.sum(results[:, 0])
    dqs_surf = np.sum(results[:, 1])
    dqs_3d_json = print_scaler(dqs_3d)
    dqs_surf_json = print_scaler(dqs_surf)
    pool.close()
    pool.join()
    return dqs_3d_json, dqs_surf_json


def fit_era5_scaler_time(time, era5_file=None, vars_3d=None, vars_surf=None, levels=None, var_levels=None, times=None):  
    eds = xr.open_zarr(era5_file)
    dqs_3d = DQuantileScaler(distribution="normal", channels_last=False)
    dqs_surf = DQuantileScaler(distribution="normal", channels_last=False)
    var_slices = []
    for var in vars_3d:
        for level in levels:
            var_slices.append(eds[var].loc[time, level])
    e3d = xr.concat(var_slices, pd.Index(var_levels, name="variable")
                        ).load()
    e3d = e3d.expand_dims(dim="time", axis=0)
    print(time, np.searchsorted(times, time), times.size)
    dqs_3d.fit(e3d)
    e_surf = xr.concat([eds[v].loc[time] for v in vars_surf], pd.Index(vars_surf, name="variable")
                           ).load()
    e_surf = e_surf.expand_dims(dim="time", axis=0)
    dqs_surf.fit(e_surf)
    eds.close()
    del eds
    return dqs_3d, dqs_surf

if __name__ == '__main__':
    main()
