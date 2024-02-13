import os
import numpy as np
import xarray as xr
import pandas as pd
import yaml
import argparse
from glob import glob
from multiprocessing import Pool
from bridgescaler.distributed import DQuantileTransformer
from bridgescaler.backend import print_scaler
from os.path import exists, join


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config file")
    parser.add_argument("-o", "--out", help="Path to save scaler files.")
    parser.add_argument("-p", "--procs", type=int, help="Number of processors")
    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("config")
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
    all_era5_files = sorted(glob(conf["data"]["save_loc"]))
    for e5 in all_era5_files:
        if "_small_" in e5:
            print(e5)
            all_era5_files.remove(e5)
    all_era5_filenames = [f.split("/")[-1] for f in all_era5_files]
    with Pool(args.procs) as p:
        all_scalers = p.map(fit_era5_scaler_year, all_era5_files)
    all_scalers_df = pd.DataFrame(all_scalers, columns=["scaler_3d", "scaler_surface"],
                                  index=all_era5_filenames)
    if not exists(args.out):
        os.makedirs(args.out)
    now = pd.Timestamp.utcnow().strftime("%Y-%m-%d_%H:%M")
    all_scalers_df.to_parquet(join(args.out, f"era5_quantile_scalers_{now}.parquet"))
    return


def fit_era5_scaler_year(era5_file):
    n_times = 300
    eds = xr.open_zarr(era5_file)
    vars_3d = ['U', 'V', 'T', 'Q']
    vars_surf = ['SP', 't2m', 'V500', 'U500', 'T500', 'Z500', 'Q500']
    levels = eds.level.values
    var_levels = []
    for var in vars_3d:
        for level in levels:
            var_levels.append(f"{var}_{level:d}")
    dqs_3d = DQuantileTransformer(distribution="normal")
    dqs_surf = DQuantileTransformer(distribution="normal")
    rand_times = np.sort(np.random.choice(eds["time"].values, size=n_times, replace=False))
    for time in rand_times:
        print(time)
        var_slices = []
        for var in vars_3d:
            for level in levels:
                var_slices.append(eds[var].sel(time=time, level=level))
        e3d = xr.concat(var_slices, pd.Index(var_levels, name="variable")
                        ).transpose("latitude", "longitude", "variable")
        dqs_3d.fit(e3d)
        e_surf = xr.concat([eds[v].sel(time=time) for v in vars_surf], pd.Index(vars_surf, name="variable")
                           ).transpose("latitude", "longitude", "variable")
        dqs_surf.fit(e_surf)
    dqs_3d_json = print_scaler(dqs_3d)
    dqs_surf_json = print_scaler(dqs_surf)
    return dqs_3d_json, dqs_surf_json

if __name__ == '__main__':
    main()
