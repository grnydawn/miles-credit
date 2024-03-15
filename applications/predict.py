'''
Run model and produce outputs 
------------------------------
Output tensor size: (variables, latitude, longitude)
Atmospheric level arrangement: top-of-atmosphere--> near-surface --> single layer
An example: U (top-of-atmosphere) --> U (near-surface) --> V (top-of-atmosphere) --> V (near-surface)

Yingkai Sha
ksha@ucar.edu
'''


# ---------- #
# System
import gc
import os
import sys
import yaml
import glob
import shutil
import logging
import warnings
import subprocess
from os.path import join
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from collections import defaultdict
from argparse import ArgumentParser

# ---------- #
# Numerics
import datetime
import numpy as np
import pandas as pd
import xarray as xr

# ---------- #
# AI libs
import torch
#import torch.fft
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms

# ---------- #
# credit
from credit.data import PredictForecast
from credit.loss import VariableTotalLoss2D
from credit.models import load_model
from credit.metrics import LatWeightedMetrics
from credit.transforms import ToTensor, NormalizeState
from credit.seed import seed_everything
from credit.pbs import launch_script, launch_script_mpi

# ---------- #
# visualization_tools is part of the credit now, but it requires a pip update
try:
    from credit.visualization_tools import draw_sigma_level, draw_diagnostics, draw_surface
except:
    from visualization_tools import draw_sigma_level, draw_diagnostics, draw_surface
    
# import wandb

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def setup(rank, world_size, mode):
    logging.info(f"Running {mode.upper()} on rank {rank} with world_size {world_size}.")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def split_and_reshape(tensor, conf):
    # return x, surf for B, c, lat, lon output 
    levels = conf['model']['levels']
    channels = len(conf['data']['variables'])
    surface_channels = len(conf['data']['surface_variables'])

    tensor1 = tensor[:, :int(channels * levels), :, :]
    tensor2 = tensor[:, -int(surface_channels):, :, :]
    tensor1 = tensor1.view(tensor1.shape[0], channels, levels, tensor1.shape[-2], tensor1.shape[-1])
    return tensor1, tensor2

def predict(rank, world_size, conf, pool):

    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        setup(rank, world_size, conf["trainer"]["mode"])

    # infer device id from rank

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Config settings
    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    history_len = conf["data"]["history_len"]
    forecast_len = conf["data"]["forecast_len"]
    time_step = conf["data"]["time_step"] if "time_step" in conf["data"] else None

    # Load paths to all ERA5 data available
    all_ERA_files = sorted(glob.glob(conf["data"]["save_loc"]))

    # Preprocessing transformations
    state_transformer = NormalizeState(conf["data"]["mean_path"], conf["data"]["std_path"])
    transform = transforms.Compose([
        state_transformer,
        ToTensor(history_len=history_len, forecast_len=forecast_len)
    ])

    dataset = PredictForecast(
        filenames=all_ERA_files,
        forecasts=conf['predict']['forecasts'],
        history_len=history_len,
        forecast_len=forecast_len,
        skip_periods=time_step,
        transform=transform,
        rank=rank,
        world_size=world_size,
        shuffle=False
    )

    # setup the dataloder for this process
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        drop_last=False
    )

    # load model
    model = load_model(conf, load_weights=True).to(device)

    # Warning -- see next line
    if conf["trainer"]["mode"] == "ddp":  # A new field needs to be added to predict
        model = DDP(model, device_ids=[device])

    model.eval()

    # Set up metrics and containers
    metrics = LatWeightedMetrics(conf)
    metrics_results = defaultdict(list)
    loss_fn = VariableTotalLoss2D(conf, validation=True)
    
    latds = xr.open_dataset(conf["loss"]["latitude_weights"])
    # Rollout
    with torch.no_grad():

        forecast_count = 0
        x_dataArrs, surf_dataArrs = [], []
        pool_jobs = []
        sigma_levels = conf['visualization']['sigma_level_visualize']['visualize_levels']
        
        y_pred = None
        for batch in data_loader:
            date_time = batch["datetime"].item()
            forecast_hour = batch["forecast_hour"].item()
            
            if forecast_hour == 1:
                # Initialize x and x_surf with the first time step
                x = model.concat_and_reshape(
                    batch["x"],
                    batch["x_surf"]
                ).to(device)

                # setup save directory for images
                init_time = datetime.datetime.utcfromtimestamp(date_time).strftime('%Y-%m-%dT%HZ')
                img_save_loc = os.path.join(os.path.expandvars(conf["save_loc"]), f'forecasts/images_{init_time}')
                os.makedirs(img_save_loc, exist_ok=True)
            
            y = model.concat_and_reshape(
                batch["y"],
                batch["y_surf"]
                ).to(device)
            
            # Predict
            y_pred = model(x)
            
            # Update the input
            if history_len == 1:
                x = y_pred.detach()
            else:
                x_detach = x[:, :, 1:].detach()
                x = torch.cat([x_detach, y_pred.detach()], dim=2)
                y = y.squeeze(2)
                y_pred = y_pred.squeeze(2)
            
            # Convert back to quantites with physical units before computing metrics
            y = state_transformer.inverse_transform(y.cpu())
            y_pred = state_transformer.inverse_transform(y_pred.cpu())
            
            # Compute metrics
            mae = loss_fn(y, y_pred)
            metrics_dict = metrics(y_pred.float(), y.float())
            for k, m in metrics_dict.items():
                metrics_results[k].append(m.item())
            metrics_results["forecast_hour"].append(forecast_hour)
            metrics_results["datetime"].append(date_time)
            
            utc_datetime = datetime.datetime.utcfromtimestamp(date_time)
            print_str = f"Forecast: {forecast_count} "
            print_str += f"Date: {utc_datetime.strftime('%Y-%m-%d %H:%M:%S')} "
            print_str += f"Hour: {batch['forecast_hour'].item()} "
            print_str += f"MAE: {mae.item()} "
            print_str += f"ACC: {metrics_dict['acc']}"
            print(print_str)

            da_x, da_surf = make_xarray(y_pred, utc_datetime, latds.latitude.values, latds.longitude.values, conf)         
            x_dataArrs.append(da_x)
            surf_dataArrs.append(da_surf)

            # submit jobs to pool
            f = partial(draw_sigma_level, conf=conf, save_location=img_save_loc)
            da_x_rmved_t = da_x.isel(datetime=0)
            job_result = pool.map_async(f, [da_x_rmved_t.sel(level=lvl) for lvl in sigma_levels])
            pool_jobs.append(job_result)
            
            # Explicitly release GPU memory
            torch.cuda.empty_cache()
            gc.collect()
            
            if batch['stop_forecast'][0]:
                break

    return x_dataArrs, surf_dataArrs, pool_jobs, img_save_loc

def make_xarray(pred, forecast_datetime, lat, lon, conf):
    x, surf = split_and_reshape(pred, conf)
    da_x = xr.DataArray(x, 
                    dims=['datetime', 'var', 'level', 'lat', 'lon'],
                    coords=dict(
                        var = conf['data']['variables'],
                        datetime = [forecast_datetime],
                        level = range(conf['model']['levels']),
                        lat = lat,
                        lon = lon)
                    )
    da_surf = xr.DataArray(surf, 
                    dims=['datetime', 'var', 'lat', 'lon'],
                    coords=dict(
                        var = conf['data']['surface_variables'],
                        datetime = [forecast_datetime],
                        lat = lat,
                        lon = lon)
                    )
    return da_x, da_surf

def save_netcdf(x_dataArrs, surf_dataArrs, conf):
    da_x = xr.concat(x_dataArrs, dim='datetime')
    da_surf = xr.concat(surf_dataArrs, dim='datetime')

    init_datetime_str = np.datetime_as_string(da_x.datetime[0], unit='h', timezone='UTC')
    
    # create save directory for xarrays
    save_location = os.path.join(os.path.expandvars(conf['save_loc']), "forecasts")
    os.makedirs(save_location, exist_ok=True)

    x_save_loc = os.path.join(save_location, f'pred_x_{init_datetime_str}.nc')
    surf_save_loc = os.path.join(save_location, f'pred_surf_{init_datetime_str}.nc')
    da_x.to_netcdf(path=x_save_loc)
    da_surf.to_netcdf(path=surf_save_loc)
    print(f'wrote .nc files for column and surface vars:\n{x_save_loc}\n{surf_save_loc}')
    
    return x_save_loc, surf_save_loc, 

def make_images_from_xarray(x_save_loc, surf_save_loc, conf):
    da_x = xr.load_dataarray(x_save_loc)
    da_surf = xr.load_dataarray(surf_save_loc)
    
    # Create directories to save images, overwrite files if already exists, filenames should uniquely id
    init_time = np.datetime_as_string(da_x.datetime[0], unit='h', timezone='UTC')
    save_loc = os.path.join(os.path.expandvars(conf["save_loc"]), f'forecasts/images_{init_time}')
    os.makedirs(save_loc, exist_ok=True)

    # model levels to plot
    sigma_levels = conf['visualization']['sigma_level_visualize']['visualize_levels']
    
    # todo: parallelize over times
    for level in sigma_levels:
        datetimes = da_x.datetime.to_numpy()
        with Pool(processes=8) as pool:
            f = partial(draw_sigma_level, conf=conf, save_location=save_loc)
            da_level = da_x.sel(level=level)
            pool.map(f, [da_level.sel(datetime=dt) for dt in datetimes])

    return save_loc

def make_movie(filenames, conf, save_location): #level, datetime
    sigma_levels = conf['visualization']['sigma_level_visualize']['visualize_levels']
    for level_idx, sigma_level in enumerate(sigma_levels):
        level_image_filenames = [filename_list[level_idx] for filename_list in filenames]

        ## send all png files to the gif maker 
        gif_name = '{}_level{:02d}.gif'.format(gif_name_prefix, sigma_level)
        command_str = f'convert -delay 20 -loop 0 {" ".join(level_image_filenames)} {save_location}/{gif_name}'
        out = subprocess.Popen(command_str, shell=True, 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE).communicate()
        print(out)

'''
    # =============================================================================== #
    # Data visualization for diagnostics
    N_vars = len(conf['visualization']['diagnostic_variable_visualize']['variable_indices'])
    
    if N_vars > 0:
    
        # collect forecast outputs
        forecast_paths = os.path.join(save_location, f"{forecast_count}_*_*_pred.npy")
        # generator of file_name, file_count
        file_list = enumerate(sorted(glob.glob(forecast_paths)))
        
        # the output session begins
        print('Preparing diagnostic outputs')
        video_files = []
        
        with Pool(processes=8) as pool:
            f = partial(draw_diagnostics, conf=conf, times=forecast_datetimes,
                        forecast_count=forecast_count, save_location=save_location)
            # collect output png file names
            video_files = pool.map(f, file_list)
            
        ## collect all png file names
        ## video_files[0] = f; video_files[1] = file_names
        video_files_all = [x[1] for x in sorted(video_files)]
        
        ## gif outpout name = png output name
        gif_name_prefix = conf['visualization']['diagnostic_variable_visualize']['file_name_prefix']
        
        ## send all png files to the gif maker 
        gif_name = '{}.gif'.format(gif_name_prefix)
        command_str = f'convert -delay 20 -loop 0 {" ".join(video_files_all)} {save_location}/{gif_name}'
        out = subprocess.Popen(command_str, shell=True, 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE).communicate()
        print(out)

    # =============================================================================== #
    # Data visualization for diagnostics
    N_vars = len(conf['visualization']['surface_visualize']['variable_indices'])
    
    if N_vars > 0:
    
        # collect forecast outputs
        forecast_paths = os.path.join(save_location, f"{forecast_count}_*_*_pred.npy")
        # generator of file_name, file_count
        file_list = enumerate(sorted(glob.glob(forecast_paths)))
        
        # the output session begins
        print('Preparing surface outputs')
        video_files = []
        
        with Pool(processes=8) as pool:
            f = partial(draw_surface, conf=conf, times=forecast_datetimes,
                        forecast_count=forecast_count, save_location=save_location)
            # collect output png file names
            video_files = pool.map(f, file_list)
            
        ## collect all png file names
        ## video_files[0] = f; video_files[1] = file_names
        video_files_all = [x[1] for x in sorted(video_files)]
        
        ## gif outpout name = png output name
        gif_name_prefix = conf['visualization']['surface_visualize']['file_name_prefix']
        
        ## send all png files to the gif maker 
        gif_name = '{}.gif'.format(gif_name_prefix)
        command_str = f'convert -delay 20 -loop 0 {" ".join(video_files_all)} {save_location}/{gif_name}'
        out = subprocess.Popen(command_str, shell=True, 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE).communicate()
        print(out)
        
    forecast_count += 1
    metrics_results = defaultdict(list)
    pred_files = []
    # true_files = []
'''

if __name__ == "__main__":

    description = "Rollout AI-NWP forecasts"
    parser = ArgumentParser(description=description)
    # -------------------- #
    # parser args: -c, -l, -w
    parser.add_argument("-c", dest="model_config", type=str, default=False,
                        help="Path to the model configuration (yml) containing your inputs.",)
    
    parser.add_argument("-l", dest="launch", type=int, default=0,
                        help="Submit workers to PBS.",)
    
    parser.add_argument("-w", "--world-size", type=int, default=4,
                        help="Number of processes (world size) for multiprocessing")
    # parse
    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("model_config")
    launch = int(args_dict.pop("launch"))

    # Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Load the configuration and get the relevant variables
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    x_save_loc = conf['predict']['x_save_loc']
    surf_save_loc = conf['predict']['surf_save_loc']

    # if xarray already exists, just make images and movies
    if x_save_loc and surf_save_loc:
        print(f'making image from xarray files:\n{x_save_loc}\nsurf_save_loc\n')
        dir = make_images_from_xarray(x_save_loc, surf_save_loc, conf)
        #make_movie(dir, conf)
        sys.exit()

    # Launch PBS jobs
    if launch:
        # Where does this script live?
        script_path = Path(__file__).absolute()
        if conf['pbs']['queue'] == 'casper':
            logging.info("Launching to PBS on Casper")
            launch_script(config, script_path)
        else:
            logging.info("Launching to PBS on Derecho")
            launch_script_mpi(config, script_path)
        sys.exit()

#     wandb.init(
#         # set the wandb project where this run will be logged
#         project="Derecho parallelism",
#         name=f"Worker {os.environ["RANK"]} {os.environ["WORLD_SIZE"]}"
#         # track hyperparameters and run metadata
#         config=conf
#     )

    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)
    
    with Pool(processes=4) as pool:
        if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
            x_dataArrs, surf_dataArrs, pool_jobs, img_save_loc = predict(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), conf, pool)
        else:
            x_dataArrs, surf_dataArrs, pool_jobs, img_save_loc = predict(0, 1, conf, pool)
        # save out to file
        x_save_loc, surf_save_loc = save_netcdf(x_dataArrs, surf_dataArrs, conf)
        
        print('waiting for all image files to write before making movies')
        print(f'num pool jobs: {len(pool_jobs)}')

        # now check if everything was successful
        try:
            print("\n filepaths written")
            print([res.get() for res in pool_jobs])
        except:
            print("\n errors:")
            print([res._value for res in pool_jobs])
            raise
                

    #write make_movie to use directory as arg so its reuseable
    #make_movie(img_save_loc, conf)