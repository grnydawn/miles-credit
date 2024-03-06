'''
Credit model output summary
------------------------------
Output tensor size: (variables, latitude, longitude)
Variables = top-of-atmosphere variables --> near-surface variables --> single layer variables

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
# Graph
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import cartopy.crs as ccrs
import cartopy.mpl.geoaxes
import cartopy.feature as cfeature

# import wandb

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def setup(rank, world_size, mode):
    logging.info(f"Running {mode.upper()} on rank {rank} with world_size {world_size}.")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def draw_forecast(data, N_level=15, level_num=10, var_num=4, 
                  conf=None, times=None, forecast_count=None, save_location=None):
    '''
    This function produces 4-panel figures 
    '''
    # ------------------------------ #
    # visualization settings
    ## variable rage limit with units of m/s, m/s, K, g/kg
    var_lims = [[-20, 20], [-20, 20], [273.15-35, 273.15+35], [0, 1e-2]]
    ## colormap
    colormaps = [plt.cm.Spectral, plt.cm.Spectral, plt.cm.RdBu_r, plt.cm.YlGn]
    ## colorbar extend
    colorbar_extends = ['both', 'both', 'both', 'max']
    ## title
    title_strings = ['U wind [m/s]\ntime: {}; step: {}', 
                     'V wind [m/s]\ntime: {}; step: {}', 
                     'Air temperature [K$^\circ$]\ntime: {}; step: {}', 
                     'Specific humidity [g/kg]\ntime: {}; step: {}']
    # ------------------------------ #
    # get timestep and filename
    k, fn = data
    t = times[k]
    pred = np.load(fn)
    pred = pred[45]
    # ------------------------------ #
    # get lat/lon grids
    lat_lon_weights = xr.open_dataset(conf['loss']['latitude_weights'])
    longitude = lat_lon_weights["longitude"]
    latitude = lat_lon_weights["latitude"]

    # Figure
    fig = plt.figure(figsize=(13, 6.5))
    
    # 2-by-2 subplots
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
    proj_ = ccrs.EckertIII()
    
    # subplot ax
    ax0 = plt.subplot(gs[0, 0], projection=proj_)
    ax1 = plt.subplot(gs[0, 1], projection=proj_)
    ax2 = plt.subplot(gs[1, 0], projection=proj_)
    ax3 = plt.subplot(gs[1, 1], projection=proj_)
    AX = [ax0, ax1, ax2, ax3]
    # panel gaps
    plt.subplots_adjust(0, 0, 1, 1, hspace=0.2, wspace=0.05)
    
    # lat/lon gridlines and labeling
    for ax in AX:
        GL = ax.gridlines(crs=ccrs.PlateCarree(), 
                          draw_labels=True, x_inline=False, y_inline=False, 
                          color='k', linewidth=0.5, linestyle=':', zorder=5)
        GL.top_labels = None; GL.bottom_labels = None
        GL.right_labels = None; GL.left_labels = None
        GL.xlabel_style = {'size': 14}; GL.ylabel_style = {'size': 14}
        GL.rotate_labels = False
    
        ax.add_feature(cfeature.COASTLINE.with_scale('110m'), edgecolor='k', linewidth=1.0, zorder=5)
        ax.spines['geo'].set_linewidth(2.5)
    
    CBar_collection = []
    
    for i_var in range(var_num):
        # get the current axis
        ax = AX[i_var]
        # get the current variable
        var_ind = i_var*N_level + level_num
        pred_draw = pred[var_ind]
        # get visualization settings
        var_lim = var_lims[i_var]
        colormap = colormaps[i_var]
        # pcolormesh
        cbar = ax.pcolormesh(longitude, latitude, pred_draw, vmin=var_lim[0], vmax=var_lim[1], 
                             cmap=colormap, transform=ccrs.PlateCarree())
        # colorbar operations
        CBar_collection.append(cbar)
        CBar = fig.colorbar(cbar, location='right', orientation='vertical', 
                            pad=0.02, fraction=0.025, shrink=0.6, aspect=15, extend=colorbar_extends[i_var], ax=ax)
        CBar.ax.tick_params(axis='y', labelsize=14, direction='in', length=0)
        CBar.outline.set_linewidth(2.5)
        # title
        ax.set_title(title_strings[i_var].format(t, k), fontsize=14)
    
    filename = join(save_location, f"global_q_{forecast_count}_{k}.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    return k, filename

def predict(rank, world_size, conf):

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

    # create save directory for numpy arrays
    save_location = os.path.join(conf['save_loc'], "forecasts")
    os.makedirs(save_location, exist_ok=True)

    # Rollout
    with torch.no_grad():

        forecast_count = 0
        forecast_datetimes = []
        pred_files, true_files = [], []

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

            formatted_datetime = datetime.datetime.utcfromtimestamp(date_time).strftime('%Y-%m-%d %H:%M:%S')
            forecast_datetimes.append(formatted_datetime)

            print_str = f"Forecast: {forecast_count} "
            print_str += f"Date: {formatted_datetime} "
            print_str += f"Hour: {batch['forecast_hour'].item()} "
            print_str += f"MAE: {mae.item()} "
            print_str += f"ACC: {metrics_dict['acc']}"
            print(print_str)

            # Save as numpy arrays for now
            # save_arr = state_transformer.inverse_transform(y_pred.cpu()).squeeze(0)
            np.save(os.path.join(save_location, f"{forecast_count}_{date_time}_{forecast_hour}_pred.npy"), y_pred.squeeze(0))
            # np.save(os.path.join(save_location, f"{forecast_count}_{date_time}_{forecast_hour}_true.npy"), y.cpu())

            pred_files.append(os.path.join(save_location, f"{forecast_count}_{date_time}_{forecast_hour}_pred.npy"))
            # true_files.append(os.path.join(save_location, f"{forecast_count}_{date_time}_{forecast_hour}_true.npy"))

            # Explicitly release GPU memory
            torch.cuda.empty_cache()
            gc.collect()

            # Make a movie
            if batch['stop_forecast'][0]:

                df = pd.DataFrame(metrics_results)
                df.to_csv(os.path.join(save_location, "metrics.csv"))

                video_files = []
                # collect forecast outputs
                forecast_paths = os.path.join(save_location, f"{forecast_count}_*_*_pred.npy")
                # generator of file_name, file_count
                file_list = enumerate(sorted(glob.glob(forecast_paths)))
                # parallelize draw_forecast func
                with Pool(processes=8) as pool:
                    f = partial(draw_forecast, conf=conf, times=forecast_datetimes,
                                forecast_count=forecast_count, save_location=save_location)
                    # collect output png file names
                    video_files = pool.map(f, file_list)
                    
                # generate gif
                video_files = [x[1] for x in sorted(video_files)]
                command_str = f'convert -delay 20 -loop 0 {" ".join(video_files)} {save_location}/global.gif'
                
                out = subprocess.Popen(command_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
                print(out)
                
                forecast_count += 1
                metrics_results = defaultdict(list)
                pred_files = []
                # true_files = []


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

    # Create directories if they do not exist and copy yml file
    os.makedirs(conf["save_loc"], exist_ok=True)
    if not os.path.exists(os.path.join(conf["save_loc"], "model.yml")):
        shutil.copy(config, os.path.join(conf["save_loc"], "model.yml"))

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

    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        predict(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), conf)
    else:
        predict(0, 1, conf)
