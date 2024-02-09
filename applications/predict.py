import warnings
import torch
import torch.distributed as dist
from argparse import ArgumentParser
from pathlib import Path

import torch.fft
import logging
import shutil

import wandb
import glob
import os
import sys
import yaml

from torch.distributed.fsdp import StateDictType
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from credit.vit2d import ViT2D
from credit.rvt import RViT
from credit.loss import VariableTotalLoss2D
from credit.data import ToTensor, NormalizeState, PredictForecast
from credit.metrics import LatWeightedMetrics
from credit.pbs import launch_script, launch_script_mpi
from credit.seed import seed_everything
import xarray as xr
from collections import defaultdict
import numpy as np
import gc

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from os.path import join
import datetime
import subprocess
from multiprocessing import Pool
from functools import partial
import pandas as pd


warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def setup(rank, world_size, mode):
    logging.info(f"Running {mode.upper()} on rank {rank} with world_size {world_size}.")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def draw_forecast(data, conf=None, times=None, forecast_count=None, save_location=None):
    k, fn = data
    lat_lon_weights = xr.open_dataset(conf['loss']['latitude_weights'])
    means = xr.open_dataset(conf['data']['mean_path'])
    sds = xr.open_dataset(conf['data']['std_path'])
    pred = np.load(fn)
    t = times[k]
    pred = pred[0, 59, :, :]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.EckertIII())
    ax.set_global()
    ax.coastlines('110m', alpha=0.5)

    pout = ax.pcolormesh(
        lat_lon_weights["longitude"],
        lat_lon_weights["latitude"],
        (pred * sds["Q"].values + means["Q"].values) * 1000,
        transform=ccrs.PlateCarree(),
        vmin=0,
        vmax=20,
        cmap='RdBu'
    )
    plt.colorbar(pout, ax=ax, orientation="horizontal", fraction=0.05, pad=0.01)
    plt.title(f"Q (g/kg) D ({t}) H ({k})")
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
    time_step = conf["data"]["time_step"]

    # Load paths to all ERA5 data available
    all_ERA_files = sorted(glob.glob(conf["data"]["save_loc"]))

    # Preprocessing transformations
    transform = transforms.Compose([
        NormalizeState(conf["data"]["mean_path"],conf["data"]["std_path"]),
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
    if 'use_rotary' in conf['model'] and conf['model']['use_rotary']:
        model = RViT.load_model(conf).to(device)
    else:
        if 'use_rotary' in conf['model']:
            del conf['model']['use_rotary']
            del conf['model']['use_ds_conv']
            del conf['model']['use_glu']
        model = ViT2D.load_model(conf).to(device)

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

            if forecast_hour == 0:
                # Initialize x and x_surf with the first time step
                x_atmo = batch["x"].squeeze(1)
                x_surf = batch["x_surf"].squeeze(1)
                x = model.concat_and_reshape(x_atmo, x_surf).to(device)
            else:
                x = y_pred.detach()

            y_atmo = batch["y"].squeeze(1)
            y_surf = batch["y_surf"].squeeze(1)
            y = model.concat_and_reshape(y_atmo, y_surf).to(device)

            # Predict

            y_pred = model(x)

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
            np.save(os.path.join(save_location, f"{forecast_count}_{date_time}_{forecast_hour}_pred.npy"), y_pred.cpu())
            #np.save(os.path.join(save_location, f"{forecast_count}_{date_time}_{forecast_hour}_true.npy"), y.cpu())

            pred_files.append(os.path.join(save_location, f"{forecast_count}_{date_time}_{forecast_hour}_pred.npy"))
            #true_files.append(os.path.join(save_location, f"{forecast_count}_{date_time}_{forecast_hour}_true.npy"))

            # Explicitly release GPU memory
            torch.cuda.empty_cache()
            gc.collect()

            # Make a movie
            if batch['stop_forecast'][0]:

                df = pd.DataFrame(metrics_results)
                df.to_csv(os.path.join(save_location, "metrics.csv"))

                video_files = []
                forecast_paths = os.path.join(save_location, f"{forecast_count}_*_*_pred.npy")
                file_list = enumerate(sorted(glob.glob(forecast_paths)))

                with Pool(processes=8) as pool:
                    f = partial(
                        draw_forecast,
                        conf=conf,
                        times=forecast_datetimes,
                        forecast_count=forecast_count,
                        save_location=save_location
                    )
                    video_files = pool.map(f, file_list)

                # for k, fn in enumerate(sorted(glob.glob(forecast_paths))):
                #     pred = np.load(fn)
                #     t = forecast_datetimes[k]
                #     pred = pred[0, 59, :, :]
                #     fig = plt.figure(figsize=(10, 8))
                #     ax = fig.add_subplot(1, 1, 1, projection=ccrs.EckertIII())
                #     ax.set_global()
                #     ax.coastlines('110m', alpha=0.5)

                #     pout = ax.pcolormesh(
                #         lat_lon_weights["longitude"],
                #         lat_lon_weights["latitude"],
                #         (pred * sds["Q"].values + means["Q"].values) * 1000,
                #         transform=ccrs.PlateCarree(),
                #         vmin=0,
                #         vmax=20,
                #         cmap='RdBu'
                #     )
                #     plt.colorbar(pout, ax=ax, orientation="horizontal", fraction=0.05, pad=0.01)
                #     plt.title(f"Q (g/kg) F ({t})")
                #     video_files.append(join(f"{save_location}", f"global_q_{forecast_count}_{k}.png"))
                #     plt.savefig(join(f"{save_location}", f"global_q_{forecast_count}_{k}.png"), dpi=300, bbox_inches="tight")
                #     plt.close()

                video_files = [x[1] for x in sorted(video_files)]
                command_str = f'convert -delay 20 -loop 0 {" ".join(video_files)} {save_location}/global_q_fixed.gif'
                out = subprocess.Popen(command_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
                print(out)

                forecast_count += 1
                metrics_results = defaultdict(list)
                pred_files = []
                true_files = []


if __name__ == "__main__":

    description = "Rollout AI-NWP forecasts"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        dest="model_config",
        type=str,
        default=False,
        help="Path to the model configuration (yml) containing your inputs.",
    )
    parser.add_argument(
        "-l",
        dest="launch",
        type=int,
        default=0,
        help="Submit workers to PBS.",
    )
    parser.add_argument(
        "-w",
        "--world-size",
        type=int,
        default=4,
        help="Number of processes (world size) for multiprocessing"
    )
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
