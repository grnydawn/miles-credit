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
from visualization import draw_upper_air

# import wandb

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def setup(rank, world_size, mode):
    logging.info(f"Running {mode.upper()} on rank {rank} with world_size {world_size}.")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

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

                # =============================================================================== #
                # Data visualization for upper air variables
                upper_air_levels = conf['visualization']['upper_air_visualize']['visualize_levels']
                
                ## parallelize draw_forecast func
                with Pool(processes=8) as pool:
                    f = partial(draw_upper_air, conf=conf, times=forecast_datetimes,
                                forecast_count=forecast_count, save_location=save_location)
                    # collect output png file names
                    video_files = pool.map(f, file_list)
                    
                ## collect all png file names
                ## video_files[0] = f; video_files[1] = file_names
                video_files_all = [x[1] for x in sorted(video_files)]
                
                ## gif outpout name = png output name
                gif_name_prefix = conf['visualization']['upper_air_visualize']['file_name_prefix']
                
                ## separate file names based on verticial levels
                ## one gif per level 
                for n_gif, upper_air_level in enumerate(upper_air_levels):
                    video_files_signle = []
                    for filename_list in video_files_all:
                        video_files_signle.append(filename_list[n_gif])
                        
                    ## send all png files to the gif maker 
                    gif_name = '{}_level{:02d}.gif'.format(gif_name_prefix, upper_air_level)
                    command_str = f'convert -delay 20 -loop 0 {" ".join(video_files_signle)} {save_location}/{gif_name}'
                    out = subprocess.Popen(command_str, shell=True, 
                                           stdout=subprocess.PIPE, 
                                           stderr=subprocess.PIPE).communicate()
                    print(out)
                # =============================================================================== #
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
