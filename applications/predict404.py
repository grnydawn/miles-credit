# ---------- #
# System
import gc
import os
import sys
import yaml
import glob
import copy
import logging
import warnings
import functools
import subprocess
from pathlib import Path
from functools import partial
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
from torchvision import transforms

# ---------- #
# credit
from credit.data404 import CONUS404Dataset
from credit.loss import VariableTotalLoss2D
from credit.models import load_model
from credit.transforms404 import ToTensor, NormalizeState
from credit.seed import seed_everything
from credit.pbs import launch_script, launch_script_mpi
from credit.mixed_precision import parse_dtype

# ---------- #

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def load_model_state(conf, model, device):
    save_loc = os.path.expandvars(conf['save_loc'])
    #  Load optimizer, gradient scaler, and learning rate scheduler
    #  Optimizer must come after wrapping model using FSDP
    ckpt = os.path.join(save_loc, "checkpoint.pt")
    checkpoint = torch.load(ckpt, map_location=device)
    if conf["trainer"]["mode"] == "fsdp":
        logging.info(f"Loading FSDP model, optimizer, grad scaler, and"
                     f"learning rate scheduler states from {save_loc}")
        checkpoint_io = TorchFSDPCheckpointIO()
        checkpoint_io.load_unsharded_model(model, os.path.join(save_loc, "model_checkpoint.pt"))
    else:
        if conf["trainer"]["mode"] == "ddp":
            logging.info(f"Loading DDP model, optimizer, grad scaler, and"
                         f"learning rate scheduler states from {save_loc}")
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            logging.info(f"Loading model, optimizer, grad scaler, and "
                         f"learning rate scheduler states from {save_loc}")
            model.load_state_dict(checkpoint["model_state_dict"])
    return model



def predict(rank, world_size, conf):

    # infer device id from rank
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        device = torch.device("cpu")

    # Config settings
    #seed = 1000 if "seed" not in conf else conf["seed"]
    #seed_everything(seed)

    #history_len = conf["data"]["history_len"]
    #forecast_len = conf["data"]["forecast_len"]
    #time_step = conf["data"]["time_step"] if "time_step" in conf["data"] else None

    # Preprocessing transformations
    if conf["data"]["scaler_type"] == "std":
        state_transformer = NormalizeState(conf)
    else:
        state_transformer = NormalizeState_Quantile(conf)
    transform = transforms.Compose(
        [
            state_transformer,
            ToTensor(conf),
        ]
    )

    dataset = CONUS404Dataset(
        varnames = conf["data"]["variables"],
        history_len = conf["data"]["history_len"],
        forecast_len = conf["data"]["forecast_len"],
        transform=transform,
        start = conf["predict"]["start"],
        finish = conf["predict"]["finish"]
        )

    # set up the dataloader for this process
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
    )

    # load model
    model = load_model(conf, load_weights=True).to(device)

    # switch to evaluation mode
    model.eval()

    # Rollout
    with torch.no_grad():

        xarraylist = []
                
        # model inference loop
        for index, batch in enumerate(data_loader):

            xin = batch["x"].to(device)
            yout = model(xin)
            y = state_transformer.inverse_transform(yout.cpu())

            rawdata = dataset.get_data(index, do_transform=False)            
            t = rawdata["y"].coords[dataset.tdimname]

            outdims = ["Time","vars","z","y","x"]  ##todo: squeeze bottom_top
            xarr = xr.DataArray(y, dims=outdims,
                                coords=dict(vars=dataset.varnames, Time=t))
            #todo: use dataset.tdimname
            
            xarraylist.append(xarr)
            
            ## rollout: xin will be a stack of previous timesteps,
            ## at each timestep, drop oldest & add newest, predict next
            ## also have to use the first N timesteps as input-only;
            ## first prediction is timestep N+1            

            ## initialization on the first forecast hour
            #if forecast_hour == 1:
            #    # Initialize x and x_surf with the first time step
            #    x = model.concat_and_reshape(batch["x"], batch["x_surf"]).to(device)
            #
            #    # setup save directory for images
            #    init_time = datetime.datetime.utcfromtimestamp(date_time).strftime(
            #        "%Y-%m-%dT%HZ"
            #    )
            #    img_save_loc = os.path.join(
            #        os.path.expandvars(conf["save_loc"]),
            #        f"forecasts/images_{init_time}",
            #    )
            #    if N_vars > 0:
            #        os.makedirs(img_save_loc, exist_ok=True)

            ## Update the input
            ## setup for next iteration, transform to z-space and send to device
            #y_pred = state_transformer.transform_array(y_pred).to(device)
            #
            #if history_len == 1:
            #    x = y_pred.detach()
            #else:
            #    # use multiple past forecast steps as inputs
            #    static_dim_size = abs(x.shape[1] - y_pred.shape[1])  # static channels will get updated on next pass
            #    x_detach = x[:, :-static_dim_size, 1:].detach()
            #    x = torch.cat([x_detach, y_pred.detach()], dim=2)
            #
            ## Explicitly release GPU memory
            #torch.cuda.empty_cache()
            #gc.collect()
            

    if distributed:
        torch.distributed.barrier()


    return xarraylist

if __name__ == "__main__":

    description = "Rollout AI-NWP forecasts"
    parser = ArgumentParser(description=description)
    # -------------------- #
    # parser args: -c, -l, -w
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
        help="Number of processes (world size) for multiprocessing",
    )

    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default=0,
        help="Update the config to use none, DDP, or FSDP",
    )

    parser.add_argument(
        "-nd",
        "--no-data",
        type=str,
        default=0,
        help="If set to True, only pandas CSV files will we saved for each forecast",
    )

    # parse
    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("model_config")
    launch = int(args_dict.pop("launch"))
    mode = str(args_dict.pop("mode"))
    no_data = 0 if "no-data" not in args_dict else int(args_dict.pop("no-data"))

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

    # Update config using override options
    if mode in ["none", "ddp", "fsdp"]:
        logger.info(f"Setting the running mode to {mode}")
        conf["trainer"]["mode"] = mode

    # Launch PBS jobs
    if launch:
        # Where does this script live?
        script_path = Path(__file__).absolute()
        if conf["pbs"]["queue"] == "casper":
            logging.info("Launching to PBS on Casper")
            launch_script(config, script_path)
        else:
            logging.info("Launching to PBS on Derecho")
            launch_script_mpi(config, script_path)
        sys.exit()


    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    xarraylist = predict(
        rank = int(os.environ["RANK"]),
        world_size = int(os.environ["WORLD_SIZE"]),
        conf = conf
    )

    xcat = xr.concat(xarraylist, dim="Time")
    ds_out = xcat.to_dataset(dim="vars")  ##todo: move to_dataset inside loop

    sep = "."
    filename = sep.join([os.path.basename(conf["save_loc"]),
                         "C404",
                         conf["predict"]["start"],
                         conf["predict"]["finish"],
                         "nc"])
    save_path = os.path.join(conf["save_loc"], filename)
    
    ds_out.to_netcdf(path=save_path,
                     format="NETCDF4",
                     engine="netcdf4",
                     encoding={v: {"zlib":True,
                                   "complevel": 1,
                                   "dtype":"float"} for v in conf["data"]["variables"]},
                     unlimited_dims="Time",
                     compute=True
                     )
        
