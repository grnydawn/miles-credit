import torch
import yaml
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys

from credit.vit2d import ViT2D
from credit.rvt import RViT
from credit.data import ERA5Dataset, ToTensor, NormalizeState, NormalizeTendency, DistributedSequentialDataset
from torchvision import transforms
from credit.metrics import LatWeightedMetrics
from credit.loss import VariableTotalLoss2D
import gc
from math import ceil, floor


from collections import defaultdict
from credit.metrics import anomaly_correlation_coefficient as ACC


#### Plotting.
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
from os.path import exists, join
import os

########################## SETTINGS #####################################
# Define settings arguments you can pass to the script
#self.allvarsdo = ['U'(0-14),'V' (15-29),
#'T'(30-44),'Q'(45-59),'SP'(60),'t2m'(61),
#'V500'(62),'U500'(63),'T500'(64),'Z500'(65),
#'Q500'(66)]
import argparse

parser = argparse.ArgumentParser(description='Script settings')
parser.add_argument('--Dovar', type=int, default=59, help='Value of Dovar')
parser.add_argument('--save_loc', type=str, default="/glade/work/schreck/repos/global/miles-credit/results/rvt_pe", help='Save location')
parser.add_argument('--config_file', type=str, default=None, help='Config file location')
args = parser.parse_args()

# If config_file argument is not provided, use the default location
config_file = args.config_file if args.config_file else f"{args.save_loc}/model.yml"
########################## SETTINGS #####################################
device = "cuda:0"

with open(config_file) as cf:
    conf = yaml.load(cf, Loader=yaml.FullLoader)


# Define a dictionary mapping Dovar values to varvar values
varvar_mapping = {
    0: 'U',
    1: 'U',
    2: 'U',
    3: 'U',
    4: 'U',
    5: 'U',
    6: 'U',
    7: 'U',
    8: 'U',
    9: 'U',
    10: 'U',
    11: 'U',
    12: 'U',
    13: 'U',
    14: 'U',
    15: 'V',
    16: 'V',
    17: 'V',
    18: 'V',
    19: 'V',
    20: 'V',
    21: 'V',
    22: 'V',
    23: 'V',
    24: 'V',
    25: 'V',
    26: 'V',
    27: 'V',
    28: 'V',
    29: 'V',
    30: 'T',
    31: 'T',
    32: 'T',
    33: 'T',
    34: 'T',
    35: 'T',
    36: 'T',
    37: 'T',
    38: 'T',
    39: 'T',
    40: 'T',
    41: 'T',
    42: 'T',
    43: 'T',
    44: 'T',
    45: 'Q',
    46: 'Q',
    47: 'Q',
    48: 'Q',
    49: 'Q',
    50: 'Q',
    51: 'Q',
    52: 'Q',
    53: 'Q',
    54: 'Q',
    55: 'Q',
    56: 'Q',
    57: 'Q',
    58: 'Q',
    59: 'Q',
    60: 'SP',
    61: 't2m',
    62: 'V500',
    63: 'U500',
    64: 'T500',
    65: 'Z500',
    66: 'Q500',
}

# Check if Dovar is within the valid range
if args.Dovar < 0 or args.Dovar > 66:
    print('Error: Dovar must be between 0 and 66')
    print('[U(0-14),V (15-29),T(30-44),Q(45-59),SP(60),t2m(61),')
    print('V500(62),U500(63),T500(64),Z500(65),Q500(66)]')
    sys.exit(1)  # Exit with a non-zero status code to indicate an error

varvar = varvar_mapping[args.Dovar]
print(f'varvar: {varvar}')
Dovar = args.Dovar

if __name__ == "__main__":

    history_len = 23
    forecast_len = 24
    time_step = 1
        
    # datasets (zarr reader) 
    all_ERA_files = sorted(glob.glob(conf["data"]["save_loc"]))
        
    # Specify the years for each set
    test_years = [str(year) for year in range(2018, 2022)] # same as graphcast -- always hold out
    
    # Filter the files for each set (2018-2019)
    test_files = [file for file in all_ERA_files if any(year in file for year in test_years)][:2]
    
    transform = transforms.Compose([
        NormalizeState(conf["data"]["mean_path"],conf["data"]["std_path"]),
        ToTensor(history_len=history_len, forecast_len=forecast_len)
    ])
    
    test_dataset = DistributedSequentialDataset(
        filenames=test_files,
        history_len=history_len,
        forecast_len=forecast_len,
        skip_periods=time_step,
        transform=transform,
        rank=0,
        world_size=1,
        shuffle=True
    )
    
    # setup the dataloder for this process
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=0,
        drop_last=True)
    
    if 'use_rotary' in conf['model'] and conf['model']['use_rotary']:
        model = RViT.load_model(conf).to(device)
    else:
        if 'use_rotary' in conf['model']:
            del conf['model']['use_rotary']
            del conf['model']['use_ds_conv']
            del conf['model']['use_glu']
        model = ViT2D.load_model(conf).to(device)
    
    model.eval()
    results = defaultdict(list)
    metrics_results = defaultdict(list)
    
    history_len = 3
    with torch.no_grad(): 
    
        true_arrays = []  # List to store true arrays
        pred_arrays = []  # List to store predicted arrays
        
        #loss_fn = torch.nn.L1Loss()
        loss_fn = VariableTotalLoss2D(conf, validation=True)
        metrics = LatWeightedMetrics(conf)
    
        loss = 0.0
        for batch in test_loader:
    
            if batch["forecast_hour"].item() == 0:
                # Initialize x and x_surf with the first time step
                x_atmo = batch["x"].squeeze(1)
                x_surf = batch["x_surf"].squeeze(1)
                x = model.concat_and_reshape(x_atmo, x_surf).to(device)
    
            y_atmo = batch["y"].squeeze(1)
            y_surf = batch["y_surf"].squeeze(1)
            y = model.concat_and_reshape(y_atmo, y_surf).to(device)
            
            # The model's output y_pred becomes the new x for the next time step
            y_pred = model(x)
            x = y_pred.detach()
    
            mae = loss_fn(y, y_pred)
            metrics_dict = metrics(y_pred.float(), y.float())
    
            for k, m in metrics_dict.items():
                metrics_results[k].append(m.item())
    
            loss += mae
    
            true_arrays.append(y[0, Dovar, :, :].to('cpu').numpy())
            pred_arrays.append(y_pred[0, Dovar, :, :].to('cpu').numpy())
    
            results["forecast_hour"].append(batch["forecast_hour"].item())
            results["mae"].append(mae.item())
            results["acc"].append(ACC(y, y_pred))
    
            print(batch["index"], batch["forecast_hour"], mae, results["acc"][-1])
    
            #del y_pred, y
            # Explicitly release GPU memory
            torch.cuda.empty_cache()
            gc.collect()
    
            if batch["forecast_hour"].item() == 23:
                break
    
    # Convert the lists to NumPy arrays
    true_arrays = np.array(true_arrays)
    pred_arrays = np.array(pred_arrays)
    np.save('true_arrays.npy', true_arrays)
    np.save('pred_arrays.npy', pred_arrays)
    
    preds = np.load("./pred_arrays.npy")
    true = np.load("./true_arrays.npy")
    lat_lon_weights = xr.open_dataset('/glade/u/home/wchapman/MLWPS/DataLoader/static_variables_ERA5_zhght.nc')
    means = xr.open_dataset("/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/All_2010_staged.SLO.mean.nc")
    sds = xr.open_dataset("/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/All_2010_staged.SLO.std.nc")
    
    out_dir = "pred_images"
    if not exists(out_dir):
        os.makedirs(out_dir)

    vvmin = np.round(np.min((preds[0] * sds[varvar].values + means[varvar].values)),4)
    vvmax = np.round(np.max((preds[0] * sds[varvar].values + means[varvar].values)),4)

    if vvmax>1:
        vvmax=ceil(vvmax)

    if vvmin<-1:
        vvmin=floor(vvmin)
    
    for t in range(23):
        print(t)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.EckertIII())
        ax.set_global()
        ax.coastlines('110m', alpha=0.5)
        
        pout = ax.pcolormesh(lat_lon_weights["longitude"], 
                     lat_lon_weights["latitude"], (preds[t] * sds[varvar].values + means[varvar].values), 
                             transform=ccrs.PlateCarree(), vmin=vvmin, vmax=vvmax,cmap='RdBu')
        plt.colorbar(pout, ax=ax, orientation="horizontal", fraction=0.05, pad=0.01)
        plt.title(f"{varvar} F{t:02d}")
        plt.savefig(join(f"{out_dir}", f"global_{varvar}{Dovar}_{t:02d}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    print('run this command:')
    print('run this command:')
    print('run this command:')
    print('run this command:')
    print('run this command:')
    print('run this command:')
    print(f'convert -delay 20 -loop 0 pred_images/global_{varvar}{Dovar}_*.png global_{varvar}{Dovar}_fixed.gif')
    print(f'convert -delay 20 -loop 0 pred_images/global_{varvar}{Dovar}_*.png global_{varvar}{Dovar}_fixed.gif')
    print(f'convert -delay 20 -loop 0 pred_images/global_{varvar}{Dovar}_*.png global_{varvar}{Dovar}_fixed.gif')
    print(f'convert -delay 20 -loop 0 pred_images/global_{varvar}{Dovar}_*.png global_{varvar}{Dovar}_fixed.gif')
    print('did you run that command?')
    print('did you run that command?')
    print('did you run that command?')
    
