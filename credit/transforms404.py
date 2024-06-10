import torch
import logging
import numpy as np
import xarray as xr
from credit.data import Sample
from typing import Dict
from torchvision import transforms as tforms


logger = logging.getLogger(__name__)


## Not currently used
#def load_transforms(conf):
#    if conf["data"]["scaler_type"] == 'std':
#        transform_scaler = NormalizeState(conf)
#    else:
#        logger.log('scaler type not supported check data: scaler_type in config file')
#        raise
#
#    to_tensor_scaler = ToTensor(conf=conf)
#
#    return tforms.Compose([
#            transform_scaler,
#            to_tensor_scaler,
#        ])


class NormalizeState:
    def __init__(
        self,
        conf
    ):
        self.mean_ds = xr.open_dataset(conf['data']['mean_path'])
        self.std_ds = xr.open_dataset(conf['data']['std_path'])
        self.variables = conf['data']['variables']

        logger.info("Loading preprocessing object for transform/inverse transform states into z-scores")

    def __call__(self, sample: Sample, inverse: bool = False) -> Sample:
        if inverse:
            return self.inverse_transform(sample)
        else:
            return self.transform(sample)

    #def transform_array(self, x: torch.Tensor) -> torch.Tensor:
    #    device = x.device
    #    tensor = x[:, :len(self.variables), :, :]
    #
    #    # Reverse z-score normalization using the pre-loaded mean and std
    #    transformed_tensor = tensor.clone()
    #    k = 0
    #    for name in self.variables:
    #        for level in range(self.levels):
    #            mean = self.mean_ds[name].values[level]
    #            std = self.std_ds[name].values[level]
    #            transformed_tensor[:, k] = (tensor[:, k] - mean) / std
    #            k += 1
    #
    #    return transformed_tensor.to(device)

    def transform(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        normalized_sample = {}
        for key, value in sample.items():
            if isinstance(value, xr.Dataset):
                normalized_sample[key] = (value - self.mean_ds) / self.std_ds
        return normalized_sample

    
    #def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
    #    device = x.device
    #    tensor = x[:, :len(self.variables), :, :]
    #
    #    # Reverse z-score normalization using the pre-loaded mean and std
    #    transformed_tensor = tensor.clone()
    #    k = 0
    #    for name in self.variables:
    #        for level in range(self.levels):
    #            mean = self.mean_ds[name].values[level]
    #            std = self.std_ds[name].values[level]
    #            transformed_tensor[:, k] = tensor[:, k] * std + mean
    #            k += 1
    #
    #    return transformed_tensor.to(device)
    
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        # Reverse z-score normalization using the pre-loaded mean and std
        xbar = x[:,:len(self.variables),:,:,:] # this may be unnecessary
        result = xbar.clone().detach() # this is very necessary
        k = 0
        for name in self.variables:
            mean = self.mean_ds[name].values
            std  = self.std_ds[name].values
            result[:,k,:,:,:] = result[:,k,:,:,:] * std + mean
            k += 1

        return result.to(x.device)
    
    
    
class ToTensor:
    #def __init__(self, conf):
    def __init__(self, conf, x0 = 120, xsize = 512, y0=300, ysize=512):
        self.conf = conf
        self.hist_len = int(conf["data"]["history_len"])
        self.for_len = int(conf["data"]["forecast_len"])
        self.variables = conf["data"]["variables"]
        self.static_variables = conf["data"]["static_variables"]
        # self.x = 1016
        # self.y = 1638
        #self.slice_x = slice(120, 632, None)
        #self.slice_y = slice(300, 812, None)
        self.slice_x = slice(x0, x0+xsize, None)
        self.slice_y = slice(y0, y0+ysize, None)
        
    def __call__(self, sample: Sample) -> Sample:

        return_dict = {}
        for key, value in sample.items():
            if key == 'historical_ERA5_images' or key == 'x':
                self.datetime = value['Time']
                self.doy = value['Time.dayofyear']
                self.hod = value['Time.hour']

            if isinstance(value, xr.DataArray):
                value_var = value.values

            elif isinstance(value, xr.Dataset):
                concatenated_vars = []
                for vv in self.variables:
                    value_var = value[vv].values
                    if len(value_var.shape) == 4:  # some seem to have extra single dimensions
                        value_var = value_var.squeeze(1)
                    concatenated_vars.append(value_var[:, self.slice_x, self.slice_y])
                concatenated_vars = np.array(concatenated_vars)
            else:
                value_var = value
                
            if key == 'x':
                x = torch.as_tensor(np.vstack([np.expand_dims(v, axis=0) for v in concatenated_vars]))
                return_dict['x'] = x

            elif key == 'y':
                y = torch.as_tensor(np.vstack([np.expand_dims(v, axis=0) for v in concatenated_vars]))
                return_dict['y'] = y
                
        if self.static_variables:
            pass

        return return_dict
