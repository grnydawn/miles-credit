import torch
import logging
import numpy as np
import xarray as xr
import netCDF4 as nc
from credit.data import Sample
from typing import Dict


logger = logging.getLogger(__name__)


class NormalizeState:
    def __init__(
        self,
        mean_file,
        std_file,
        variables=['U', 'V', 'T', 'Q'],
        surface_variables=['SP', 't2m', 'V500', 'U500', 'T500', 'Z500', 'Q500'],
        levels=15
    ):

        self.mean_ds = xr.open_dataset(mean_file)
        self.std_ds = xr.open_dataset(std_file)
        self.variables = variables
        self.surface_variables = surface_variables
        self.levels = levels

        logger.info("Loading preprocessing object for transform/inverse transform states into z-scores")

    def __call__(self, sample: Sample, inverse: bool = False) -> Sample:
        if inverse:
            return self.inverse_transform(sample)
        else:
            return self.transform(sample)
        
    def transform_array(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        tensor = x[:, :(len(self.variables)*self.levels), :, :]
        surface_tensor = x[:, (len(self.variables)*self.levels):, :, :]

        # Reverse z-score normalization using the pre-loaded mean and std
        transformed_tensor = tensor.clone()
        k = 0
        for name in self.variables:
            for level in range(self.levels):
                mean = self.mean_ds[name].values[level]
                std = self.std_ds[name].values[level]
                transformed_tensor[:, k] = (tensor[:, k] - mean) / std
                k += 1

        transformed_surface_tensor = surface_tensor.clone()
        for k, name in enumerate(self.surface_variables):
            mean = self.mean_ds[name].values
            std = self.std_ds[name].values
            transformed_surface_tensor[:, k] = (surface_tensor[:, k] - mean) / std

        transformed_x = torch.cat((transformed_tensor, transformed_surface_tensor), dim=1)

        return transformed_x.to(device)

    def transform(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        normalized_sample = {}
        for key, value in sample.items():
            if isinstance(value, xr.Dataset):
                normalized_sample[key] = (value - self.mean_ds) / self.std_ds
        return normalized_sample

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        tensor = x[:, :(len(self.variables)*self.levels), :, :]
        surface_tensor = x[:, (len(self.variables)*self.levels):, :, :]

        # Reverse z-score normalization using the pre-loaded mean and std
        transformed_tensor = tensor.clone()
        k = 0
        for name in self.variables:
            for level in range(self.levels):
                mean = self.mean_ds[name].values[level]
                std = self.std_ds[name].values[level]
                transformed_tensor[:, k] = tensor[:, k] * std + mean
                k += 1

        transformed_surface_tensor = surface_tensor.clone()
        for k, name in enumerate(self.surface_variables):
            mean = self.mean_ds[name].values
            std = self.std_ds[name].values
            transformed_surface_tensor[:, k] = surface_tensor[:, k] * std + mean

        transformed_x = torch.cat((transformed_tensor, transformed_surface_tensor), dim=1)

        return transformed_x.to(device)


class NormalizeTendency:
    def __init__(self, variables, surface_variables, base_path):
        self.variables = variables
        self.surface_variables = surface_variables
        self.base_path = base_path

        # Load the NetCDF files and store the data
        self.mean = {}
        self.std = {}
        for name in self.variables + self.surface_variables:
            mean_dataset = nc.Dataset(f'{self.base_path}/All_NORMtend_{name}_2010_staged.mean.nc')
            std_dataset = nc.Dataset(f'{self.base_path}/All_NORMtend_{name}_2010_staged.STD.nc')
            self.mean[name] = torch.from_numpy(mean_dataset.variables[name][:])
            self.std[name] = torch.from_numpy(std_dataset.variables[name][:])

        logger.info("Loading preprocessing object for transform/inverse transform tendencies into z-scores")

    def transform(self, tensor, surface_tensor):
        device = tensor.device

        # Apply z-score normalization using the pre-loaded mean and std
        for name in self.variables:
            mean = self.mean[name].view(1, 1, self.mean[name].size(0), 1, 1).to(device)
            std = self.std[name].view(1, 1, self.std[name].size(0), 1, 1).to(device)
            transformed_tensor = (tensor - mean) / std

        for name in self.surface_variables:
            mean = self.mean[name].view(1, 1, 1, 1).to(device)
            std = self.std[name].view(1, 1, 1, 1).to(device)
            transformed_surface_tensor = (surface_tensor - mean) / std

        return transformed_tensor, transformed_surface_tensor

    def inverse_transform(self, tensor, surface_tensor):
        device = tensor.device

        # Reverse z-score normalization using the pre-loaded mean and std
        for name in self.variables:
            mean = self.mean[name].view(1, 1, self.mean[name].size(0), 1, 1).to(device)
            std = self.std[name].view(1, 1, self.std[name].size(0), 1, 1).to(device)
            transformed_tensor = tensor * std + mean

        for name in self.surface_variables:
            mean = self.mean[name].view(1, 1, 1, 1).to(device)
            std = self.std[name].view(1, 1, 1, 1).to(device)
            transformed_surface_tensor = surface_tensor * std + mean

        return transformed_tensor, transformed_surface_tensor


class ToTensor:
    def __init__(
        self,
        history_len=1,
        forecast_len=2,
        variables=['U', 'V', 'T', 'Q'],
        surface_variables=['SP', 't2m', 'V500', 'U500', 'T500', 'Z500', 'Q500']
    ):

        self.hist_len = int(history_len)
        self.for_len = int(forecast_len)
        self.variables = variables
        self.surface_variables = surface_variables
        self.allvars = variables + surface_variables

    def __call__(self, sample: Sample) -> Sample:

        return_dict = {}

        for key, value in sample.items():

            if isinstance(value, xr.DataArray):
                value_var = value.values

            elif isinstance(value, xr.Dataset):
                surface_vars = []
                concatenated_vars = []
                for vv in self.allvars:
                    value_var = value[vv].values
                    if vv in self.surface_variables:
                        surface_vars_temp = value_var
                        surface_vars.append(surface_vars_temp)
                    else:
                        concatenated_vars.append(value_var)
                surface_vars = np.array(surface_vars)
                concatenated_vars = np.array(concatenated_vars)

            else:
                value_var = value

            if key == 'historical_ERA5_images' or key == 'x':
                x_surf = torch.as_tensor(surface_vars).squeeze()
                return_dict['x_surf'] = x_surf.permute(1, 0, 2, 3) if len(x_surf.shape) == 4 else x_surf.unsqueeze(0)
                return_dict['x'] = torch.as_tensor(np.hstack([np.expand_dims(x, axis=1) for x in concatenated_vars]))

            elif key == 'target_ERA5_images' or key == 'y':
                y_surf = torch.as_tensor(surface_vars)
                y = torch.as_tensor(np.hstack([np.expand_dims(x, axis=1) for x in concatenated_vars]))
                return_dict['y_surf'] = y_surf.permute(1, 0, 2, 3)
                return_dict['y'] = y

        return return_dict
