import os
from os.path import join, expandvars
import typing as t
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import torch

from credit.data_conversions import dataConverter
from weatherbench2.derived_variables import ZonalEnergySpectrum


class LatWeightedMetrics:

    def __init__(self, conf, predict_mode=False):
        self.conf = conf
        self.predict_mode = predict_mode
        lat_file = conf['loss']['latitude_weights']
        atmos_vars = conf['data']['variables']
        surface_vars = conf['data']['surface_variables']
        levels = conf['model']['levels'] if 'levels' in conf['model'] else conf['model']['frames']

        self.vars = [f"{v}_{k}" for v in atmos_vars for k in range(levels)]
        self.vars += surface_vars

        self.w_lat = None
        if conf["loss"]["use_latitude_weights"]:
            lat = xr.open_dataset(lat_file)["latitude"].values
            w_lat = np.cos(np.deg2rad(lat))
            w_lat = w_lat / w_lat.mean()
            self.w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1)

        self.w_var = None
        if conf["loss"]["use_variable_weights"]:
            var_weights = [value if isinstance(value, list) else [value] for value in
                           conf["loss"]["variable_weights"].values()]
            var_weights = [item for sublist in var_weights for item in sublist]
            self.w_var = torch.from_numpy(var_weights).unsqueeze(0).unsqueeze(-1)
        
        if self.predict_mode:
            self.zonal_metrics = ZonalSpectrumMetric(self.conf)

    def __call__(self, pred, y, clim=None, transform=None, forecast_datetime=0):
        if transform is not None:
            pred = transform(pred)
            y = transform(y)

        # Get latitude and variable weights
        w_lat = self.w_lat.to(dtype=pred.dtype, device=pred.device) if self.w_lat is not None else 1.
        w_var = self.w_var.to(dtype=pred.dtype, device=pred.device) if self.w_var is not None else 1.

        if clim is not None:
            clim = clim.to(device=y.device).unsqueeze(0)
            pred = pred - clim
            y = y - clim

        loss_dict = {}
        with torch.no_grad():
            error = (pred - y)
            for i, var in enumerate(self.vars):
                pred_prime = pred[:, i] - torch.mean(pred[:, i])
                y_prime = y[:, i] - torch.mean(y[:, i])

                # Add epsilon to avoid division by zero
                epsilon = 1e-7

                denominator = torch.sqrt(
                    torch.sum(w_var * w_lat * pred_prime ** 2) * torch.sum(w_var * w_lat * y_prime ** 2)
                ) + epsilon

                loss_dict[f"acc_{var}"] = torch.sum(w_var * w_lat * pred_prime * y_prime) / denominator
                loss_dict[f"rmse_{var}"] = torch.mean(
                    torch.sqrt(torch.mean(error[:, i] ** 2 * w_lat * w_var, dim=(-2, -1)))
                )
                loss_dict[f"mse_{var}"] = (error[:, i] ** 2 * w_lat * w_var).mean()
                loss_dict[f"mae_{var}"] = (torch.abs(error[:, i]) * w_lat * w_var).mean()

        # Calculate metrics averages
        loss_dict["acc"] = np.mean([loss_dict[k].cpu().item() for k in loss_dict.keys() if "acc_" in k])
        loss_dict["rmse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys() if "rmse_" in k])
        loss_dict["mse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys() if "mse_" in k and "rmse_" not in k])
        loss_dict["mae"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys() if "mae_" in k])

        # additional metrics where xarray computations are needed
        # put metric configs here
        # convert to xarray:
        if self.predict_mode:
            self.converter = dataConverter(self.conf)
            pred_ds = self.converter.tensor_to_Dataset(pred, [forecast_datetime])
            y_ds = self.converter.tensor_to_Dataset(y, [forecast_datetime])
            loss_dict = loss_dict | self.zonal_metrics(pred_ds, y_ds)  # merge two dictionaries

        return loss_dict


class ZonalSpectrumMetric:
    def __init__(self, conf,  #w_var - not implemented yet
                 x_variables=["U", "V", "T", "Q"], single_level_variables=["SP", "t2m"],
                 spectrum_vis_levels=[8, 10], figsize=(50, 5)):
        '''
        _variables arguments determine which data vars to compute spectra metric
        '''
        self.conf = conf
        self.variables = x_variables + single_level_variables
        self.x_variables = x_variables
        self.single_level_variables = single_level_variables
        self.zonal_spectrum_calculator = ZonalEnergySpectrum(self.variables)
        self.spectrum_vis_levels = spectrum_vis_levels
        if conf["loss"]["use_latitude_weights"]:
            lat = xr.open_dataset(conf['loss']['latitude_weights'])["latitude"]
            w_lat = np.cos(np.deg2rad(lat))
            self.w_lat = w_lat / w_lat.mean()
        self.ifs_levels = xr.open_dataset('/glade/derecho/scratch/dkimpara/nwp_files/ifs_levels.nc')
        if figsize:
            self.figsize = figsize
        else:
            num_vars = len(x_variables) * len(spectrum_vis_levels) + len(single_level_variables)
            self.figsize = (5 * num_vars, 5)
        
        # save directory for spectra plots
        os.makedirs(join(expandvars(self.conf['save_loc']), 'forecasts/spectra/'), exist_ok=True)

    def __call__(self, pred_ds, y_ds):
        '''
        pred, y can be normalized or unnormalized tensors.
        trying to achieve minimal interface with LatWeightedMetrics 
        '''
        # first dim is the batch dim
        loss_dict = {}

        # compute spectrum and add epsilon to avoid division by zero
        epsilon = 1e-7
        pred_spectrum = self.zonal_spectrum_calculator.compute(pred_ds) + epsilon
        y_spectrum = self.zonal_spectrum_calculator.compute(y_ds) + epsilon
        loss = self.lat_weighted_spectrum_diff(pred_spectrum, y_spectrum)
        loss_dict = self.store_loss(loss, loss_dict, 'spectrum_mse')

        # compute average zonal spectrum
        avg_pred_spectrum = self.get_avg_spectrum(pred_spectrum)
        avg_y_spectrum = self.get_avg_spectrum(y_spectrum)

        # visualize
        fig, axs = plt.subplots(
            ncols=len(self.x_variables) * len(self.spectrum_vis_levels) + len(self.single_level_variables),
                      figsize=self.figsize)
        fig.suptitle(f't={avg_pred_spectrum.datetime.values[0]}')
        for ax in axs:
            ax.set_yscale('log')
            ax.set_xscale('log')

        curr_ax = 0
        for level in self.spectrum_vis_levels:
            for variable in self.x_variables:
                avg_pred_spectrum[variable].sel(level=level).plot(x='wavelength', ax=axs[curr_ax], color='r')
                avg_y_spectrum[variable].sel(level=level).plot(x='wavelength', ax=axs[curr_ax], color='0')

                axs[curr_ax].set_title(f'{variable} {self.ifs_levels.ref_hPa.sel(level=level).values}')
                axs[curr_ax].set_ylabel('Power')
                curr_ax += 1

        for variable in self.single_level_variables:
            avg_pred_spectrum[variable].plot(x='wavelength', ax=axs[curr_ax], color='r')
            avg_y_spectrum[variable].plot(x='wavelength', ax=axs[curr_ax], color='0')
            axs[curr_ax].set_title(variable)
            axs[curr_ax].set_ylabel('Power')
            curr_ax += 1

        fig.savefig(os.path.join(os.path.expandvars(self.conf['save_loc']),
                                 f'forecasts/spectra/spectra_t{avg_pred_spectrum.datetime.values[0]:03}'))

        return loss_dict

    def get_avg_spectrum(self, ds_spectrum):
        ds_spectrum = ds_spectrum.sel(level=self.spectrum_vis_levels)
        ds_spectrum = interpolate_spectral_frequencies(ds_spectrum, 'zonal_wavenumber')
        ds_spectrum = ds_spectrum.weighted(self.w_lat).mean(dim='latitude')
        return ds_spectrum

    def store_loss(self, loss, loss_dict, metric_header_str):
        '''
        loss: dataset
            w/ each variable dim must include level if atmos var, 
            ow no need for single level vars
        sums over remaining dimensions, and writes to loss_dict
        '''
        keys = []
        for v in self.x_variables:
            for k in range(self.conf['model']['levels']):
                label = f"{metric_header_str}_{v}_{k}"
                keys.append(label)
                loss_dict[label] = loss[v].sel(level=k).sum()  #latitudes already weighted

        for v in self.single_level_variables:
            label = f"{metric_header_str}_{v}"
            keys.append(label)
            loss_dict[label] = loss[v].sum()  #latitudes already weighted

        loss_dict[f"{metric_header_str}"] = np.mean([loss_dict[k] for k in keys])
        return loss_dict

    def lat_weighted_spectrum_diff(self, pred_spectrum, y_spectrum):
        # using squared distance
        # variables to compute spectra on determined by class init
        sq_diff = np.square(np.log10(pred_spectrum) - np.log10(y_spectrum))
        sq_diff = sq_diff.sum(dim=['datetime', 'zonal_wavenumber'])
        return sq_diff * self.w_lat


# from weatherbench, slightly modified
def interpolate_spectral_frequencies(
        spectrum: xr.DataArray,
        wavenumber_dim: str,
        frequencies: t.Optional[t.Sequence[float]] = None,
        method: str = 'linear',
        **interp_kwargs: t.Optional[dict[str, t.Any]],
) -> xr.DataArray:
    """Interpolate frequencies in `spectrum` to common values.
    
    Args:
    spectrum: Data as produced by ZonalEnergySpectrum.compute.
    wavenumber_dim: Dimension that indexes wavenumber, e.g. 'zonal_wavenumber'
      if `spectrum` is produced by ZonalEnergySpectrum.
    frequencies: Optional 1-D sequence of frequencies to interpolate to. By
      default, use the most narrow range of frequencies in `spectrum`.
    method: Interpolation method passed on to DataArray.interp.
    **interp_kwargs: Additional kwargs passed on to DataArray.interp.
    
    Returns:
    New DataArray with dimension "frequency" replacing the "wavenumber" dim in
      `spectrum`.
    """

    if set(spectrum.frequency.dims) != set((wavenumber_dim, 'latitude')):
        raise ValueError(
            f'{spectrum.frequency.dims=} was not a permutation of '
            f'("{wavenumber_dim}", "latitude")'
        )

    if frequencies is None:
        freq_min = spectrum.frequency.max('latitude').min(wavenumber_dim).data
        freq_max = spectrum.frequency.min('latitude').max(wavenumber_dim).data
        frequencies = np.linspace(
            freq_min, freq_max, num=spectrum.sizes[wavenumber_dim]
        )
        frequencies = np.asarray(frequencies)
    if frequencies.ndim != 1:
        raise ValueError(f'Expected 1-D frequencies, found {frequencies.shape=}')

    def interp_at_one_lat(da: xr.DataArray) -> xr.DataArray:
        da = (
            da.squeeze('latitude')
            .swap_dims({wavenumber_dim: 'frequency'})  # pytype: disable=wrong-arg-types
            .drop_vars(wavenumber_dim)
            .interp(frequency=frequencies, method=method, **interp_kwargs)
        )
        # Interp didn't deal well with the infinite wavelength, so just reset λ as..
        da['wavelength'] = 1 / da.frequency
        da['wavelength'] = da['wavelength'].assign_attrs(units='m')
        return da

    return spectrum.groupby('latitude').apply(interp_at_one_lat)


def anomaly_correlation_coefficient(pred, true):
    pred = pred.float()
    true = true.float()

    B, C, H, W = pred.size()

    # Flatten the spatial dimensions
    pred_flat = pred.view(B, C, -1)
    true_flat = true.view(B, C, -1)

    # Mean over spatial dimensions
    pred_mean = torch.mean(pred_flat, dim=-1, keepdim=True)
    true_mean = torch.mean(true_flat, dim=-1, keepdim=True)

    # Anomaly calculation
    pred_anomaly = pred_flat - pred_mean
    true_anomaly = true_flat - true_mean

    # Covariance matrix
    covariance_matrix = torch.bmm(pred_anomaly, true_anomaly.transpose(1, 2)) / (H * W - 1)

    # Variance terms
    pred_var = torch.bmm(pred_anomaly, pred_anomaly.transpose(1, 2)) / (H * W - 1)
    true_var = torch.bmm(true_anomaly, true_anomaly.transpose(1, 2)) / (H * W - 1)

    # Anomaly Correlation Coefficient
    acc_numerator = torch.einsum('bii->b', covariance_matrix).sum()
    acc_denominator = torch.sqrt(torch.einsum('bii->b', pred_var).sum() * torch.einsum('bii->b', true_var).sum())

    # Avoid division by zero
    epsilon = 1e-8
    acc = acc_numerator / (acc_denominator + epsilon)

    return acc.item()
