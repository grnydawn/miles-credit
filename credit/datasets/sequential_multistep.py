from credit.data import (Sample, ERA5Dataset, find_key_for_number,
                         get_forward_data, get_forward_data_netCDF4,
                         drop_var_from_dataset, extract_month_day_hour,
                         find_common_indices)
from concurrent.futures import ProcessPoolExecutor as Pool
# https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
from functools import partial
from torch.utils.data import get_worker_info
from torch.utils.data.distributed import DistributedSampler
from typing import Any, Callable, Dict, List, Optional, Tuple
import xarray as xr
import numpy as np
import logging
import torch
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistributedSequentialDatasetV1(torch.utils.data.IterableDataset):
    # https://colab.research.google.com/drive/1OFLZnX9y5QUFNONuvFsxOizq4M-tFvk-?usp=sharing#scrollTo=CxSCQPOMHgwo

    def __init__(self, filenames, history_len, forecast_len, skip_periods, rank, world_size, shuffle=False,
                 transform=None, rollout_p=0.0):

        self.dataset = ERA5Dataset(
            filenames=filenames,
            history_len=history_len,
            forecast_len=forecast_len,
            skip_periods=skip_periods,
            transform=transform
        )
        self.meta_data_dict = self.dataset.meta_data_dict
        self.all_fils = self.dataset.all_fils
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.filenames = filenames
        self.transform = transform
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.skip_periods = skip_periods
        self.current_epoch = 0
        self.rollout_p = rollout_p

    def __len__(self):
        tlen = 0
        for bb in self.all_fils:
            tlen += (len(bb['time']) - self.forecast_len)
        return tlen

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        sampler = DistributedSampler(self, num_replicas=num_workers * self.world_size,
                                     rank=self.rank * num_workers + worker_id, shuffle=self.shuffle)
        sampler.set_epoch(self.current_epoch)

        for index in iter(sampler):
            result_key = find_key_for_number(index, self.meta_data_dict)
            true_ind = index - self.meta_data_dict[result_key][1]

            if true_ind > (len(self.all_fils[int(result_key)]['time']) - (self.history_len + self.forecast_len + 1)):
                true_ind = len(self.all_fils[int(result_key)]['time']) - (self.history_len + self.forecast_len + 3)

            indices = list(range(true_ind, true_ind + self.history_len + self.forecast_len))
            stop_forecast = False

            for k, ind in enumerate(indices):

                concatenated_samples = {'x': [], 'x_surf': [], 'y': [], 'y_surf': [], "static": [], "TOA": []}
                sliced = xr.open_zarr(self.filenames[int(result_key)], consolidated=True).isel(
                    time=slice(ind, ind + self.history_len + self.forecast_len + 1, self.skip_periods))

                historical_data = sliced.isel(time=slice(0, self.history_len)).load()
                target_data = sliced.isel(time=slice(self.history_len, self.history_len + 1)).load()

                sample = {
                    "x": historical_data,
                    "y": target_data,
                    "t": [
                        int(historical_data.time.values[0].astype('datetime64[s]').astype(int)),
                        int(target_data.time.values[0].astype('datetime64[s]').astype(int))
                    ]
                }

                if self.transform:
                    sample = self.transform(sample)

                for key in concatenated_samples.keys():
                    concatenated_samples[key] = sample[key].squeeze()

                stop_forecast = (k == self.forecast_len)

                concatenated_samples['forecast_hour'] = k
                concatenated_samples['index'] = index
                concatenated_samples['stop_forecast'] = stop_forecast
                concatenated_samples["datetime"] = [
                    int(historical_data.time.values[0].astype('datetime64[s]').astype(int)),
                    int(target_data.time.values[0].astype('datetime64[s]').astype(int))
                ]

                if self.history_len == 1:
                    concatenated_samples['x'] = concatenated_samples['x'].unsqueeze(0)
                    concatenated_samples['x_surf'] = concatenated_samples['x_surf'].unsqueeze(0)

                concatenated_samples['y'] = concatenated_samples['y'].unsqueeze(0)
                concatenated_samples['y_surf'] = concatenated_samples['y_surf'].unsqueeze(0)

                yield concatenated_samples

                if stop_forecast:
                    break

                if (k == self.forecast_len):
                    break


class DistributedSequentialDatasetV2(torch.utils.data.IterableDataset):
    # https://colab.research.google.com/drive/1OFLZnX9y5QUFNONuvFsxOizq4M-tFvk-?usp=sharing#scrollTo=CxSCQPOMHgwo

    def __init__(
        self,
        varname_upper_air: List[str],
        varname_surface: List[str],
        varname_forcing: List[str],
        varname_static: List[str],
        varname_diagnostic: List[str],
        filenames: List[str],
        filename_surface: Optional[List[str]] = None,
        filename_forcing: Optional[str] = None,
        filename_static: Optional[str] = None,
        filename_diagnostic: Optional[List[str]] = None,
        rank: int = 0,
        world_size: int = 1,
        history_len: int = 2,
        forecast_len: int = 0,
        transform: Optional[Callable] = None,
        seed: int = 42,
        skip_periods: Optional[int] = None,
        one_shot: Optional[bool] = None,
        max_forecast_len: Optional[int] = None,
        shuffle: bool = True,
        num_workers: int = 0
    ):

        """
        Initialize the DistributedSequentialDatasetV2.

        Parameters:
        - varname_upper_air (list): List of upper air variable names.
        - varname_surface (list): List of surface variable names.
        - varname_forcing (list): List of forcing variable names.
        - varname_static (list): List of static variable names.
        - varname_diagnostic (list): List of diagnostic variable names.
        - filenames (list): List of filenames for upper air data.
        - filename_surface (list, optional): List of filenames for surface data.
        - filename_forcing (str, optional): Filename for forcing data.
        - filename_static (str, optional): Filename for static data.
        - filename_diagnostic (list, optional): List of filenames for diagnostic data.
        - rank (int, optional): Rank of the current process. Default is 0.
        - world_size (int, optional): Total number of processes. Default is 1.
        - history_len (int, optional): Length of the history sequence. Default is 2.
        - forecast_len (int, optional): Length of the forecast sequence. Default is 0.
        - transform (callable, optional): Transformation function to apply to the data.
        - seed (int, optional): Random seed for reproducibility. Default is 42.
        - skip_periods (int, optional): Number of periods to skip between samples.
        - one_shot (bool, optional): Whether to use one-shot sampling.
        - max_forecast_len (int, optional): Maximum length of the forecast sequence.
        - shuffle (bool, optional): Whether to shuffle the data. Default is True.
        - num_workers (int, optional): Number of worker processes. Default is 0.

        Returns:
        - sample (dict): A dictionary containing historical ERA5 images, target ERA5 images, datetime index, and additional information.
        """

        self.history_len = history_len
        self.forecast_len = forecast_len
        self.transform = transform
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.current_epoch = 0
        self.num_workers = num_workers

        # skip periods
        self.skip_periods = skip_periods
        if self.skip_periods is None:
            self.skip_periods = 1

        # one shot option
        self.one_shot = one_shot

        # total number of needed forecast lead times
        self.total_seq_len = self.history_len + self.forecast_len

        # set random seed
        self.rng = np.random.default_rng(seed=seed)

        # max possible forecast len
        self.max_forecast_len = max_forecast_len

        # ======================================================== #
        # ERA5 operations
        all_files = []
        filenames = sorted(filenames)

        for fn in filenames:
            # drop variables if they are not in the config
            xarray_dataset = get_forward_data(filename=fn)
            xarray_dataset = drop_var_from_dataset(xarray_dataset, varname_upper_air)

            # collect yearly datasets within a list
            all_files.append(xarray_dataset)

        self.all_files = all_files

        # set data places:
        indo = 0
        self.meta_data_dict = {}
        for ee, bb in enumerate(self.all_files):
            self.meta_data_dict[str(ee)] = [len(bb['time']), indo, indo + len(bb['time'])]
            indo += len(bb['time']) + 1

        # get sample indices from ERA5 upper-air files:
        ind_start = 0
        self.ERA5_indices = {}
        for ind_file, ERA5_xarray in enumerate(self.all_files):
            # [number of samples, ind_start, ind_end]
            self.ERA5_indices[str(ind_file)] = [len(ERA5_xarray['time']),
                                                ind_start,
                                                ind_start + len(ERA5_xarray['time'])]
            ind_start += len(ERA5_xarray['time']) + 1

        # ======================================================== #
        # forcing file
        self.filename_forcing = filename_forcing

        if self.filename_forcing is not None:
            assert os.path.isfile(filename_forcing), 'Cannot find forcing file [{}]'.format(filename_forcing)

            # drop variables if they are not in the config
            xarray_dataset = get_forward_data_netCDF4(filename_forcing)
            xarray_dataset = drop_var_from_dataset(xarray_dataset, varname_forcing)

            self.xarray_forcing = xarray_dataset
        else:
            self.xarray_forcing = False

        # ======================================================== #
        # static file
        self.filename_static = filename_static

        if self.filename_static is not None:
            assert os.path.isfile(filename_static), 'Cannot find static file [{}]'.format(filename_static)

            # drop variables if they are not in the config
            xarray_dataset = get_forward_data_netCDF4(filename_static)
            xarray_dataset = drop_var_from_dataset(xarray_dataset, varname_static)

            self.xarray_static = xarray_dataset
        else:
            self.xarray_static = False

        # ======================================================== #
        # diagnostic file
        self.filename_diagnostic = filename_diagnostic

        if self.filename_diagnostic is not None:

            diagnostic_files = []
            filename_diagnostic = sorted(filename_diagnostic)

            for fn in filename_diagnostic:

                # drop variables if they are not in the config
                xarray_dataset = get_forward_data(filename=fn)
                xarray_dataset = drop_var_from_dataset(xarray_dataset, varname_diagnostic)

                diagnostic_files.append(xarray_dataset)

            self.diagnostic_files = diagnostic_files

            assert len(self.diagnostic_files) == len(self.all_files), \
                'Mismatch between the total number of diagnostic files and upper-air files'
        else:
            self.diagnostic_files = False

        # ======================================================== #
        # surface files
        if filename_surface is not None:

            surface_files = []
            filename_surface = sorted(filename_surface)

            for fn in filename_surface:

                # drop variables if they are not in the config
                xarray_dataset = get_forward_data(filename=fn)
                xarray_dataset = drop_var_from_dataset(xarray_dataset, varname_surface)

                surface_files.append(xarray_dataset)

            self.surface_files = surface_files

            assert len(self.surface_files) == len(self.all_files), \
                'Mismatch between the total number of surface files and upper-air files'
        else:
            self.surface_files = False

    def __post_init__(self):
        # Total sequence length of each sample.
        self.total_seq_len = self.history_len + self.forecast_len

    def __len__(self) -> int:
        # compute the total number of length
        total_len = 0
        for ERA5_xarray in self.all_files:
            total_len += len(ERA5_xarray['time']) - self.total_seq_len + 1
        return total_len

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        sampler = DistributedSampler(self, num_replicas=num_workers * self.world_size,
                                     rank=self.rank * num_workers + worker_id, shuffle=self.shuffle)
        sampler.set_epoch(self.current_epoch)

        process_index_partial = partial(
            process_index,
            ERA5_indices=self.ERA5_indices,
            all_files=self.all_files,
            surface_files=self.surface_files,
            history_len=self.history_len,
            forecast_len=self.forecast_len,
            skip_periods=self.skip_periods,
            xarray_forcing=self.xarray_forcing,
            xarray_static=self.xarray_static,
            diagnostic_files=self.diagnostic_files,
            one_shot=self.one_shot,
            transform=self.transform
        )

        # Dont use multi-processing
        if self.num_workers <= 1:
            for index in iter(sampler):
                # Explicit inner (time step) loop
                indices = list(range(index, index + self.history_len + self.forecast_len))
                for ind in indices:
                    sample = process_index_partial((index, ind))
                    yield sample
                    if sample['stop_forecast']:
                        break
        else:  # use multi-processing
            with Pool(self.num_workers) as p:
                for index in iter(sampler):
                    # Use pool.map to parallelize the inner loop
                    indices = list(range(index, index + self.history_len + self.forecast_len))
                    for sample in p.map(process_index_partial, [(index, ind) for ind in indices]):
                        yield sample
                        if sample['stop_forecast']:
                            break


def process_index(
    tuple_index: Tuple[int, int],
    ERA5_indices: Dict[str, List[int]],
    all_files: List[Any],
    surface_files: Optional[List[Any]],
    history_len: int,
    forecast_len: int,
    skip_periods: int,
    xarray_forcing: Optional[Any],
    xarray_static: Optional[Any],
    diagnostic_files: Optional[List[Any]],
    one_shot: Optional[bool],
    transform: Optional[Callable]
) -> Dict[str, Any]:

    index, ind = tuple_index

    try:
        # select the ind_file based on the iter index
        ind_file = find_key_for_number(ind, ERA5_indices)

        # get the ind within the current file
        ind_start = ERA5_indices[ind_file][1]
        ind_start_in_file = ind - ind_start

        # handle out-of-bounds
        ind_largest = len(all_files[int(ind_file)]['time']) - (history_len + forecast_len + 1)
        if ind_start_in_file > ind_largest:
            ind_start_in_file = ind_largest

        # subset xarray on time dimension & load it to the memory
        ind_end_in_file = ind_start_in_file + history_len + forecast_len

        ERA5_subset = all_files[int(ind_file)].isel(
            time=slice(ind_start_in_file, ind_end_in_file + 1)).load()

        if surface_files:
            surface_subset = surface_files[int(ind_file)].isel(
                time=slice(ind_start_in_file, ind_end_in_file + 1)).load()
            ERA5_subset = ERA5_subset.merge(surface_subset)

        ind_end_time = len(ERA5_subset['time'])
        datetime_as_number = ERA5_subset.time.values.astype('datetime64[s]').astype(int)

        historical_ERA5_images = ERA5_subset.isel(time=slice(0, history_len, skip_periods))

        if xarray_forcing:
            month_day_forcing = extract_month_day_hour(np.array(xarray_forcing['time']))
            month_day_inputs = extract_month_day_hour(np.array(historical_ERA5_images['time']))
            ind_forcing, _ = find_common_indices(month_day_forcing, month_day_inputs)
            forcing_subset_input = xarray_forcing.isel(time=ind_forcing).load()
            forcing_subset_input['time'] = historical_ERA5_images['time']
            historical_ERA5_images = historical_ERA5_images.merge(forcing_subset_input)

        if xarray_static:
            N_time_dims = len(ERA5_subset['time'])
            static_subset_input = xarray_static.expand_dims(dim={"time": N_time_dims})
            static_subset_input = static_subset_input.assign_coords({'time': ERA5_subset['time']})
            static_subset_input = static_subset_input.isel(time=slice(0, history_len, skip_periods)).load()
            static_subset_input['time'] = historical_ERA5_images['time']
            historical_ERA5_images = historical_ERA5_images.merge(static_subset_input)

        target_ERA5_images = ERA5_subset.isel(time=slice(history_len, history_len + skip_periods, skip_periods))

        if diagnostic_files:
            diagnostic_subset = diagnostic_files[int(ind_file)].isel(
                time=slice(ind_start_in_file, ind_end_in_file + 1)).load()
            target_diagnostic = diagnostic_subset.isel(time=slice(history_len, ind_end_time, skip_periods))
            target_ERA5_images = target_ERA5_images.merge(target_diagnostic)

        if one_shot is not None:
            target_ERA5_images = target_ERA5_images.isel(time=slice(0, 1))

        sample = Sample(
            historical_ERA5_images=historical_ERA5_images,
            target_ERA5_images=target_ERA5_images,
            datetime_index=datetime_as_number
        )

        if transform:
            sample = transform(sample)

        sample["index"] = index
        stop_forecast = ((ind - index) == forecast_len)
        sample['forecast_hour'] = ind - index + 1
        sample['index'] = index
        sample['stop_forecast'] = stop_forecast
        sample["datetime"] = [
            int(historical_ERA5_images.time.values[0].astype('datetime64[s]').astype(int)),
            int(target_ERA5_images.time.values[0].astype('datetime64[s]').astype(int))
        ]

    except Exception as e:
        logger.error(f"Error processing index {tuple_index}: {e}")
        raise

    return sample
