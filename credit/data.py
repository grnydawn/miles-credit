from typing import Optional, Callable, TypedDict, Union, Iterable, NamedTuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import get_worker_info
from torch.utils.data.distributed import DistributedSampler
import torch.utils.data
import datetime
import os
from glob import glob
from itertools import repeat


def get_forward_data(filename) -> xr.DataArray:
    """Lazily opens the Zarr store on gladefilesystem.
    """
    dataset = xr.open_zarr(filename, consolidated=True)
    return dataset


Array = Union[np.ndarray, xr.DataArray]
IMAGE_ATTR_NAMES = ('historical_ERA5_images', 'target_ERA5_images')


class Sample(TypedDict):
    """Simple class for structuring data for the ML model.

    Using typing.TypedDict gives us several advantages:
      1. Single 'source of truth' for the type and documentation of each example.
      2. A static type checker can check the types are correct.

    Instead of TypedDict, we could use typing.NamedTuple,
    which would provide runtime checks, but the deal-breaker with Tuples is that they're immutable
    so we cannot change the values in the transforms.
    """
    # IMAGES
    # Shape: batch_size, seq_length, lat, lon, lev
    historical_ERA5_images: Array
    target_ERA5_images: Array

    # METADATA
    datetime_index: Array


@dataclass
class Reshape_Data():
    size: int = 128  #: Size of the cropped image.

    def __call__(self, sample: Sample) -> Sample:
        for attr_name in IMAGE_ATTR_NAMES:
            image = sample[attr_name]
            # TODO: Random crop!
            cropped_image = image[..., :self.size, :self.size]
            sample[attr_name] = cropped_image
        return sample


class CheckForBadData():
    def __call__(self, sample: Sample) -> Sample:
        for attr_name in IMAGE_ATTR_NAMES:
            image = sample[attr_name]
            if np.any(image < 0):
                raise ValueError(f'\n{attr_name} has negative values at {image.time.values}')
        return sample


# class NormalizeState():
#     def __init__(self, mean_file, std_file):
#         self.mean_ds = xr.open_dataset(mean_file)
#         self.std_ds = xr.open_dataset(std_file)

#     def __call__(self, sample: Sample) -> Sample:
#         normalized_sample = {}
#         for key, value in sample.items():
#             if isinstance(value, xr.Dataset):
#                 #key_change = key
#                 #value_change = (value - self.mean_ds)/self.std_ds
#                 #sample[key]=value_change
#                 normalized_sample[key] = (value - self.mean_ds) / self.std_ds
#         return normalized_sample


class Segment(NamedTuple):
    """Represents the start and end indicies of a segment of contiguous samples."""
    start: int
    end: int


def get_contiguous_segments(dt_index: pd.DatetimeIndex, min_timesteps: int, max_gap: pd.Timedelta) -> Iterable[Segment]:
    """Chunk datetime index into contiguous segments, each at least min_timesteps long.

    max_gap defines the threshold for what constitutes a 'gap' between contiguous segments.

    Throw away any timesteps in a sequence shorter than min_timesteps long.
    """
    gap_mask = np.diff(dt_index) > max_gap
    gap_indices = np.argwhere(gap_mask)[:, 0]

    # gap_indicies are the indices into dt_index for the timestep immediately before the gap.
    # e.g. if the datetimes at 12:00, 12:05, 18:00, 18:05 then gap_indicies will be [1].
    segment_boundaries = gap_indices + 1

    # Capture the last segment of dt_index.
    segment_boundaries = np.concatenate((segment_boundaries, [len(dt_index)]))

    segments = []
    start_i = 0
    for end_i in segment_boundaries:
        n_timesteps = end_i - start_i
        if n_timesteps >= min_timesteps:
            segment = Segment(start=start_i, end=end_i)
            segments.append(segment)
        start_i = end_i

    return segments


def get_zarr_chunk_sequences(
    n_chunks_per_disk_load: int,
    zarr_chunk_boundaries: Iterable[int],
    contiguous_segments: Iterable[Segment]) -> Iterable[Segment]:
    """

    Args:
      n_chunks_per_disk_load: Maximum number of Zarr chunks to load from disk in one go.
      zarr_chunk_boundaries: The indicies into the Zarr store's time dimension which define the Zarr chunk boundaries.
        Must be sorted.
      contiguous_segments: Indicies into the Zarr store's time dimension that define contiguous timeseries.
        That is, timeseries with no gaps.

    Returns zarr_chunk_sequences: a list of Segments representing the start and end indicies of contiguous sequences of multiple Zarr chunks,
    all exactly n_chunks_per_disk_load long (for contiguous segments at least as long as n_chunks_per_disk_load zarr chunks),
    and at least one side of the boundary will lie on a 'natural' Zarr chunk boundary.

    For example, say that n_chunks_per_disk_load = 3, and the Zarr chunks sizes are all 5:


                  0    5   10   15   20   25   30   35
                  |....|....|....|....|....|....|....|

    INPUTS:
                     |------CONTIGUOUS SEGMENT----|

    zarr_chunk_boundaries:
                  |----|----|----|----|----|----|----|

    OUTPUT:
    zarr_chunk_sequences:
           3 to 15:  |-|----|----|
           5 to 20:    |----|----|----|
          10 to 25:         |----|----|----|
          15 to 30:              |----|----|----|
          20 to 32:                   |----|----|-|

    """
    assert n_chunks_per_disk_load > 0

    zarr_chunk_sequences = []

    for contig_segment in contiguous_segments:
        # searchsorted() returns the index into zarr_chunk_boundaries at which contig_segment.start
        # should be inserted into zarr_chunk_boundaries to maintain a sorted list.
        # i_of_first_zarr_chunk is the index to the element in zarr_chunk_boundaries which defines
        # the start of the current contig chunk.
        i_of_first_zarr_chunk = np.searchsorted(zarr_chunk_boundaries, contig_segment.start)

        # i_of_first_zarr_chunk will be too large by 1 unless contig_segment.start lies
        # exactly on a Zarr chunk boundary.  Hence we must subtract 1, or else we'll
        # end up with the first contig_chunk being 1 + n_chunks_per_disk_load chunks long.
        if zarr_chunk_boundaries[i_of_first_zarr_chunk] > contig_segment.start:
            i_of_first_zarr_chunk -= 1

        # Prepare for looping to create multiple Zarr chunk sequences for the current contig_segment.
        zarr_chunk_seq_start_i = contig_segment.start
        zarr_chunk_seq_end_i = None  # Just a convenience to allow us to break the while loop by checking if zarr_chunk_seq_end_i != contig_segment.end.
        while zarr_chunk_seq_end_i != contig_segment.end:
            zarr_chunk_seq_end_i = zarr_chunk_boundaries[i_of_first_zarr_chunk + n_chunks_per_disk_load]
            zarr_chunk_seq_end_i = min(zarr_chunk_seq_end_i, contig_segment.end)
            zarr_chunk_sequences.append(Segment(start=zarr_chunk_seq_start_i, end=zarr_chunk_seq_end_i))
            i_of_first_zarr_chunk += 1
            zarr_chunk_seq_start_i = zarr_chunk_boundaries[i_of_first_zarr_chunk]

    return zarr_chunk_sequences


Array = Union[np.ndarray, xr.DataArray]
IMAGE_ATTR_NAMES = ('historical_ERA5_images', 'target_ERA5_images')


class Sample(TypedDict):
    """Simple class for structuring data for the ML model.

    Using typing.TypedDict gives us several advantages:
      1. Single 'source of truth' for the type and documentation of each example.
      2. A static type checker can check the types are correct.

    Instead of TypedDict, we could use typing.NamedTuple,
    which would provide runtime checks, but the deal-breaker with Tuples is that they're immutable
    so we cannot change the values in the transforms.
    """
    # IMAGES
    # Shape: batch_size, seq_length, lat, lon, lev
    historical_ERA5_images: Array
    target_ERA5_images: Array

    # METADATA
    datetime_index: Array


def flatten_list(list_of_lists):
    """
    Flatten a list of lists.

    Parameters:
    - list_of_lists (list): A list containing sublists.

    Returns:
    - flattened_list (list): A flattened list containing all elements from sublists.
    """
    return [item for sublist in list_of_lists for item in sublist]


def generate_integer_list_around(number, spacing=10):
    """
    Generate a list of integers on either side of a given number with a specified spacing.

    Parameters:
    - number (int): The central number around which the list is generated.
    - spacing (int): The spacing between consecutive integers in the list. Default is 10.

    Returns:
    - integer_list (list): List of integers on either side of the given number.
    """
    lower_limit = number - spacing
    upper_limit = number + spacing + 1  # Adding 1 to include the upper limit
    integer_list = list(range(lower_limit, upper_limit))

    return integer_list


def find_key_for_number(input_number, data_dict):
    """
    Find the key in the dictionary based on the given number.

    Parameters:
    - input_number (int): The number to search for in the dictionary.
    - data_dict (dict): The dictionary with keys and corresponding value lists.

    Returns:
    - key_found (str): The key in the dictionary where the input number falls within the specified range.
    """
    for key, value_list in data_dict.items():
        if value_list[1] <= input_number <= value_list[2]:
            return key

    # Return None if the number is not within any range
    return None


class ERA5Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        filenames: list = ['/glade/derecho/scratch/wchapman/STAGING/TOTAL_2012-01-01_2012-12-31_staged.zarr', '/glade/derecho/scratch/wchapman/STAGING/TOTAL_2013-01-01_2013-12-31_staged.zarr'],
        history_len: int = 1,
        forecast_len: int = 2,
        transform: Optional[Callable] = None,
        seed=42,
        skip_periods=None,
        one_shot=None
    ):
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.transform = transform
        self.skip_periods = skip_periods
        self.one_shot = one_shot
        self.total_seq_len = self.history_len + self.forecast_len
        all_fils = []
        filenames = sorted(filenames)
        for fn in filenames:
            all_fils.append(get_forward_data(filename=fn))
        self.all_fils = all_fils
        self.data_array = all_fils[0]
        self.rng = np.random.default_rng(seed=seed)

        # set data places:
        indo = 0
        self.meta_data_dict = {}
        for ee, bb in enumerate(self.all_fils):
            self.meta_data_dict[str(ee)] = [len(bb['time']), indo, indo+len(bb['time'])]
            indo += len(bb['time'])+1

        # set out of bounds indexes...
        OOB = []
        for kk in self.meta_data_dict.keys():
            OOB.append(generate_integer_list_around(self.meta_data_dict[kk][2]))
        self.OOB = flatten_list(OOB)

    def __post_init__(self):
        # Total sequence length of each sample.
        self.total_seq_len = self.history_len + self.forecast_len

    def __len__(self):
        tlen = 0
        for bb in self.all_fils:
            tlen += len(bb['time']) - self.total_seq_len + 1
        return tlen

    def __getitem__(self, index):

        # find the result key:
        result_key = find_key_for_number(index, self.meta_data_dict)
        # get the data selection:
        true_ind = index-self.meta_data_dict[result_key][1]

        if true_ind > (len(self.all_fils[int(result_key)]['time'])-(self.history_len+self.forecast_len+1)):
            true_ind = len(self.all_fils[int(result_key)]['time'])-(self.history_len+self.forecast_len+1)

        datasel = self.all_fils[int(result_key)].isel(time=slice(true_ind, true_ind+self.history_len+self.forecast_len+1)).load()

        if self.skip_periods is not None:
            sample = Sample(
                historical_ERA5_images=datasel.isel(time=slice(0, self.history_len, self.skip_periods)),
                target_ERA5_images=datasel.isel(time=slice(self.history_len, len(datasel['time']), self.skip_periods)),
                datetime_index=datasel.time.values.astype('datetime64[s]').astype(int)
            )
        elif self.one_shot is not None:
            sample = Sample(
                historical_ERA5_images=datasel.isel(time=slice(0, self.history_len)),
                target_ERA5_images=datasel.isel(time=-1),
                datetime_index=datasel.time.values.astype('datetime64[s]').astype(int)
            )
        else:
            sample = Sample(
                historical_ERA5_images=datasel.isel(time=slice(0, self.history_len)),
                target_ERA5_images=datasel.isel(time=slice(self.history_len, len(datasel['time']))),
                datetime_index=datasel.time.values.astype('datetime64[s]').astype(int)
            )

        if self.transform:
            sample = self.transform(sample)
        return sample


    
from functools import reduce
                                    
## flatten list-of-list
def flatten(array):
  return reduce(lambda a,b: a+b, array) 

## lazy-load & merge zarr stores
def lazymerge(zlist):
    return xr.merge([get_forward_data(z) for z in zlist])

## todo: dataclass decorator

class CONUS404Dataset(torch.utils.data.Dataset):
    """Each Zarr store for the CONUS-404 data contains one year of
    hourly data for one variable.

    When we're sampling data, we only want to load from a single zarr
    store; we don't want the samples to span zarr store boundaries.
    This lets us leave years out from across the entire span for
    validation during training.

    So to do this, we segment the dataset by year.  We figure out how
    many samples we could have, then subtract all the ones that start
    in one year and end in another (or end past the end of the
    dataset).  Then we create an index of which segment each sample
    belongs to, and the number of that sample within the segment.

    Then, for the __getitem__ method, we look up which segment the
    sample is in and its numbering within the segment, then open the
    corresponding zarr store and read only the data we want with an
    isel() call.

    For multiple variables, we do end up reading from multiple stores,
    but they're merged into a single xarray Dataset, so hopefully
    that's okay?
    
    """


    def __init__(
            self,
            zarrpath:     str = "/glade/campaign/ral/risc/DATA/conus404/zarr",
            varnames:     list = [],
            history_len:  int = 1,
            forecast_len: int = 2,
            transform:    Optional[Callable] = None,
            seed:         int = 22,
            skip_periods: int = None,
            one_shot:     bool = False,
            tdimname:     str = "Time"
    ):
        self.zarrpath     = zarrpath
        self.history_len  = history_len
        self.forecast_len = forecast_len
        self.transform    = transform
        self.skip_periods = skip_periods
        self.one_shot     = one_shot
        self.tdimname     = tdimname
        
        self.sample_len = sample_len = history_len + forecast_len
        self.stride = stride = 1 if skip_periods is None else skip_periods + 1

        self.rng = np.random.default_rng(seed)

        ## CONUS404 data is organized into directories by variable,
        ## with a set of yearly zarr stores for each variable
        if len(varnames) == 0:
            varnames = os.listdir(zarrpath)
        self.varnames = sorted(varnames)
        
        ## get file paths
        zdict = {}
        for v in varnames:
            zdict[v] = sorted(glob(zarrpath+"/"+v+"/"+v+".*.zarr"))

        ## check that lists align
        zlen = [len(z) for z in zdict.values()]
        assert all([zlen[i] == zlen[0] for i in range(len(zlen))])

        ## transpose list-of-lists; sort by key to ensure var order constant
        zlol = list(zip(*sorted(zdict.values())))
        
        ## lazy-load & merge zarr stores
        self.zarrs = [lazymerge(z) for z in zlol]

        ## construct indexing arrays
        zarrlen = [z.dims[tdimname] for z in self.zarrs]
        whichseg = [list(repeat(s,z)) for s,z in zip(range(len(zarrlen)), zarrlen)]
        segindex = [list(range(z)) for z in zarrlen]
        
        ## subset to samples that don't overlap a segment boundary
        ## (sample size N = can't use last N-1 samples)
        N = self.sample_len - 1
        self.segments = flatten([s[:-N] for s in whichseg])
        self.zindex   = flatten([i[:-N] for i in segindex])

        ## precompute mask arrays for subsetting data for samples
        self.histmask = list(range(0, history_len, stride))
        foreind = list(range(sample_len))
        if one_shot:
            self.foremask = foreind[-1]
        else:
            self.foremask =  foreind[slice(history_len, sample_len, stride)]

    def __len__(self):
        return(len(self.zindex))

    def __getitem__(self, index):
        time = self.tdimname
        first = self.zindex[index]
        last = first + self.sample_len
        seg = self.segments[index]
        subset = self.zarrs[seg].isel({time:slice(first, last)}).load()
        sample=Sample(history=subset.isel({time:self.histmask}),
                      target=subset.isel({time:self.foremask}),
                      datetime=subset[time])
        
        if self.transform:
            sample = self.transform(sample)

        return sample

        
## Note: DistributedSequentialDataset & DistributedSequentialDataset
## are legacy; they wrap ERA5Dataset to send data batches to GPUs for
## (1 class of?) huge sharded models, but otherwise have been
## superseded by ERA5Dataset.

class SequentialDataset(torch.utils.data.Dataset):

    def __init__(self, filenames, history_len=1, forecast_len=2, skip_periods=1, transform=None, random_forecast=True):
        self.dataset = ERA5Dataset(
            filenames=filenames,
            history_len=history_len,
            forecast_len=forecast_len,
            transform=transform
        )
        self.meta_data_dict = self.dataset.meta_data_dict
        self.all_fils = self.dataset.all_fils
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.filenames = filenames
        self.transform = transform
        self.skip_periods = skip_periods
        self.random_forecast = random_forecast
        self.iteration_count = 0
        self.current_epoch = 0
        self.adjust_forecast = 0

        self.index_list = []
        for i, x in enumerate(self.all_fils):
            times = x['time'].values
            slices = np.arange(0, times.shape[0] - (self.forecast_len + 1))
            self.index_list += [(i, slice) for slice in slices]

    def __len__(self):
        return len(self.index_list)

    def set_params(self, epoch):
        self.current_epoch = epoch
        self.iteration_count = 0

    def __getitem__(self, index):

        if self.random_forecast and (self.iteration_count % self.forecast_len == 0):
            # Randomly choose a starting point within a valid range
            max_start = len(self.index_list) - (self.forecast_len + 1)
            self.adjust_forecast = np.random.randint(0, max_start + 1)

        index = (index + self.adjust_forecast) % self.__len__()
        file_id, slice_idx = self.index_list[index]

        dataset = xr.open_zarr(self.filenames[file_id], consolidated=True).isel(time=slice(slice_idx, slice_idx + self.skip_periods + 1, self.skip_periods))

        sample = {
            'x': dataset.isel(time=slice(0, 1, 1)),
            'y': dataset.isel(time=slice(1, 2, 1)),
        }

        if self.transform:
            sample = self.transform(sample)

        sample['forecast_hour'] = self.iteration_count
        sample['forecast_datetime'] = dataset.time.values.astype('datetime64[s]').astype(int)
        sample['stop_forecast'] = False

        if self.iteration_count == self.forecast_len - 1:
            sample['stop_forecast'] = True

        # Increment the iteration count
        self.iteration_count += 1

        return sample


class DistributedSequentialDataset(torch.utils.data.IterableDataset):
    # https://colab.research.google.com/drive/1OFLZnX9y5QUFNONuvFsxOizq4M-tFvk-?usp=sharing#scrollTo=CxSCQPOMHgwo

    def __init__(self, filenames, history_len, forecast_len, skip_periods, rank, world_size, shuffle=False, transform=None, rollout_p=0.0):

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

    def set_rollout_prob(self, p):
        self.rollout_p = p

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        sampler = DistributedSampler(self, num_replicas=num_workers*self.world_size, rank=self.rank*num_workers+worker_id, shuffle=self.shuffle)
        sampler.set_epoch(self.current_epoch)

        for index in iter(sampler):
            result_key = find_key_for_number(index, self.meta_data_dict)
            true_ind = index - self.meta_data_dict[result_key][1]

            if true_ind > (len(self.all_fils[int(result_key)]['time'])-(self.history_len+self.forecast_len+1)):
                true_ind = len(self.all_fils[int(result_key)]['time'])-(self.history_len+self.forecast_len+3)

            indices = list(range(true_ind, true_ind+self.history_len+self.forecast_len+1))
            self.seq_len = self.history_len
            indices = indices[:self.seq_len]

            stop_forecast = False
            for k, ind in enumerate(indices):

                concatenated_samples = {'x': [], 'x_surf': [], 'y': [], 'y_surf': []}
                sliced = xr.open_zarr(self.filenames[int(result_key)], consolidated=True).isel(time=slice(ind, ind+self.history_len+1, self.skip_periods))
                sample = {
                    'x': sliced.isel(time=slice(0, 1, 1)),
                    'y': sliced.isel(time=slice(1, 2, 1)),
                    't': sliced.time.values.astype('datetime64[s]').astype(int),
                }

                if self.transform:
                    sample = self.transform(sample)

                for key in concatenated_samples.keys():
                    concatenated_samples[key] = sample[key].squeeze()

                stop_forecast = (torch.rand(1).item() < self.rollout_p)

                concatenated_samples['forecast_hour'] = k
                concatenated_samples['index'] = index
                concatenated_samples['stop_forecast'] = stop_forecast

                yield concatenated_samples

                if stop_forecast:
                    break

                if (k == self.history_len):
                    break


class PredictForecast(torch.utils.data.IterableDataset):
    def __init__(self,
                 filenames,
                 forecasts,
                 history_len,
                 forecast_len,
                 skip_periods,
                 rank,
                 world_size,
                 shuffle=False,
                 transform=None,
                 rollout_p=0.0,
                 start_time=None,
                 stop_time=None):

        self.dataset = ERA5Dataset(
            filenames=filenames,
            history_len=history_len,
            forecast_len=forecast_len,
            skip_periods=skip_periods,
            transform=transform
        )
        self.meta_data_dict = self.dataset.meta_data_dict
        self.all_files = self.dataset.all_fils
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
        self.forecasts = forecasts

    def find_start_stop_indices(self, index):
        datetime_objs = [np.datetime64(date) for date in self.forecasts[index]]
        start_time, stop_time = [str(datetime_obj) + '.000000000' for datetime_obj in datetime_objs]
        self.start_time = np.datetime64(start_time).astype(datetime.datetime)
        self.stop_time = np.datetime64(stop_time).astype(datetime.datetime)

        info = {}

        for idx, dataset in enumerate(self.all_files):
            start_time = np.datetime64(dataset['time'].min().values).astype(datetime.datetime)
            stop_time = np.datetime64(dataset['time'].max().values).astype(datetime.datetime)
            track_start = False
            track_stop = False

            if start_time <= self.start_time <= stop_time:
                # Start time is in this file, use start time index
                dataset = np.array([np.datetime64(x.values).astype(datetime.datetime) for x in dataset['time']])
                start_idx = np.searchsorted(dataset, self.start_time)
                start_idx = max(0, min(start_idx, len(dataset)-1))
                track_start = True

            elif start_time < self.stop_time and stop_time > self.start_time:
                # File overlaps time range, use full file
                start_idx = 0
                track_start = True

            if start_time <= self.stop_time <= stop_time:
                # Stop time is in this file, use stop time index
                if isinstance(dataset, np.ndarray):
                    pass
                else:
                    dataset = np.array([np.datetime64(x.values).astype(datetime.datetime) for x in dataset['time']])
                stop_idx = np.searchsorted(dataset, self.stop_time)
                stop_idx = max(0, min(stop_idx, len(dataset)-1))
                track_stop = True

            elif start_time < self.stop_time and stop_time > self.start_time:
                # File overlaps time range, use full file
                stop_idx = len(dataset) - 1
                track_stop = True

            # Only include files that overlap the time range
            if track_start and track_stop:
                info[idx] = ((idx, start_idx), (idx, stop_idx))

        indices = []
        for dataset_idx, (start, stop) in info.items():
            for i in range(start[1], stop[1]+1):
                indices.append((start[0], i))
        return indices

    def __len__(self):
        return len(self.forecasts)

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        sampler = DistributedSampler(self, num_replicas=num_workers*self.world_size, rank=self.rank*num_workers+worker_id, shuffle=self.shuffle)

        for index in sampler:
            data_lookup = self.find_start_stop_indices(index)

            for k, (file_key, time_key) in enumerate(data_lookup):
                concatenated_samples = {'x': [], 'x_surf': [], 'y': [], 'y_surf': []}
                sliced_x = xr.open_zarr(self.filenames[file_key], consolidated=True).isel(time=slice(time_key, time_key+1, self.skip_periods))
                sample_x = {
                    'x': sliced_x.isel(time=slice(0, 1, 1)),
                    #'t': sliced_x.time.values.astype('datetime64[s]').astype(int),
                }
                # Fetch the next pair to create the y tensor
                next_k = k + 1
                if next_k < len(data_lookup):
                    next_file_key, next_time_key = data_lookup[next_k]
                    sliced_y = xr.open_zarr(self.filenames[next_file_key], consolidated=True).isel(time=slice(next_time_key, next_time_key+1, self.skip_periods))
                    sample_x['y'] = sliced_y.isel(time=slice(0, 1, 1))

                if self.transform:
                    sample_x = self.transform(sample_x)

                for key in concatenated_samples.keys():
                    concatenated_samples[key] = sample_x[key].squeeze()

                concatenated_samples['forecast_hour'] = k
                concatenated_samples['stop_forecast'] = (k == (len(data_lookup)-2))
                concatenated_samples['datetime'] = sliced_x.time.values.astype('datetime64[s]').astype(int)[0]

                yield concatenated_samples

                if concatenated_samples['stop_forecast']:
                    break

############  test:


#from data import *

#zarrpath = "/glade/work/mcginnis/ML/GWC/testdata/zarr"
#varnames = ["prec","tmin","tmax"]
#tdimname = "time"

#zarrpath = "/glade/campaign/ral/risc/DATA/conus404/zarr"
#varnames = ["Q850","T850","U850","V850","Z850"]
#tdimname = "Time"

#c4 = CONUS404Dataset(zarrpath, varnames, tdimname=tdimname)

#len(c4)

#c4[22]
