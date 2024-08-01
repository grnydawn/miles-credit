import numpy as np
import torch
import random
from typing import Optional, Callable, List
from credit.data import get_forward_data, generate_integer_list_around, flatten_list, find_key_for_number


class MultiStepERA5(torch.utils.data.Dataset):

    def __init__(
            self,
            filenames: List[str] = ['/glade/derecho/scratch/wchapman/STAGING/TOTAL_2012-01-01_2012-12-31_staged.zarr',
                                    '/glade/derecho/scratch/wchapman/STAGING/TOTAL_2013-01-01_2013-12-31_staged.zarr'],
            history_len: int = 1,
            forecast_len: int = 2,
            transform: Optional[Callable] = None,
            seed=42,
            skip_periods=None,
            one_shot=None,
            max_forecast_len=None,
            rank=0,
            world_size=1
    ):
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.transform = transform
        self.skip_periods = skip_periods
        self.one_shot = one_shot
        self.total_seq_len = self.history_len + self.forecast_len
        self.max_forecast_len = max_forecast_len
        self.rank = rank
        self.world_size = world_size
        np.random.seed(seed + rank)

        all_fils = []
        filenames = sorted(filenames)
        for fn in filenames:
            all_fils.append(get_forward_data(filename=fn))
        self.all_fils = all_fils
        self.data_array = all_fils[0]

        # Set data places
        indo = 0
        self.meta_data_dict = {}
        for ee, bb in enumerate(self.all_fils):
            self.meta_data_dict[str(ee)] = [len(bb['time']), indo, indo + len(bb['time'])]
            indo += len(bb['time']) + 1

        # Set out of bounds indexes...
        OOB = []
        for kk in self.meta_data_dict.keys():
            OOB.append(generate_integer_list_around(self.meta_data_dict[kk][2]))
        self.OOB = flatten_list(OOB)

        # Generate sequences based on rank and world_size
        self.sequence_indices = self.generate_sequences()
        self.forecast_hour = 0

    def generate_sequences(self):
        # Calculate the total length manually
        total_length = sum(len(bb['time']) - self.total_seq_len + 1 for bb in self.all_fils)
        all_indices = list(range(total_length))

        chunk_size = len(all_indices) // self.world_size
        start_idx = self.rank * chunk_size
        end_idx = start_idx + chunk_size if self.rank != self.world_size - 1 else len(all_indices)

        random.shuffle(all_indices)

        # Select the start times
        random_start_times = all_indices[start_idx:end_idx]
        sequence_indices = []

        for start_time in random_start_times:
            if start_time == 0:
                continue
            for i in range(self.forecast_len + 1):
                sequence_indices.append(start_time + i)

        return sequence_indices

    def __post_init__(self):
        # Total sequence length of each sample.
        self.total_seq_len = self.history_len + self.forecast_len

    def __len__(self):
        return len(self.sequence_indices)

    def is_end_of_forecast(self, index: int) -> bool:
        """
        Determine if the current index is the last index in a forecast sequence.

        Parameters:
            index (int): The current index in sequence_indices.

        Returns:
            bool: True if the current index is the last index in a forecast sequence, otherwise False.
        """
        # Get the index of the current position in the sequence_indices list
        current_pos = self.sequence_indices.index(index)

        # Check if it's the last index in the sequence
        if current_pos == len(self.sequence_indices) - 1:
            return 1

        # Determine if the next index starts a new forecast
        next_index = self.sequence_indices[current_pos + 1]
        if next_index - index != 1:
            return 1

        return 0

    def __getitem__(self, index):
        index = self.sequence_indices[index]
        # The rest of your existing __getitem__ implementation remains unchanged
        # find the result key:
        result_key = find_key_for_number(index, self.meta_data_dict)

        # get the data selection:
        true_ind = index - self.meta_data_dict[result_key][1]

        if true_ind > (len(self.all_fils[int(result_key)]['time']) - (self.history_len + self.forecast_len + 1)):
            true_ind = len(self.all_fils[int(result_key)]['time']) - (self.history_len + self.forecast_len + 1)

        datasel = self.all_fils[int(result_key)].isel(
            time=slice(true_ind, true_ind + self.history_len + self.forecast_len + 1))

        historical_data = datasel.isel(time=slice(0, self.history_len)).load()
        target_data = datasel.isel(time=slice(self.history_len, self.history_len + 1)).load()

        sample = {
            "historical_ERA5_images": historical_data,
            "target_ERA5_images": target_data,
            "datetime_index": [int(historical_data.time.values[0].astype('datetime64[s]').astype(int)),
                               int(target_data.time.values[0].astype('datetime64[s]').astype(int))]
        }

        if self.transform:
            sample = self.transform(sample)

        sample["index"] = index
        sample["stop_forecast"] = self.is_end_of_forecast(index)
        sample["forecast_hour"] = self.forecast_hour
        sample["datetime_index"] = [
            int(historical_data.time.values[0].astype('datetime64[s]').astype(int)),
            int(target_data.time.values[0].astype('datetime64[s]').astype(int))
        ]

        if sample["stop_forecast"]:
            self.forecast_hour = 0
        else:
            self.forecast_hour += 1

        return sample
