import torch
import logging
import numpy as np
from functools import partial
from credit.data import (
    drop_var_from_dataset,
    get_forward_data
)
from credit.datasets.era5_multistep import worker


logger = logging.getLogger(__name__)


class ERA5_MultiStep_Batcher(torch.utils.data.Dataset):
    """
    A Pytorch Dataset class that works on:
        - upper-air variables (time, level, lat, lon)
        - surface variables (time, lat, lon)
        - dynamic forcing variables (time, lat, lon)
        - foring variables (time, lat, lon)
        - diagnostic variables (time, lat, lon)
        - static variables (lat, lon)
    """

    def __init__(
        self,
        varname_upper_air,
        varname_surface,
        varname_dyn_forcing,
        varname_forcing,
        varname_static,
        varname_diagnostic,
        filenames,
        filename_surface=None,
        filename_dyn_forcing=None,
        filename_forcing=None,
        filename_static=None,
        filename_diagnostic=None,
        history_len=2,
        forecast_len=0,
        transform=None,
        seed=42,
        rank=0,
        world_size=1,
        skip_periods=None,
        max_forecast_len=None,
        batch_size=1
    ):
        """
        Initialize the ERA5_and_Forcing_Dataset

        Parameters:
        - varname_upper_air (list): List of upper air variable names.
        - varname_surface (list): List of surface variable names.
        - varname_dyn_forcing (list): List of dynamic forcing variable names.
        - varname_forcing (list): List of forcing variable names.
        - varname_static (list): List of static variable names.
        - varname_diagnostic (list): List of diagnostic variable names.
        - filenames (list): List of filenames for upper air data.
        - filename_surface (list, optional): List of filenames for surface data.
        - filename_dyn_forcing (list, optional): List of filenames for dynamic forcing data.
        - filename_forcing (str, optional): Filename for forcing data.
        - filename_static (str, optional): Filename for static data.
        - filename_diagnostic (list, optional): List of filenames for diagnostic data.
        - history_len (int, optional): Length of the history sequence. Default is 2.
        - forecast_len (int, optional): Length of the forecast sequence. Default is 0.
        - transform (callable, optional): Transformation function to apply to the data.
        - seed (int, optional): Random seed for reproducibility. Default is 42.
        - skip_periods (int, optional): Number of periods to skip between samples.
        - max_forecast_len (int, optional): Maximum length of the forecast sequence.
        - shuffle (bool, optional): Whether to shuffle the data. Default is True.

        Returns:
        - sample (dict): A dictionary containing historical_ERA5_images,
                                                 target_ERA5_images,
                                                 datetime index, and additional information.
        """

        self.history_len = history_len
        self.forecast_len = forecast_len
        self.transform = transform
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

        # skip periods
        self.skip_periods = skip_periods
        if self.skip_periods is None:
            self.skip_periods = 1

        # total number of needed forecast lead times
        self.total_seq_len = self.history_len + self.forecast_len

        # set random seed
        self.rng = np.random.default_rng(seed=seed)

        # max possible forecast len
        self.max_forecast_len = max_forecast_len

        # =================================================================== #
        # flags to determin if any of the [surface, dyn_forcing, diagnostics]
        # variable groups share the same file as upper air variables
        flag_share_surf = False
        flag_share_dyn = False
        flag_share_diag = False

        all_files = []
        filenames = sorted(filenames)

        # ------------------------------------------------------------------ #
        # blocks that can handle no-sharing (each group has it own file)
        # surface
        if filename_surface is not None:
            surface_files = []
            filename_surface = sorted(filename_surface)

            if filenames == filename_surface:
                flag_share_surf = True
            else:
                for fn in filename_surface:
                    # drop variables if they are not in the config
                    ds = get_forward_data(filename=fn)
                    ds_surf = drop_var_from_dataset(ds, varname_surface)
                    surface_files.append(ds_surf)

                self.surface_files = surface_files
        else:
            self.surface_files = False

        # dynamic forcing
        if filename_dyn_forcing is not None:
            dyn_forcing_files = []
            filename_dyn_forcing = sorted(filename_dyn_forcing)

            if filenames == filename_dyn_forcing:
                flag_share_dyn = True
            else:
                for fn in filename_dyn_forcing:
                    # drop variables if they are not in the config
                    ds = get_forward_data(filename=fn)
                    ds_dyn = drop_var_from_dataset(ds, varname_dyn_forcing)
                    dyn_forcing_files.append(ds_dyn)

                self.dyn_forcing_files = dyn_forcing_files
        else:
            self.dyn_forcing_files = False

        # diagnostics
        if filename_diagnostic is not None:
            diagnostic_files = []
            filename_diagnostic = sorted(filename_diagnostic)

            if filenames == filename_diagnostic:
                flag_share_diag = True
            else:
                for fn in filename_diagnostic:
                    # drop variables if they are not in the config
                    ds = get_forward_data(filename=fn)
                    ds_diag = drop_var_from_dataset(ds, varname_diagnostic)
                    diagnostic_files.append(ds_diag)

                self.diagnostic_files = diagnostic_files
        else:
            self.diagnostic_files = False

        # ------------------------------------------------------------------ #
        # blocks that can handle file sharing (share with upper air file)
        for fn in filenames:
            # drop variables if they are not in the config
            ds = get_forward_data(filename=fn)
            ds_upper = drop_var_from_dataset(ds, varname_upper_air)

            if flag_share_surf:
                ds_surf = drop_var_from_dataset(ds, varname_surface)
                surface_files.append(ds_surf)

            if flag_share_dyn:
                ds_dyn = drop_var_from_dataset(ds, varname_dyn_forcing)
                dyn_forcing_files.append(ds_dyn)

            if flag_share_diag:
                ds_diag = drop_var_from_dataset(ds, varname_diagnostic)
                diagnostic_files.append(ds_diag)

            all_files.append(ds_upper)

        self.all_files = all_files

        if flag_share_surf:
            self.surface_files = surface_files
        if flag_share_dyn:
            self.dyn_forcing_files = dyn_forcing_files
        if flag_share_diag:
            self.diagnostic_files = diagnostic_files

        # -------------------------------------------------------------------------- #
        # get sample indices from ERA5 upper-air files:
        ind_start = 0
        self.ERA5_indices = {}  # <------ change
        for ind_file, ERA5_xarray in enumerate(self.all_files):
            # [number of samples, ind_start, ind_end]
            self.ERA5_indices[str(ind_file)] = [
                len(ERA5_xarray["time"]),
                ind_start,
                ind_start + len(ERA5_xarray["time"]),
            ]
            ind_start += len(ERA5_xarray["time"]) + 1

        # ======================================================== #
        # forcing file
        self.filename_forcing = filename_forcing

        if self.filename_forcing is not None:
            # drop variables if they are not in the config
            xarray_dataset = get_forward_data(filename_forcing)
            xarray_dataset = drop_var_from_dataset(xarray_dataset, varname_forcing)

            self.xarray_forcing = xarray_dataset
        else:
            self.xarray_forcing = False

        # ======================================================== #
        # static file
        self.filename_static = filename_static

        if self.filename_static is not None:
            # drop variables if they are not in the config
            xarray_dataset = get_forward_data(filename_static)
            xarray_dataset = drop_var_from_dataset(xarray_dataset, varname_static)

            self.xarray_static = xarray_dataset
        else:
            self.xarray_static = False

        self.worker = partial(
            worker,
            ERA5_indices=self.ERA5_indices,
            all_files=self.all_files,
            surface_files=self.surface_files,
            dyn_forcing_files=self.dyn_forcing_files,
            diagnostic_files=self.diagnostic_files,
            xarray_forcing=self.xarray_forcing,
            xarray_static=self.xarray_static,
            history_len=self.history_len,
            forecast_len=self.forecast_len,
            skip_periods=self.skip_periods,
            transform=self.transform,
        )

        self.total_length = len(self.ERA5_indices)
        self.current_epoch = None
        self.current_index = None
        self.initial_index = None

        # Initialize state variables for batch management
        self.batch_size = batch_size
        self.batch_indices = None  # To track initial indices for each batch item
        self.time_steps = None  # Tracks time steps for each batch index
        self.forecast_step_counts = None  # Track forecast step counts for each batch item
        self.initial_indices = None  # Tracks the initial index for each forecast item in the batch

        # Initialize batch once when the dataset is created
        self.initialize_batch()

    def initialize_batch(self):
        """
        Initializes random starting indices for each batch item and resets their time steps and forecast counts.
        This must be called before accessing the dataset or when resetting the batch.
        """
        # Randomly sample indices for the batch
        self.batch_indices = np.random.choice(
            range(self.__len__() - self.forecast_len),
            size=self.batch_size,
            replace=False,
        )
        self.time_steps = [0 for idx in self.batch_indices]  # Initialize time to 0 for each item
        self.forecast_step_counts = [0 for idx in self.batch_indices]  # Initialize forecast step counts
        self.initial_indices = list(self.batch_indices)  # Track initial indices for each batch item

    def __post_init__(self):
        # Total sequence length of each sample.
        self.total_seq_len = self.history_len + self.forecast_len

    def __len__(self):
        # compute the total number of length
        total_len = 0
        for ERA5_xarray in self.all_files:
            total_len += len(ERA5_xarray["time"]) - self.total_seq_len + 1
        return total_len

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        self.current_index = None
        self.initial_index = None

    def __getitem__(self, _):
        """
        Fetches the current forecast step data for each item in the batch.
        Resets items when their forecast length is exceeded.
        """
        batch = {}

        # If the forecast_step_count exceeds forecast_len, reset the item
        # If one exceeds, they all exceed and all neet reset
        if self.forecast_step_counts[0] == self.forecast_len + 1:
            # Get a new starting index for this item (randomly selected)
            self.initialize_batch()

        for k, idx in enumerate(self.batch_indices):
            # Get the current time step for this batch item
            current_t = self.time_steps[k]
            initial_idx = self.initial_indices[k]  # Correctly find the initial index
            index_pair = (initial_idx, idx + current_t)  # Correctly construct the index_pair

            # Fetch the current sample for this batch item
            sample = self.worker(index_pair)

            # Add index to the sample
            sample["index"] = idx

            # Concatenate data by common keys in sample
            for key, value in sample.items():
                if isinstance(value, np.ndarray):  # If it's a numpy array
                    value = torch.tensor(value)
                elif isinstance(value, np.int64):  # If it's a numpy scalar (int64)
                    value = torch.tensor(value, dtype=torch.int64)
                elif isinstance(value, (int, float)):  # If it's a native Python scalar (int/float)
                    value = torch.tensor(value, dtype=torch.float32)  # Ensure tensor is float for scalar
                elif not isinstance(value, torch.Tensor):  # If it's not already a tensor
                    value = torch.tensor(value)  # Ensure conversion to tensor

                # Convert zero-dimensional tensor (scalar) to 1D tensor
                if value.ndimension() == 0:
                    value = value.unsqueeze(0)  # Unsqueeze to make it a 1D tensor

                if key not in batch:
                    batch[key] = value  # Initialize the key in the batch dictionary
                else:
                    batch[key] = torch.cat((batch[key], value), dim=0)  # Concatenate values along the batch dimension

            # Increment time steps and forecast step counts for this batch item
            self.time_steps[k] += 1
            self.forecast_step_counts[k] += 1

        batch["forecast_step"] = self.forecast_step_counts[0]
        batch["stop_forecast"] = batch["forecast_step"] == self.forecast_len + 1
        batch["datetime"] = batch["datetime"].view(-1, self.batch_size)  # reshape

        return batch


if __name__ == "__main__":

    import logging
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from credit.transforms import load_transforms
    from credit.parser import credit_main_parser, training_data_check
    from credit.datasets import setup_data_loading, set_globals

    # Set up the logger
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)  # Create a logger with the module name

    with open(
        "/glade/derecho/scratch/schreck/repos/miles-credit/production/multistep/wxformer_6h/model.yml"
    ) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    conf = credit_main_parser(
        conf, parse_training=True, parse_predict=False, print_summary=False
    )
    training_data_check(conf, print_summary=False)
    data_config = setup_data_loading(conf)

    batch_size = 2
    data_config["forecast_len"] = 6

    set_globals(data_config, namespace=globals())

    # globals().update(data_config)
    # for key, value in data_config.items():
    #     globals()[key] = value
    #     logger.info(f"Creating global variable in the namespace: {key}")

    dataset_multi = ERA5_MultiStep_Batcher(
        varname_upper_air=data_config['varname_upper_air'],
        varname_surface=data_config['varname_surface'],
        varname_dyn_forcing=data_config['varname_dyn_forcing'],
        varname_forcing=data_config['varname_forcing'],
        varname_static=data_config['varname_static'],
        varname_diagnostic=data_config['varname_diagnostic'],
        filenames=data_config['all_ERA_files'],
        filename_surface=data_config['surface_files'],
        filename_dyn_forcing=data_config['dyn_forcing_files'],
        filename_forcing=data_config['forcing_files'],
        filename_static=data_config['static_files'],
        filename_diagnostic=data_config['diagnostic_files'],
        history_len=data_config['history_len'],
        forecast_len=data_config['forecast_len'],
        skip_periods=data_config['skip_periods'],
        max_forecast_len=data_config['max_forecast_len'],
        transform=load_transforms(conf),
        batch_size=batch_size
    )

    dataloader = DataLoader(
        dataset_multi,
        batch_size=1,  # Adjust the batch size as needed
        shuffle=False,   # Shuffle the dataset if needed
        num_workers=1,  # Number of subprocesses to use for data loading (adjust as needed)
        drop_last=True,  # Drop the last incomplete batch if not divisible by batch_size,
        prefetch_factor=4
    )

    dataloader.dataset.set_epoch(0)
    for (k, sample) in enumerate(dataloader):
        print(k, sample['index'], sample['datetime'], sample['forecast_step'], sample['stop_forecast'])
        if k == 20:
            break
