import numpy as np
from credit.data import (
    MPASA_and_Forcing_Dataset,
    Sample,
    find_key_for_number,
    extract_month_day_hour,
    find_common_indices,
)


class MPASA_and_Forcing_SingleStep(MPASA_and_Forcing_Dataset):

    def __getitem__(self, index):
        # cross-year indices --> the index of the year + indices within that year

        # select the ind_file based on the iter index
        ind_file = find_key_for_number(index, self.MPASA_indices)

        # get the ind within the current file
        ind_start = self.MPASA_indices[ind_file][1]
        ind_start_in_file = index - ind_start

        # handle out-of-bounds
        ind_largest = len(self.all_files[int(ind_file)]["time"]) - (
            self.history_len + self.forecast_len + 1
        )
        if ind_start_in_file > ind_largest:
            ind_start_in_file = ind_largest

        # subset xarray on time dimension

        ind_end_in_file = ind_start_in_file + self.history_len + self.forecast_len

        # ERA5_subset: a xarray dataset that contains training input and target (for the current batch)
        MPASA_subset = self.all_files[int(ind_file)].isel(
            time=slice(ind_start_in_file, ind_end_in_file + 1)
        )  # .load() NOT load into memory

        # merge surface into the dataset

        if self.surface_files:
            # subset surface variables
            surface_subset = self.surface_files[int(ind_file)].isel(
                time=slice(ind_start_in_file, ind_end_in_file + 1)
            )

            # YSK ORG: merge upper-air and surface here:
            MPASA_subset = MPASA_subset.merge(
                surface_subset
            )

        # split MPASA_subset into training inputs and targets
        #   + merge with dynamic forcing, forcing, and static
        # the ind_end of the MPASA_subset
        ind_end_time = len(MPASA_subset["time"])

        # datetiem information as int number (used in some normalization methods)
        datetime_as_number = MPASA_subset.time.values.astype("datetime64[s]").astype(int)

        # xarray dataset as input
        # historical_MPASA_images: the final input
        historical_MPASA_images = MPASA_subset.isel(
            # YSK ORG: time=slice(0, self.history_len, self.skip_periods)
            time=slice(0, self.history_len, 1)
        ).load()

        if self.one_shot is not None:
            # one_shot is True (on), go straight to the last element
            target_MPASA_images = MPASA_subset.isel(
                time=slice(-1, None)
            ).load()

        else:
            # one_shot is None (off), get the full target length based on forecast_len
            target_MPASA_images = MPASA_subset.isel(
                # YSK ORG: time=slice(self.history_len, ind_end_time, self.skip_periods)
                time=slice(self.history_len, ind_end_time, 1)
            ).load()  # <-- load into memory

        # pipe xarray datasets to the sampler
        sample = Sample(
            historical_MPASA_images=historical_MPASA_images,
            target_MPASA_images=target_MPASA_images,
            datetime_index=datetime_as_number,
        )

        # data normalization
        if self.transform:
            sample = self.transform(sample)

        # add datetime for convenice
        sample["datetime"] = [
            int(
                historical_MPASA_images.time.values[0]
                .astype("datetime64[s]")
                .astype(int)
            ),
            int(target_MPASA_images.time.values[0].astype("datetime64[s]").astype(int)),
        ]

        # assign sample index
        sample["index"] = index
        # These are here for consistency with trainers.
        # Forecast_step is always 1
        sample["forecast_step"] = 1
        # Hence stop_forecast is always true
        sample["stop_forecast"] = True

        return sample


if __name__ == "__main__":

    import yaml
    from torch.utils.data import DataLoader
    from credit.transforms import load_transforms
    from credit.parser import credit_main_parser, training_data_check
    from credit.datasets import setup_data_loading, set_globals

    with open(
        "/glade/derecho/scratch/schreck/repos/miles-credit/production/multistep/wxformer_6h/model.yml"
    ) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    conf = credit_main_parser(
        conf, parse_training=True, parse_predict=False, print_summary=False
    )
    training_data_check(conf, print_summary=False)

    data_config = setup_data_loading(conf)

    data_config["forecast_len"] = 1

    set_globals(data_config, namespace=globals())

    dataset_multi = MPASA_and_Forcing_SingleStep(
        varname_upper_air=data_config['varname_upper_air'],
        varname_surface=data_config['varname_surface'],
        history_len=data_config['history_len'],
        forecast_len=data_config['forecast_len'],
        one_shot=False,
        transform=load_transforms(conf)
    )

    dataloader = DataLoader(
        dataset_multi,
        batch_size=2,  # Adjust the batch size as needed
        shuffle=True,   # Shuffle the dataset if needed
        num_workers=4,  # Number of subprocesses to use for data loading (adjust as needed)
        drop_last=True,  # Drop the last incomplete batch if not divisible by batch_size,
        prefetch_factor=4
    )

    for (k, sample) in enumerate(dataloader):
        print(k, sample['index'], sample['datetime'], sample['forecast_step'], sample['stop_forecast'], sample["x"].shape, sample["x_surf"].shape)
        if k == 20:
            break
