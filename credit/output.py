"""
output.py
-------------------------------------------------------
Content:
    - load_metadata()
    - make_xarray()
    - save_netcdf_increment()
"""

import os
import yaml
import logging
import traceback
import xarray as xr
from credit.data import drop_var_from_dataset
from credit.interp import full_state_pressure_interpolation
import numpy as np

logger = logging.getLogger(__name__)


def load_metadata(conf: dict):
    """
    Load metadata attributes from yaml file in credit/metadata directory

    Args:
        conf (dict): Configuration dictionary

    """
    # set priorities for user-specified metadata
    if conf["predict"]["metadata"]:
        meta_file = conf["predict"]["metadata"]
        with open(meta_file) as f:
            meta_data = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        print("conf['predict']['metadata'] not given. Skip.")
        meta_data = False

    return meta_data


def split_and_reshape(tensor, conf):
    """
    Split the output tensor of the model to upper air variables and diagnostics/surface variables.

    Upperair level arrangement: top-of-atmosphere--> near-surface --> single layer
    An example: U (top-of-atmosphere) --> U (near-surface) --> V (top-of-atmosphere) --> V (near-surface)
    The shape of the output tensor is (variables, latitude, longitude)

    Args:
        tensor: PyTorch Tensor containing output of the AI NWP model
        conf: config file for the model

    """

    # get the number of levels
    levels = conf["model"]["levels"]

    # get number of channels
    channels = len(conf["data"]["variables"])
    single_level_channels = len(conf["data"]["surface_variables"])

    # subset upper air variables
    tensor_upper_air = tensor[:, : int(channels * levels), :, :]

    shape_upper_air = tensor_upper_air.shape
    tensor_upper_air = tensor_upper_air.view(
        shape_upper_air[0], channels, levels, shape_upper_air[-2], shape_upper_air[-1]
    )

    # subset surface variables
    tensor_single_level = tensor[:, -int(single_level_channels) :, :, :]

    # return x, surf for B, c, lat, lon output
    return tensor_upper_air, tensor_single_level


def make_xarray(pred, forecast_datetime, lat, lon, conf):
    """
    Convert prediction tensor to xarray DataArrays for later saving.

    Args:
        pred (torch.Tensor): full tensor containing output of the AI NWP model
        forecast_datetime (pd.Timestamp or datetime.datetime): valid time of the forecast
        lat: latitude coordinate array
        lon: longitude coordinate array
        conf (dict): config dictionary for training/rollout

    Returns:
        xr.DataArray: upper air predictions, xr.DataArray: surface variable predictions
    """
    # subset upper air and surface variables
    tensor_upper_air, tensor_single_level = split_and_reshape(pred, conf)

    if "level_ids" in conf["data"].keys():
        level_ids = conf["data"]["level_ids"]
    else:
        level_ids = np.array(
            [10, 30, 40, 50, 60, 70, 80, 90, 95, 100, 105, 110, 120, 130, 136, 137],
            dtype=np.int64,
        )

    # save upper air variables
    darray_upper_air = xr.DataArray(
        tensor_upper_air,
        dims=["time", "vars", "level", "lat", "lon"],
        coords=dict(
            vars=conf["data"]["variables"],
            time=[forecast_datetime],
            level=level_ids,
            lat=lat,
            lon=lon,
        ),
    )

    # save surface variables
    # !!! need to add diagnostic vars !!!
    darray_single_level = xr.DataArray(
        tensor_single_level.squeeze(2),
        dims=["time", "vars", "lat", "lon"],
        coords=dict(
            vars=conf["data"]["surface_variables"],
            time=[forecast_datetime],
            lat=lat,
            lon=lon,
        ),
    )
    # return DataArrays as outputs
    return darray_upper_air, darray_single_level


def save_netcdf_increment(
    darray_upper_air: xr.DataArray,
    darray_single_level: xr.DataArray,
    nc_filename: str,
    forecast_hour: int,
    meta_data: dict,
    conf: dict,
):
    """
    Save CREDIT model prediction output to netCDF file. Also performs pressure level
    interpolation on the output if you wish.

    Args:
        darray_upper_air (xr.DataArray): upper air variable predictions
        darray_single_level (xr.DataArray): surface variable predictions
        nc_filename (str): file description to go into output filenames
        forecast_hour (int):  how many hours since the initialization of the model.
        meta_data (dict): metadata dictionary for output variables
        conf (dict): configuration dictionary for training and/or rollout
    """
    try:
        """
        Save increment to a unique NetCDF file using Dask for parallel processing.
        """
        # Convert DataArrays to Datasets
        ds_upper = darray_upper_air.to_dataset(dim="vars")
        ds_single = darray_single_level.to_dataset(dim="vars")

        # Merge datasets
        ds_merged = xr.merge([ds_upper, ds_single])

        # Add forecast_hour coordinate
        ds_merged["forecast_hour"] = forecast_hour

        # Add CF convention version
        ds_merged.attrs["Conventions"] = "CF-1.11"

        if "interp_pressure" in conf["predict"].keys():
            if "surface_geopotential_var" in conf["predict"]["interp_pressure"].keys():
                surface_geopotential_var = conf["predict"]["interp_pressure"][
                    "surface_geopotential_var"
                ]
            else:
                surface_geopotential_var = "Z_GDS4_SFC"
            with xr.open_dataset(conf["data"]["save_loc_static"]) as static_ds:
                ds_merged[surface_geopotential_var] = static_ds[
                    surface_geopotential_var
                ]
            pressure_interp = full_state_pressure_interpolation(
                ds_merged, **conf["predict"]["interp_pressure"]
            )
            ds_merged = xr.merge([ds_merged, pressure_interp])

        logger.info(f"Trying to save forecast hour {forecast_hour} to {nc_filename}")

        save_location = os.path.join(conf["predict"]["save_forecast"], nc_filename)
        os.makedirs(save_location, exist_ok=True)

        unique_filename = os.path.join(
            save_location, f"pred_{nc_filename}_{forecast_hour:03d}.nc"
        )
        # ---------------------------------------------------- #
        # If conf['predict']['save_vars'] provided --> drop useless vars
        if "save_vars" in conf["predict"]:
            if len(conf["predict"]["save_vars"]) > 0:
                ds_merged = drop_var_from_dataset(
                    ds_merged, conf["predict"]["save_vars"]
                )

        # when there's no metafile --> meta_data = False
        if meta_data is not False:
            # Add metadata attributes to every model variable if available
            for var in ds_merged.variables:
                if var in meta_data.keys():
                    if var != "time":
                        # use attrs.update for non-datetime variables
                        ds_merged[var].attrs.update(meta_data[var])
                    else:
                        # use time.encoding for datetime variables/coords
                        for metadata_time in meta_data["time"]:
                            ds_merged.time.encoding[metadata_time] = meta_data["time"][
                                metadata_time
                            ]
        encoding_dict = {}
        if "ua_var_encoding" in conf["predict"].keys():
            for ua_var in conf["data"]["variables"]:
                encoding_dict[ua_var] = conf["predict"]["ua_var_encoding"]
        if "surface_var_encoding" in conf["predict"].keys():
            for surface_var in conf["data"]["variables"]:
                encoding_dict[surface_var] = conf["predict"]["surface_var_encoding"]
        if "pressure_var_encoding" in conf["predict"].keys():
            for pres_var in conf["data"]["variables"]:
                encoding_dict[pres_var] = conf["predict"]["pressure_var_encoding"]
        # Use Dask to write the dataset in parallel
        ds_merged.to_netcdf(unique_filename, encoding=encoding_dict)

        logger.info(f"Saved forecast hour {forecast_hour} to {unique_filename}")
    except Exception:
        print(traceback.format_exc())
