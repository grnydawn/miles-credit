import numpy as np
from numba import njit
import xarray as xr
from tqdm import tqdm
from .physics_constants import RDGAS, RVGAS


def full_state_pressure_interpolation(
    state_dataset: xr.Dataset,
    pressure_levels: np.ndarray,
    model_a: np.ndarray,
    model_b: np.ndarray,
    interp_fields: tuple[str] = ("U", "V", "T", "Q"),
    temperature_var: str = "T",
    q_var: str = "Q",
    surface_pressure_var: str = "SP",
    surface_geopotential_var: str = "Z_GDS4_SFC",
    geopotential_var: str = "Z",
    time_var: str = "time",
    lat_var: str = "latitude",
    lon_var: str = "longitude",
    pres_var: str = "pressure",
    verbose: int = 1,
) -> xr.Dataset:
    """
    Interpolate full model state variables from model levels to pressure levels.

    Args:
        state_dataset (xr.Dataset): state variables being interpolated
        pressure_levels (np.ndarray): pressure levels for interpolation in Pa.
        model_a (np.ndarray): model level a coefficients.
        model_b (np.ndarray): model level b coefficients.
        interp_fields (tuple[str]): fields to be interpolated.
        temperature_var (str): temperature variable to be interpolated (units K).
        q_var (str): mixing ratio/specific humidity variable to be interpolated (units kg/kg).
        surface_pressure_var (str): surface pressure variable (units Pa).
        surface_geopotential_var (str): surface geoptential variable (units m^2/s^2).
        geopotential_var (str): geopotential variable being derived (units m^2/s^2).
        time_var (str): time coordinate
        lat_var (str): latitude coordinate
        lon_var (str): longitude coordinate
        pres_var (str): pressure coordinate
        verbose (int): verbosity level. If verbose > 0, print progress.
    Returns:
        pressure_ds (xr.Dataset): Dataset containing pressure interpolated variables.
    """
    pres_dims = (time_var, pres_var, lat_var, lon_var)
    coords = {
        time_var: state_dataset[time_var],
        pres_var: pressure_levels,
        lat_var: state_dataset[lat_var],
        lon_var: state_dataset[lon_var],
    }
    pressure_ds = xr.Dataset(
        data_vars={
            f: xr.DataArray(
                coords=coords, dims=pres_dims, name=f, attrs=state_dataset[f].attrs
            )
            for f in interp_fields
        },
        coords=coords,
    )
    pressure_ds[geopotential_var] = xr.DataArray(
        coords=coords, dims=pres_dims, name=geopotential_var
    )
    disable = False
    if verbose == 0:
        disable = True
    for t, time in tqdm(enumerate(state_dataset[time_var]), disable=disable):
        pressure_grid = create_pressure_grid(
            state_dataset[surface_pressure_var][t].values, model_a, model_b
        )
        geopotential_grid = geopotential_from_model_vars(
            state_dataset[surface_geopotential_var][t].values,
            state_dataset[surface_pressure_var][t].values,
            state_dataset[temperature_var][t].values,
            state_dataset[q_var][t].values,
            model_a,
            model_b,
        )
        for interp_field in interp_fields:
            pressure_ds[interp_field][t] = interp_hybrid_to_pressure_levels(
                state_dataset[interp_field][t].values, pressure_grid, pressure_levels
            )
        pressure_ds[geopotential_var][t] = interp_hybrid_to_pressure_levels(
            geopotential_grid, pressure_grid, pressure_levels
        )
    return pressure_ds


@njit
def create_pressure_grid(surface_pressure, model_a, model_b):
    """
    Create a 3D pressure field at model levels from the surface pressure field and the hybrid sigma-pressure
    coefficients from ECMWF. Conversion is `pressure_3d = a + b * SP`.

    Args:
        surface_pressure (np.ndarray): (time, latitude, longitude) or (latitude, longitude) grid in units of Pa.
        model_a (np.ndarray): a coefficients at each model level being used in units of Pa.
        model_b (np.ndarray): b coefficients at each model level being used (unitness).

    Returns:
        pressure_3d: 3D pressure field with dimensions of surface_pressure and number of levels from model_a and model_b.
    """
    assert (
        model_a.size == model_b.size
    ), "Model pressure coefficient arrays do not match."
    if surface_pressure.ndim == 3:
        # Generate the 3D pressure field for a time series of surface pressure grids
        pressure_3d = np.zeros(
            (
                surface_pressure.shape[0],
                model_a.shape[0],
                surface_pressure.shape[1],
                surface_pressure.shape[2],
            ),
            dtype=surface_pressure.dtype,
        )
        model_a_3d = model_a.reshape(-1, 1, 1)
        model_b_3d = model_b.reshape(-1, 1, 1)
        for i in range(surface_pressure.shape[0]):
            pressure_3d[i] = model_a_3d + model_b_3d * surface_pressure[i]
    else:
        # Generate the 3D pressure field for a single surface pressure grid.
        model_a_3d = model_a.reshape(-1, 1, 1)
        model_b_3d = model_b.reshape(-1, 1, 1)
        pressure_3d = model_a_3d + model_b_3d * surface_pressure
    return pressure_3d


@njit
def interp_hybrid_to_pressure_levels(model_var, model_pressure, interp_pressures):
    """
    Interpolate data field from hybrid sigma-pressure vertical coordinates to pressure levels.

    Args:
        model_var (np.ndarray): 3D field on hybrid sigma-pressure levels with shape (levels, y, x).
        model_pressure (np.ndarray): 3D pressure field with shape (levels, y, x) in units Pa
        interp_pressures: (np.ndarray): pressure levels for interpolation in units Pa.

    Returns:
        pressure_var (np.ndarray): 3D field on pressure levels with shape (len(interp_pressures), y, x).
    """
    pressure_var = np.zeros(
        (interp_pressures.shape[0], model_var.shape[1], model_var.shape[2]),
        dtype=model_var.dtype,
    )
    log_interp_pressures = np.log(interp_pressures)
    for (i, j), v in np.ndenumerate(model_var[0]):
        pressure_var[:, i, j] = np.interp(
            log_interp_pressures, np.log(model_pressure[:, i, j]), model_var[:, i, j]
        )
    return pressure_var


@njit
def geopotential_from_model_vars(
    surface_geopotential, surface_pressure, temperature, mixing_ratio, model_a, model_b
):
    """
    Calculate geopotential from the base state variables. Geopotential height is calculated by adding thicknesses
    calculated within each half-model-level to account for variations in temperature and moisture between grid cells.
    Note that this function is calculating geopotential in units of (m^2 s^-2) not geopential height.

    To convert geopotential to geopotential height, divide geopotential by g (9.806 m s^-2).

    Geopotential height is defined as the height above mean sea level. To get height above ground level, substract
    the surface geoptential height field from the 3D geopotential height field.

    Args:
        surface_geopotential (np.ndarray): Surface geopotential in shape (y,x) and units m^2 s^-2.
        surface_pressure (np.ndarray): Surface pressure in shape (y, x) and units Pa
        temperature (np.ndarray): temperature in shape (levels, y, x) and units K
        mixing_ratio (np.ndarray): mixing ratio in shape (levels, y, x) and units kg/kg.
        model_a (np.ndarray): a coefficients at each model level being used in units of Pa.
        model_b (np.ndarray): b coefficients at each model level being used (unitness).

    Returns:
        model_geoptential (np.ndarray): geopotential on model levels in shape (levels, y, x)
    """
    gamma = RVGAS / RDGAS - 1.0
    half_a = 0.5 * (model_a[:-1] + model_a[1:])
    half_b = 0.5 * (model_b[:-1] + model_b[1:])
    model_pressure = create_pressure_grid(surface_pressure, model_a, model_b)
    half_pressure = create_pressure_grid(surface_pressure, half_a, half_b)
    model_geopotential = np.zeros(model_pressure.shape, dtype=surface_pressure.dtype)
    half_geopotential = np.zeros(half_pressure.shape, dtype=surface_pressure.dtype)
    virtual_temperature = temperature * (1.0 + gamma * mixing_ratio)
    m = model_geopotential.shape[-3] - 1
    h = half_geopotential.shape[-3] - 1
    model_geopotential[m] = surface_geopotential + RDGAS * virtual_temperature[
        m
    ] * np.log(surface_pressure / model_pressure[m])
    for i in range(1, model_geopotential.shape[-3]):
        half_geopotential[h] = model_geopotential[m] + RDGAS * virtual_temperature[
            m
        ] * np.log(model_pressure[m] / half_pressure[h])
        m -= 1
        model_geopotential[m] = half_geopotential[h] + RDGAS * virtual_temperature[
            m
        ] * np.log(half_pressure[h] / model_pressure[m])
        h -= 1
    return model_geopotential
