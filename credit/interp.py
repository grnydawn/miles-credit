import numpy as np
from numba import njit
import xarray as xr


@njit
def create_pressure_grid(surface_pressure, model_a, model_b):
    """
    Create a 3D pressure field at model levels from the surface pressure field and the hybrid sigma-pressure
    coefficients from ECMWF. Conversion is `pressure_3d = a + b * SP`.

    Args:
        surface_pressure: (time, latitude, longitude) or (latitude, longitude) grid in units of Pa.
        model_a: a coefficients at each model level being used.
        model_b: b coefficients at each model level being used.

    Returns:
        pressure_3d: 3D pressure field with dimensions of surface_pressure and number of levels from model_a and model_b.
    """
    assert model_a.size == model_b.size, "Model pressure coefficient arrays do not match."
    if len(surface_pressure.shape) == 3:
        # Generate the 3D pressure field for a time series of surface pressure grids
        pressure_3d = np.zeros((surface_pressure.shape[0], model_a.shape[0],
                                surface_pressure.shape[1], surface_pressure.shape[2]), dtype=surface_pressure.dtype)
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
    pressure_var = np.zeros((interp_pressures.shape[0],
                             model_var.shape[1], model_var.shape[2]), dtype=model_var.dtype)
    log_interp_pressures = np.log(interp_pressures)
    for (i, j), v in np.ndenumerate(model_var[0]):
        pressure_var[:, i, j] = np.interp(log_interp_pressures,
                                          np.log(model_pressure[:, i, j]),
                                          model_var[:, i, j])
    return pressure_var


@njit
def geopotential_height_from_model_vars(surface_geopotential, surface_pressure,
                                        temperature, mixing_ratio,
                                        model_a, model_b):
    Rd = 287.0
    half_a = 0.5 * (model_a[:-1] + model_a[1:])
    half_b = 0.5 * (model_b[:-1] + model_b[1:])
    model_pressure = create_pressure_grid(surface_pressure, model_a, model_b)
    half_pressure = create_pressure_grid(surface_pressure, half_a, half_b)
    model_geopotential = np.zeros(model_pressure.shape, dtype=surface_pressure.dtype)
    half_geopotential = np.zeros(half_pressure.shape, dtype=surface_pressure.dtype)
    virtual_temperature = temperature * (1.0 + 0.608 * mixing_ratio)
    m = model_geopotential.shape[-3] - 1
    h = half_geopotential.shape[-3] - 1
    model_geopotential[m] = (surface_geopotential +
                             Rd * virtual_temperature[m] * np.log(surface_pressure / model_pressure[m]))
    for i in range(1, model_geopotential.shape[-3]):
        half_geopotential[h] = (model_geopotential[m] +
                                Rd * virtual_temperature[m] * np.log(model_pressure[m] / half_pressure[h]))
        m -= 1
        model_geopotential[m] = (half_geopotential[h] +
                                 Rd * virtual_temperature[m] * np.log(half_pressure[h] / model_pressure[m]))
        h -= 1
    return model_geopotential
