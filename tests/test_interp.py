from credit.interp import full_state_pressure_interpolation
import xarray as xr
import os
import numpy as np


def test_full_state_pressure_interpolation():
    path_to_test = os.path.abspath(os.path.dirname(__file__))
    input_file = os.path.join(path_to_test, "data/test_interp.nc")
    model_level_file = os.path.join(path_to_test, "../credit/metadata/ERA5_Lev_Info.nc")
    ds = xr.open_dataset(input_file)
    model_levels = xr.open_dataset(model_level_file)
    pressure_levels = np.array([200.0, 500.0, 700.0, 850.0, 1000.0]) * 100.0
    model_a = model_levels["a_model"].loc[ds["level"]].values
    model_b = model_levels["b_model"].loc[ds["level"]].values
    interp_ds = full_state_pressure_interpolation(ds, 
                                                  pressure_levels, 
                                                  model_a, 
                                                  model_b,
                                                  lat_var="lat",
                                                  lon_var="lon")
    assert interp_ds["U"].shape[1] == pressure_levels.size, "Pressure level mismatch"
    return
