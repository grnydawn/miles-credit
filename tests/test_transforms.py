from credit.transforms import BridgescalerScaleState
from credit.data import Sample
import numpy as np
import xarray as xr

def test_BridgescalerScaleState():
    conf = {"data": {"quant_path": "data/era5_standard_scalers_2024-07-27_10:30.parquet",
                     "variables": ["U", "V"],
                     "surface_variables": ["U500", "V500"],
                    },
            "model": {"levels": 15}
            }
    data = xr.Dataset()
    d_shape = (1, 15, 16, 32)
    d2_shape = (1, 16, 32)

    data["U"] = (("time", "level", "latitude", "longitude"),
                 np.random.normal(1, 12, size=d_shape))
    data["V"] = (("time", "level", "latitude", "longitude"),
                 np.random.normal(-2, 20, size=d_shape))
    data["U500"] = (("time", "latitude", "longitude"),
                 np.random.normal(1, 13, size=d2_shape))
    data["V500"] = (("time", "latitude", "longitude"),
                    np.random.normal(-2, 20, size=d2_shape))
    data.coords["level"] = np.array([10, 30, 40, 50, 60, 70, 80, 90, 95, 100, 105, 110, 120, 130, 136],
                                      dtype=np.int64)
    samp = Sample()
    samp["historical_ERA5_images"] = data
    transform = BridgescalerScaleState(conf)
    transformed = transform.transform(samp)
    tdvars = transformed["historical_ERA5_images"].data_vars()
    assert tdvars == ["U", "V", "U500", "V500"]
    return