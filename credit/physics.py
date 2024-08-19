'''
Physics-based constraints and derivations for CREDIT models 
'''


import numpy as np

import torch


LATENT_HEAT_OF_VAPORIZATION = 2.5e6  # J/kg
GRAVITY = 9.80665  # m/s^2
SPECIFIC_HEAT_OF_DRY_AIR_CONST_PRESSURE = 1004.6  # J/kg/K


def vertical_integral(
    integrand: torch.Tensor,
    surface_pressure: torch.Tensor,
    upper_air_pressure: torch.Tensor,
) -> torch.Tensor:
    '''
    Computes a vertical integral:

    (1 / g) * ∫ x dp

    where
    - g = acceleration due to gravity
    - x = integrad
    - p = pressure level

    Args:
        integrand (lat, lon, vertical_level), (kg/kg)
        surface_pressure: (lat, lon), (Pa)
        upper_air_pressure: (vertical_level) (Pa)

    Returns:
        Vertical integral of the integrand (lat, lon).
    '''
    pressure_thickness = upper_air_pressure.diff(dim=-1)
    integral = torch.sum(pressure_thickness * integrand, axis=-1)  # type: ignore
    return 1 / GRAVITY * integral



