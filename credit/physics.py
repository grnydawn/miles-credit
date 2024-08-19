'''
Physics-based constraints and derivations for pressure-level CREDIT models
--------------------------------------------------------------------------
Content:
    - evaporation_rate
    - horizontal_advection
    - pressure_level_integral
    - surface_pressure_for_dry_air
    - geopotential_height
        - pressure_level_thickness
    - surface_pressure_to_slp

Usage:
    - Conservation of dry air mass (strong constraint)
        - Before and after a single-step, surface_pressure_for_dry_air (Pa) must equal
        
    - Conservation of water (strong constraint)
        - Before and after a single-step, evaporation_rate - precipitation + horizontal_advection(TWC) [mm] must equal
        - TWC = pressure_level_integral(Q + cloud + rain + snow)
        
    - hydrostatic equilibrium (weak constraint)
        - GPH derived from geopotential_height ~ ERA5 GPH on a set of levels [300 hPa, 500 hPa, 800 hPa, 1000 hPa]

Yingkai Sha
ksha@ucar.edu
'''

import numpy as np

import torch

# Earth's radius
RAD_EARTH = 6371000  # in meters

# ideal gas constant of water vapor
RVGAS = 461.5  # J/kg/K

# ideal gas constant of dry air
RDGAS = 287.05  # J/kg/K

# gravity
GRAVITY = 9.80665  # m/s^2

# latent heat caused by the phase change of water
# from liquid to gas
LATENT_HEAT_OF_VAPORIZATION = 2.5e6  # J/kg

SPECIFIC_HEAT_OF_DRY_AIR_CONST_PRESSURE = 1004.6  # J/kg/K

def virtual_temperature(air_temperature: torch.Tensor,
                       specific_humidity: torch.Tensor) -> torch.Tensor:
    '''
    Compute virtual_temperature

    Tv = T * (1 + gamma*Q)
    gamma = (Rv / Rd) - 1

    where
    - T: air temperature
    - Q: specific humidity
    - Rv, Rd: ideal gas constant for water vapor and dry air
    
    Args:
    
    Return:
    
    '''
    
    gamma = (RVGAS / RDGAS - 1.0)
    Tv = air_temperature * (1 + gamma*specific_humidity)
    return Tv

def evaporation_rate(latent_heat_flux) -> torch.Tensor:
    '''
    Compute evaporation rate based on the latent heat flux [kg m-2 s-1.]

    Args:
    Return:
    '''
    # latent_heat_flux [W/m^2]
    # convert to evaporation rate: (W/m^2) / (J/kg) = (J s^-1 m^-2) / (J/kg) = kg/m^2/s
    return latent_heat_flux / LATENT_HEAT_OF_VAPORIZATION
    
def horizontal_advection(u: torch.Tensor, 
                         v: torch.Tensor, 
                         T: torch.Tensor, 
                         lon: torch.Tensor, 
                         lat: torch.Tensor) -> torch.Tensor:
    '''
    Compute the horizonal advection of the quantity:

    adv = - (u * dT/dx + v * dT/dy)

    where
    - T = the quantity to be computed
    - u, v = horizonal wind component
    - lon, lat = longitude and latitude
    '''

    # latitude grid spacing [m]
    dy = torch.gradient(lat * (torch.pi/180) * RAD_EARTH, dim=0)[0]  # Convert lat degrees to meters

    # longitude grid spacing [m], adjusted by the cosine of latitude
    dx = torch.gradient(lon * (torch.pi/180) * RAD_EARTH * torch.cos(torch.radians(lat).unsqueeze(-1)), dim=1)[0]
    
    # Calculate the gradient of the quantity
    dT_dx = torch.gradient(T, spacing=(dx,), dim=-1)  # Gradient in the longitude direction
    dT_dy = torch.gradient(T, spacing=(dy,), dim=-2)  # Gradient in the latitude direction
    
    # Calculate the advection term: - (u * dT/dx + v * dT/dy)
    advection = -(u * dT_dx + v * dT_dy)
    
    return advection

def pressure_level_integral(T: torch.Tensor,
                            upper_air_pressure: torch.Tensor,) -> torch.Tensor:
    '''
    Computes a vertical integral with given pressure levels:

    (1 / g) * \int T dp

    where
    - g = gravity
    - T = the integrad
    - p = pressure level

    Args:
        T: (lat, lon, vertical_level), ()
        surface_pressure: (lat, lon), (Pa)
        upper_air_pressure: (vertical_level + 1) (Pa)

    Returns:
        Vertical integral of the integrand (lat, lon).
    '''
    pressure_thickness = upper_air_pressure.diff()
    integral = torch.sum(pressure_thickness * T, axis=-1)  # type: ignore
    return 1 / GRAVITY * integral


def surface_pressure_for_dry_air(specific_total_water: torch.Tensor,
                                 surface_pressure: torch.Tensor,
                                 upper_air_pressure: torch.Tensor,) -> torch.Tensor:
    '''
    Compute the surface pressure of dry air (Pa)

    P_dry = P_surf - g * TWC

    where
    - P_dry: surface pressure due to dry air
    - P_surf: surface pressure of all gases, including water vapor
    - TWC: total water content, derived from specific total water
           specific total water = specific humidity + cloud ice + cloud liquid + rain + snow
    - g: gravity
    
    Args:
        specific_total_water (lat, lon, vertical_level), (kg/kg)
        surface_pressure (lat, lon), (Pa)
        upper_air_pressure: (vertical_level + 1) (Pa)
    
    Returns:
        P_dry (Pa)
    '''
    
    TWC = pressure_level_integral(specific_total_water, 
                                  upper_air_pressure)
    
    p_dry = surface_pressure - GRAVITY * TWC
    return p_dry

def geopotential_height(upper_air_pressure: torch.Tensor,
                        air_temperature: torch.Tensor,
                        specific_humidity: torch.Tensor,
                        surface_height: torch.tensor) -> torch.Tensor:
    '''
    Compute geopotential height (GPH) for a given pressure level 
    using air temperature and specific_humidity.

    Args:
    Returns:
    '''
    layer_thickness = pressure_level_thickness(
        upper_air_pressure, air_temperature, specific_humidity)

    # flip so it is now from bottom to top
    layer_thickness = layer_thickness.flip(dims=(-1,))
    
    # cumulate thickness to height                                     
    cumulative_thickness = torch.cumsum(layer_thickness, dim=-1)

    # flip back so it is now from top to bottom
    cumulative_thickness.flip(dims=(-1,))
    
    # fill negative surface height with 0
    H_surf = torch.where(surface_height < 0.0, 0, surface_height).reshape(*surface_height.shape, 1)

    # combine upper-air thickness with surface height (broadcast on vertical dim)
    H_upper = cumulative_thickness + H_surf.broadcast_to(cumulative_thickness.shape)

    # compute geopotential height
    # H_upper is top to bottom, so concat surface in the end
    GPH = torch.concat([H_upper, H_surf], dim=-1)
    # e.g. Z500 = GPH[..., index_500hPa]
    
    return GPH
    
def pressure_level_thickness(upper_air_pressure: torch.Tensor,
                             air_temperature: torch.Tensor,
                             specific_humidity: torch.Tensor,) -> torch.Tensor:
    
    '''
    Computes pressure level thickness using hydrostatic equilibrium.

    thickness = (RDGAS * Tv / g) * (log(p1) - log(p2))

    where
    - Tv: virtual temperature
    - g: gravity
    - p1, p2: two pressure levels

    Args:
    
    Returns:
    
    '''
    # Compute Tv
    Tv = virtual_temperature(air_temperature, specific_humidity)
    
    # Compute logP diff
    dlogp = torch.log(upper_air_pressure).diff()
    
    # thickness
    thickness = (RDGAS * Tv / GRAVITY) * dlogp
    
    return thickness

def surface_pressure_to_slp(surface_pressure: torch.Tensor,
                            air_temperature: torch.Tensor,
                            specific_humidity: torch.Tensor,
                            surface_height: torch.tensor,):
    '''
    Compute surface pressure using barometric equation

    slp = p_surf * exp((g * h) / (Rd * Tv))

    where
    - slp: sea level pressure
    - p_surf: surface pressure
    - g: gravity
    - h: surface_height
    - Rd: ideal gas constant for dry air
    - Tv: virtual temperature
    
    Args:
    
    Returns:
    
    '''
    # Compute Tv
    Tv = virtual_temperature(air_temperature, specific_humidity)
    
    slp = surface_pressure * np.exp((GRAVITY * surface_height) / (RDGAS * Tv))

    return slp




