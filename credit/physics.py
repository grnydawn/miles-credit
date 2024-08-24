'''
Physics-based constraints and derivations for CREDIT models
--------------------------------------------------------------------------
Content:
    - physics_pressure_level
        - virtual_temperature
        - evaporation_from_latent_heat
        - latent_heat_from_evaporation
        - horizontal_advection
        - pressure_level_integral
        - surface_pressure_for_dry_air
        - pressure_level_thickness
        - geopotential_height
        
Usage:
    - Conservation of dry air mass (constraint)
        - Before and after a single-step, surface_pressure_for_dry_air (Pa) must equal
        
    - Conservation of water (constraint)
        - Before and after a single-step, 
          evaporation_rate - precipitation + horizontal_advection(TWC) [mm] 
          must equal to (dTWC / dt)
        - TWC = (1/g) * pressure_level_integral(specific_humidity)
        
    - hydrostatic equilibrium (weak constraint)
        - GPH derived from geopotential_height ~ predicted ERA5 GPH

Yingkai Sha
ksha@ucar.edu
'''

import numpy as np
import torch

# =============================================== #
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
# =============================================== #

class physics_pressure_level:
    '''
    Pressure level physics

    Attributes:
        upper_air_pressure (torch.Tensor): 
    
    Methods:
        virtual_temperature():
            xxx
        
        evaporation_from_latent_heat():
            xxx

        latent_heat_from_evaporation():
            xxx
            
        horizontal_advection():
            xxx

        pressure_level_integral():
            xxx

        surface_pressure_for_dry_air():
            xxx

        pressure_level_thickness():
            xxx
            
        geopotential_height():
            xxx
    '''
    
    def __init__(self, 
                 upper_air_pressure: torch.Tensor):
        
        # get pressure (Pa) per level directly
        self.upper_air_pressure = upper_air_pressure
        # get pressure level thickness
        self.pressure_thickness = self.upper_air_pressure.diff()
        
    def virtual_temperature(self, 
                            air_temperature: torch.Tensor, 
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

    def evaporation_from_latent_heat(self, 
                                     latent_heat_flux: torch.Tensor) -> torch.Tensor:
        '''
        Compute evaporation rate based on the latent heat flux [kg m-2 s-1.]
    
        Args:
        
        Return:
        
        '''
        # latent_heat_flux [W/m^2]
        # convert to evaporation rate: (W/m^2) / (J/kg) = (J s^-1 m^-2) / (J/kg) = kg/m^2/s
        return latent_heat_flux / LATENT_HEAT_OF_VAPORIZATION

    def latent_heat_from_evaporation(self, 
                                     evaporation: torch.Tensor) -> torch.Tensor:
        '''
        Compute latent heat from evaporation
        
        Args:
        
        Return:
        
        '''
        return evaporation * LATENT_HEAT_OF_VAPORIZATION

    def horizontal_advection(self, 
                             u: torch.Tensor, 
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

    def pressure_level_integral(self, 
                                T: torch.Tensor) -> torch.Tensor:
        '''
        Computes a vertical integral with given pressure levels:
    
        (1 / g) * \int T dp
    
        where
        - g = gravity
        - T = the integrad
        - p = pressure level
    
        Args:
            T: (lat, lon, vertical_level), ()
            
        Returns:
            Vertical integral of the integrand (lat, lon).
        '''
        integral = torch.sum(self.pressure_thickness * T, dim=-1)  # type: ignore
        return integral

    def surface_pressure_for_dry_air(self, 
                                     specific_humidity: torch.Tensor, 
                                     surface_pressure: torch.Tensor) -> torch.Tensor:
        '''
        Compute the surface pressure of dry air (Pa)
    
        P_dry = P_surf - g * TWC
    
        where
        - P_dry: surface pressure due to dry air
        - P_surf: surface pressure of all gases, including water vapor
        - TWC: total water content, derived from specific_humidity
        - g: gravity
        
        Args:
            specific_humidity (lat, lon, vertical_level), (kg/kg)
            surface_pressure (lat, lon), (Pa)
        
        Returns:
            P_dry (Pa)
        '''
        
        TWC = (1 / GRAVITY) * self.pressure_level_integral(specific_humidity)
        
        p_dry = surface_pressure - GRAVITY * TWC
        return p_dry

    def pressure_level_thickness(self, 
                                 air_temperature: torch.Tensor, 
                                 specific_humidity: torch.Tensor) -> torch.Tensor:
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
        Tv = self.virtual_temperature(air_temperature, specific_humidity)
        
        # Compute logP diff
        dlogp = torch.log(self.upper_air_pressure).diff()
        
        # thickness
        thickness = (RDGAS * Tv / GRAVITY) * dlogp
        
        return thickness
    
    def geopotential_height(self,
                            air_temperature: torch.Tensor, 
                            specific_humidity: torch.Tensor, 
                            surface_height: torch.Tensor) -> torch.Tensor:
        '''
        Compute geopotential height (GPH) for a given pressure level 
        using air temperature and specific humidity.
    
        Args:
        Returns:
        '''
        layer_thickness = self.pressure_level_thickness(air_temperature, specific_humidity)
    
        # flip so it is now from bottom to top
        layer_thickness = layer_thickness.flip(dims=(-1,))
        
        # cumulate thickness to height                                     
        cumulative_thickness = torch.cumsum(layer_thickness, dim=-1)
    
        # flip back so it is now from top to bottom
        cumulative_thickness = cumulative_thickness.flip(dims=(-1,))
        
        # fill negative surface height with 0
        H_surf = torch.where(surface_height < 0.0, 0, surface_height).reshape(*surface_height.shape, 1)
    
        # combine upper-air thickness with surface height (broadcast on vertical dim)
        H_upper = cumulative_thickness + H_surf.broadcast_to(cumulative_thickness.shape)
    
        # compute geopotential height
        # H_upper is top to bottom, so concat surface in the end
        GPH = torch.concat([H_upper, H_surf], dim=-1)
        # e.g. Z500 = GPH[..., index_500hPa]
        
        return GPH
        
    # def surface_pressure_to_slp(self, 
    #                             surface_pressure: torch.Tensor, 
    #                             air_temperature: torch.Tensor, 
    #                             specific_humidity: torch.Tensor, 
    #                             surface_height: torch.Tensor) -> torch.Tensor:
    #     '''
    #     Compute surface pressure using barometric equation
    
    #     slp = p_surf * exp((g * h) / (Rd * Tv))
    
    #     where
    #     - slp: sea level pressure
    #     - p_surf: surface pressure
    #     - g: gravity
    #     - h: surface_height
    #     - Rd: ideal gas constant for dry air
    #     - Tv: virtual temperature
        
    #     Args:
        
    #     Returns:
        
    #     '''
    #     # Compute Tv
    #     Tv = self.virtual_temperature(air_temperature, specific_humidity)
        
    #     slp = surface_pressure * np.exp((GRAVITY * surface_height) / (RDGAS * Tv))
    
    #     return slp



