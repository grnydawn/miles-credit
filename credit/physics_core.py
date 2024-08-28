'''
Physics-based constraints and derivations for CREDIT models
--------------------------------------------------------------------------
Content:
    - physics_pressure_level
        - virtual_temperature
        - evaporation_from_latent_heat
        - latent_heat_from_evaporation
        - divergence_flux
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
          E - P - pressure_level_integral(divergence_flux(q)) = (dTWC/dt)
        - TWC = pressure_level_integral(q)
        
    - hydrostatic equilibrium (weak constraint)
        - GPH derived from geopotential_height ~ predicted ERA5 GPH

Reference:
    - https://journals.ametsoc.org/view/journals/clim/34/10/JCLI-D-20-0676.1.xml
    - https://doi.org/10.1175/JCLI-D-13-00018.1
    - https://github.com/ai2cm/ace/tree/main/fme/fme/core

Yingkai Sha
ksha@ucar.edu
'''



import torch
from credit.physics_constants import *

# Earth's radius
RAD_EARTH = 6371000 # m

# ideal gas constant of water vapor
RVGAS = 461.5 # J/kg/K

# ideal gas constant of dry air
RDGAS = 287.05 # J/kg/K

# gravity
GRAVITY = 9.80665 # m/s^2

# density of water
RHO_WATER = 1000.0 # kg/m^3

# latent heat caused by the phase change of water
# from liquid to gas
LATENT_HEAT_OF_VAPORIZATION = 2.5e6  # J/kg

class physics_pressure_level:
    '''
    Pressure level physics

    Attributes:
        upper_air_pressure (torch.Tensor): Pressure levels in Pa.
        lon (torch.Tensor): Longitude in degrees.
        lat (torch.Tensor): Latitude in degrees.
    
    Methods:
    
    '''
    
    def __init__(self,
                 lon: torch.Tensor,
                 lat: torch.Tensor,
                 upper_air_pressure: torch.Tensor):
        '''
        Initialize the class with longitude, latitude, and pressure levels.
        
        Args:
            lon (torch.Tensor): Longitude in degrees.
            lat (torch.Tensor): Latitude in degrees.
            upper_air_pressure (torch.Tensor): Pressure levels in Pa.
        '''
        self.lon = lon
        self.lat = lat
        self.upper_air_pressure = upper_air_pressure
        self.pressure_thickness = self.upper_air_pressure.diff(dim=-1)
        
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
            air_temperature (torch.Tensor): Air temperature (K).
            specific_humidity (torch.Tensor): Specific humidity (kg/kg).
        
        Returns:
            torch.Tensor: Virtual temperature (K).
        '''
        gamma = (RVGAS / RDGAS - 1.0)
        Tv = air_temperature * (1 + gamma*specific_humidity)
        return Tv

    def evaporation_from_latent_heat(self, 
                                     latent_heat_flux: torch.Tensor) -> torch.Tensor:
        '''
        Compute evaporation rate based on the latent heat flux [kg m^-2 s^-1.]
    
        Args:
            latent_heat_flux (torch.Tensor): Latent heat flux (W/m^2).
        
        Returns:
            torch.Tensor: Evaporation rate (kg/m^2/s).
        '''
        # latent_heat_flux [W/m^2]
        # convert to evaporation rate: (W/m^2) / (J/kg) = (J s^-1 m^-2) / (J/kg) = kg/m^2/s
        return latent_heat_flux / LATENT_HEAT_OF_VAPORIZATION

    def latent_heat_from_evaporation(self, 
                                     evaporation: torch.Tensor) -> torch.Tensor:
        '''
        Compute latent heat from evaporation.
        
        Args:
            evaporation (torch.Tensor): Evaporation rate (kg/m^2/s).
        
        Returns:
            torch.Tensor: Latent heat (W/m^2).
        '''
        return evaporation * LATENT_HEAT_OF_VAPORIZATION

    def horizontal_advection(self, 
                             T: torch.Tensor,
                             u: torch.Tensor, 
                             v: torch.Tensor) -> torch.Tensor:
        '''
        Compute the horizonal advection of the quantity:
    
        adv = - (u * dT/dx + v * dT/dy)
    
        where
        - T = the quantity to be computed
        - u, v = horizonal wind component
        - lon, lat = longitude and latitude
        
        Args:
            u (torch.Tensor): Zonal (east-west) wind component (m/s).
            v (torch.Tensor): Meridional (north-south) wind component (m/s).
            T (torch.Tensor): Scalar field (e.g., temperature) to be advected.
        
        Returns:
            torch.Tensor: Horizontal advection (same units as T).
        '''
        # degree to rad
        lat_rad = torch.deg2rad(self.lat)
        lon_rad = torch.deg2rad(self.lon)
        # latitude grid spacing [m]
        
        dy = torch.gradient(lat_rad * RAD_EARTH, dim=0)[0]  # Convert lat degrees to meters
    
        # longitude grid spacing [m], adjusted by the cosine of latitude
        dx = torch.gradient(lon_rad * RAD_EARTH * torch.cos(lat_rad).unsqueeze(-1), dim=1)[0]
        
        # Calculate the gradient of the quantity
        dT_dx = torch.gradient(T, dim=-1)[0] / dx  # Gradient in the longitude direction
        dT_dy = torch.gradient(T, dim=-2)[0] / dy  # Gradient in the latitude direction
        
        # Calculate the advection term: - (u * dT/dx + v * dT/dy)
        advection = -(u * dT_dx + v * dT_dy)
        
        return advection
    
    def pressure_level_integral(self, 
                                T: torch.Tensor) -> torch.Tensor:
        '''
        Computes a vertical integral with given pressure levels.
        
        Args:
            T (torch.Tensor): Scalar field to integrate (lat, lon, vertical_level).
        
        Returns:
            torch.Tensor: Vertical integral of the integrand (lat, lon).
        '''
        return torch.sum(self.pressure_thickness * T, dim=-1)
    
    def divergence_of_moisture_flux_integral(self,
                                             q: torch.Tensor,
                                             u: torch.Tensor,
                                             v: torch.Tensor) -> torch.Tensor:
        '''
        Compute the divergence of vertical integral of water vapour flux

        div(F) = [1/(R*cos(lat))] * d[int(u*q)*cos(lat)]/d(lon) + [1/R] * d[int(v*q)]/d(lat)

        where
        - q: specific humidity
        - (u, v): horizontal wind components
        - int(u*q), int(v*q): the vertical integral of water vapour flux
        - lon, lat: longitude and latitude
        - R: RAD_EARTH

        Args:
            q (torch.Tensor): Scalar quantity (e.g., specific humidity).
            u (torch.Tensor): Zonal (east-west) wind component (m/s).
            v (torch.Tensor): Meridional (north-south) wind component (m/s).
        
        Returns:
            torch.Tensor: Divergence of the flux (1/m).
        '''
        # Convert lat and lon to radians
        lat_rad = torch.deg2rad(self.lat)
        lon_rad = torch.deg2rad(self.lon)
    
        # compute the vertical integrals of the moisture fluxes
        # flux m/s * pressure kg/(ms^2) * gravity m/s^2 = kg/m/s
        Fx = (1/GRAVITY) * self.pressure_level_integral(q * u)  
        Fy = (1/GRAVITY) * self.pressure_level_integral(q * v)

        # ------------------------------------------------------------------- #
        # Compute the partial derivatives of the fluxes
        dlon = torch.gradient(lon_rad, dim=-1)[0]
        dlat = torch.gradient(lat_rad, dim=-2)[0]

        # kg/m/s divided by RAD_EARTH: kg/m^2/s
        dFx_dlon = torch.gradient(Fx * torch.cos(lat_rad), dim=-1)[0] / (dlon * RAD_EARTH * torch.cos(lat_rad))
        dFy_dlat = torch.gradient(Fy, dim=-2)[0] / (dlat * RAD_EARTH)
    
        # Compute the divergence of the moisture flux
        div_Fq = dFx_dlon + dFy_dlat
        # ------------------------------------------------------------------- #
        
        return div_Fq
        
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
            specific_humidity (torch.Tensor): Specific humidity (lat, lon, vertical_level), (kg/kg).
            surface_pressure (torch.Tensor): Surface pressure (lat, lon), (Pa).
        
        Returns:
            torch.Tensor: Surface pressure due to dry air (Pa).
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
            air_temperature (torch.Tensor): Air temperature (K).
            specific_humidity (torch.Tensor): Specific humidity (kg/kg).
        
        Returns:
            torch.Tensor: Pressure level thickness (m).
        '''
        # Compute Tv
        Tv = self.virtual_temperature(air_temperature, specific_humidity)
        
        # Compute logP diff
        dlogp = torch.log(self.upper_air_pressure).diff(dim=-1)
        
        # thickness
        thickness = (RDGAS * Tv / GRAVITY) * dlogp
        
        return thickness
    
    def geopotential_height(self,
                            air_temperature: torch.Tensor, 
                            specific_humidity: torch.Tensor, 
                            gp_surface: torch.Tensor) -> torch.Tensor:
        '''
        Compute geopotential height (GPH) for a given pressure level 
        using air temperature and specific humidity.

        GPH(lev_t) = \sum_{lev0, lev_t}{Z} + Phi_surf/g

        where:
        - Z:  pressure level thickness
        - Phi_surf: geopotential at surface
        - g: gravity
        
        Args:
            air_temperature (torch.Tensor): Air temperature (K).
            specific_humidity (torch.Tensor): Specific humidity (kg/kg).
            gp_surface (torch.Tensor): geopotential at surface (m^2/s^2).
        
        Returns:
            torch.Tensor: Geopotential height (m).
        '''
        layer_thickness = self.pressure_level_thickness(air_temperature, specific_humidity)
    
        # flip so it is now from bottom to top
        layer_thickness = layer_thickness.flip(dims=(-1,))
        
        # cumulate thickness to height                                     
        cumulative_thickness = torch.cumsum(layer_thickness, dim=-1)
    
        # flip back so it is now from top to bottom
        cumulative_thickness = cumulative_thickness.flip(dims=(-1,))
        
        # fill negative surface height with 0
        H_surf = torch.where(gp_surface < 0.0, 0, gp_surface).reshape(*gp_surface.shape, 1)
        H_surf = H_surf / GRAVITY
        
        # combine upper-air thickness with surface height (broadcast on vertical dim)
        H_upper = cumulative_thickness + H_surf.broadcast_to(cumulative_thickness.shape)
    
        # compute geopotential height
        # H_upper is top to bottom, so concat surface in the end
        GPH = torch.concat([H_upper, H_surf], dim=-1)
        # e.g. Z500 = GPH[..., index_500hPa]
        
        return GPH


