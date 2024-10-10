'''
postblock.py 
-------------------------------------------------------
Content:
    - PostBlock
    - tracer_fixer
    - global_mass_fixer
    - global_energy_fixer
    - SKEBS
    
'''

import torch
from torch import nn
import numpy as np

from credit.data import get_forward_data
from credit.transforms import load_transforms
from credit.physics_core import physics_pressure_level
from credit.physics_constants import (RAD_EARTH, GRAVITY, 
                                      RHO_WATER, LH_WATER, 
                                      RVGAS, RDGAS, CP_DRY, CP_VAPOR)

import logging
logger = logging.getLogger(__name__)


class PostBlock(nn.Module):
    def __init__(self, 
                 post_conf):
        """
            post_conf: dictionary with config options for PostBlock.
                       if post_conf is not specified in config, 
                       defaults are set in the parser

            This class is a wrapper for all post-model operations.
            Registered modules:
                - SKEBS
                - tracer_fixer
                - global_mass_fixer
                
        """
        super().__init__()

        self.operations = nn.ModuleList()

        # The general order of postblock processes:
        # (1) negative tracer fixer --> global mass fixer --> SKEB --> global energy fixer
        
        # negative tracer fixer
        if post_conf['tracer_fixer']['activate']:
            opt = tracer_fixer(post_conf)
            self.operations.append(opt)
            
        # stochastic kinetic energy backscattering (SKEB)
        if post_conf["skebs"]["activate"]:
            logging.info("using SKEBS")
            self.operations.append(SKEBS(post_conf))
            
        # global mass fixer
        if post_conf['global_mass_fixer']['activate']:
            opt = global_mass_fixer(post_conf)
            self.operations.append(opt)

        # global energy fixer
        if post_conf['global_energy_fixer']['activate']:
            opt = global_energy_fixer(post_conf)
            self.operations.append(opt)

    def forward(self, x):
        for op in self.operations:
            x = op(x)
            
        if isinstance(x, dict):
            # if output is a dict, return y_pred (if it exists), otherwise return x
            return x.get("y_pred", x)
        else:
            # if output is not a dict (assuming tensor), return x
            return x

class tracer_fixer(nn.Module):
    '''
    This module fixes tracer values by replacing their values to a given threshold 
    (e.g., `tracer[tracer<thres] = thres`).

    Args:
        post_conf (dict): config dictionary that includes all specs for the tracer fixer.
    '''
    def __init__(self, post_conf):
        super().__init__()
        
        # ------------------------------------------------------------------------------ #
        # identify variables of interest
        self.tracer_indices = post_conf['tracer_fixer']['tracer_inds']
        self.tracer_thres = post_conf['tracer_fixer']['tracer_thres']

        # ------------------------------------------------------------------------------ #
        # setup a scaler
        if post_conf['tracer_fixer']['denorm']:
            
            self.state_trans = load_transforms(post_conf, scaler_only=True)
        else:
            self.state_trans = None
    
    def forward(self, x):
        # ------------------------------------------------------------------------------ #
        # get y_pred
        # y_pred is channel first: (batch, var, time, lat, lon)
        y_pred = x["y_pred"]
        
        # if denorm is needed
        if self.state_trans:
            y_pred = self.state_trans.inverse_transform(y_pred)
            
        # ------------------------------------------------------------------------------ #
        # tracer correction
        for i, i_var in enumerate(self.tracer_indices):
            # get the tracers
            tracer_vals = y_pred[:, i_var, ...]

            # in-place modification of y_pred
            thres = self.tracer_thres[i]
            tracer_vals[tracer_vals < thres] = thres
            
        if self.state_trans:
            y_pred = self.state_trans.transform_array(y_pred)
            
        # give it back to x
        x["y_pred"] = y_pred

        # return dict, 'x' is not touched
        return x

class global_mass_fixer(nn.Module):
    '''
    This module applies global mass conservation fixes for both dry air and water budget.
    The output ensures that the global dry air mass and global water budgets are conserved 
    through correction ratios applied during model runs. Variables `specific total water`
    and `precipitation` will be corrected to close the budget. All corrections are done
    using float32 PyTorch tensors.
    
    Args:
        post_conf (dict): config dictionary that includes all specs for the global mass fixer.
    '''
    
    def __init__(self, post_conf):
        super().__init__()

        # ------------------------------------------------------------------------------------ #
        # initialize physics computation

        # provide example data if it is a unit test
        if post_conf['global_mass_fixer']['simple_demo']:
            y_demo = np.array([90, 70, 50, 30, 10, -10, -30, -50, -70, -90])
            x_demo = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 
                               200, 220, 240, 260, 280, 300, 320, 340])
            
            lon_demo, lat_demo = np.meshgrid(x_demo, y_demo)
            lon_demo = torch.from_numpy(lon_demo)
            lat_demo = torch.from_numpy(lat_demo)

            p_level_demo = torch.from_numpy(np.array([100, 30000, 50000, 70000, 80000, 90000, 100000]))
            self.core_compute = physics_pressure_level(lon_demo, lat_demo, p_level_demo, 
                                                       midpoint=post_conf['global_mass_fixer']['midpoint'])
            self.N_levels = len(p_level_demo)
            self.N_seconds = int(post_conf['data']['lead_time_periods']) * 3600
            self.ind_fix = len(p_level_demo) - int(post_conf['global_mass_fixer']['fix_level_num']) + 1
            
        else:
            # the actual setup for model runs
            ds_physics = get_forward_data(post_conf['data']['save_loc_physics'])
            
            lon_lat_level_names = post_conf['global_mass_fixer']['lon_lat_level_name']
            lon2d = torch.from_numpy(ds_physics[lon_lat_level_names[0]].values).float()
            lat2d = torch.from_numpy(ds_physics[lon_lat_level_names[1]].values).float()
            p_level = torch.from_numpy(ds_physics[lon_lat_level_names[2]].values).float()
            
            self.core_compute = physics_pressure_level(lon2d, lat2d, p_level, 
                                                       midpoint=post_conf['global_mass_fixer']['midpoint'])
            self.N_levels = len(p_level)
            self.N_seconds = int(post_conf['data']['lead_time_periods']) * 3600
            self.ind_fix = len(p_level) - int(post_conf['global_mass_fixer']['fix_level_num']) + 1
            
        # ------------------------------------------------------------------------------------ #
        # identify variables of interest
        self.q_ind_start = int(post_conf['global_mass_fixer']['q_inds'][0])
        self.q_ind_end = int(post_conf['global_mass_fixer']['q_inds'][-1]) + 1
        self.precip_ind = int(post_conf['global_mass_fixer']['precip_ind'])
        self.evapor_ind = int(post_conf['global_mass_fixer']['evapor_ind'])
        
        # ------------------------------------------------------------------------------------ #
        # setup a scaler
        if post_conf['global_mass_fixer']['denorm']:
            self.state_trans = load_transforms(post_conf, scaler_only=True)
        else:
            self.state_trans = None
            
    def forward(self, x):
        # ------------------------------------------------------------------------------ #
        # get tensors

        # x_input (batch, var, time, lat, lon)
        # x_input does not have precip and evapor
        x_input = x['x']
        y_pred = x["y_pred"]
        
        # if denorm is needed
        if self.state_trans:
            x_input = self.state_trans.inverse_transform_input(x_input)
            y_pred = self.state_trans.inverse_transform(y_pred)
            
        q_input = x_input[:, self.q_ind_start:self.q_ind_end, -1, ...]
        
        # y_pred (batch, var, time, lat, lon) 
        # pick the first time-step, y_pred is expected to have the next step only
        q_pred = y_pred[:, self.q_ind_start:self.q_ind_end, 0, ...]
        precip = y_pred[:, self.precip_ind, 0, ...]
        evapor = y_pred[:, self.evapor_ind, 0, ...]
        
        # ------------------------------------------------------------------------------ #
        # global dry air mass conservation

        # total mass from q_input
        mass_dry_sum_t0 = self.core_compute.total_dry_air_mass(q_input)

        # total mass from q_pred
        mass_dry_sum_t1_hold = self.core_compute.weighted_sum(
            self.core_compute.integral_sliced(1-q_pred, 0, self.ind_fix) / GRAVITY, 
            axis=(-2, -1))
        
        mass_dry_sum_t1_fix = self.core_compute.weighted_sum(
            self.core_compute.integral_sliced(1-q_pred, self.ind_fix-1, self.N_levels) / GRAVITY, 
            axis=(-2, -1))

        q_correct_ratio = (mass_dry_sum_t0 - mass_dry_sum_t1_hold) / mass_dry_sum_t1_fix
        q_correct_ratio = torch.clamp(q_correct_ratio, min=0.9, max=1.1)
        
        # broadcast: (batch, 1, 1, 1, 1)
        q_correct_ratio = q_correct_ratio.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # ===================================================================== #
        # Compute the corrected part of q_pred without in-place modifications
        q_pred_to_correct = q_pred[:, self.ind_fix-1:, ...]
        q_pred_corrected_part = 1 - (1 - q_pred_to_correct) * q_correct_ratio
    
        # Extract the unmodified part of q_pred
        q_pred_unchanged_part = q_pred[:, :self.ind_fix-1, ...]
    
        # Concatenate the unmodified and corrected parts
        q_pred_corrected = torch.cat([q_pred_unchanged_part, q_pred_corrected_part], dim=1)  # Along levels dimension

        # ------------------------------------------------------------------------------ #
        # global water balance
        precip_flux = precip * RHO_WATER / self.N_seconds
        evapor_flux = evapor * RHO_WATER / self.N_seconds
        
        # total water content (batch, var, time, lat, lon)
        TWC_input = self.core_compute.total_column_water(q_input)
        TWC_pred = self.core_compute.total_column_water(q_pred_corrected)  # Use corrected q_pred
        
        dTWC_dt = (TWC_pred - TWC_input) / self.N_seconds
        
        # global sum of total water content tendency
        TWC_sum = self.core_compute.weighted_sum(dTWC_dt, axis=(-2, -1))
        
        # global evaporation source
        E_sum = self.core_compute.weighted_sum(evapor_flux, axis=(-2, -1))
        
        # global precip sink
        P_sum = self.core_compute.weighted_sum(precip_flux, axis=(-2, -1))
        
        # global water balance residual
        residual = -TWC_sum - E_sum - P_sum
        
        # compute correction ratio
        P_correct_ratio = (P_sum + residual) / P_sum
        P_correct_ratio = torch.clamp(P_correct_ratio, min=0.9, max=1.1)
        # broadcast: (batch_size, 1, 1, 1)
        P_correct_ratio = P_correct_ratio.unsqueeze(-1).unsqueeze(-1)
        
        # apply correction on precip
        precip_corrected = precip * P_correct_ratio  # Apply correction to precip

        # --------------------------------------------------------------------------------------------- #
        # return corrected values back
        q_pred_corrected = q_pred_corrected.unsqueeze(2)
        
        # precip has shape (batch, lat, lon)
        # We need to expand it to (batch, 1, 1, lat, lon)
        precip_corrected = precip_corrected.unsqueeze(1).unsqueeze(2)  # Insert variable and time dimensions

        # Initialize a list to collect slices and corrected variables
        variables_list = []
        
        # Variables before q_pred indices
        if self.q_ind_start > 0:
            variables_before_q = y_pred[:, :self.q_ind_start, :, :, :]
            variables_list.append(variables_before_q)
        
        # Append corrected q_pred
        variables_list.append(q_pred_corrected)
        
        # Variables between q_pred and precip indices
        if self.q_ind_end < self.precip_ind:
            variables_between_q_and_precip = y_pred[:, self.q_ind_end:self.precip_ind, :, :, :]
            variables_list.append(variables_between_q_and_precip)
        
        # Append corrected precip
        variables_list.append(precip_corrected)
        
        # Variables after precip index
        if self.precip_ind + 1 < y_pred.size(1):
            variables_after_precip = y_pred[:, self.precip_ind + 1:, :, :, :]
            variables_list.append(variables_after_precip)
        
        # Concatenate all parts along the variable dimension (dim=1)
        y_pred_corrected = torch.cat(variables_list, dim=1)
        
        if self.state_trans:
            y_pred_corrected = self.state_trans.transform_array(y_pred_corrected)
        
        # give it back to x
        x["y_pred"] = y_pred_corrected

        # return dict, 'x' is not touched
        return x

class global_energy_fixer(nn.Module):
    '''
    This module applys global energy conservation fixes. The output ensures that the global sum
    of total energy in the atmosphere is balanced by radiantion and energy fluxes at the top of 
    the atmosphere and the surface. Variables `air temperature` will be modified to close the
    budget. All corrections are done using float32 Pytorch tensors.

    Args:
        post_conf (dict): config dictionary that includes all specs for the global energy fixer.
    '''
    def __init__(self, post_conf):
        super().__init__()

        # ------------------------------------------------------------------------------------ #
        # initialize physics computation

        # provide example data if it is a unit test
        if post_conf['global_energy_fixer']['simple_demo']:
            y_demo = np.array([90, 70, 50, 30, 10, -10, -30, -50, -70, -90])
            x_demo = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 
                               200, 220, 240, 260, 280, 300, 320, 340])
            
            lon_demo, lat_demo = np.meshgrid(x_demo, y_demo)
            lon_demo = torch.from_numpy(lon_demo)
            lat_demo = torch.from_numpy(lat_demo)

            p_level_demo = torch.from_numpy(np.array([100, 30000, 50000, 70000, 80000, 90000, 100000]))
            self.core_compute = physics_pressure_level(lon_demo, lat_demo, p_level_demo, 
                                                       midpoint=post_conf['global_energy_fixer']['midpoint'])
            self.N_seconds = int(post_conf['data']['lead_time_periods']) * 3600

            gph_surf_demo = np.ones((10, 18))
            self.GPH_surf = torch.from_numpy(gph_surf_demo)
            
        else:
            # the actual setup for model runs
            ds_physics = get_forward_data(post_conf['data']['save_loc_physics'])

            lon_lat_level_names = post_conf['global_energy_fixer']['lon_lat_level_name']
            
            lon2d = torch.from_numpy(ds_physics[lon_lat_level_names[0]].values).float()
            lat2d = torch.from_numpy(ds_physics[lon_lat_level_names[1]].values).float()
            p_level = torch.from_numpy(ds_physics[lon_lat_level_names[2]].values).float()
        
            self.core_compute = physics_pressure_level(lon2d, lat2d, p_level, 
                                                       midpoint=post_conf['global_energy_fixer']['midpoint'])
            self.N_seconds = int(post_conf['data']['lead_time_periods']) * 3600
            
            varname_gph = post_conf['global_energy_fixer']['surface_geopotential_name']
            self.GPH_surf = torch.from_numpy(ds_physics[varname_gph[0]].values).float()
            
        # ------------------------------------------------------------------------------------ #
        # identify variables of interest
        self.T_ind_start = int(post_conf['global_energy_fixer']['T_inds'][0])
        self.T_ind_end = int(post_conf['global_energy_fixer']['T_inds'][-1]) + 1
        
        self.q_ind_start = int(post_conf['global_energy_fixer']['q_inds'][0])
        self.q_ind_end = int(post_conf['global_energy_fixer']['q_inds'][-1]) + 1
        
        self.U_ind_start = int(post_conf['global_energy_fixer']['U_inds'][0])
        self.U_ind_end = int(post_conf['global_energy_fixer']['U_inds'][-1]) + 1
        
        self.V_ind_start = int(post_conf['global_energy_fixer']['V_inds'][0])
        self.V_ind_end = int(post_conf['global_energy_fixer']['V_inds'][-1]) + 1

        self.TOA_solar_ind = int(post_conf['global_energy_fixer']['TOA_rad_inds'][0])
        self.TOA_OLR_ind = int(post_conf['global_energy_fixer']['TOA_rad_inds'][1])

        self.surf_solar_ind = int(post_conf['global_energy_fixer']['surf_rad_inds'][0])
        self.surf_LR_ind = int(post_conf['global_energy_fixer']['surf_rad_inds'][1])

        self.surf_SH_ind = int(post_conf['global_energy_fixer']['surf_flux_inds'][0])
        self.surf_LH_ind = int(post_conf['global_energy_fixer']['surf_flux_inds'][1])
        
        # ------------------------------------------------------------------------------------ #
        # setup a scaler
        if post_conf['global_energy_fixer']['denorm']:
            self.state_trans = load_transforms(post_conf, scaler_only=True)
        else:
            self.state_trans = None
            
    def forward(self, x):
        # ------------------------------------------------------------------------------ #
        # get tensors

        # x_input (batch, var, time, lat, lon)
        # x_input does not have precip and evapor
        x_input = x['x']
        y_pred = x["y_pred"]
        
        # if denorm is needed
        if self.state_trans:
            x_input = self.state_trans.inverse_transform_input(x_input)
            y_pred = self.state_trans.inverse_transform(y_pred)
        
        T_input = x_input[:, self.T_ind_start:self.T_ind_end, -1, ...]
        q_input = x_input[:, self.q_ind_start:self.q_ind_end, -1, ...]
        U_input = x_input[:, self.U_ind_start:self.U_ind_end, -1, ...]
        V_input = x_input[:, self.V_ind_start:self.V_ind_end, -1, ...]
        
        # y_pred (batch, var, time, lat, lon) 
        # pick the first time-step, y_pred is expected to have the next step only
        T_pred = y_pred[:, self.T_ind_start:self.T_ind_end, 0, ...]
        q_pred = y_pred[:, self.q_ind_start:self.q_ind_end, 0, ...]
        U_pred = y_pred[:, self.U_ind_start:self.U_ind_end, 0, ...]
        V_pred = y_pred[:, self.V_ind_start:self.V_ind_end, 0, ...]
        
        TOA_solar_pred = y_pred[:, self.TOA_solar_ind, 0, ...]
        TOA_OLR_pred = y_pred[:, self.TOA_OLR_ind, 0, ...]
        
        surf_solar_pred = y_pred[:, self.surf_solar_ind, 0, ...]
        surf_LR_pred = y_pred[:, self.surf_LR_ind, 0, ...]
        surf_SH_pred = y_pred[:, self.surf_SH_ind, 0, ...]
        surf_LH_pred = y_pred[:, self.surf_LH_ind, 0, ...]
        
        # ------------------------------------------------------------------------------ #
        # Latent heat, potential energy, kinetic energy

        # heat capacity on constant pressure
        CP_t0 = (1 - q_input) * CP_DRY + q_input * CP_VAPOR
        CP_t1 = (1 - q_pred) * CP_DRY + q_pred * CP_VAPOR
        
        # kinetic energy
        ken_t0 = 0.5 * (U_input ** 2 + V_input ** 2)
        ken_t1 = 0.5 * (U_pred ** 2 + V_pred ** 2)

        # packing latent heat + potential energy + kinetic energy
        E_qgk_t0 = LH_WATER * q_input + self.GPH_surf + ken_t0
        E_qgk_t1 = LH_WATER * q_input + self.GPH_surf + ken_t1

        # ------------------------------------------------------------------------------ #
        # energy source and sinks
        
        # TOA energy flux
        R_T = (TOA_solar_pred + TOA_OLR_pred) / self.N_seconds
        R_T_sum = self.core_compute.weighted_sum(R_T, axis=(-2, -1))
        
        # surface net energy flux
        F_S = (surf_solar_pred + surf_LR_pred + surf_SH_pred + surf_LH_pred) / self.N_seconds
        F_S_sum = self.core_compute.weighted_sum(F_S, axis=(-2, -1))
        
        # ------------------------------------------------------------------------------ #
        # thermal energy correction

        # total energy per level
        E_level_t0 = CP_t0 * T_input + E_qgk_t0
        E_level_t1 = CP_t1 * T_pred + E_qgk_t1

        # column integrated total energy
        TE_t0 = self.core_compute.integral(E_level_t0) / GRAVITY
        TE_t1 = self.core_compute.integral(E_level_t1) / GRAVITY
        # dTE_dt = (TE_t1 - TE_t0) / self.N_seconds

        global_TE_t0 = self.core_compute.weighted_sum(TE_t0, axis=(-2, -1)) 
        global_TE_t1 = self.core_compute.weighted_sum(TE_t1, axis=(-2, -1))

        # total energy correction ratio
        E_correct_ratio = (self.N_seconds * (R_T_sum - F_S_sum) + global_TE_t0) / global_TE_t1
        E_correct_ratio = torch.clamp(E_correct_ratio, min=0.9, max=1.1)
        # broadcast: (batch, 1, 1, 1, 1)
        E_correct_ratio = E_correct_ratio.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # apply total energy correction
        E_t1_correct = E_level_t1 * E_correct_ratio

        # let thermal energy carry the corrected total energy amount
        T_pred = (E_t1_correct - E_qgk_t1) / CP_t1
        
        # Give T_pred back to y_pred
        # (batch, levels, 1, lat, lon)
        T_pred_correct = T_pred.unsqueeze(2)  
    
        # Initialize a list to collect slices and corrected variables
        variables_list = []
    
        # Variables before T_pred indices
        if self.T_ind_start > 0:
            variables_before_T = y_pred[:, :self.T_ind_start, :, :, :]
            variables_list.append(variables_before_T)
    
        # Append corrected T_pred
        variables_list.append(T_pred_correct)
    
        # Variables after T_pred indices
        if self.T_ind_end < y_pred.size(1):
            variables_after_T = y_pred[:, self.T_ind_end:, :, :, :]
            variables_list.append(variables_after_T)
    
        # Concatenate all parts along the variable dimension (dim=1)
        y_pred_correct = torch.cat(variables_list, dim=1)
        
        if self.state_trans:
            y_pred_correct = self.state_trans.transform_array(y_pred_correct)
        
        # give it back to x
        x["y_pred"] = y_pred_correct

        # return dict, 'x' is not touched
        return x
        
class SKEBS(nn.Module):
    """
        post_conf: dictionary with config options for PostBlock.
                    if post_conf is not specified in config, 
                    defaults are set in the parser

        This class is currently a placeholder for SKEBS
    """
    def __init__(self, post_conf):
        super().__init__()
        self.image_width = post_conf['model']['image_width']
        final_layer_size = self.image_width
        self.additional_layer = nn.Linear(final_layer_size, final_layer_size)#.to(self.device) # Example: another layer
    
    def forward(self, x):
        x = x["y_pred"]
        return self.additional_layer(x)
    
if __name__ == "__main__":
    image_width = 100
    conf = {"post_conf": {"use_skebs": True, "image_width": image_width}}

    input_tensor = torch.randn(image_width)
    postblock = PostBlock(**conf)
    assert any([isinstance(module, SKEBS) for module in postblock.modules()])

    y_pred = postblock(input_tensor)
    print("Predicted shape:", y_pred.shape)


