'''
A collection of functions for visualizing the forecasts 
-------------------------------------------------------
Yingkai Sha
ksha@ucar.edu
'''

# ---------- #
# System
from os.path import join

# ---------- #
# Numerics
import datetime
import numpy as np
import xarray as xr
# ---------- #
# matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colormaps as plt_cmaps
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# cartopy
import cartopy.crs as ccrs
import cartopy.mpl.geoaxes
import cartopy.feature as cfeature


def get_colormap(cmap_strings):
    colormap_obj = []
    for cmap_name in cmap_strings:
        colormap_obj.append(plt_cmaps[cmap_name])
    return colormap_obj

def get_colormap_extend(var_range):
    if var_range[0] == 0.0:
        return 'max'
    elif var_range[1] == 0.0:
        return 'min'
    else:
        return 'both'
    return colorbar_extends

def get_variable_range(data):
    data_ravel = data.ravel()
    
    data_max = np.quantile(data_ravel, 0.98)
    if np.abs(np.min(data)) < 1e-2:
        data_min = 0
    else:
        data_min = np.quantile(data_ravel, 0.02)
        
    # rounding
    if data_max > 10:
        round_val = 10
    elif data_max > 1:
        round_val = 1
    elif data_max > 0.1:
        round_val = 0.1
    elif data_max > 0.01:
        round_val = 0.01
    elif data_max > 0.001:
        round_val = 0.001
    elif data_max > 0.0001:
        round_val = 0.0001
    else:
        round_val = 0.00001
        
    data_max = int(np.ceil(data_max / round_val)) * round_val
    if data_min != 0:
        data_min = int(np.floor(data_min / round_val)) * round_val
        
    # 0 in the middle
    if data_min < 0 and data_max > 0:
        data_limit = np.max([-data_min, data_max])
        data_min = -data_limit
        data_max = data_limit
    
    return [data_min, data_max]

def draw_forecast(data, conf=None, times=None, forecast_count=None, save_location=None):
    '''
    This function produces 4-panel figures 
    '''
    # ------------------------------ #
    # visualization settings
    ## colormap
    colormaps = get_colormap(conf['visualization']['upper_air_visualize']['colormaps'])
    ## variable names
    var_names = conf['visualization']['upper_air_visualize']['variable_names']
    title_string = '{}; level {}\ntime: {}; step: {}' #.format(var_name, level, datetime, step)
    ## number of levels
    N_levels = int(len(conf['loss']['variable_weights']['U']))
    ## number of variables to plot
    var_num = int(len(var_names))
    level_nums = conf['visualization']['upper_air_visualize']['visualize_levels']
    ## output figure options and names
    save_options = conf['visualization']['upper_air_visualize']['save_options']
    save_name_head = conf['visualization']['upper_air_visualize']['file_name_prefix']
    ## collect figure names
    filenames = []
    
    # ------------------------------ #
    # get forecast step and file name
    k, fn = data
    t = times[k]
    pred = np.load(fn)
    
    # ------------------------------ #
    # get lat/lon grids
    lat_lon_weights = xr.open_dataset(conf['loss']['latitude_weights'])
    longitude = lat_lon_weights["longitude"]
    latitude = lat_lon_weights["latitude"]
    
    
    for level_num in level_nums:
        # ------------------------------ #
        # Figure
        fig = plt.figure(figsize=(13, 6.5))
        
        # 2-by-2 subplots
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
        proj_ = ccrs.EckertIII()
        
        # subplot ax
        ax0 = plt.subplot(gs[0, 0], projection=proj_)
        ax1 = plt.subplot(gs[0, 1], projection=proj_)
        ax2 = plt.subplot(gs[1, 0], projection=proj_)
        ax3 = plt.subplot(gs[1, 1], projection=proj_)
        AX = [ax0, ax1, ax2, ax3]
        plt.subplots_adjust(0, 0, 1, 1, hspace=0.2, wspace=0.05)
        
        # lat/lon gridlines and labeling
        for ax in AX:
            GL = ax.gridlines(crs=ccrs.PlateCarree(), 
                              draw_labels=True, x_inline=False, y_inline=False, 
                              color='k', linewidth=0.5, linestyle=':', zorder=5)
            GL.top_labels = None; GL.bottom_labels = None
            GL.right_labels = None; GL.left_labels = None
            GL.xlabel_style = {'size': 14}; GL.ylabel_style = {'size': 14}
            GL.rotate_labels = False
        
            ax.add_feature(cfeature.COASTLINE.with_scale('110m'), edgecolor='k', linewidth=1.0, zorder=5)
            ax.spines['geo'].set_linewidth(2.5)
        
        # pcolormesh / colorbar / title in loops
        for i_var in range(var_num):
            # get the current axis
            ax = AX[i_var]
            # get the current variable
            var_ind = i_var*N_levels + level_num
            pred_draw = pred[var_ind]
            # get visualization settings
            var_lim = get_variable_range(pred_draw)
            cbar_extend = get_colormap_extend(var_lim)
            colormap = colormaps[i_var]
            # pcolormesh
            cbar = ax.pcolormesh(longitude, latitude, pred_draw, vmin=var_lim[0], vmax=var_lim[1], 
                                 cmap=colormap, transform=ccrs.PlateCarree())
            # colorbar operations
            CBar = fig.colorbar(cbar, location='right', orientation='vertical', 
                                pad=0.02, fraction=0.025, shrink=0.6, aspect=15, extend=cbar_extend, ax=ax)
            CBar.ax.tick_params(axis='y', labelsize=14, direction='in', length=0)
            CBar.outline.set_linewidth(2.5)
            # title
            var_name = var_names[i_var]
            ax.set_title(title_string.format(var_name, level_num, t, k), fontsize=14)
    
        save_name = '{}_level{:02d}_fcst{:03d}_step{:03d}.png'.format(save_name_head, level_num, forecast_count, k)
        filename = join(save_location, save_name)
        plt.savefig(filename, **save_options)
        plt.close()
        filenames.append(filename)
    return k, filenames