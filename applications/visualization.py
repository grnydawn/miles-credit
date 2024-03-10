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
#import pandas as pd
import xarray as xr

# ---------- #
# matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# cartopy
import cartopy.crs as ccrs
import cartopy.mpl.geoaxes
import cartopy.feature as cfeature

def draw_forecast(data, N_level=15, level_num=10, var_num=4, 
                  conf=None, times=None, forecast_count=None, save_location=None):
    '''
    This function produces 4-panel figures 
    '''
    # ------------------------------ #
    # visualization settings
    ## variable range limit with units of m/s, m/s, K, g/kg
    var_lims = [[-20, 20], [-20, 20], [273.15-35, 273.15+35], [0, 1e-2]]
    ## colormap
    colormaps = [plt.cm.Spectral, plt.cm.Spectral, plt.cm.RdBu_r, plt.cm.YlGn]
    ## colorbar extend
    colorbar_extends = ['both', 'both', 'both', 'max']
    ## title
    title_strings = ['U wind [m/s]; level {}\ntime: {}; step: {}', 
                     'V wind [m/s]; level {}\ntime: {}; step: {}', 
                     'Air temperature [$^\circ$K]; level {}\ntime: {}; step: {}', 
                     'Specific humidity [kg/kg]; level {}\ntime: {}; step: {}']
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
        var_ind = i_var*N_level + level_num
        pred_draw = pred[var_ind]
        # get visualization settings
        var_lim = var_lims[i_var]
        colormap = colormaps[i_var]
        cbar_extend = colorbar_extends[i_var]
        # pcolormesh
        cbar = ax.pcolormesh(longitude, latitude, pred_draw, vmin=var_lim[0], vmax=var_lim[1], 
                             cmap=colormap, transform=ccrs.PlateCarree())
        # colorbar operations
        CBar = fig.colorbar(cbar, location='right', orientation='vertical', 
                            pad=0.02, fraction=0.025, shrink=0.6, aspect=15, extend=cbar_extend, ax=ax)
        CBar.ax.tick_params(axis='y', labelsize=14, direction='in', length=0)
        CBar.outline.set_linewidth(2.5)
        # title
        ax.set_title(title_strings[i_var].format(level_num, t, k), fontsize=14)
    
    filename = join(save_location, f"global_q_{forecast_count}_{k}.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    return k, filename