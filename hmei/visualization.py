import os
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", message="Unable to decode time axis into full numpy.datetime64 objects, continuing using cftime.datetime objects instead, reason: dates out of range")
warnings.filterwarnings("ignore", message="invalid value encountered in reduce")

import xarray as xr
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


#################################################################################
#################################################################################
def ctrl_overlay_plot(reg_mean, title, ylabel, size=(10,5)):
    """
    Description:
        Generates a timeseries plot of the annual mean with all regions overlaid.
    
    Parameters:
        reg_mean - list of xarray.DataArray objects corresponding to each region
        title    - string title
        ylabel   - string y-axis label
        size     - tuple size of plot
    """
    
    ## get masks
    write_rootdir = '/home/bbuchovecky/storage/so_predict_derived/'
    reg_masks = xr.open_dataset(write_rootdir+'regional_global_masks.nc')

    ## axes formatting
    time = np.arange(1,reg_mean[0].size+1,1)
    ens_yrs = [22,64,106,170,232,295]
    xlim = [1,reg_mean[0].size]
    ls = '-'
    primary_width = 2
    secondary_width = 1
    # cmap = plt.get_cmap('tab10') ## need to use cmap(i)
    # cmap = ['#003f5c', '#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600'] ## equidistant colors
    cmap = ['black', 'red', 'blue', 'green', 'orange', 'magenta']
    
    ## create figure
    fig,ax = plt.subplots(figsize=size)
    
    ## plot all regions
    ax.plot(time, reg_mean[0], label='Southern Ocean', color=cmap[0], ls=ls, linewidth=primary_width, zorder=6)

    for (reg, i) in zip(reg_masks.data_vars, range(0,6)):
        if i !=0:
            reg_title = reg_masks[reg].attrs['long_name']
            ax.plot(time, reg_mean[i], label=reg_title, color=cmap[i], ls=ls, linewidth=secondary_width)

    ax.set(title=title, xlabel = 'Time (yr)', ylabel=ylabel, xlim=xlim);
    
    leg = ax.legend(bbox_to_anchor = (1.02, 1));
    for line in leg.get_lines():
        line.set_linewidth(4.0)


#################################################################################
#################################################################################
def ctrl_grid_plot(reg_mean, title, ylabel, style='equal'):
    """
    Description:
        Generates a grid of individual plots, each with the annual mean for one
        region.
    
    Parameters:
        reg_mean - list of xarray.DataArray objects corresponding to each region
        title    - string title
        ylabel   - string y-axis label
        style    - choose the layout of the plots
    """
    
    ## get masks
    write_rootdir = '/home/bbuchovecky/storage/so_predict_derived/'
    reg_masks = xr.open_dataset(write_rootdir+'regional_global_masks.nc')
    
    ## figure settings
    if style == 'center':
        rows = 4
        cols = 2
    else:
        rows = 3
        cols = 2
    size = (10,10)

    fig = plt.figure(constrained_layout=True, figsize=size)
    spec = gridspec.GridSpec(ncols=cols, nrows=rows, figure=fig)

    if style == 'center':
        ax1 = fig.add_subplot(spec[0,:])
        ax2 = fig.add_subplot(spec[1,0])
        ax3 = fig.add_subplot(spec[1,1])
        ax4 = fig.add_subplot(spec[2,1])
        ax5 = fig.add_subplot(spec[2,0])
        ax6 = fig.add_subplot(spec[3,0])
    else:
        ax1 = fig.add_subplot(spec[0,0])
        ax2 = fig.add_subplot(spec[0,1])
        ax3 = fig.add_subplot(spec[1,0])
        ax4 = fig.add_subplot(spec[1,1])
        ax5 = fig.add_subplot(spec[2,0])
        ax6 = fig.add_subplot(spec[2,1])

    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    ## axes settings
    label_loc = 'left'
    time = np.arange(1,reg_mean[0].size+1,1)
    ens_yrs = [22,64,106,170,232,295]
    xlim = [1,reg_mean[0].size]
    size = (10,5)
    ls = '-'
    primary_width = 2
    secondary_width = 1.5
    # cmap = plt.get_cmap('tab10') ## need to use cmap(i)
    # cmap = ['#003f5c', '#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600'] ## equidistant colors
    cmap = ['black', 'red', 'blue', 'green', 'orange', 'magenta']
    fignum = 97
    
    min_ylim = float('inf')
    max_ylim = float('-inf')

    ## plot all regions
    for (reg, i) in zip(reg_masks.data_vars, range(6)):
        reg_title = reg_masks[reg].attrs['long_name']
        axes[i].plot(time, reg_mean[i], color=cmap[i], ls=ls, linewidth=secondary_width)
        axes[i].set(xlabel = 'Time (yr)', ylabel=ylabel, xlim=xlim, xticks=ens_yrs)
        axes[i].set_title('('+chr(fignum)+') '+reg_title, loc=label_loc)
        axes[i].autoscale(enable=True,axis='x',tight=True)
        fignum += 1
        
        this_ylim = axes[i].get_ylim()
        
        diff = this_ylim[1] - this_ylim[0]
        diff = math.floor(math.log(diff, 10))
        
        if this_ylim[0] < min_ylim:
            min_ylim = round(this_ylim[0], abs(diff)+1)
        if this_ylim[1] > max_ylim:
            max_ylim = round(this_ylim[1], abs(diff)+1)
    
    for i in range(6):
        axes[i].set_ylim(min_ylim, max_ylim)

    fig.suptitle(title, fontsize=16)