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
def ctrl_overlay_plot(
    reg_mean, title, ylabel, size=(10,5)):
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
def ctrl_grid_plot(
    reg_mean, title, ylabel, style='equal'):
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
    
    
#################################################################################
#################################################################################
def grid_month_anom(
    so_monthly_anom, reg, ylabel='', title='', size=(10,10)):
    
    fig,axes = plt.subplots(4, 3, sharey=True, sharex=True, figsize=size)

    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
              'August', 'September', 'October', 'November', 'December']
    ens_yrs = [22,64,106,170,232,295]
    years = np.arange(1,301)
    xlim=[0,300]
    regions = ['SouthernOcean', 'Weddell', 'Indian', 
               'WestPacific', 'Ross', 'AmundBell']
    cmap = ['black', 'red', 'blue', 'green', 'orange', 'magenta']
    iColor = regions.index(reg)
    month_anom = np.zeros((12,300))
    
    max_ylim = float('-inf')

    for m in range(3600):
        month_anom[m%12,int(m/12)] = so_monthly_anom[reg].isel(month=m)

    for i in range(12):
        axes[int(i/3),i%3].plot(years, month_anom[i], color=cmap[iColor])
        axes[int(i/3),i%3].set(title=months[i], xlim=xlim);
        
        ## add irregular ticks for ensemble start years
        axes[int(i/3),i%3].set_xticks(ens_yrs);

        axes[int(i/3),i%3].grid()
#         axes[int(i/3),i%3].xaxis.grid(True)
#         axes[int(i/3),i%3].yaxis.grid(True)
    
#         axes[int(i/3),i%3].spines['right'].set_color('none')
#         axes[int(i/3),i%3].spines['top'].set_color('none')

        if int(i/3) == 3:
            axes[int(i/3),i%3].set(xlabel='Time (yrs)');
#         if i%3 == 0:
#             axes[int(i/3),i%3].set(ylabel=so_monthly_anom[reg].attrs['label']);
        
        ## make sure y-axis is symmetric
        this_ylim = axes[int(i/3),i%3].get_ylim()
        diff = this_ylim[1] - this_ylim[0]
        diff = math.floor(math.log(diff, 10))
        if abs(this_ylim[0]) > max_ylim:
            max_ylim = round(abs(this_ylim[0]), abs(diff)+1)
        if abs(this_ylim[1]) > max_ylim:
            max_ylim = round(abs(this_ylim[1]), abs(diff)+1)
          
    ## set y-axis limits and plot vertical/horizontal lines
    for i in range(12):
        axes[int(i/3),i%3].hlines(0, 1, 300, color='gray', ls='-')
#         axes[int(i/3),i%3].vlines(ens_yrs, -max_ylim, max_ylim, color='silver', ls='-', linewidth=0.75)
        axes[int(i/3),i%3].set_ylim(-max_ylim, max_ylim)

    if title == '':
        fig.suptitle(so_monthly_anom.attrs['name']+' - '+reg, fontsize=16)
    else:
        fig.suptitle(title, fontsize=16)
    
    fig.tight_layout()


#################################################################################
#################################################################################
def stdev_plot(
    so_ds, reg, title='', ylabel='', size=(10,5)):
    
    write_rootdir = '/home/bbuchovecky/storage/so_predict_derived/'
    subdir_ctrl = 'CTRL/'
    
    if so_ds[reg].dims[0] == 'year':
        xlim = [0,300]
        time = np.arange(1,301)
        xlabel = 'Time (years)'
        ens_yrs = [22,64,106,170,232,295]
        duration = 10
    if so_ds[reg].dims[0] == 'month':
        xlim = [0,3600]
        time = np.arange(1,3601)
        xlabel = 'Time (months)'
        ens_yrs = np.array([22,64,106,170,232,295])*12
        duration = 120
    
    reg_masks = xr.open_dataset(write_rootdir+'regional_global_masks.nc')
    ireg = list(reg_masks.data_vars).index(reg)
    
    mean = so_ds[reg].mean()
    stdev = so_ds[reg].std()

    fig,ax = plt.subplots(figsize=size)

    cmap = ['black', 'red', 'blue', 'green', 'orange', 'magenta']

    ax.plot(time, so_ds[reg], color=cmap[ireg])
    ax.set(xlim=xlim)

#     for yr in ens_yrs:
#         if yr == 22:
#             ax.axvspan(yr, yr+duration, alpha=0.25, color='gray', label='Ensemble Runs')
#         else:
#             ax.axvspan(yr, yr+duration, alpha=0.25, color='gray')
# #         ax.axvspan(yr, yr+10, color='gray', fill=False, hatch='xx', alpha=0.5)
#     ax.legend(loc='upper left')

    max_sigma = 0
    while (max_sigma*stdev) <= abs(so_ds[reg]-mean).max():
        max_sigma += 1
    
    stdev_lines = []
    sigma_labels = []
    sigma_num = np.arange(0,max_sigma+1)
    
    if max_sigma >= 6:
        sigma_num = np.arange(0,max_sigma+1,2)
        
    for s in sigma_num:
        if s == 0:
            stdev_lines.append(mean)
            sigma_labels.append('0')
        else:
            stdev_lines.append(mean + (s*stdev))
            stdev_lines.append(mean - (s*stdev))
            sigma_labels.insert(0,'-'+str(s)+'$\sigma$')
            sigma_labels.append('+'+str(s)+'$\sigma$')

    
    ax.hlines(stdev_lines, xlim[0], xlim[1], color='gray', ls='-.', alpha=0.5)
#     ax.hlines(mean, xlim[0], xlim[1], color='gray', ls='-', alpha=0.75)
    
    ax2 = ax.twinx()
    ax2.set(ylim=ax.get_ylim());
    ax2.set_yticks(sorted(stdev_lines));
    ax2.set_yticklabels(sigma_labels);
    
    ax.set_xlabel(xlabel)
    
    if title == '':
        ax.set_title(so_ds.attrs['name'].rstrip('.nc')+' - '+reg)
    if title != '':
        ax.set_title(title)
    if ylabel == '':
        ax.set_ylabel(so_ds[reg].attrs['label'])
    if ylabel != '':
        ax.set_ylabel(ylabel)
        
    return fig,ax

#################################################################################
#################################################################################
def open_ppp(
    var, reg, timescale='monthly'):
    
    writedir = '/home/bbuchovecky/storage/so_predict_derived/'
    subdir = 'PPP/'+var.upper()+'/'
    filename = var.lower()+'_ts_'+reg+'_'+timescale+'_ppp.nc'
    return xr.open_dataset(writedir+subdir+filename)

#################################################################################
#################################################################################

def format_ppp_axes(
    ax, timescale='monthly', summer_span=True, threshold=0.183, ymin=-0.2, ppp=None):
    
    if timescale == 'monthly':
        ax.hlines(threshold, 0, 121, color='k', linestyle='-.', label='Predictability threshold ('+str(threshold)+')')
        ax.set_xlim(1,120)
        
    if timescale == 'annual':
        ax.hlines(threshold, 0, 11, color='k', linestyle='-.', label='Predictability threshold ('+str(threshold)+')')
        ax.set_ylim(1,10)
        
    ax.set_ylim(ymin, 1.0)
    if ppp != None:
        if ppp.min() < 0:
            ax.set_ylim(round(ppp.min().values-0.0, 1)-0.1, 1.0)
        if ppp.min() > 0:
            ax.set_ylim(0.0,1.0)
    
    if summer_span and timescale == 'monthly':
        for m in np.arange(-1,120,12):
            if m < 0:
                ax.axvspan(m, m+4, alpha=0.25, color='gray', label='Months DJFM')
            else:
                ax.axvspan(m, m+4, alpha=0.25, color='gray')
    
    ## set xticks and labels
    yrs = np.array([2,4,6,8,10])
    if timescale == 'monthly':
        ax.set_xticks(yrs*12)
    ax.set_xticklabels(yrs)
    
    return ax

#################################################################################
#################################################################################

def plot_ppp(
    var, reg, so_reg='SouthernOcean', timescale='monthly', summer_span=True, threshold=0.183, figsize=(10,5), leg_loc='upper right'):
    
    reg_colors = dict({'SouthernOcean':'black', 'Weddell':'red', 'Indian':'blue', 'WestPacific':'green', 
                       'Ross':'orange', 'AmundBell':'magenta'})
    var_colors = dict({'npp':'limegreen', 'mld':'black', 'sie':'blue', 'sst':'red', 'sss':'darkorange', 
                       'cn_inv':'deepskyblue', 'pco2surf':'magenta', 'siv':'darkslateblue'})
    reg_names = dict({'SouthernOcean':'Pan-Antarctic', 'Weddell':'Weddell', 'Indian':'Indian', 
                      'WestPacific':'West Pacific', 'Ross':'Ross', 'AmundBell':'A and B'})
    var_su_names = dict({'npp':'NPP', 'mld':'MLD', 'sie':'SIE', 'sst':'SST', 'sss':'SSS', 
                         'cn_inv':'SIC', 'pco2surf':'Surface pCO$_2$', 'siv':'SIV'})
    var_lu_names = dict({'npp':'Net Primary Production', 'mld':'Mixed Layer Depth', 'sie':'Sea Ice Extent',
                         'sst':'Sea Surface Temperature', 'sss':'Sea Surface Salinity', 
                         'cn_inv':'Sea Ice Concentration', 'pco2surf':'Surface pCO$_2$', 'siv':'Sea Ice Volume'})
    var_ll_names = dict({'npp':'Net primary production', 'mld':'Mixed layer depth', 
                         'sie':'Sea ice extent', 'sst':'Sea surface temperature', 'sss':'Sea surface salinity',
                         'cn_inv':'Sea ice concentration', 'pco2surf':'Surface pCO$_2$', 'siv':'Sea ice volume'})
        
    if type(var) == str and type(so_reg) == str:
        fig,ax = plt.subplots(figsize=figsize)
        
        if reg == 'global':
            title = 'Global '+var_lu_names[var.lower()]
            r = 'Global'

        if reg == 'so':
            title = reg_names[so_reg]+' '+var_lu_names[var.lower()]
            r = so_reg
        
        ppp = open_ppp(var, reg, timescale)
        ax.plot(ppp['nT'], ppp[r], color='red', label=var_su_names[var])

        ax = format_ppp_axes(ax, timescale=timescale, summer_span=summer_span, threshold=threshold)
        
        
    if type(var) == list and type(so_reg) == str:
        fig,ax = plt.subplots(figsize=figsize)
        
        if reg == 'global':
            title = 'Global'
            r = 'Global'

        if reg == 'so':
            title = reg_names[so_reg]
            r = so_reg
        
        for v in var:
            ppp = open_ppp(v, reg, timescale)
            ax.plot(ppp['nT'], ppp[r], color=var_colors[v], label=var_su_names[v.lower()])
        
        ax = format_ppp_axes(ax, timescale=timescale, summer_span=summer_span, threshold=threshold)
        
    if type(so_reg) == list and type(var) == str:
        fig,ax = plt.subplots(figsize=figsize)
        
        ppp = open_ppp(var, reg, timescale)
        for r in so_reg:
            ax.plot(ppp['nT'], ppp[r], color=reg_colors[r], label=reg_names[r])

        title = var_lu_names[var]
        
        ax = format_ppp_axes(ax, timescale=timescale, summer_span=summer_span, threshold=threshold)

    ax.set_xlabel('Lead time (yr)')
    ax.set_ylabel('PPP')
    ax.set_title(title)
    
    ## set up legend
    leg = ax.legend(loc=leg_loc);
    for line in leg.get_lines():
        line.set_linewidth(2.0)
        line.set_linestyle(line.get_linestyle())
        
    return fig,ax

#################################################################################
#################################################################################

def ppp_var_grid(
    reg, timescale='monthly', summer_span=True, ylim=None, threshold=0.183, leg_loc='outside', figsize=(13,10)):
    
    reg_colors = dict({'SouthernOcean':'black', 'Weddell':'red', 'Indian':'blue', 'WestPacific':'green', 
                       'Ross':'orange', 'AmundBell':'magenta'})
    var_colors = dict({'npp':'limegreen', 'mld':'black', 'sie':'blue', 'sst':'red', 'sss':'darkorange', 
                       'cn_inv':'deepskyblue', 'pco2surf':'magenta', 'siv':'darkslateblue'})
    reg_names = dict({'SouthernOcean':'Pan-Antarctic', 'Weddell':'Weddell', 'Indian':'Indian', 
                      'WestPacific':'West Pacific', 'Ross':'Ross', 'AmundBell':'A and B'})
    var_su_names = dict({'npp':'NPP', 'mld':'MLD', 'sie':'SIE', 'sst':'SST', 'sss':'SSS', 
                         'cn_inv':'SIC', 'pco2surf':'Surface pCO$_2$', 'siv':'SIV'})
    var_lu_names = dict({'npp':'Net Primary Production', 'mld':'Mixed Layer Depth', 'sie':'Sea Ice Extent',
                         'sst':'Sea Surface Temperature', 'sss':'Sea Surface Salinity', 
                         'cn_inv':'Sea Ice Concentration', 'pco2surf':'Surface pCO$_2$', 'siv':'Sea Ice Volume'})
    var_ll_names = dict({'npp':'Net primary production', 'mld':'Mixed layer depth', 
                         'sie':'Sea ice extent', 'sst':'Sea surface temperature', 'sss':'Sea surface salinity',
                         'cn_inv':'Sea ice concentration', 'pco2surf':'Surface pCO$_2$', 'siv':'Sea ice volume'})
    
    variables = ['sst', 'sss', 'npp', 'mld', 'pco2surf', 'cn_inv', 'sie', 'siv']

    fig,axes = plt.subplots(4, 2, figsize=figsize, sharex=True)

    min_ylim = float('inf')
    for (i,var) in zip(range(8), variables):
        if type(reg) == str:
            ppp = open_ppp(var, 'so')[reg]

            axes[int(i/2),i%2].plot(ppp['nT'], ppp, color=var_colors[var])
            axes[int(i/2),i%2] = format_ppp_axes(axes[int(i/2),i%2], timescale=timescale, summer_span=summer_span, threshold=threshold)
            axes[int(i/2),i%2].set_title(var_su_names[var.lower()])

            if axes[int(i/2),i%2].get_ylim()[0] < min_ylim:
                min_ylim = axes[int(i/2),i%2].get_ylim()[0]

            if int(i/2) == 3:
                axes[int(i/2),i%2].set_xlabel('Lead time (yr)')

            if i%2 == 0:
                axes[int(i/2),i%2].set_ylabel('PPP')
                
        if type(reg) == list:
            for r in reg:
                ppp = open_ppp(var, 'so')[r]

                axes[int(i/2),i%2].plot(ppp['nT'], ppp, color=reg_colors[r], label=reg_names[r])

                if axes[int(i/2),i%2].get_ylim()[0] < min_ylim:
                    min_ylim = axes[int(i/2),i%2].get_ylim()[0]
                    
            axes[int(i/2),i%2] = format_ppp_axes(axes[int(i/2),i%2], timescale=timescale, summer_span=summer_span, threshold=threshold)
            axes[int(i/2),i%2].set_title(var_su_names[var.lower()])

            if int(i/2) == 3:
                axes[int(i/2),i%2].set_xlabel('Lead time (yr)')

            if i%2 == 0:
                axes[int(i/2),i%2].set_ylabel('PPP')

    if ylim == 'same':
        for i in range(8):
            axes[int(i/2),i%2].set_ylim(min_ylim, 1.0)

    if leg_loc == 'outside':
        leg = axes[0,1].legend(bbox_to_anchor = (1.02, 1));
    if leg_loc != 'outside':
        leg = axes[0,1].legend(loc=leg_loc);
    for line in leg.get_lines():
        line.set_linewidth(2.0)
        line.set_linestyle(line.get_linestyle())

    if type(reg) == str:
        fig.suptitle(reg_names[reg]+' PPP', fontsize=16)
    if type(reg) == list:
        fig.suptitle('Regional PPP', fontsize=16)
    fig.tight_layout()
    
    return fig,axes

#################################################################################
#################################################################################

def ppp_reg_grid(
    var, timescale='monthly', summer_span=True, ylim=None, threshold=0.183, leg_loc='outside', figsize=(13,8)):
    
    reg_colors = dict({'SouthernOcean':'black', 'Weddell':'red', 'Indian':'blue', 'WestPacific':'green', 
                       'Ross':'orange', 'AmundBell':'magenta'})
    var_colors = dict({'npp':'limegreen', 'mld':'black', 'sie':'blue', 'sst':'red', 'sss':'darkorange', 
                       'cn_inv':'deepskyblue', 'pco2surf':'magenta', 'siv':'darkslateblue'})
    reg_names = dict({'SouthernOcean':'Pan-Antarctic', 'Weddell':'Weddell', 'Indian':'Indian', 
                      'WestPacific':'West Pacific', 'Ross':'Ross', 'AmundBell':'A and B'})
    var_su_names = dict({'npp':'NPP', 'mld':'MLD', 'sie':'SIE', 'sst':'SST', 'sss':'SSS', 
                         'cn_inv':'SIC', 'pco2surf':'Surface pCO$_2$', 'siv':'SIV'})
    var_lu_names = dict({'npp':'Net Primary Production', 'mld':'Mixed Layer Depth', 'sie':'Sea Ice Extent',
                         'sst':'Sea Surface Temperature', 'sss':'Sea Surface Salinity', 
                         'cn_inv':'Sea Ice Concentration', 'pco2surf':'Surface pCO$_2$', 'siv':'Sea Ice Volume'})
    var_ll_names = dict({'npp':'Net primary production', 'mld':'Mixed layer depth', 
                         'sie':'Sea ice extent', 'sst':'Sea surface temperature', 'sss':'Sea surface salinity',
                         'cn_inv':'Sea ice concentration', 'pco2surf':'Surface pCO$_2$', 'siv':'Sea ice volume'})
    
    regions = ['SouthernOcean', 'Weddell', 'Indian', 'WestPacific', 'Ross', 'AmundBell']
    
    fig,axes = plt.subplots(3, 2, figsize=figsize, sharex=True)

    min_ylim = float('inf')
    for (i,reg) in zip(range(6), regions):
        if type(var) == str:    
            ppp = open_ppp(var, 'so')[reg]

            axes[int(i/2),i%2].plot(ppp['nT'], ppp, color=reg_colors[reg])
            axes[int(i/2),i%2] = format_ppp_axes(axes[int(i/2),i%2], timescale=timescale, summer_span=summer_span, threshold=threshold)
            axes[int(i/2),i%2].set_title(reg_names[reg])

            if axes[int(i/2),i%2].get_ylim()[0] < min_ylim:
                min_ylim = axes[int(i/2),i%2].get_ylim()[0]

            if int(i/2) == 2:
                axes[int(i/2),i%2].set_xlabel('Lead time (yr)')

            if i%2 == 0:
                axes[int(i/2),i%2].set_ylabel('PPP')
                
        if type(var) == list:
            for v in var:
                ppp = open_ppp(v, 'so')[reg]

                axes[int(i/2),i%2].plot(ppp['nT'], ppp, color=var_colors[v], label=var_su_names[v])

                if axes[int(i/2),i%2].get_ylim()[0] < min_ylim:
                    min_ylim = axes[int(i/2),i%2].get_ylim()[0]

            axes[int(i/2),i%2] = format_ppp_axes(axes[int(i/2),i%2], timescale=timescale, summer_span=summer_span, threshold=threshold)
            axes[int(i/2),i%2].set_title(reg_names[reg])
            
            if int(i/2) == 2:
                axes[int(i/2),i%2].set_xlabel('Lead time (yr)')

            if i%2 == 0:
                axes[int(i/2),i%2].set_ylabel('PPP')

    if ylim == 'same':
        for i in range(6):
            axes[int(i/2),i%2].set_ylim(min_ylim, 1.0)

    if leg_loc == 'outside':
        leg = axes[0,1].legend(bbox_to_anchor = (1.02, 1));
    if leg_loc != 'outside':
        leg = axes[0,1].legend(loc=leg_loc);
    for line in leg.get_lines():
        line.set_linewidth(2.0)
        line.set_linestyle(line.get_linestyle())
            
    if type(var) == str:
        fig.suptitle(var_lu_names[var.lower()]+' PPP', fontsize=16)
    if type(var) == list:
        fig.suptitle('Regional PPP', fontsize=16)
    fig.tight_layout()
    
    return fig,axes

#################################################################################
#################################################################################

def ppp_grid(
    variables, regions=['SouthernOcean', 'Weddell', 'Indian', 'WestPacific', 'Ross', 'AmundBell']):
    
    cols = len(regions)
    rows = len(variables)

    fig,axes = plt.subplots(rows, cols, figsize=(17,2*rows), sharex=True, sharey=True)

    reg_colors = dict({'SouthernOcean':'black', 'Weddell':'red', 'Indian':'blue', 'WestPacific':'green', 
                       'Ross':'orange', 'AmundBell':'magenta'})
    var_colors = dict({'npp':'limegreen', 'mld':'black', 'sie':'blue', 'sst':'red', 'sss':'darkorange', 
                       'cn_inv':'deepskyblue', 'pco2surf':'magenta', 'siv':'darkslateblue'})
    reg_names = dict({'SouthernOcean':'Pan-Antarctic', 'Weddell':'Weddell', 'Indian':'Indian', 
                      'WestPacific':'West Pacific', 'Ross':'Ross', 'AmundBell':'A and B'})
    var_su_names = dict({'npp':'NPP', 'mld':'MLD', 'sie':'SIE', 'sst':'SST', 'sss':'SSS', 
                         'cn_inv':'SIC', 'pco2surf':'Surface pCO$_2$', 'siv':'SIV'})
    var_lu_names = dict({'npp':'Net Primary Production', 'mld':'Mixed Layer Depth', 'sie':'Sea Ice Extent',
                         'sst':'Sea Surface Temperature', 'sss':'Sea Surface Salinity', 
                         'cn_inv':'Sea Ice Concentration', 'pco2surf':'Surface pCO$_2$', 'siv':'Sea Ice Volume'})
    var_ll_names = dict({'npp':'Net primary production', 'mld':'Mixed layer depth', 
                         'sie':'Sea ice extent', 'sst':'Sea surface temperature', 'sss':'Sea surface salinity',
                         'cn_inv':'Sea ice concentration', 'pco2surf':'Surface pCO$_2$', 'siv':'Sea ice volume'})

    ls = ['solid', 'dashed', 'dashdot', 'dotted']

    for (ireg, reg) in zip(range(cols), regions):

        for (ivar, var) in zip(range(rows), variables):

            ppp = open_ppp(var, 'so')[reg]

            axes[ivar, ireg].plot(ppp['nT'], ppp, color=var_colors[var])

            axes[ivar, ireg] = format_ppp_axes(axes[ivar, ireg], summer_span=False)
            axes[ivar, ireg].set(ylabel='', ylim=[-0.2,1])

            if ivar == 0:
                axes[ivar, ireg].set_title(reg_names[reg], fontweight='bold', fontsize=14)

            if ireg == 0:
                axes[ivar, ireg].set_ylabel(var_su_names[var.lower()], fontweight='bold', fontsize=14)

            if ivar == rows-1:
                axes[ivar, ireg].set_xlabel('Lead time (yr)')

    fig.suptitle('PPP in the Southern Ocean', fontsize=16, y=1.001)
    fig.tight_layout()
    
    
#################################################################################
#################################################################################
    
def plot_clim(
    var, reg, so_reg='SouthernOcean', leg_loc='outside', figsize=(10,5)):
    
    reg_colors = dict({'SouthernOcean':'black', 'Weddell':'red', 'Indian':'blue', 'WestPacific':'green', 
                       'Ross':'orange', 'AmundBell':'magenta'})
    var_colors = dict({'npp':'limegreen', 'mld':'black', 'sie':'blue', 'sst':'red', 'sss':'darkorange', 
                       'cn_inv':'deepskyblue', 'pco2surf':'magenta', 'siv':'darkslateblue'})

    reg_names = dict({'SouthernOcean':'Pan-Antarctic', 'Weddell':'Weddell', 'Indian':'Indian', 
                      'WestPacific':'West Pacific', 'Ross':'Ross', 'AmundBell':'A and B'})
    var_su_names = dict({'npp':'NPP', 'mld':'MLD', 'sie':'SIE', 'sst':'SST', 'sss':'SSS', 'cn_inv':'SIC', 
                         'pco2surf':'Surface pCO$_2$', 'siv':'SIV'})
    var_lu_names = dict({'npp':'Net Primary Production', 'mld':'Mixed Layer Depth', 'sie':'Sea Ice Extent', 
                         'sst':'Sea Surface Temperature', 'sss':'Sea Surface Salinity',
                       'cn_inv':'Sea Ice Concentration', 'pco2surf':'Surface pCO$_2$', 'siv':'Sea Ice Volume'})
    var_ll_names = dict({'npp':'Net primary production', 'mld':'Mixed layer depth', 'sie':'Sea ice extent', 
                         'sst':'Sea surface temperature', 'sss':'Sea surface salinity','cn_inv':'Sea ice concentration', 
                         'pco2surf':'Surface pCO$_2$', 'siv':'Sea ice volume'})
    var_units = dict({'npp':'mol C m$^{-2}$ s$^{-1}$', 'mld':'m', 'sie':'m$^2$', 'sst':'$^\circ$C', 
                      'sss':'psu', 'cn_inv':'%', 'pco2surf':'uatm', 'siv':'m$^3$'})
    
    writedir = '/home/bbuchovecky/storage/so_predict_derived/CTRL/'
    
    fig,ax = plt.subplots(figsize=figsize)
    month_ticks = np.arange(1,13)
    var = var.lower()
    ls = '-'
    
    if type(var) == list:
        assert len(var) <= 2, '\'var\' length is too long'
    
    if type(var) == str and type(so_reg) == str and so_reg != 'all':
        subdir = var.upper()+'/'
        filename = var.lower()+'_ts_'+reg+'_clim.nc'
        ds = xr.open_dataset(writedir+subdir+filename)
        
        if var == 'cn_inv':
            ds = ds*100
        
        if reg == 'global':
            clim = ds['Global']
            title = 'Global '+var_lu_names[var]+' Climatology'
        if reg == 'so':
            clim = ds[so_reg]
            title = reg_names[so_reg]+' '+var_su_names[var]+' Climatology'
            if so_reg == 'SouthernOcean':
                    ls = '-.'
            else:
                ls = '-'
            
        ax.plot(month_ticks, clim, color=var_colors[var], ls=ls)
        
        ax.set_title(title)
        ax.set_ylabel(var_su_names[var]+' ('+var_units[var]+')')
    
    if type(var) == list and type(so_reg) == str:
        for v in var:
            subdir = v.upper()+'/'
            filename = v.lower()+'_ts_'+reg+'_clim.nc'
            ds = xr.open_dataset(writedir+subdir+filename)
            
            if v == 'cn_inv':
                ds = ds*100
            
            if reg == 'global':
                clim = ds['Global']
            if reg == 'so':
                clim = ds[so_reg]
                if so_reg == 'SouthernOcean':
                    ls = '-.'
                else:
                    ls = '-'
            
            if var.index(v) == 0:
                ax.plot(month_ticks, clim, color=var_colors[v], label=var_su_names[v], ls=ls)
                ax.set_ylabel(var_su_names[v]+' ('+var_units[v]+')')
            
            if var.index(v) == 1:
                twinax = ax.twinx()
                twinax.plot(month_ticks, clim, color=var_colors[v], label=var_su_names[v], ls=ls)
                twinax.set_ylabel(var_su_names[v]+' ('+var_units[v]+')')
            
        ax.set_title(reg_names[so_reg]+' Climatology')        
        ax.legend(bbox_to_anchor = (1.04,1))
        twinax.legend(bbox_to_anchor = (1.15,0.91))
        
    if (type(so_reg) == list or so_reg == 'all') and type(var) == str:
        subdir = var.upper()+'/'
        filename = var.lower()+'_ts_'+reg+'_clim.nc'
        ds = xr.open_dataset(writedir+subdir+filename)
        
        if var == 'cn_inv':
            ds = ds*100
        
        if so_reg == 'all':
            so_reg = ['SouthernOcean', 'Weddell', 'Indian', 'WestPacific', 'Ross', 'AmundBell']
        
        for r in so_reg:
            clim = ds[r]
            if r == 'SouthernOcean':
                    ls = '-.'
            else:
                ls = '-'
            ax.plot(month_ticks, clim, color=reg_colors[r], label=reg_names[r], ls=ls)
            
        if leg_loc == 'outside':
            leg = ax.legend(bbox_to_anchor = (1.02,1))
        if leg_loc != 'outside':
            leg = ax.legend(loc=leg_loc)
        for line in leg.get_lines():
            line.set_linewidth(2.0)
            line.set_linestyle(line.get_linestyle())
            
        ax.set_title(var_lu_names[var]+' Climatology')
        ax.set_ylabel(var_su_names[var]+' ('+var_units[var]+')')
            
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(ds['abbrv_month'].values)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    
    ax.set_xlim(1,12)
            
    return fig,ax


#################################################################################
#################################################################################

def plot_grid_clim(
    figsize=(12,7)):
    
    reg_colors = dict({'SouthernOcean':'black', 'Weddell':'red', 'Indian':'blue', 'WestPacific':'green', 
                       'Ross':'orange', 'AmundBell':'magenta'})
    var_colors = dict({'npp':'limegreen', 'mld':'black', 'sie':'blue', 'sst':'red', 'sss':'darkorange', 
                       'cn_inv':'deepskyblue', 'pco2surf':'magenta', 'siv':'darkslateblue'})

    reg_names = dict({'SouthernOcean':'Pan-Antarctic', 'Weddell':'Weddell', 'Indian':'Indian', 
                      'WestPacific':'West Pacific', 'Ross':'Ross', 'AmundBell':'A and B'})
    var_su_names = dict({'npp':'NPP', 'mld':'MLD', 'sie':'SIE', 'sst':'SST', 'sss':'SSS', 'cn_inv':'SIC', 
                         'pco2surf':'Surface pCO$_2$', 'siv':'SIV'})
    var_lu_names = dict({'npp':'Net Primary Production', 'mld':'Mixed Layer Depth', 'sie':'Sea Ice Extent', 
                         'sst':'Sea Surface Temperature', 'sss':'Sea Surface Salinity',
                       'cn_inv':'Sea Ice Concentration', 'pco2surf':'Surface pCO$_2$', 'siv':'Sea Ice Volume'})
    var_ll_names = dict({'npp':'Net primary production', 'mld':'Mixed layer depth', 'sie':'Sea ice extent', 
                         'sst':'Sea surface temperature', 'sss':'Sea surface salinity','cn_inv':'Sea ice concentration', 
                         'pco2surf':'Surface pCO$_2$', 'siv':'Sea ice volume'})
    var_units = dict({'npp':'mol C m$^{-2}$ s$^{-1}$', 'mld':'m', 'sie':'m$^2$', 'sst':'$^\circ$C', 
                      'sss':'psu', 'cn_inv':'%', 'pco2surf':'uatm', 'siv':'m$^3$'})
    
    variables = ['sst', 'sss', 'npp', 'mld', 'pco2surf', 'cn_inv', 'sie', 'siv']
    regions=['SouthernOcean', 'Weddell', 'Indian', 'WestPacific', 'Ross', 'AmundBell']
    
    writedir = '/home/bbuchovecky/storage/so_predict_derived/CTRL/'
    
    fig,axes = plt.subplots(2, 4, figsize=figsize, sharex=True)
    month_ticks = np.arange(1,13)
    
    for (iv,v) in zip(range(len(variables)), variables):
        subdir = v.upper()+'/'
        filename = v.lower()+'_ts_so_clim.nc'
        clim = xr.open_dataset(writedir+subdir+filename)
        
        if v == 'cn_inv':
            clim = clim*100
        
        for (ir,r) in zip(range(len(regions)), regions):   
            if r == 'SouthernOcean':
                axes[int(iv/4),iv%4].plot(month_ticks, clim[r], label=reg_names[r], color=reg_colors[r], ls='-.')
            else:
                axes[int(iv/4),iv%4].plot(month_ticks, clim[r], label=reg_names[r], color=reg_colors[r])
            
        axes[int(iv/4),iv%4].set_title(var_su_names[v])
        axes[int(iv/4),iv%4].set_ylabel(var_su_names[v]+' ('+var_units[v]+')')
        axes[int(iv/4),iv%4].set_xlim(1,12)
        
        if int(iv/4) == 1:
            axes[int(iv/4),iv%4].set_xticks(month_ticks[::2])
            axes[int(iv/4),iv%4].set_xticklabels(clim['abbrv_month'].values[::2])
            for tick in axes[int(iv/4),iv%4].get_xticklabels():
                tick.set_rotation(45)
    
    leg = axes[1,3].legend(bbox_to_anchor=(0.95, -0.25));
    for line in leg.get_lines():
        line.set_linewidth(2.0)
        line.set_linestyle(line.get_linestyle())
    
    fig.suptitle('Climatology', fontsize=16)
    fig.tight_layout()


#################################################################################
#################################################################################   
    
def plot_ppp_szncycle(
    var, reg, so_reg='SouthernOcean', threshold=0.183, leg_loc='upper right', figsize=(10,5)):
        
    reg_colors = dict({'SouthernOcean':'black', 'Weddell':'red', 'Indian':'blue', 'WestPacific':'green', 
                       'Ross':'orange', 'AmundBell':'magenta'})
    var_colors = dict({'npp':'limegreen', 'mld':'black', 'sie':'blue', 'sst':'red', 'sss':'darkorange', 
                       'cn_inv':'deepskyblue', 'pco2surf':'magenta', 'siv':'darkslateblue'})

    reg_names = dict({'SouthernOcean':'Pan-Antarctic', 'Weddell':'Weddell', 'Indian':'Indian', 
                      'WestPacific':'West Pacific', 'Ross':'Ross', 'AmundBell':'A and B'})
    var_su_names = dict({'npp':'NPP', 'mld':'MLD', 'sie':'SIE', 'sst':'SST', 'sss':'SSS', 'cn_inv':'SIC', 
                         'pco2surf':'Surface pCO$_2$', 'siv':'SIV'})
    var_lu_names = dict({'npp':'Net Primary Production', 'mld':'Mixed Layer Depth', 'sie':'Sea Ice Extent', 
                         'sst':'Sea Surface Temperature', 'sss':'Sea Surface Salinity',
                       'cn_inv':'Sea Ice Concentration', 'pco2surf':'Surface pCO$_2$', 'siv':'Sea Ice Volume'})
    var_ll_names = dict({'npp':'Net primary production', 'mld':'Mixed layer depth', 'sie':'Sea ice extent', 
                         'sst':'Sea surface temperature', 'sss':'Sea surface salinity','cn_inv':'Sea ice concentration', 
                         'pco2surf':'Surface pCO$_2$', 'siv':'Sea ice volume'})
    var_units = dict({'npp':'mol C m$^{-2}$ s$^{-1}$', 'mld':'m', 'sie':'m$^2$', 'sst':'$^\circ$C', 
                      'sss':'psu', 'cn_inv':'%', 'pco2surf':'uatm', 'siv':'m$^3$'})
    
    writedir = '/home/bbuchovecky/storage/so_predict_derived/PPP/'
    
    fig,ax = plt.subplots(figsize=figsize)
    month_ticks = np.arange(1,13)
    ls = '-'
    
    if type(var) == str and type(so_reg) == str and so_reg != 'all':
        subdir = var.upper()+'/'
        filename = var.lower()+'_ts_'+reg+'_szncycle_ppp.nc'
        ds = xr.open_dataset(writedir+subdir+filename)
        
        if var == 'cn_inv':
            ds = ds*100
        
        if reg == 'global':
            clim = ds['Global']
            title = 'Global '+var_lu_names[var]+' PPP Seasonal Cycle'
        if reg == 'so':
            clim = ds[so_reg]
            title = reg_names[so_reg]+' '+var_su_names[var]+' PPP Seasonal Cycle'
            
        ax.plot(month_ticks, clim, color=var_colors[var], ls=ls)
        
        ax.set_title(title)
        ax.set_ylabel('PPP')
    
    if type(var) == list and type(so_reg) == str:
        for v in var:
            subdir = v.upper()+'/'
            filename = v.lower()+'_ts_'+reg+'_szncycle_ppp.nc'
            ds = xr.open_dataset(writedir+subdir+filename)
            
            if v == 'cn_inv':
                ds = ds*100
            
            if reg == 'global':
                clim = ds['Global']
            if reg == 'so':
                clim = ds[so_reg]
            
            ax.plot(month_ticks, clim, color=var_colors[v], label=var_su_names[v], ls=ls)
            ax.set_ylabel('PPP')
            
        ax.set_title(reg_names[so_reg]+' PPP Seasonal Cycle')        
        ax.legend(loc=leg_loc)
        
    if (type(so_reg) == list or so_reg == 'all') and type(var) == str:
        subdir = var.upper()+'/'
        filename = var.lower()+'_ts_'+reg+'_szncycle_ppp.nc'
        ds = xr.open_dataset(writedir+subdir+filename)
        
        if so_reg == 'all':
            so_reg = ['SouthernOcean', 'Weddell', 'Indian', 'WestPacific', 'Ross', 'AmundBell']
        
        for r in so_reg:
            clim = ds[r]
            ax.plot(month_ticks, clim, color=reg_colors[r], label=reg_names[r], ls=ls)
            
        if leg_loc == 'outside':
            leg = ax.legend(bbox_to_anchor = (1.02,1))
        if leg_loc != 'outside':
            leg = ax.legend(loc=leg_loc)
        for line in leg.get_lines():
            line.set_linewidth(2.0)
            line.set_linestyle(line.get_linestyle())
            
        ax.set_title(var_lu_names[var]+' PPP Seasonal Cycle')
        ax.set_ylabel('PPP')
            
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(ds['abbrv_month'].values)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    
    if threshold != None:
        ax.hlines(threshold, 0, 13, color='k', linestyle='-.', label='Predictability threshold ('+str(threshold)+')')
    ax.set_xlim(1,12)
    ax.set_ylim(-0.2, 1.0)
    
    fig.tight_layout()
            
    return fig,ax


#################################################################################
################################################################################# 

def plot_grid_ppp_szncycle(
    threshold=0.183, figsize=(12,7)):
    
    reg_colors = dict({'SouthernOcean':'black', 'Weddell':'red', 'Indian':'blue', 'WestPacific':'green', 
                       'Ross':'orange', 'AmundBell':'magenta'})
    var_colors = dict({'npp':'limegreen', 'mld':'black', 'sie':'blue', 'sst':'red', 'sss':'darkorange', 
                       'cn_inv':'deepskyblue', 'pco2surf':'magenta', 'siv':'darkslateblue'})

    reg_names = dict({'SouthernOcean':'Pan-Antarctic', 'Weddell':'Weddell', 'Indian':'Indian', 
                      'WestPacific':'West Pacific', 'Ross':'Ross', 'AmundBell':'A and B'})
    var_su_names = dict({'npp':'NPP', 'mld':'MLD', 'sie':'SIE', 'sst':'SST', 'sss':'SSS', 'cn_inv':'SIC', 
                         'pco2surf':'Surface pCO$_2$', 'siv':'SIV'})
    var_lu_names = dict({'npp':'Net Primary Production', 'mld':'Mixed Layer Depth', 'sie':'Sea Ice Extent', 
                         'sst':'Sea Surface Temperature', 'sss':'Sea Surface Salinity',
                       'cn_inv':'Sea Ice Concentration', 'pco2surf':'Surface pCO$_2$', 'siv':'Sea Ice Volume'})
    var_ll_names = dict({'npp':'Net primary production', 'mld':'Mixed layer depth', 'sie':'Sea ice extent', 
                         'sst':'Sea surface temperature', 'sss':'Sea surface salinity','cn_inv':'Sea ice concentration', 
                         'pco2surf':'Surface pCO$_2$', 'siv':'Sea ice volume'})
    var_units = dict({'npp':'mol C m$^{-2}$ s$^{-1}$', 'mld':'m', 'sie':'m$^2$', 'sst':'$^\circ$C', 
                      'sss':'psu', 'cn_inv':'%', 'pco2surf':'uatm', 'siv':'m$^3$'})
    
    variables = ['sst', 'sss', 'npp', 'mld', 'pco2surf', 'cn_inv', 'sie', 'siv']
    regions=['SouthernOcean', 'Weddell', 'Indian', 'WestPacific', 'Ross', 'AmundBell']
    
    writedir = '/home/bbuchovecky/storage/so_predict_derived/PPP/'
    
    fig,axes = plt.subplots(2, 4, figsize=figsize, sharex=True)
    month_ticks = np.arange(1,13)
    
    for (iv,v) in zip(range(len(variables)), variables):
        subdir = v.upper()+'/'
        filename = v.lower()+'_ts_so_szncycle_ppp.nc'
        clim = xr.open_dataset(writedir+subdir+filename)
        
        for (ir,r) in zip(range(len(regions)), regions):   
            axes[int(iv/4),iv%4].plot(month_ticks, clim[r], label=reg_names[r], color=reg_colors[r])
        
        if threshold != None:
            axes[int(iv/4),iv%4].hlines(threshold, 0, 13, color='k', linestyle='-.', label='Predictability threshold ('+str(threshold)+')')
        
        axes[int(iv/4),iv%4].set_title(var_su_names[v])
        axes[int(iv/4),iv%4].set_ylabel('PPP')
        axes[int(iv/4),iv%4].set_xlim(1,12)
        axes[int(iv/4),iv%4].set_ylim(-0.2, 1.0)
        
        if int(iv/4) == 1:
            axes[int(iv/4),iv%4].set_xticks(month_ticks[::2])
            axes[int(iv/4),iv%4].set_xticklabels(clim['abbrv_month'].values[::2])
            for tick in axes[int(iv/4),iv%4].get_xticklabels():
                tick.set_rotation(45)
    
    leg = axes[1,3].legend(bbox_to_anchor=(0.95, -0.25));
    for line in leg.get_lines():
        line.set_linewidth(2.0)
        line.set_linestyle(line.get_linestyle())
    
    fig.suptitle('PPP Seasonal Cycle', fontsize=16)
    fig.tight_layout()