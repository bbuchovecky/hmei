import os
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", message="Unable to decode time axis into full numpy.datetime64 objects, continuing using cftime.datetime objects instead, reason: dates out of range")
warnings.filterwarnings("ignore", message="invalid value encountered in reduce")

import xarray as xr
import numpy as np
import math

## return the size of the dataset in gigabytes GB (not gibibytes GiB)
def xr_size(da):
    """
    Description:
        Returns the size of a xarray.Dataset or xarray.DataArray object in
        Gigabytes GB (not Gibibytes (GiB)
    
    Parameters:
        da - xarray.Dataset or xarray.DataArray
    """

    return str(da.nbytes / (1000**3))+' gigabytes'


## view contents of a directory
def dir_inspect(path):
    """
    Description:
        Views the sorted contents of a directory.
    
    Parameters:
        path - directory path
    """
    
    return sorted(os.listdir(path))


## opens raw data files from Froelicher et al. 2020 
## and formats the coordinate/variable names
def open_raw_ctrl(var):
    """
    Description:
        Opens raw control simulation model data from Froelicher et al. 2020 as
        <xarray.Dataset> objects. Formats the coordinate and variable names to
        ensure coords 'time', 'xt_ocean', 'yt_ocean', 'geolon_t', and 'geolat_t'.
        
        path = /local/projects/so_predict/esm2m_froelicher/CTRL/[VAR]/
    
    Parameters:
        var - string variable name
    """

    rootdir = '/local/projects/so_predict/esm2m_froelicher/'
    subdir_ctrl = 'CTRL/'
    subdir_grid = 'GRID/'
    
    ocean_grid = xr.open_dataset(rootdir+subdir_grid+'ocean.static.nc')
    geolat_t = ocean_grid.geolat_t
    geolon_t = ocean_grid.geolon_t

    if (var.lower() == 'sst'):
        path = rootdir+subdir_ctrl+'SST/sst_*.nc'
        print(path)
        ds = xr.open_mfdataset(path)
        return ds
    
    elif (var.lower() == 'sss'):
        path = rootdir+subdir_ctrl+'SSS/sss_*.nc'
        print(path)
        ds = xr.open_mfdataset(path)
        ds = ds_sss.assign_coords({'geolon_t':geolon_t, 'geolat_t':geolat_t})
        return ds
        
    elif (var.upper() == 'CN_INV'):
        path = rootdir+subdir_ctrl+'CN/CN_inv_*.nc'
        print(path)
        ds = xr.open_mfdataset(path)
        ds = ds.rename({'XT':'xt_ocean', 'YT':'yt_ocean', 'TIME':'time'})
        ds = ds.assign_coords({'xt_ocean':ocean_grid.xt_ocean, 'yt_ocean':ocean_grid.yt_ocean})
        ds = ds.assign_coords({'geolon_t':geolon_t, 'geolat_t':geolat_t})
        return ds
        
    elif (var.upper() == 'CN_INV'):
        path = rootdir+subdir_ctrl+'NPP/NPP_*.nc'
        print(path)
        ds = xr.open_mfdataset(path)
        ds = ds.rename({'XT_OCEAN':'xt_ocean', 'YT_OCEAN':'yt_ocean', 'TIME':'time'})
        ds = ds.assign_coords({'geolon_t':geolon_t, 'geolat_t':geolat_t})
        return ds
    
    elif (var.lower() == 'pco2surf'):
        path = rootdir+subdir_ctrl+'PCO2SURF/pco2surf_*.nc'
        print(path)
        ds = xr.open_mfdataset(path)
        return ds

    else:
        print('<invalid parameters>')
        return


def open_gridcell_ctrl(var, metric=False, reg=False):
    """
    Description:
        Opens processed grid-cell data (either climatology, anomaly, or variance)
        as <xarray.Dataset> objects.
        
        path = /home/bbuchovecky/storage/so_predict_derived/CTRL/[VAR].
    
    Parameters:
        var    - string variable name
        metric - 'climatology', 'anomaly', 'variance'
                 (or False for SIE and SIV)
        reg    - 'global', 'so', 'southernocean'
                 (or False for global, True for Southern Ocean)
    """
    
    write_rootdir = '/home/bbuchovecky/storage/so_predict_derived/'
    subdir_ctrl = 'CTRL/'
    
    if (not reg) or (reg == 'global'):
        if (var.lower() == 'sie') or (var.lower() == 'siv'):
            filename = write_rootdir+subdir_ctrl+var.upper()+'/'+var.lower()+'_ctrl_so_timeseries.nc'
        else:
            filename = write_rootdir+subdir_ctrl+var.upper()+'/'+var.lower()+'_ctrl_global_'+metric.lower()+'.nc'
        print(filename)
        return xr.open_dataset(filename)
    
    elif (reg) or (reg.lower() == 'so') or (reg.lower() == 'southernocean'):
        if (var.lower() == 'sie') or (var.lower() == 'siv'):
            filename = write_rootdir+subdir_ctrl+var.upper()+'/'+var.lower()+'_ctrl_so_timeseries.nc'
        else:
            filename = write_rootdir+subdir_ctrl+var.upper()+'/'+var.lower()+'_ctrl_so_'+metric.lower()+'.nc'
        print(filename)
        return xr.open_dataset(filename)
    
    else:
        print('<invalid parameters>')
        return


## computes the regional or global mean of a variable
def reg_annual_mean(ds, var, ocean_grid=False, masks=False, reg=False):
    """
    Description:
        Computes the annual regional or global mean of a variable. Returns an
        xarray.[]
        
    STILL WORKING
    """
    
    if not ocean_grid:
        ocean_grid = xr.open_dataset('/local/projects/so_predict/esm2m_froelicher/GRID/ocean.static.nc')
    
    area = xr.where(np.isnan(ds[var][0]), np.nan, ocean_grid['area_t'])
    
    mask_bool = True
    if type(masks) == type(True):
        mask_bool = masks
    
    ## global mean
    if (not mask_bool) and (reg == False):
        area_sum = area.sum(dim={'xt_ocean', 'yt_ocean'})
        global_var = ds[var]
        annual_mean = (global_var * ocean_grid['area_t']).groupby('time.year').mean(dim='time').sum(dim={'xt_ocean', 'yt_ocean'}) / area_sum
        annual_mean.name = 'Global_' + var + '_annual_mean'
        return annual_mean
    
    ## mean for each region from masks parameter, returns a list of timeseries
    elif (mask_bool) and (reg == False):        
        annual_mean = []
        for (reg, i) in zip(masks.data_vars, range(6)):
            area = area.where(masks[reg] == 1)
            area_sum = area.sum(dim={'xt_ocean', 'yt_ocean'})
            reg_var = ds[var].where(masks[reg] == 1)
            reg_mean = (reg_var * ocean_grid['area_t']).groupby('time.year').mean(dim='time').sum(dim={'xt_ocean', 'yt_ocean'}) / area_sum
            reg_mean.name = reg + '_' + var + '_annual_mean'
            annual_mean.append(reg_mean)
        return annual_mean
    
    ## mean for a single specified region from masks parameter
    elif (mask_bool) and (reg != False):
        area = area.where(masks[reg] == 1)
        area_sum = area.sum(dim={'xt_ocean', 'yt_ocean'})
        reg_var = ds[var].where(masks[reg] == 1)
        annual_mean = (reg_var * ocean_grid['area_t']).groupby('time.year').mean(dim='time').sum(dim={'xt_ocean', 'yt_ocean'}) / area_sum
        annual_mean.name = reg + '_' + var + '_annual_mean'
        return annual_mean
    
    else:
        print('<invalid parameters>')
        return
    
    
## computes the regional or global mean of a variable
def reg_annual_anom(ds, var, ocean_grid=False, masks=False, reg=False):
    if not ocean_grid:
        ocean_grid = xr.open_dataset('/local/projects/so_predict/esm2m_froelicher/GRID/ocean.static.nc')
    
    area = xr.where(np.isnan(ds[var][0]), np.nan, ocean_grid['area_t'])
    
    mask_bool = True
    if type(masks) == type(True):
        mask_bool = masks

    ## global anomaly
    if (not mask_bool) and (reg == False):
        annual_mean = reg_annual_mean(ds, var, ocean_grid)
        single_mean = annual_mean.mean(dim='year')
        annual_anom = annual_mean - single_mean
        annual_anom.name = annual_anom.name.removesuffix('annual_mean') + 'yearly_anomaly'
        return annual_anom
        
    ## anomaly for each region from masks parameter, returns a list of timeseries
    elif (mask_bool) and (reg == False):
        annual_mean = reg_annual_mean(ds, var, ocean_grid, masks)
        single_mean = []
        for reg in annual_mean:
            single_mean.append(reg.mean(dim='year'))
            
        annual_anom = []
        for (t,m) in zip(annual_mean, single_mean):
            anom = t - m
            anom.name = anom.name.removesuffix('annual_mean') + 'yearly_anomaly'
            annual_anom.append(anom)
        return annual_anom
        
    ## anomaly for a single specified region from masks parameter
    elif (mask_bool) and (reg != False):
        annual_mean = reg_annual_mean(ds, var, ocean_grid, masks, reg)
        single_mean = annual_mean.mean(dim='year')
        annual_anom = annual_mean - single_mean
        annual_anom.name = anom.name.removesuffix('annual_mean') + 'yearly_anomaly'
        return annual_anom
        
    else:
        print('<invalid parameters>')
        return