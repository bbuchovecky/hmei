import os
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", message="Unable to decode time axis into full numpy.datetime64 objects, continuing using cftime.datetime objects instead, reason: dates out of range")
warnings.filterwarnings("ignore", message="invalid value encountered in reduce")

import xarray as xr
import numpy as np
import math


#################################################################################
#################################################################################
def xr_size(da):
    """
    Description:
        Returns the size of a xarray.Dataset or xarray.DataArray object in
        Gigabytes GB (not Gibibytes (GiB)
    
    Parameters:
        da - xarray.Dataset or xarray.DataArray
    """

    return str(da.nbytes / (1000**3))+' gigabytes'


#################################################################################
#################################################################################
def dir_inspect(path):
    """
    Description:
        Views the sorted contents of a directory.
    
    Parameters:
        path - directory path
    """
    
    return sorted(os.listdir(path))


#################################################################################
#################################################################################
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
        ds = ds.assign_coords({'geolon_t':geolon_t, 'geolat_t':geolat_t})
        return ds
        
    elif (var.upper() == 'CN_INV'):
        path = rootdir+subdir_ctrl+'CN/CN_inv_*.nc'
        print(path)
        ds = xr.open_mfdataset(path)
        ds = ds.rename({'XT':'xt_ocean', 'YT':'yt_ocean', 'TIME':'time'})
        ds = ds.assign_coords({'xt_ocean':ocean_grid.xt_ocean, 'yt_ocean':ocean_grid.yt_ocean})
        ds = ds.assign_coords({'geolon_t':geolon_t, 'geolat_t':geolat_t})
        return ds
        
    elif (var.upper() == 'NPP'):
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
    
    elif (var.lower() == 'mld'):
        path = rootdir+subdir_ctrl+'MLD/mld_0*.nc'
        print(path)
        ds = xr.open_mfdataset(path)
        return ds

    else:
        print('<invalid parameters>')
        return

    
#################################################################################
#################################################################################
# def open_gridcell_ctrl(var, metric=False, reg='global'):
#     """
#     Description:
#         Opens processed grid-cell data (either climatology, anomaly, or variance)
#         as <xarray.Dataset> objects.
        
#         path = /home/bbuchovecky/storage/so_predict_derived/CTRL/[VAR].
    
#     Parameters:
#         var    - string variable name
#         metric - 'climatology', 'anomaly', 'variance'
#                  (or False for SIE and SIV)
#         reg    - 'global', 'so', 'southernocean'
#                  (or False for global, True for Southern Ocean)
#     """
    
#     write_rootdir = '/home/bbuchovecky/storage/so_predict_derived/'
#     subdir_ctrl = 'CTRL/'
    
#     if reg == 'global':
#         if (var.lower() == 'sie') or (var.lower() == 'siv'):
#             filename = write_rootdir+subdir_ctrl+var.upper()+'/'+var.lower()+'_ctrl_so_timeseries.nc'
#         else:
#             filename = write_rootdir+subdir_ctrl+var.upper()+'/'+var.lower()+'_ctrl_global_'+metric.lower()+'.nc'
#         print(filename)
#         return xr.open_dataset(filename)
    
#     elif (reg.lower() == 'so') or (reg.lower() == 'southernocean'):
#         if (var.lower() == 'sie') or (var.lower() == 'siv'):
#             filename = write_rootdir+subdir_ctrl+var.upper()+'/'+var.lower()+'_ctrl_so_timeseries.nc'
#         else:
#             filename = write_rootdir+subdir_ctrl+var.upper()+'/'+var.lower()+'_ctrl_so_'+metric.lower()+'.nc'
#         print(filename)
#         return xr.open_dataset(filename)
    
#     else:
#         print('<invalid parameters>')
#         return


#################################################################################
#################################################################################
def gridcell_calc(ds, var, metric=False):
    da = ds[var]
    
    if (type(metric) == bool) and (metric == False):            
        clim = da.groupby('time.month').mean(dim='time')
        clim = clim.rename(da.name+'_gridcell_clim')

        anom = da.groupby('time.month') - clim
        anom = anom.rename(da.name+'_gridcell_anom')

#         var = anom.var(dim='time') ## single value, variance of all 3600 months
#         var = anom.groupby('time.month').var(dim='time') ## 12 values, variance for each month
        var = var.rename(da.name+'_gridcell_var')

        metrics = xr.merge([clim, anom, var])
        metrics.attrs = da.attrs
        return metrics
    
    elif metric == 'clim':
        clim = da.groupby('time.month').mean(dim='time')
        clim = clim.rename(da.name+'_gridcell_clim')
        return clim
    
    elif metric == 'anom':
        clim = da.groupby('time.month').mean(dim='time')
        anom = da.groupby('time.month') - clim
        anom = anom.rename(da.name+'_gridcell_anom')
        return anom
    
    elif metric == 'var':
        clim = da.groupby('time.month').mean(dim='time')
        anom = da.groupby('time.month') - clim
#         var = anom.var(dim='time') ## single value, variance of all 3600 months
#         var = anom.groupby('time.month').var(dim='time') ## 12 values, variance for each month
        var = var.rename(da.name+'_gridcell_var')
        return var

    else:
        print('<invalid parameters>')
        return

    
#################################################################################
#################################################################################
def save_gridcell_calc(ds, var, write_rootdir, subdir):
    filename = var.lower()+'_gridcell_metrics.nc'
    path = write_rootdir+subdir+var.upper()+'/'+filename
    ds.to_netcdf(path)
    print(path)
    

#################################################################################
#################################################################################
def reg_annual_mean(ds, var, masks=False, reg='global'):
    """
    Description:
        Computes the annual mean timeseries of a variable over a specific region
        or globally.
    
    Parameters:
        ds         - xarray.Dataset containing raw model data
        var        - string variable name
        reg        - string region name
        masks      - xarray.Dataset of regional masks
    """
    
    mask_bool = True
    if type(masks) == bool:
        mask_bool = masks
    
    ## get ocean model grid information
    ocean_grid = xr.open_dataset('/local/projects/so_predict/esm2m_froelicher/GRID/ocean.static.nc')
    
    ## get grid-cell area for the ocean only
    area = xr.where(np.isnan(ds[var][0]), np.nan, ocean_grid['area_t'])
    
    ## global mean
    if (not mask_bool) and (reg == 'global'):
        area_sum = area.sum(dim={'xt_ocean', 'yt_ocean'})
        global_var = ds[var]
        annual_mean = (global_var * ocean_grid['area_t']).groupby('time.year').mean(dim='time').sum(dim={'xt_ocean', 'yt_ocean'}) / area_sum
        annual_mean.name = 'Global'
        annual_mean = annual_mean.to_dataset()
        annual_mean.attrs['name'] = var.lower() + '_global_annual_mean'
        return annual_mean
    
    ## mean for a single specified region from masks parameter
    elif (mask_bool) and (reg != 'global'):
        area = area.where(masks[reg] == 1)
        area_sum = area.sum(dim={'xt_ocean', 'yt_ocean'})
        reg_var = ds[var].where(masks[reg] == 1)
        annual_mean = (reg_var * ocean_grid['area_t']).groupby('time.year').mean(dim='time').sum(dim={'xt_ocean', 'yt_ocean'}) / area_sum
        annual_mean.name = reg + '_' + var + '_annual_mean'
        return annual_mean
    
    else:
        print('<invalid parameters>')
        return
    

#################################################################################
#################################################################################
def reg_annual_anom(ds, var, masks=False, reg='global'):
    """
    Description:
        Computes the annual anomaly timeseries of a variable over a specific region
        or globally.
    
    Parameters:
        ds         - xarray.Dataset containing raw model data
        var        - string variable name
        reg        - string region name
        masks      - xarray.Dataset of regional masks
    """

    mask_bool = True
    if type(masks) == type(True):
        mask_bool = masks
    
    ## get ocean model grid information
    ocean_grid = xr.open_dataset('/local/projects/so_predict/esm2m_froelicher/GRID/ocean.static.nc')
    
    ## get grid-cell area for the ocean only
    area = xr.where(np.isnan(ds[var][0]), np.nan, ocean_grid['area_t'])

    ## global anomaly
    if (not mask_bool) and (reg == 'global'):
        annual_mean = reg_annual_mean(ds, var)
        single_mean = annual_mean.mean(dim='year')
        annual_anom = annual_mean - single_mean
        annual_anom.name = annual_anom.name.removesuffix('annual_mean') + 'yearly_anomaly'
        return annual_anom
        
    ## anomaly for a single specified region from masks parameter
    elif (mask_bool) and (reg != 'global'):
        annual_mean = reg_annual_mean(ds, var, masks, reg)
        single_mean = annual_mean.mean(dim='year')
        annual_anom = annual_mean - single_mean
        annual_anom.name = anom.name.removesuffix('annual_mean') + 'yearly_anomaly'
        return annual_anom
        
    else:
        print('<invalid parameters>')
        return


#################################################################################
#################################################################################
def reg_monthly_mean(ds, var, masks=False, reg='global'):
    
    mask_bool = True
    if type(masks) == type(True):
        mask_bool = masks
    
    ## get ocean model grid information
    ocean_grid = xr.open_dataset('/local/projects/so_predict/esm2m_froelicher/GRID/ocean.static.nc')
    
    ## get grid-cell area for the ocean only
    area = xr.where(np.isnan(ds[var][0]), np.nan, ocean_grid['area_t'])

    ## global mean
    if (not mask_bool) and (reg == 'global'):
        area_sum = area.sum(dim={'xt_ocean', 'yt_ocean'})
        global_var = ds[var]
        monthly_mean = (global_var * ocean_grid['area_t']).sum(dim={'xt_ocean', 'yt_ocean'}) / area_sum
        monthly_mean.name = 'Global'
        monthly_mean = monthly_mean.to_dataset()
        monthly_mean.attrs['name'] = var.lower() + '_global_monthly_mean'
        return monthly_mean
    
    ## mean for a single specified region from masks parameter
    elif (mask_bool) and (reg != 'global'):
        area = area.where(masks[reg] == 1)
        area_sum = area.sum(dim={'xt_ocean', 'yt_ocean'})
        reg_var = ds[var].where(masks[reg] == 1)
        monthly_mean = (reg_var * ocean_grid['area_t']).sum(dim={'xt_ocean', 'yt_ocean'}) / area_sum
        monthly_mean.name = reg + '_' + var + '_monthly_mean'
        return monthly_mean
    
    else:
        print('<invalid parameters>')
        return


#################################################################################
#################################################################################
def reg_monthly_anom(ds_mean):
    """
    Description:
        Computes the monthly anomaly timeseries of a variable.
    
    Parameters:
        ds_mean - dataset containing monthly mean timeseries
    """

    ds_anom = ds_mean.copy(deep=True)
    ds_anom['month'] = np.arange(1,3601)
    for reg in ds_anom.data_vars:
        ds_anom[reg] = (('month'), ds_anom[reg])
    ds_anom = ds_anom.drop('time')
    ds_anom.attrs['name'] = ds_anom.attrs['name'].replace('mean', 'anom')
    
    for reg in ds_mean.data_vars:
        np_clim = ds_mean[reg].groupby('time.month').mean(dim='time').values
        np_mean = ds_mean[reg].values

        reg_anom = np.zeros(3600)
        for m in range(0,3600):
            reg_anom[m] = np_mean[m] - np_clim[m%12]

        ds_anom[reg] = (('month'), reg_anom)
    
    return ds_anom


#################################################################################
#################################################################################
def reg_monthly_var(ds_anom):
    return