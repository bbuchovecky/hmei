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
def xr_size(
    da):
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
def dir_inspect(
    path):
    """
    Description:
        Views the sorted contents of a directory.
    
    Parameters:
        path - directory path
    """
    
    return sorted(os.listdir(path))


#################################################################################
#################################################################################
def open_raw_ctrl(
    var):
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
def open_gridcell_ctrl(
    var, metric=False, reg='global'):
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
    
    if reg == 'global':
        if (var.lower() == 'sie') or (var.lower() == 'siv'):
            filename = write_rootdir+subdir_ctrl+var.upper()+'/'+var.lower()+'_ctrl_so_timeseries.nc'
        else:
            filename = write_rootdir+subdir_ctrl+var.upper()+'/'+var.lower()+'_ctrl_global_'+metric.lower()+'.nc'
        print(filename)
        return xr.open_dataset(filename)
    
    elif (reg.lower() == 'so') or (reg.lower() == 'southernocean'):
        if (var.lower() == 'sie') or (var.lower() == 'siv'):
            filename = write_rootdir+subdir_ctrl+var.upper()+'/'+var.lower()+'_ctrl_so_timeseries.nc'
        else:
            filename = write_rootdir+subdir_ctrl+var.upper()+'/'+var.lower()+'_ctrl_so_'+metric.lower()+'.nc'
        print(filename)
        return xr.open_dataset(filename)
    
    else:
        print('<invalid parameters>')
        return


#################################################################################
#################################################################################
def gridcell_calc(
    ds, var, metric=False, resize=False):
    
    da = ds[var]
    
    if resize == True:
        da = SouthernOcean_resize(da, time=True)
    
    if (type(metric) == bool) and (metric == False):            
        clim = da.groupby('time.month').mean(dim='time')
        clim = clim.rename(da.name+'_gridcell_clim')

        anom = da.groupby('time.month') - clim
        anom = anom.rename(da.name+'_gridcell_anom')

#         var = anom.var(dim='time') ## single value, variance of all 3600 months
        var = anom.groupby('time.month').var(dim='time') ## 12 values, variance for each month
        var = var.rename(da.name+'_gridcell_var')
        var.attrs['description'] = 'variance of each month over the 300 year control simulation'

        if resize == True:
            metrics = xr.merge([da, clim, anom, var])
            metrics.attrs = da.attrs
            metrics.attrs['region'] = 'latitude <= -40.5 degN'
        if resize == False:
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
        var = anom.groupby('time.month').var(dim='time') ## 12 values, variance for each month
        var = var.rename(da.name+'_gridcell_var')
        var.attrs['description'] = 'variance of each month over the 300 year control simulation'
        return var

    else:
        print('<invalid parameters>')
        return

    
#################################################################################
#################################################################################
def save_gridcell_calc(
    ds, var, write_rootdir, subdir):
    
    filename = var.lower()+'_so_gridcell_metrics.nc'
    path = write_rootdir+subdir+var.upper()+'/'+filename
    ds.to_netcdf(path)
    print(path)


#################################################################################
#################################################################################
# Resize DataArray to Southern Ocean region (latitude <= -40.5degN)
# The Southern Ocean mask is south of -55degN, but this resizing will provide room to play around with latitude bounds.
# For this model data, `yt_ocean` is equal to `geolat_t` in the Southern Hemisphere, so `yt_ocean` is used to resize the data.
# The `da` DataArray parameter requires the following coordinates - case sensitive: {'time', 'yt_ocean', 'xt_ocean', 'geolat_t', 'geolon_t'}
def SouthernOcean_resize(
    da, time):
    
    if time == True:
        so = da.drop({'time', 'xt_ocean', 'yt_ocean', 'geolon_t', 'geolat_t'})
        so = so[:,:42].assign_coords({'time':da.time})
        so = so.assign_coords({'yt_ocean':da.yt_ocean[:42], 'xt_ocean':da.xt_ocean})
        so = so.assign_coords({'geolon_t':da.geolon_t[:42,:], 'geolat_t':da.geolat_t[:42,:]})
        
    elif time == False:
        so = da.drop({'xt_ocean', 'yt_ocean', 'geolon_t', 'geolat_t'})
        so = so[:42,:]
        so = so.assign_coords({'yt_ocean':da.yt_ocean[:42], 'xt_ocean':da.xt_ocean})
        so = so.assign_coords({'geolon_t':da.geolon_t[:42,:], 'geolat_t':da.geolat_t[:42,:]})
    
    else:
        print('the parameter \'time\' must be a boolean')
        return
    
    return so
    

#################################################################################
#################################################################################
def reg_annual_mean(
    ds, var, masks=False, reg='global'):
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
        if var == 'SIE_area' or var == 'SIV_area':
            annual_mean = global_var.groupby('time.year').mean(dim='time').sum(dim={'xt_ocean', 'yt_ocean'}).groupby('time.year').mean(dim='time')
        else:
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
        if var == 'SIE_area' or var == 'SIV_area':
            annual_mean = reg_var.groupby('time.year').mean(dim='time').sum(dim={'xt_ocean', 'yt_ocean'})
        else:
            annual_mean = (reg_var * ocean_grid['area_t']).groupby('time.year').mean(dim='time').sum(dim={'xt_ocean', 'yt_ocean'}) / area_sum
        annual_mean.name = reg + '_' + var + '_annual_mean'
        return annual_mean
    
    else:
        print('<invalid parameters>')
        return
    

#################################################################################
#################################################################################
def reg_annual_anom(
    ds, var, masks=False, reg='global'):
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
        annual_anom.name = annual_anom.name.removesuffix('annual_mean') + 'annual_anomaly'
        return annual_anom
        
    ## anomaly for a single specified region from masks parameter
    elif (mask_bool) and (reg != 'global'):
        annual_mean = reg_annual_mean(ds, var, masks, reg)
        single_mean = annual_mean.mean(dim='year')
        annual_anom = annual_mean - single_mean
        annual_anom.name = anom.name.removesuffix('annual_mean') + 'annual_anomaly'
        return annual_anom
        
    else:
        print('<invalid parameters>')
        return


#################################################################################
#################################################################################
def reg_monthly_mean(
    ds, var, masks=False, reg='global'):
    
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
def reg_monthly_anom(
    ds_mean):
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
def reg_monthly_var(
    ds_anom):
    
    ## copy and reshape the anomaly dataset
    ds_var = ds_anom.copy(deep=True)
    reg = True
    
    try:
        ds_var = ds_var.drop({'month','SouthernOcean','Weddell','Indian',
                              'WestPacific','Ross','AmundBell'})
    except:
        ds_var = ds_var.drop({'month','Global'})
        reg = False
    
    ds_var['month'] = np.arange(1,13)
    ds_var.attrs['name'] = ds_var.attrs['name'].replace('anom', 'var')

    ## create numpy array for grouping anomaly data by month
    ## i.e. [ [Jan], [Feb], ..., [Dec]]
    if reg:
        np_anom = np.zeros((len(ds_anom.data_vars),12,300))
        
        ## group anomaly data by month
        for (reg,i) in zip(ds_anom.data_vars,range(6)):
            for m in range(0,3600):           
                np_anom[i][m%12][int(m/12)] = ds_anom[reg].isel(month=m)
        
        ## compute variance
        np_var = np.var(np_anom, axis=2)
        if np.shape(np_var) != (6,12):
            raise Exception('variance array is the wrong shape')
        
        ## migrate data from numpy array to dataset
        for (reg,i) in zip(ds_anom.data_vars, range(6)):
            ds_var[reg] = (('month'), np_var[i])

    if not reg:
        np_anom = np.zeros((12,300))
        
        ## group anomaly data by month
        for reg in ds_anom.data_vars:
            for m in range(0,3600):           
                np_anom[m%12][int(m/12)] = ds_anom[reg].isel(month=m)
        
        ## compute variance
        np_var = np.var(np_anom, axis=1)
        if np.shape(np_var) != (12,):
            raise Exception('variance array is the wrong shape')
            
        ## migrate data from numpy array to dataset
        ds_var['Global'] = (('month'), np_var)
    
#     if reg:
#         np_var = np.var(np_anom, axis=2)
#         print(np.shape(np_var))
#         if np.shape(np_var) != (6,12):
#             raise Exception('variance array is the wrong shape')
#     if not reg:
#         np_var = np.var(np_anom)
#         print(np.shape(np_var))
#         if np.shape(np_var) != (1,12):
#             raise Exception('variance array is the wrong shape')
            
    ## migrate data from numpy array to dataset
#     for (reg,i) in zip(ds_anom.data_vars, range(6)):
#         ds_var[reg] = (('month'), np_var[i])
    
    return ds_var


#################################################################################
#################################################################################

def comp_ppp(
    var, reg, save=False, timescale='monthly'):

    ## import necessary data
    writedir = '/home/bbuchovecky/storage/so_predict_derived/'
    
    ## variance of the control simulation for each month
    ## dimensions (month: 12)
    subdir = 'CTRL/'+var.upper()+'/'
    filename = var.lower()+'_ts_'+reg+'_'+timescale+'_var.nc'
    ctrl_var = xr.open_dataset(writedir+subdir+filename)
    
    ## ensemble anomalies
    ## dimensions (nStart: 6, nEns: 40, nT: 120)
    subdir = var.upper()+'_ENSEMBLE/'
    filename = var.lower()+'_ens_'+reg+'_'+timescale+'_anom.nc'
    ens_anom = xr.open_dataset(writedir+subdir+filename)
    
    regions = ctrl_var.data_vars

    ## constants
    N = 6
    M = 40

    ## numerator coefficient
    num_coeff = 1 / (N * (M - 1))

    ## numerator outer sum - across all ensembles
    num_outer_sum = np.zeros((6,120))

    PPP = np.zeros( (len(regions), 120) )

    for (iReg,r) in zip(range(len(regions)), regions):

        for j in range(N):
            ## numerator inner sum - across all ensemble members
            num_inner_sum = np.zeros((40,120))

            ## ensemble mean
            ens_mean = ens_anom[r][j].mean(dim='nEns')

            for i in range(M):
                num_inner_sum[i] = np.square(ens_anom[r][j,i] - ens_mean)

            num_outer_sum[j] = np.sum(num_inner_sum, axis=0)

        ## numerator total sum
        num_sum = np.sum(num_outer_sum, axis=0)

        ## compute PPP
        for m in range(120):
            PPP[iReg][m] = 1 - ( (num_coeff * num_sum[m]) / ctrl_var[r][m%12] )
            
    ## organize PPP data in a Dataset
    PPP_da = []
    for (iReg,r) in zip(range(len(regions)), regions):
        
        PPP_reg = xr.DataArray(
                    data=PPP[iReg],
                    dims=['nT'],
                    coords=dict(
                        nT=np.arange(1,121)
                    ),
                    name=r
                )
        
        PPP_da.append(PPP_reg.copy(deep=True))
    
    PPP_ds = xr.merge(PPP_da)
    PPP_ds.attrs['name'] = var.lower()+'_ts_'+reg+'_'+timescale+'_ppp.nc'
        
    if save:
        subdir = 'PPP/'+var.upper()+'/'
        filename = var.lower()+'_ts_'+reg+'_'+timescale+'_ppp.nc'
        PPP_ds.to_netcdf(writedir+subdir+filename)
        print(writedir+subdir+filename)
        
    elif not save:
        return PPP_ds

    
#################################################################################
#################################################################################    
    
def comp_clim(var, reg, save=False): 
    writedir = '/home/bbuchovecky/storage/so_predict_derived/CTRL/'
    subdir = var.upper()+'/'
    filename = var.lower()+'_ts_'+reg+'_monthly_mean.nc'
    
    mean = xr.open_dataset(writedir+subdir+filename)
    clim = mean.groupby('time.month').mean(dim='time')
    
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 
                   'August', 'September', 'October', 'November', 'December']
    abbrv_month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 
                         'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    clim['month'] = month_names
    clim['abbrv_month'] = abbrv_month_names
    
    filename = var.lower()+'_ts_'+reg+'_clim.nc'
    clim.attrs['name'] = filename
    
    if save:
        clim.to_netcdf(writedir+subdir+filename)
        print(writedir+subdir+filename)
    if not save:
        print('not saved - '+writedir+subdir+filename)
        return clim