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
        Gigabytes GB, not Gibibytes (GiB)
    
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
    ds, var, metric, resize=False, save=False):
    
    da = ds[var]
    
    if resize == True:
        da = SouthernOcean_resize(da, time=True)

    if metric == 'clim':
        clim = da.groupby('time.month').mean(dim='time')
        clim = clim.rename(da.name+'__clim')
        return clim
    
    if metric == 'anom':
        clim = da.groupby('time.month').mean(dim='time')
        anom = da.groupby('time.month') - clim
        anom = anom.rename(da.name+'_gridcell_anom')
        return anom
    
    if metric == 'var':
        clim = da.groupby('time.month').mean(dim='time')
        anom = da.groupby('time.month') - clim
        var = anom.groupby('time.month').var(dim='time') ## 12 values, variance for each month
        var = var.rename(da.name+'_gridcell_var')
        var.attrs['description'] = 'variance of each month over the 300 year control simulation'
        return var


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
            annual_mean = global_var.sum(dim={'xt_ocean', 'yt_ocean'}).groupby('time.year').mean(dim='time')
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
    
    subdir = 'PPP/'+var.upper()+'/'
    filename = var.lower()+'_ts_'+reg+'_'+timescale+'_ppp.nc'
        
    if save:
        PPP_ds.to_netcdf(writedir+subdir+filename)
        print(writedir+subdir+filename)
        
    elif not save:
        print(writedir+subdir+filename)
        return PPP_ds

    
#################################################################################
#################################################################################    
    
def comp_clim(
    var, reg, save=False): 
    
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

    
#################################################################################
#################################################################################

def comp_gc_metrics(
    ds, variable, resize=False, save=False):
    
    writedir = '/home/bbuchovecky/storage/so_predict_derived/CTRL/'
    subdir = variable.upper()+'/'
    reg = 'global'
    
    da = ds[variable.lower()]
    
    if resize:
        da = SouthernOcean_resize(da, time=True)
        reg = 'so'
    
    ## global climatology
    clim = da.groupby('time.month').mean(dim='time')
    clim.name = variable.lower()+'_clim'
    clim.attrs = da.attrs
    clim.attrs['description'] = 'monthly climatology'
    
    clim_name = variable.lower()+'_gc_'+reg+'_clim.nc'
    clim = clim.to_dataset()
    clim.attrs['name'] = clim_name
    
    ## global monthly anomaly
    anom = da.groupby('time.month') - clim[variable.lower()+'_clim']
    anom.name = variable.lower()+'_anom'
    anom.attrs = da.attrs
    anom.attrs['description'] = 'monthly anomaly'
    
    anom_name = variable.lower()+'_gc_'+reg+'_monthly_anom.nc'
    anom = anom.to_dataset()
    anom.attrs['name'] = anom_name
    
    ## global monthly variance
    var = anom[variable.lower()+'_anom'].groupby('time.month').var(dim='time')
    var.name = variable.lower()+'_var'
    var.attrs = da.attrs
    var.attrs['units'] = '( '+da.attrs['units']+' )^2'
    var.attrs['description'] = 'variance of monthly anomaly'
    
    var_name = variable.lower()+'_gc_'+reg+'_monthly_var.nc'
    var = var.to_dataset()
    var.attrs['name'] = var_name
    
    if save:
        clim.to_netcdf(writedir+subdir+clim_name)
        anom.to_netcdf(writedir+subdir+anom_name)
        var.to_netcdf(writedir+subdir+var_name)
        
        print(writedir+subdir+clim_name)
        print(writedir+subdir+anom_name)
        print(writedir+subdir+var_name)
        
    if not save:
        print(writedir+subdir+clim_name)
        print(writedir+subdir+anom_name)
        print(writedir+subdir+var_name)
        return clim,anom,var
    
    
#################################################################################
#################################################################################

def comp_annual_mean(
    ds, var, reg, save=False):
    
    writedir = '/home/bbuchovecky/storage/so_predict_derived/'
    reg_masks = xr.open_dataset(writedir+'regional_global_masks.nc')
    
    ## get ocean model grid information
    ocean_grid = xr.open_dataset('/local/projects/so_predict/esm2m_froelicher/GRID/ocean.static.nc')
    
    ## get grid-cell area for the ocean only
    total_area = xr.where(np.isnan(ds[var][0]), np.nan, ocean_grid['area_t'])
    
    ## global mean
    if reg == 'global':
        area_sum = total_area.sum(dim={'xt_ocean', 'yt_ocean'})
        
        if var == 'SIE_area' or var == 'SIV_area':
            annual_mean = ds[var].groupby('time.year').mean(dim='time').sum(dim={'xt_ocean', 'yt_ocean'})
        else:
            annual_mean = (ds[var] * ocean_grid['area_t']).groupby('time.year').mean(dim='time').sum(dim={'xt_ocean', 'yt_ocean'}) / area_sum
        
        annual_mean.name = 'Global'
        annual_mean.attrs = ds[var].attrs
        annual_mean.attrs['description'] = 'annual mean'
        annual_mean = annual_mean.to_dataset()
        annual_mean = annual_mean.drop('time')
        
        filename = var.lower()+'_ts_'+reg+'_annual_mean.nc'
        annual_mean.attrs['name'] = filename
        
        if save:
            annual_mean.to_netcdf(writedir+'CTRL/'+var.upper()+'/'+filename)
            print(writedir+'CTRL/'+var.upper()+'/'+filename)
        if not save:
            print(writedir+'CTRL/'+var.upper()+'/'+filename)
            return annual_mean
    
    ## regional mean
    if reg == 'so':
        regions = ['SouthernOcean', 'Weddell', 'Indian', 'WestPacific', 'Ross', 'AmundBell']
        reg_mean = []
        
        for r in regions:
            area = total_area.where(reg_masks[r] == 1)
            area_sum = area.sum(dim={'xt_ocean', 'yt_ocean'})
            reg_var = ds[var].where(reg_masks[r] == 1)
            
            if var == 'SIE_area' or var == 'SIV_area':
                annual_mean = reg_var.groupby('time.year').mean(dim='time').sum(dim={'xt_ocean', 'yt_ocean'})
            else:
                annual_mean = (reg_var * ocean_grid['area_t']).groupby('time.year').mean(dim='time').sum(dim={'xt_ocean', 'yt_ocean'}) / area_sum
            
            annual_mean.name = r
            annual_mean.attrs = ds[var].attrs
            annual_mean.attrs['description'] = 'annual mean'
            
            reg_mean.append(annual_mean.copy(deep=True))
            
        annual_mean = xr.merge(reg_mean)
        annual_mean = annual_mean.drop('time')
        
        filename = var.lower()+'_ts_'+reg+'_annual_mean.nc'
        annual_mean.attrs = {}
        annual_mean.attrs['name'] = filename
        
        if save:
            annual_mean.to_netcdf(writedir+'CTRL/'+var.upper()+'/'+filename)
            print(writedir+'CTRL/'+var.upper()+'/'+filename)
        if not save:
            print(writedir+'CTRL/'+var.upper()+'/'+filename)
            return annual_mean
        
        
#################################################################################
#################################################################################

def comp_monthly_mean(
    ds, var, reg, save=False):
    
    writedir = '/home/bbuchovecky/storage/so_predict_derived/'
    reg_masks = xr.open_dataset(writedir+'regional_global_masks.nc')
    
    ## get ocean model grid information
    ocean_grid = xr.open_dataset('/local/projects/so_predict/esm2m_froelicher/GRID/ocean.static.nc')
    
    ## get grid-cell area for the ocean only
    total_area = xr.where(np.isnan(ds[var][0]), np.nan, ocean_grid['area_t'])
    
    ## global mean
    if reg == 'global':
        area_sum = total_area.sum(dim={'xt_ocean', 'yt_ocean'})
        
        if var == 'SIE_area' or var == 'SIV_area':
            monthly_mean = ds[var].sum(dim={'xt_ocean', 'yt_ocean'})
        else:
            monthly_mean = (ds[var] * ocean_grid['area_t']).sum(dim={'xt_ocean', 'yt_ocean'}) / area_sum
        
        monthly_mean.name = 'Global'
        monthly_mean.attrs = ds[var].attrs
        monthly_mean.attrs['description'] = 'monthly mean'
        monthly_mean = monthly_mean.to_dataset()
        
        filename = var.lower()+'_ts_'+reg+'_monthly_mean.nc'
        monthly_mean.attrs['name'] = filename
        
        if save:
            monthly_mean.to_netcdf(writedir+'CTRL/'+var.upper()+'/'+filename)
            print(writedir+'CTRL/'+var.upper()+'/'+filename)
        if not save:
            print(writedir+'CTRL/'+var.upper()+'/'+filename)
            return monthly_mean
    
    ## regional mean
    if reg == 'so':
        regions = ['SouthernOcean', 'Weddell', 'Indian', 'WestPacific', 'Ross', 'AmundBell']
        reg_mean = []
        
        for r in regions:
            area = total_area.where(reg_masks[r] == 1)
            area_sum = area.sum(dim={'xt_ocean', 'yt_ocean'})
            reg_var = ds[var].where(reg_masks[r] == 1)
            
            if var == 'SIE_area' or var == 'SIV_area':
                monthly_mean = reg_var.sum(dim={'xt_ocean', 'yt_ocean'})
            else:
                monthly_mean = (reg_var * ocean_grid['area_t']).sum(dim={'xt_ocean', 'yt_ocean'}) / area_sum
            
            monthly_mean.name = r
            monthly_mean.attrs = ds[var].attrs
            monthly_mean.attrs['description'] = 'monthly mean'
            
            reg_mean.append(monthly_mean.copy(deep=True))
            
        monthly_mean = xr.merge(reg_mean)
        
        filename = var.lower()+'_ts_'+reg+'_monthly_mean.nc'
        monthly_mean.attrs = {}
        monthly_mean.attrs['name'] = filename
        
        if save:
            monthly_mean.to_netcdf(writedir+'CTRL/'+var.upper()+'/'+filename)
            print(writedir+'CTRL/'+var.upper()+'/'+filename)
        if not save:
            print(writedir+'CTRL/'+var.upper()+'/'+filename)
            return monthly_mean
        
        
#################################################################################
#################################################################################

def comp_monthly_anom(
    var, reg, mean=None, save=False):
    
    writedir = '/home/bbuchovecky/storage/so_predict_derived/'
    
    if os.path.isfile(writedir+'CTRL/'+var.upper()+'/'+var.lower()+'_ts_'+reg+'_monthly_mean.nc') and mean == None:
        mean = xr.open_dataset(writedir+'CTRL/'+var.upper()+'/'+var.lower()+'_ts_'+reg+'_monthly_mean.nc')
        print('Source: '+var.lower()+'_ts_'+reg+'_monthly_mean.nc')

    ds_anom = mean.copy(deep=True)
    ds_anom['month'] = np.arange(1,3601)
    
    for r in ds_anom.data_vars:
        ds_anom[r] = (('month'), ds_anom[r])
        
    ds_anom = ds_anom.drop('time')
    
    for r in mean.data_vars:
        np_clim = mean[r].groupby('time.month').mean(dim='time').values
        np_mean = mean[r].values

        reg_anom = np.zeros(3600)
        for m in range(0,3600):
            reg_anom[m] = np_mean[m] - np_clim[m%12]

        ds_anom[r] = (('month'), reg_anom)
        ds_anom[r].attrs = mean[r].attrs
        ds_anom[r].attrs['description'] = 'monthly anomaly'
    
    filename = var.lower()+'_ts_'+reg+'_monthly_anom.nc'
    ds_anom.attrs['name'] = filename
    
    if save:
        ds_anom.to_netcdf(writedir+'CTRL/'+var.upper()+'/'+filename)
        print(writedir+'CTRL/'+var.upper()+'/'+filename)
    if not save:
        print(writedir+'CTRL/'+var.upper()+'/'+filename)
        return ds_anom
    
    
#################################################################################
#################################################################################

def comp_monthly_var(
    var, reg, anom=None, save=False):
    
    writedir = '/home/bbuchovecky/storage/so_predict_derived/'
    
    if os.path.isfile(writedir+'CTRL/'+var.upper()+'/'+var.lower()+'_ts_'+reg+'_monthly_anom.nc') and anom == None:
        anom = xr.open_dataset(writedir+'CTRL/'+var.upper()+'/'+var.lower()+'_ts_'+reg+'_monthly_anom.nc')
        print('Source: '+var.lower()+'_ts_'+reg+'_monthly_anom.nc')
    
    ds_var = anom.copy(deep=True)
    
    if reg == 'global':
        ds_var = ds_var.drop({'month','Global'})
        ds_var['month'] = np.arange(1,13)
        
        np_anom = np.zeros((12,300))
        
        ## group anomaly data by month
        for r in anom.data_vars:
            for m in range(0,3600):           
                np_anom[m%12][int(m/12)] = anom[r].isel(month=m)
        
        ## compute variance
        np_var = np.var(np_anom, axis=1)
        if np.shape(np_var) != (12,):
            raise Exception('variance array is the wrong shape')
            
        ## migrate data from numpy array to dataset
        ds_var['Global'] = (('month'), np_var)
        ds_var['Global'].attrs['name'] = 'variance of monthly anomaly'
    
    if reg == 'so':
        ds_var = ds_var.drop({'month','SouthernOcean','Weddell','Indian','WestPacific','Ross','AmundBell'})
        ds_var['month'] = np.arange(1,13)
        
        np_anom = np.zeros((len(anom.data_vars),12,300))
        
        ## group anomaly data by month
        for (r,i) in zip(anom.data_vars,range(6)):
            for m in range(0,3600):           
                np_anom[i][m%12][int(m/12)] = anom[r].isel(month=m)
        
        ## compute variance
        np_var = np.var(np_anom, axis=2)
        if np.shape(np_var) != (6,12):
            raise Exception('variance array is the wrong shape')
        
        ## migrate data from numpy array to dataset
        for (r,i) in zip(anom.data_vars, range(6)):
            ds_var[r] = (('month'), np_var[i])
            ds_var[r].attrs['name'] = 'variance of monthly anomaly'
    
    filename = var.lower()+'_ts_'+reg+'_monthly_var.nc'
    ds_var.attrs['name'] = filename
    
    if save:
        ds_var.to_netcdf(writedir+'CTRL/'+var.upper()+'/'+filename)
        print(writedir+'CTRL/'+var.upper()+'/'+filename)
    if not save:
        print(writedir+'CTRL/'+var.upper()+'/'+filename)
        return ds_var
    
    
#################################################################################
#################################################################################

def comp_ens_global_mean(
    variables, timescale, save=False, SIE=False, SIV=False):
    
    rootdir = '/local/projects/so_predict/esm2m_froelicher/'
    writedir = '/home/bbuchovecky/storage/so_predict_derived/'
    
    reg_masks = xr.open_dataset(writedir+'regional_global_masks.nc')
    regions = list(reg_masks.data_vars)

    ocean_grid = xr.open_dataset(rootdir+'GRID/ocean.static.nc')
    geolat_t = ocean_grid.geolat_t
    geolon_t = ocean_grid.geolon_t
    
    months = 120

    nT_months = np.arange(1,months+1)
    nT_years = np.arange(1,(months/12)+1)
    nStart = np.arange(1,7)
    nEns = np.empty(0, dtype=str)

    start_yrs = ['0170_0179', '0022_0031', '0064_0073', '0106_0115', '0232_0241', '0295_0304']
    
    biomass_vars = ['sfc_ndi', 'sfc_nlg_diatoms', 'sfc_nlg_nondiatoms', 'sfc_nsm']
    
    ## iterate through all variables
    for var in variables:
        ## update status
        print('\n######## '+var[0]+' ########\n')

        subdir = var[0]+'_ENSEMBLE/'

        ## list with datasets for each start year (includes 40 ensembles)
        yrs_list = []

        ## iterate through the start years
        for (i,yrs) in zip(range(1,7), start_yrs):

            ## list with datasets for each ensemble run (pos/neg, 1-20)
            ens_list = []

            ## reset 'nEns' coord list
            nEns = np.empty(0, dtype=str)

            ## iterate through ensembles
            for sign in ['neg', 'pos']:
                for pert in range(1,21):
                    
                    nEns = np.append(nEns, sign+f'{pert:02}')
                    
                    if var[0] != 'BIOMASS':
                        filename = var[1]+'_ENS'+f'{i:02}'+'_'+sign+f'{pert:02}'+'_'+yrs+'.nc'
                        ds = xr.open_dataset(rootdir+subdir+filename)
                    
                    if var[0] == 'BIOMASS':
                        filename = '_ENS'+f'{i:02}'+'_'+sign+f'{pert:02}'+'_'+yrs+'.nc'
                        ndi = xr.open_dataset(rootdir+subdir+biomass_vars[0]+filename)
                        diatoms = xr.open_dataset(rootdir+subdir+biomass_vars[1]+filename)
                        nondiatoms = xr.open_dataset(rootdir+subdir+biomass_vars[2]+filename)
                        nsm = xr.open_dataset(rootdir+subdir+biomass_vars[3]+filename)

                        ds = ndi.copy(deep=True)
                        ds = ds.rename({'sfc_ndi':'sfc_biomass'})

                        ds['sfc_biomass'] = ndi['sfc_ndi'] + diatoms['sfc_nlg_diatoms'] + nondiatoms['sfc_nlg_nondiatoms'] + nsm['sfc_nsm']
                    
                    ## reassign coord names to ensure continuity between datasets
                    coords = list(ds.coords)
                    if (coords.count('XT') == 1) and (coords.count('YT') == 1) and (coords.count('TIME') == 1):
                        ds = ds.rename({'XT':'xt_ocean', 'YT':'yt_ocean', 'TIME':'time'})

                    if (coords.count('XT_OCEAN') == 1) and (coords.count('YT_OCEAN') == 1) and (coords.count('TIME') == 1):
                        ds = ds.rename({'XT_OCEAN':'xt_ocean', 'YT_OCEAN':'yt_ocean', 'TIME':'time'})
                        
                    if (coords.count('xt') == 1) and (coords.count('yt') == 1):
                        ds = ds.rename({'xt':'xt_ocean', 'yt':'yt_ocean'})
                        
                    ## truncate at 120 months for CN_INV
                    if ds['time'].size > months:
                        time = xr.cftime_range(start='0000', periods=months, freq='MS', calendar='noleap')
                        temp = ds.copy(deep=True)
                        ds = ds.drop('time')
                        ds = ds.drop_vars(var[2])
                        ds = ds.assign_coords({'time':time})
                        ds[var[2]] = (('time','yt_ocean','xt_ocean'), temp[var[2]][:months])
                                            
                    ## assign ocean_grid coords for continuity
                    ds = ds.assign_coords({'xt_ocean':ocean_grid.xt_ocean, 'yt_ocean':ocean_grid.yt_ocean})
                    
                    ## remove 'area_t' gridcells on land, keep only the ocean
                    area = xr.where(np.isnan(ds[var[2]][0]), np.nan, ocean_grid['area_t'])

                    ## total ocean area in the region
                    area_sum = area.sum(dim={'xt_ocean', 'yt_ocean'})
                    
                    ## load and format SIC data for SIV computation
                    if SIV:
                        sie_area = xr.open_dataset(rootdir+'CN_ENSEMBLE/CN_inv_ENS'+f'{i:02}'+'_'+sign+f'{pert:02}'+'_'+yrs+'.nc')
                        sie_area = sie_area.rename({'XT':'xt_ocean', 'YT':'yt_ocean', 'TIME':'time'})
                        sie_area = sie_area.assign_coords({'xt_ocean':ocean_grid.xt_ocean, 'yt_ocean':ocean_grid.yt_ocean})
        
                        if sie_area['CN_INV'].size > months:
                            temp = sie_area.copy(deep=True)
                            sie_area = sie_area.drop('time')
                            sie_area = sie_area.drop_vars('CN_INV')
                            sie_area['CN_INV'] = (('time','yt_ocean','xt_ocean'), temp['CN_INV'][:months])
                        
                        sie_area = sie_area.assign_coords({'time':ds['time']})
                        sie_area.transpose()
                        sie_area = sie_area['CN_INV'] * area
                        
                    ## threshold SIE at >15% SIC
                    if SIE:
                        gridcell = xr.where(ds[var[2]] > 0.15, 1, 0)

                    if timescale == 'annual':                      
                        if SIE:
                            mean = (gridcell * ocean_grid['area_t']).groupby('time.year').mean(dim='time').sum(dim={'xt_ocean', 'yt_ocean'})
                            
                        elif SIV:
                            mean = (ds[var[2]] * sie_area).groupby('time.year').mean(dim='time').sum(dim={'xt_ocean', 'yt_ocean'})
                        
                        else:                          
                            ## annual mean - compute areal integral for each year
                            ## (1) find annual mean value for each gridcell
                            ## (2) compute areal integral for the annual mean gridcells
                            mean = (ds[var[2]] * ocean_grid['area_t']).groupby('time.year').mean(dim='time').sum(dim={'xt_ocean', 'yt_ocean'}) / area_sum

                        ## reassign 'time' cftime coord to 'nT' integer coord from 1-10
                        if list(mean.coords).count('time') > 0:
                            mean = mean.drop('time')
                        mean = mean.rename({'year':'nT'})
                        mean = mean.assign_coords({'nT':nT_years})

                    if timescale == 'monthly':
                        if SIE:
                            mean = (gridcell * ocean_grid['area_t']).sum(dim={'xt_ocean', 'yt_ocean'})
                            
                        elif SIV:
                            mean = (ds[var[2]] * sie_area).sum(dim={'xt_ocean', 'yt_ocean'})
                        
                        else:
                            ## monthly mean - compute areal integral for each month
                            mean = (ds[var[2]] * ocean_grid['area_t']).sum(dim={'xt_ocean', 'yt_ocean'}) / area_sum

                        ## reassign 'time' cftime coord to 'nT' integer coord from 1-120
                        mean = mean.rename({'time':'nT'})
                        mean = mean.assign_coords({'nT':nT_months})

                    ## rename the array to the region's name
                    mean.name = 'Global'

                    ## convert the array to a dataset
                    this_ens = mean.to_dataset()

                    ## append a copy of the dataset to the ensembles list
                    ens_list.append(this_ens.copy(deep=True))

                    ## update status
                    if (i == 1) and (sign == 'neg') and (pert == 1):
                        print('this_ens.coords = '+str(list(this_ens.coords)))
                        print('this_ens.data_vars = '+str(list(this_ens.data_vars))+'\n')

                    print('o',end='')
                    if pert == 20:
                        print('')

                ## update status
                if sign == 'neg':
                    print('var: '+var[2]+' | nStart: '+str(i)+' | nEns: 20/40')
                if sign == 'pos':
                    print('var: '+var[2]+' | nStart: '+str(i)+' | nEns: 40/40')

            ## concatenate ensemble arrays along new 'nEns' dimension
            these_ens = xr.concat(ens_list, 'nEns')
            these_ens = these_ens.assign_coords({'nEns':nEns}) 

            ## append a copy of the dataset to the start years list
            yrs_list.append(these_ens.copy(deep=True))

            ## update status
            if i == 1:
                print('\nthese_ens.coords = '+str(list(these_ens.coords)))
                print('these_ens.data_vars = '+str(list(these_ens.data_vars))+'\n')

        ## concatenate start years arrays along new 'nStart' dimension
        all_ens = xr.concat(yrs_list, 'nStart')
        all_ens = all_ens.assign_coords({'nStart':start_yrs})

        ## copy attributes to each region data variable
        all_ens['Global'].attrs = mean.attrs

        ## add description
        if SIE:
            filename = 'sie_ens_global_'+timescale+'_mean.nc'
        if SIV:
            filename = 'siv_ens_global_'+timescale+'_mean.nc'
        if not SIE and not SIV:
            filename = var[1].lower()+'_ens_global_'+timescale+'_mean.nc'
        all_ens.attrs['name'] = filename

        print('\nall_ens.coords = '+str(list(all_ens.coords)))
        print('all_ens.data_vars = '+str(list(all_ens.data_vars))+'\n')

        if save:
            if var[0] == 'CN':
                subdir = var[2]+'_ENSEMBLE/'
            if var[0] == 'BIOMASS':
                subdir = var[2].upper()+'_ENSEMBLE/'
            if SIE:
                subdir = 'SIE_ENSEMBLE/'
            if SIV:
                subdir = 'SIV_ENSEMBLE/'
            all_ens.to_netcdf(writedir+subdir+filename)
            print(writedir+subdir+filename)

        elif not save:
            return all_ens
        
        
#################################################################################
#################################################################################

def comp_ens_so_mean(
    variables, timescale, save=False, SIE=False, SIV=False):
    
    rootdir = '/local/projects/so_predict/esm2m_froelicher/'
    writedir = '/home/bbuchovecky/storage/so_predict_derived/'
    
    reg_masks = xr.open_dataset(writedir+'regional_global_masks.nc')
    regions = list(reg_masks.data_vars)

    ocean_grid = xr.open_dataset(rootdir+'GRID/ocean.static.nc')
    geolat_t = ocean_grid.geolat_t
    geolon_t = ocean_grid.geolon_t
    
    months = 120

    nT_months = np.arange(1,months+1)
    nT_years = np.arange(1,(months/12)+1)
    nStart = np.arange(1,7)
    nEns = np.empty(0, dtype=str)

    start_yrs = ['0170_0179', '0022_0031', '0064_0073', '0106_0115', '0232_0241', '0295_0304']
    
    ## iterate through all variables
    for var in variables:
        ## update status
        print('\n######## '+var[0]+' ########\n')

        subdir = var[0]+'_ENSEMBLE/'

        ## list with datasets for each start year (includes 40 ensembles)
        yrs_list = []
        
        biomass_vars = ['sfc_ndi', 'sfc_nlg_diatoms', 'sfc_nlg_nondiatoms', 'sfc_nsm']
        
        ## iterate through the start years
        for (i,yrs) in zip(range(1,7), start_yrs):

            ## list with datasets for each ensemble run (pos/neg, 1-20)
            ens_list = []

            ## reset 'nEns' coord list
            nEns = np.empty(0, dtype=str)

            ## iterate through ensembles
            for sign in ['neg', 'pos']:
                for pert in range(1,21):
                    
                    nEns = np.append(nEns, sign+f'{pert:02}')
                    
                    if var[0] != 'BIOMASS':
                        filename = var[1]+'_ENS'+f'{i:02}'+'_'+sign+f'{pert:02}'+'_'+yrs+'.nc'
                        ds = xr.open_dataset(rootdir+subdir+filename)

                    if var[0] == 'BIOMASS':
                        filename = '_ENS'+f'{i:02}'+'_'+sign+f'{pert:02}'+'_'+yrs+'.nc'
                        ndi = xr.open_dataset(rootdir+subdir+biomass_vars[0]+filename)
                        diatoms = xr.open_dataset(rootdir+subdir+biomass_vars[1]+filename)
                        nondiatoms = xr.open_dataset(rootdir+subdir+biomass_vars[2]+filename)
                        nsm = xr.open_dataset(rootdir+subdir+biomass_vars[3]+filename)

                        ds = ndi.copy(deep=True)
                        ds = ds.rename({'sfc_ndi':'sfc_biomass'})

                        ds['sfc_biomass'] = ndi['sfc_ndi'] + diatoms['sfc_nlg_diatoms'] + nondiatoms['sfc_nlg_nondiatoms'] + nsm['sfc_nsm']
                    
                    ## reassign coord names to ensure continuity between datasets
                    coords = list(ds.coords)
                    if (coords.count('XT') == 1) and (coords.count('YT') == 1) and (coords.count('TIME') == 1):
                        ds = ds.rename({'XT':'xt_ocean', 'YT':'yt_ocean', 'TIME':'time'})

                    if (coords.count('XT_OCEAN') == 1) and (coords.count('YT_OCEAN') == 1) and (coords.count('TIME') == 1):
                        ds = ds.rename({'XT_OCEAN':'xt_ocean', 'YT_OCEAN':'yt_ocean', 'TIME':'time'})
                        
                    if (coords.count('xt') == 1) and (coords.count('yt') == 1):
                        ds = ds.rename({'xt':'xt_ocean', 'yt':'yt_ocean'})
                    
                    ## truncate at 120 months for CN_INV
                    if ds['time'].size > months:
                        time = xr.cftime_range(start='0000', periods=months, freq='MS', calendar='noleap')
                        temp = ds.copy(deep=True)
                        ds = ds.drop('time')
                        ds = ds.drop_vars(var[2])
                        ds = ds.assign_coords({'time':time})
                        ds[var[2]] = (('time','yt_ocean','xt_ocean'), temp[var[2]][:months])
                    
                    ## assign ocean_grid coords for continuity
                    ds = ds.assign_coords({'xt_ocean':ocean_grid.xt_ocean, 'yt_ocean':ocean_grid.yt_ocean})
                    
                    ## remove 'area_t' gridcells on land, keep only the ocean
                    glob_area = xr.where(np.isnan(ds[var[2]][0]), np.nan, ocean_grid['area_t'])
                    
                    ## load and format SIC data for SIV computation
                    if SIV:
                        sie_area = xr.open_dataset(rootdir+'CN_ENSEMBLE/CN_inv_ENS'+f'{i:02}'+'_'+sign+f'{pert:02}'+'_'+yrs+'.nc')
                        sie_area = sie_area.rename({'XT':'xt_ocean', 'YT':'yt_ocean', 'TIME':'time'})
                        sie_area = sie_area.assign_coords({'xt_ocean':ocean_grid.xt_ocean, 'yt_ocean':ocean_grid.yt_ocean})
        
                        if sie_area['CN_INV'].size > months:
                            temp = sie_area.copy(deep=True)
                            sie_area = sie_area.drop('time')
                            sie_area = sie_area.drop_vars('CN_INV')
                            sie_area['CN_INV'] = (('time','yt_ocean','xt_ocean'), temp['CN_INV'][:months])
                        
                        sie_area = sie_area.assign_coords({'time':ds['time']})
                        sie_area.transpose()
                        sie_area = sie_area['CN_INV'] * glob_area
                                    
                    ## compute timeseries for each region
                    reg_list = []
                    for reg in regions:
                        ## regional mask of 'area_t' 
                        area = glob_area.where(reg_masks[reg] == 1)
                        
                        ## total ocean area in the region
                        area_sum = area.sum(dim={'xt_ocean', 'yt_ocean'})
                        
                        ## regional mask of ensemble run data
                        reg_gridcell = ds[var[2]].where(reg_masks[reg] == 1)
                        
                        ## threshold SIE at >15% SIC
                        if SIE:
                            reg_gridcell = xr.where(reg_gridcell > 0.15, 1, 0)
                        
                        if timescale == 'annual':
                            if SIE:
                                reg_mean = (reg_gridcell * ocean_grid['area_t']).groupby('time.year').mean(dim='time').sum(dim={'xt_ocean', 'yt_ocean'})
                            
                            elif SIV:
                                reg_mean = (sie_area * reg_gridcell).groupby('time.year').mean(dim='time').sum(dim={'xt_ocean', 'yt_ocean'})
                            
                            else:
                                ## annual mean - compute area integral for each year
                                ## (1) find annual mean value for each gridcell
                                ## (2) compute areal integral for the annual mean gridcells
                                reg_mean = (reg_gridcell * ocean_grid['area_t']).groupby('time.year').mean(dim='time').sum(dim={'xt_ocean', 'yt_ocean'}) / area_sum
                            
                            ## reassign 'time' cftime coord to 'nT' integer coord from 1-10
                            if list(reg_mean.coords).count('time') > 0:
                                reg_mean = reg_mean.drop('time')
                            reg_mean = reg_mean.rename({'year':'nT'})
                            reg_mean = reg_mean.assign_coords({'nT':nT_years})

                        if timescale == 'monthly':
                            if SIE:
                                reg_mean = (reg_gridcell * ocean_grid['area_t']).sum(dim={'xt_ocean', 'yt_ocean'})
                            
                            elif SIV:
                                reg_mean = (sie_area * reg_gridcell).sum(dim={'xt_ocean', 'yt_ocean'})
                               
                            else:
                                ## monthly mean - compute area integral for each month
                                reg_mean = (reg_gridcell * ocean_grid['area_t']).sum(dim={'xt_ocean', 'yt_ocean'}) / area_sum
                            
                            ## reassign 'time' cftime coord to 'nT' integer coord from 1-120
                            reg_mean = reg_mean.rename({'time':'nT'})
                            reg_mean = reg_mean.assign_coords({'nT':nT_months})

                        ## rename the array to the region's name
                        reg_mean.name = reg

                        ## append a copy of the array to the regions list
                        reg_list.append(reg_mean.copy(deep=True))

                    ## merge the regions list into one dataset, separate variables for each region
                    this_ens = xr.merge(reg_list)

                    ## append a copy of the dataset to the ensembles list
                    ens_list.append(this_ens.copy(deep=True))

                    ## update status
                    if (i == 1) and (sign == 'neg') and (pert == 1):
                        print('this_ens.coords = '+str(list(this_ens.coords)))
                        print('this_ens.data_vars = '+str(list(this_ens.data_vars))+'\n')

                    print('o',end='')
                    if pert == 20:
                        print('')

                ## update status
                if sign == 'neg':
                    print('var: '+var[2]+' | nStart: '+str(i)+' | nEns: 20/40')
                if sign == 'pos':
                    print('var: '+var[2]+' | nStart: '+str(i)+' | nEns: 40/40')

            ## concatenate ensemble arrays along new 'nEns' dimension
            these_ens = xr.concat(ens_list, 'nEns')
            these_ens = these_ens.assign_coords({'nEns':nEns}) 

            ## append a copy of the dataset to the start years list
            yrs_list.append(these_ens.copy(deep=True))

            ## update status
            if i == 1:
                print('\nthese_ens.coords = '+str(list(these_ens.coords)))
                print('these_ens.data_vars = '+str(list(these_ens.data_vars))+'\n')

        ## concatenate start years arrays along new 'nStart' dimension
        all_ens = xr.concat(yrs_list, 'nStart')
        all_ens = all_ens.assign_coords({'nStart':start_yrs})

        ## copy attributes to each region data variable
        for reg in regions:
            all_ens[reg].attrs = reg_gridcell.attrs

        ## add description
        if SIE:
            filename = 'sie_ens_so_'+timescale+'_mean.nc'
        if SIV:
            filename = 'siv_ens_so_'+timescale+'_mean.nc'
        if not SIE and not SIV:
            filename = var[1].lower()+'_ens_so_'+timescale+'_mean.nc'
        all_ens.attrs['name'] = filename

        print('\nall_ens.coords = '+str(list(all_ens.coords)))
        print('all_ens.data_vars = '+str(list(all_ens.data_vars))+'\n')

        if save:
            if var[0] == 'CN':
                subdir = var[2]+'_ENSEMBLE/'
            if var[0] == 'BIOMASS':
                subdir = var[2].upper()+'_ENSEMBLE/'
            if SIE:
                subdir = 'SIE_ENSEMBLE/'
            if SIV:
                subdir = 'SIV_ENSEMBLE/'
            all_ens.to_netcdf(writedir+subdir+filename)
            print(writedir+subdir+filename)
        elif not save:
            return all_ens
        
        
#################################################################################
#################################################################################

def comp_ens_anom(
    var, timescale, reg, clim=None, save=False):
    
    writedir = '/home/bbuchovecky/storage/so_predict_derived/'
    subdir = var.upper()+'_ENSEMBLE/'
    
    if clim == None and timescale == 'monthly':
        clim = xr.open_dataset(writedir+'CTRL/'+var.upper()+'/'+var.lower()+'_ts_'+reg+'_monthly_mean.nc')
        clim = clim.groupby('time.month').mean(dim='time')
    
    ## open the ensemble timeseries
    ens_mean = xr.open_dataset(writedir+subdir+var.lower()+'_ens_'+reg+'_'+timescale+'_mean.nc')
    
    ## create list of regions (either SO regions or Global)
    regions = list(ens_mean.data_vars)
        
    ## create a copy of the ensemble timeseries to overwrite with anomaly data
    ens_anom = ens_mean.copy(deep=True)
        
    ## iterate through all start years
    for s in range(ens_anom['nStart'].size):
        ## iterate through each perturbation
        for e in range(ens_anom['nEns'].size):
            ## iterate through all regions
            for r in regions:
                ## create a NumPy array of the ensemble data
                np_mean = ens_mean[r][s,e].values

                ## if SST, adjust units of climatology back to Kelvin
                if var.lower() == 'sst':
                    np_clim = clim[r].values + 273.15
                else:
                    np_clim = clim[r].values 

                ## calculate the anomaly between the ensemble data and the
                ## control run climatology 
                ## (Jan_ens - Jan_ctrl, Feb_ens - Feb_ctrl, etc.)                    
                reg_anom = np.zeros(120)
                for m in range(120):
                    reg_anom[m] = np_mean[m] - np_clim[m%12]

                ens_anom[r][s,e] = reg_anom
                
    ## add description
    if len(regions) > 1:
        filename = var.lower()+'_ens_so_'+timescale+'_anom.nc'
    elif len(regions) == 1:
        filename = var.lower()+'_ens_global_'+timescale+'_anom.nc'
    ens_anom.attrs['name'] = filename
    
    if save:
        ens_anom.to_netcdf(writedir+subdir+filename)
        print(writedir+subdir+filename)
    elif not save:
        return ens_anom
    
