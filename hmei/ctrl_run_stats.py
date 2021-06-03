# compute the monthly climatology from the 300-year control runs
def compute_ctrl_climatology(path):
    ds = xr.open_mfdataset(path)
    
    climatology = ds.groupby('time.month').mean(dim='time')

    seq = []
    for i in range(0,300):
        seq.append(climatology)

    climatology_full = xr.concat(seq, 'month')
    climatology_full = climatology_full.assign_coords(month=ds.time.values)
    climatology_full = climatology_full.rename({'month':'time'})

    return (ds - climatology_full)