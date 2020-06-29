# S.O.M. interpolation routine
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import numba as nb
from numba import njit

model = 'E3SM'  # e3sm or mmf
# function definitions
if 'mmf' in model.lower():
    case3_topdir = '/global/project/projectdirs/m3312/crjones/e3sm/early_science/3hourly_3d_hist/'
    case2_topdir = '/global/project/projectdirs/m3312/crjones/e3sm/early_science/hourly_2d_hist/'
    case3_prefix = 'earlyscience.FC5AV1C-H01A.ne120.sp1_64x1_1000m.20190329.cam.h2.'
    case2_prefix = 'earlyscience.FC5AV1C-H01A.ne120.sp1_64x1_1000m.20190329.cam.h1.'
else:
    case3_topdir = '/global/project/projectdirs/m3312/crjones/e3sm/early_science/3hourly_3d_hist/'
    case2_topdir = '/global/project/projectdirs/m3312/crjones/e3sm/early_science/hourly_2d_hist/'
    case3_prefix = 'earlyscience.FC5AV1C-H01A.ne120.sp1_64x1_1000m.20190329.cam.h2.'
    case2_prefix = 'earlyscience.FC5AV1C-H01A.ne120.sp1_64x1_1000m.20190329.cam.h1.'
    
out_dir = '/global/cscratch1/sd/crjones/'

# For a single array
@njit
def nplev_linear_weights(p, levs=[250, 500, 925]):
    inds = np.empty((len(levs), 2), dtype=np.int64)
    weights = np.empty((len(levs), 2), dtype=np.float64)
    lev_array = np.array(levs).astype(np.float64)
    for n, ll in enumerate(lev_array):
        iup = np.searchsorted(p, ll, side='right')
        if iup == 0 or iup >= len(p):
            inds[n] = (0, 0)
            weights[n] = (np.nan, np.nan)
            continue
        idn = iup - 1
        dp = (ll - p[idn]) / (p[iup] - p[idn])
        inds[n] = (idn, iup)
        weights[n] = (1 - dp, dp)
    return inds, weights

@njit
def napply_weights_to_column(v, inds, weights):
    return np.array([np.sum(v[inds[k]] * weights[k]) for k in range(len(inds))])

@njit
def ninterp_col_to_pres_level(p, levs, da):
    shape = (*p.shape[:-1], len(levs))
    # outx = np.full(shape, fill_value=np.nan)
    outx = np.empty(shape)
    for ind in np.ndindex(p.shape[:-1]):
        inds, weights = nplev_linear_weights(p[ind], levs)
        outx[ind] = napply_weights_to_column(da[ind], inds, weights)
    return outx


def interp_and_prep_dataset(p, levs, ds, variables=['U', 'V', 'Q']):
    time = ds.time
    ncol = ds.ncol
    pres = xr.DataArray(levs, coords=[levs], dims=['pressure'], attrs={'units': 'hPa', 'long_name': 'pressure'}, name='pressure')
    out_dict = {}
    for v in variables:
        da = ds[v].transpose(*p.dims)
        interp = ninterp_col_to_pres_level(p.values, levs, da.values)
        out_dict[v] = xr.DataArray(interp, coords=[time, ncol, pres],
                                   dims=['time', 'ncol', 'pressure'],
                                   name=ds[v].name, attrs=ds[v].attrs)
    return xr.Dataset(out_dict, attrs=ds.attrs)


def write_dataset(ds, out_name, encoding_dict={'dtype': 'float32', '_FillValue': -9999.0}):
    encoding = {v: encoding_dict for v in ds.data_vars}
    # transpose for ncremap purposes
    ds.to_netcdf(out_name, encoding=encoding)
    
    
def process_datasets(ds3, ds2, variables=['U', 'V', 'Q'], levels=np.array([200, 500, 925]),
                     out_name='test.nc'):
    ps = ds2['PS'].sel(time=ds3.time)
    hyam = ds3['hyam'].load()
    hybm = ds3['hybm'].load()
    p0 = 1000
    p = ps * hybm / 100 + p0 * hyam
    p = p.transpose('time', 'ncol', 'lev')
    
    ds = interp_and_prep_dataset(p, levels, ds3, variables=variables)
    ds = ds.transpose('time', 'pressure', 'ncol')
    write_dataset(ds, out_name)


def process_from_file_template(the_date='0002-05-15-00000',
                               case3_topdir=case3_topdir,
                               case2_topdir=case2_topdir,
                               case3_prefix=case3_prefix,
                               case2_prefix=case2_prefix):
    f3 = case3_topdir + case3_prefix + the_date + '.nc'
    f2 = case2_topdir + case2_prefix + the_date + '.nc'
    out_name = out_dir + 'temp.' + the_date + '.nc'
    ds3 = xr.open_dataset(f3)
    ds2 = xr.open_dataset(f2)
    process_datasets(ds3, ds2, out_name=out_name)

    
if __name__ == "__main__": 
    process_from_file_template()

