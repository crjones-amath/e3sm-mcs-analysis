# S.O.M. interpolation routine
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import numba as nb
from numba import njit
from glob import glob
from concurrent.futures import ProcessPoolExecutor
import subprocess
import os.path

debug = True

model = 'e3sm'
if 'mmf' in model.lower():
    # E3SM-MMF
    case_prefix = 'earlyscience.FC5AV1C-H01A.ne120.sp1_64x1_1000m.3d.'
    out_dir_remap = '/global/cscratch1/sd/crjones/conus/e3sm-mmf/'
    out_prefix = 'earlyscience.FC5AV1C-H01A.ne120.sp1_64x1_1000m.850hpa.'
else:
    # E3SM
    case_prefix = 'earlyscience.FC5AV1C-H01A.ne120.E3SM.3d.'
    out_dir_remap = '/global/cscratch1/sd/crjones/conus/e3sm/'
    out_prefix = 'earlyscience.FC5AV1C-H01A.ne120.E3SM.850hpa.'

    # E3SM
    case_prefix = 'earlyscience.FC5AV1C-H01A.ne120.E3SM.3d.'
    out_dir_remap = '/global/project/projectdirs/m3312/crjones/e3sm/early_science_e3sm/3hourly_3d_hist/conus/daily/'
    out_prefix = 'earlyscience.FC5AV1C-H01A.ne120.E3SM.850hpa.'



out_dir = out_dir_remap + 'tmp/'

# For interpolation
@njit
def nplev_linear_weights(p, levs=np.array([250, 500, 925])):
    inds = np.empty((len(levs), 2), dtype=np.int64)
    weights = np.empty((len(levs), 2), dtype=np.float64)
    for n, ll in enumerate(levs):
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
    lat = ds.lat
    lon = ds.lon
    pres = xr.DataArray(levs, coords=[levs], dims=['pressure'], attrs={'units': 'hPa', 'long_name': 'pressure'}, name='pressure')
    out_dict = {}
    for v in variables:
        da = ds[v].transpose(*p.dims)
        interp = ninterp_col_to_pres_level(p.values, levs, da.values)
        out_dict[v] = xr.DataArray(interp, coords=[time, lat, lon, pres],
                                   dims=['time', 'lat', 'lon', 'pressure'],
                                   name=ds[v].name, attrs=ds[v].attrs)
    return xr.Dataset(out_dict, attrs=ds.attrs)


def write_dataset(ds, out_name, encoding_dict={'dtype': 'float32', '_FillValue': -9999.0}):
    encoding = {v: encoding_dict for v in ds.data_vars}
    # transpose for ncremap purposes
    ds.to_netcdf(out_name, encoding=encoding)
    
    
def process_datasets(ds, variables=['U', 'V', 'Q'], levels=np.array([850]),
                     out_name='test.nc'):
    ps = ds['PS']
    hyam = ds['hyam'].load()
    hybm = ds['hybm'].load()
    p0 = 1000
    p = ps * hybm / 100 + p0 * hyam
    p = p.transpose('time', 'lat', 'lon', 'lev')
    
    ds0 = interp_and_prep_dataset(p, levels, ds, variables=variables)
    ds0 = ds0.transpose('time', 'pressure', 'lat', 'lon')
    write_dataset(ds0, out_name)

# note: hacking this to just do ['Z3' at 500 mb]
def process_from_file_template(the_date,
                               topdir=out_dir_remap,
                               case_prefix=case_prefix,
                               out_prefix=out_prefix,
                               out_dir=out_dir,
                               fsuffix='.nc',
                               overwrite=False):
    fname = topdir + case_prefix + the_date + fsuffix
    out_name = out_dir + out_prefix + the_date + fsuffix
    if os.path.exists(out_name) and not overwrite:
        print('{} exists (skipping)'.format(out_name))
        return
    if debug:
        print('fname = ' + fname)
        print('out_name = ' + out_name)
    ds = xr.open_dataset(fname)
    if debug:
        print('ds loadeded - begin processing')
    process_datasets(ds, out_name=out_name)
    if debug:
        print('{} complete'.format(the_date))


def main(do_parallel=True):
    # files_to_process = sorted(glob(case3_topdir + "*.000[67]-[01][34567890]-*.nc"))
    files_to_process = sorted(glob(out_dir_remap + "*.nc"))
    dates_to_process = [f.split('.')[-2] for f in files_to_process]
    print(files_to_process)
    print(dates_to_process)

    if not do_parallel:
        process_from_file_template(dates_to_process[0])  # test
    else:
        # remap the files
        with ProcessPoolExecutor(max_workers=6) as Executor:
            Executor.map(process_from_file_template, dates_to_process)

    
if __name__ == "__main__": 
    main(do_parallel=True)
    # process_from_file_template('0002-05-15-00000')
