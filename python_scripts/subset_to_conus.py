"""Subset daily PRECT and FLNT output to CONUS"""
import xarray as xr
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import glob
from datetime import timedelta


def subset_and_write_to_file(fname, lons=[220, 300], lats=[20, 50],
                             output_vars=['PRECT', 'FLNT'],
                             transpose_dims=True, shift_time=True):
    print('Processing file ', fname)
    fout = fname.replace('.nc', '.CONUS.nc')
    fout = fout.replace('/daily/', '/daily/CONUS/')  # quick move to CONUS out-directory
    # original version:
    # ds = xr.open_dataset(fname)
    # ds[['PRECT', 'FLNT']].sel(lat=slice(*lats), lon=slice(*lons)).to_netcdf(fout)
    if output_vars is None:
        ds = xr.open_dataset(fname).sel(lat=slice(*lats), lon=slice(*lons)).load()
    else:
        ds = xr.open_dataset(fname)[output_vars].sel(lat=slice(*lats), lon=slice(*lons)).load()
    if shift_time:
        ds['time'] = ds['time'] + timedelta(days=365 * 2000)
        fout = fout.replace('h1.000', 'h1.200')
    if transpose_dims:
        ds.transpose(*('time', 'lat', 'lon')).to_netcdf(fout)
    else:
        ds.to_netcdf(fout)

def main(do_parallel=True):
    # files_sp = glob.glob('/global/project/projectdirs/m3312/crjones/e3sm/early_science/hourly_2d_hist/remap/daily/*.nc')
    files_sp = sorted(glob.glob('/global/project/projectdirs/m3312/crjones/e3sm/early_science_e3sm/hourly_2d_hist/remap/daily/*.nc'))
    # files_sp = glob.glob('/global/project/projectdirs/m3312/crjones/e3sm/early_science/hourly_2d_hist/remap/daily/*.000[2-7]*.nc')
    # files_sp = glob.glob('/global/project/projectdirs/m3312/crjones/e3sm/early_science/hourly_2d_hist/remap/daily/*.0002-05*.nc')
    # files_e3sm = glob.glob('/global/project/projectdirs/m3312/crjones/e3sm/early_science_e3sm/hourly_2d_hist/remap/daily/*.nc')
    # files_e3sm = glob.glob('/global/project/projectdirs/m3312/crjones/e3sm/early_science_e3sm/hourly_2d_hist/remap/daily/*.000[2-3]*.nc')
    # files_e3sm = glob.glob('/global/project/projectdirs/m3312/crjones/e3sm/early_science_e3sm/hourly_2d_hist/remap/daily/*.0002-04-0[1-9]*nc')

    if not do_parallel:
        subset_and_write_to_file(files_sp[0])

    # dates_to_process given as 'yyyy-mm-dd'
    if do_parallel:
        with ProcessPoolExecutor(max_workers=8) as Executor:
            Executor.map(subset_and_write_to_file, files_sp)

    # with ProcessPoolExecutor(max_workers=8) as Executor:
    #     Executor.map(subset_and_write_to_file, files_sp)

if __name__ == "__main__":
    main()