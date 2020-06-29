#!/usr/bin/env python
"""Remap global e3sm output and subset to CONUS"""
import xarray as xr
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import glob
from datetime import timedelta
import subprocess
import os.path

# step 1: remap
# step 2: load the remapped file
# step 3: subset to CONUS
# step 4: save daily files
# step 5: delete original global file

debug = True
model = 'e3sm'
if 'mmf' in model.lower():
    # E3SM-MMF
    case3_topdir = '/global/project/projectdirs/m3312/crjones/e3sm/early_science/3hourly_3d_hist/'
    case2_topdir = '/global/project/projectdirs/m3312/crjones/e3sm/early_science/hourly_2d_hist/'
    case3_prefix = 'earlyscience.FC5AV1C-H01A.ne120.sp1_64x1_1000m.20190329.cam.h2.'
    case2_prefix = 'earlyscience.FC5AV1C-H01A.ne120.sp1_64x1_1000m.20190329.cam.h1.'
    out_dir = '/global/cscratch1/sd/crjones/conus/tmp/'
    out_prefix = 'earlyscience.FC5AV1C-H01A.ne120.sp1_64x1_1000m.3d.'
    out_dir_remap = '/global/cscratch1/sd/crjones/conus/e3sm-mmf/'
else:
    # E3SM
    # case3_topdir = '/global/cfs/cdirs/m3312/crjones/e3sm/early_science_e3sm/3hourly_3d_hist/'
    case3_topdir = '/global/cscratch1/sd/crjones/e3sm/'
    case2_topdir = '/global/cfs/cdirs/m3312/crjones/e3sm/early_science_e3sm/hourly_2d_hist/'
    case3_prefix = 'earlyscience.FC5AV1C-H01A.ne120.E3SM.20190329.cam.h2.'
    case2_prefix = 'earlyscience.FC5AV1C-H01A.ne120.E3SM.20190329.cam.h1.'
    out_dir = '/global/cfs/cdirs/m3312/crjones/e3sm/early_science_e3sm/3hourly_3d_hist/tmp/'
    out_prefix = 'earlyscience.FC5AV1C-H01A.ne120.E3SM.3d.'
    out_dir_remap = '/global/cfs/cdirs/m3312/crjones/e3sm/early_science_e3sm/3hourly_3d_hist/conus/'

# for remapping
# mapfile = '/global/homes/z/zender/data/maps/map_ne120np4_to_cmip6_180x360_aave.20181001.nc'
mapfile = '/global/homes/z/zender/data/maps/map_ne120np4_to_cmip6_720x1440_aave.20181001.nc'

# step 1: load files, subset variables, save to temp file
def subset_and_stuff(date, vars_to_subset=['U', 'V', 'Q', 'T', 'hyam', 'hybm'],
                     overwrite=False, remove_remapped_file=True):
    fout = out_dir + out_prefix + date + '.nc'
    remapped_file = out_dir_remap + fout.split('/')[-1]
    processed_file = remapped_file.replace('.nc', '.CONUS.nc')
    
    if os.path.isfile(processed_file):
        if overwrite:
            print('removing ' + processed_file)
            remove_file(processed_file)
        else:
            print(processed_file + ' exists -- skipping')
            return

    do_write = True
    do_write_remap = True

    if debug:
        print(fout)
        print(remapped_file)
    # check if remapped file exists -- if so, we can skip:
    if os.path.isfile(remapped_file):
        print(remapped_file + ' already exists ...')
        if overwrite:
            remove_file(remapped_file)
        else:
            do_write_remap = False
            do_write = False
    # check if original subsetted file exists ...
    if os.path.isfile(fout):
        print(fout + ' already exists ...')
        if overwrite:
            remove_file(fout)
        else:
            # assuming we're cool, so don't try to write again ...
            do_write = False
    if do_write_remap:
        if do_write:
            ds2 = xr.open_dataset(case3_topdir + case3_prefix + date + '.nc')
            ds1 = xr.open_dataset(case2_topdir + case2_prefix + date + '.nc')
            ds = ds2[vars_to_subset]
            ds['PS'] = ds1['PS'].sel(time=ds.time)
            if debug:
                print(ds)
            write_dataset(ds, fout)
        # do the remapping here
        result = remap_file(fout, outdir=out_dir_remap)

        # once remapping is successfully completed, can delete fout
        if result == 0:
            result = remove_file(fout)
    conus_outname = subset_and_write_to_file(remapped_file)
    if os.path.isfile(conus_outname) and remove_remapped_file:
        # can remove intermediate file
        remove_file(remapped_file)

def remap_file(fname, skip_processed=True, outdir=out_dir):
    print('processing file ' + fname)
    result = subprocess.run(['ncremap', '-m', mapfile, '-a', 'conserve', '-O', outdir, fname])
    if result.returncode != 0:
        print('Remap failed for fname: ' + fname)
        print('Return code: ' + str(result.returncode))
    return result.returncode

def remove_file(fname):
    print('File {} no longer needed -- delete!'.format(fname))
    result = subprocess.run(['rm', fname])
    if result.returncode != 0:
        print('rm failed for ' + fname)
        print('Return code: ' + str(result.returncode))
    return result.returncode

def write_dataset(ds, out_name, encoding_dict={'dtype': 'float32', '_FillValue': -9999.0}):
    encoding = {v: encoding_dict.copy() for v in ds.data_vars}
    for v in encoding.keys():
        if ds[v].dtype:
            encoding[v]['dtype'] = ds[v].dtype
    ds.to_netcdf(out_name, encoding=encoding)
    return out_name


def subset_and_write_to_file(fname, lons=[220, 300], lats=[20, 50],
                             output_vars=None, split_files=False,
                             transpose_dims=False, shift_time=False):
    print('Processing file ', fname)
    fout = fname.replace('.nc', '.CONUS.nc')
    if output_vars is None:
        ds = xr.open_dataset(fname).sel(lat=slice(*lats), lon=slice(*lons))
    else:
        ds = xr.open_dataset(fname)[output_vars].sel(lat=slice(*lats), lon=slice(*lons))
    if shift_time:
        ds['time'] = ds['time'] + timedelta(days=365 * 2000)
        fout = fout.replace('h1.000', 'h1.200')
    if transpose_dims:
        ds = ds.transpose(*('time', 'lat', 'lon'))
    return write_dataset(ds, fout)
        
def split_file_to_daily(ds, out_name_prefix, out_suffix='.CONUS.nc'):
    days, dsets = zip(*ds.groupby('time.dayofyear'))
    out_files = [out_name_prefix + dsets[d].time[0].item().strftime('%Y-%m-%d').replace(' ', '0') + out_suffix for d in range(len(days))]
    xr.save_mfdataset(dsets, out_files)
        
def main(do_parallel=True):
    the_dates = [f.split('.')[-2] for f in sorted(glob.glob(case3_topdir + "*.nc"))]  # 'yyyy-mm-dd-sssss'
    print(the_dates)
    if not do_parallel:
        for date in the_dates[0:2]:
            subset_and_stuff(date)
    # dates_to_process given as 'yyyy-mm-dd'
    if do_parallel:
        with ProcessPoolExecutor(max_workers=2) as Executor:
            Executor.map(subset_and_stuff, the_dates)

if __name__ == "__main__":
    main()