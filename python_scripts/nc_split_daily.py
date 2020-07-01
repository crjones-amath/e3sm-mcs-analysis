"""Split a series of netcdf"""
import xarray as xr
# from dask_jobqueue import SLURMCluster
# from dask.distributed import Client, LocalCluster, progress
import glob
import os
from concurrent.futures import ProcessPoolExecutor

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'


years = ['0002', '0003', '0004', '0005', '0006', '0007']
# years = ['0001', '0003']

start_dates = [None] * len(years)

# years = ['0002']
# start_dates = ['0002-02-25']

# ########################
# E3SM - MMF
# ########################
# path_out = '/global/project/projectdirs/m3312/crjones/e3sm/early_science/hourly_2d_hist/remap/daily/'
# out_name = path_out + 'earlyscience.FC5AV1C-H01A.ne120.sp1_64x1_1000m.cam.h1.'
# path_in = '/global/project/projectdirs/m3312/crjones/e3sm/early_science/hourly_2d_hist/remap/'

# for 3D processing
path_out = '/global/cscratch1/sd/crjones/conus/e3sm-mmf/daily/'
path_in = '/global/cscratch1/sd/crjones/conus/e3sm-mmf/'
out_name = path_out + 'earlyscience.FC5AV1C-H01A.ne120.sp1_64x1_1000m.3d.'


# ########################
# E3SM
# ########################
# for 2D processing
# path_out = '/global/project/projectdirs/m3312/crjones/e3sm/early_science_e3sm/hourly_2d_hist/remap/daily/'
# out_name = path_out + 'earlyscience.FC5AV1C-H01A.ne120.E3SM.cam.h1.'
# path_in = '/global/project/projectdirs/m3312/crjones/e3sm/early_science_e3sm/hourly_2d_hist/remap/'

# for 3D processing
# path_out = '/global/project/projectdirs/m3312/crjones/e3sm/early_science_e3sm/3hourly_3d_hist/conus/daily/'
# path_in = '/global/project/projectdirs/m3312/crjones/e3sm/early_science_e3sm/3hourly_3d_hist/conus/'
# out_name = path_out + 'earlyscience.FC5AV1C-H01A.ne120.E3SM.3d.'

files_in = sorted(glob.glob(path_in + '*.nc'))
dates_in = [f.split('.')[-3][:10] if 'conus.nc' in f.lower()  # selects yyyy-mm-dd from $case.yyyy-mm-dd-sssss.conus.nc
            else f.split('.')[-2][:10] for f in files_in]     # selects yyyy-mm-dd from $case.yyyy-mm-dd-sssss.nc
years_in = {d[0:4] for d in dates_in}
files_by_year = {y: [f for f in files_in if y in f] for y in years_in}
dates_by_year = {y: [d for d in dates_in if y in d] for y in years_in}

assert(path_out != path_in)  # make sure you don't accidentally clobber anything

def file_list_for_given_year(year, start_date=None):
    # may need to grab one of the previous year as well, since 000{n}-12-31 will run over into 000{n+1}
    file_list = files_by_year[year]
    start_idx = dates_by_year[year].index(start_date) if start_date in dates_by_year[year] else 0
    if start_idx > 0:
        return file_list[start_idx:]
    else:
        # need to include last element of previous year
        prior_year = str(int(year) - 1).zfill(4)
        prepend = [files_by_year[prior_year][-1]] if prior_year in years_in else []
        return prepend + file_list

def main():
    """ Original (now obsolete version) that frequently crashed """
    # loop over years, split into daily files
    for yr, start_date in zip(years, start_dates):
        print('Processing year ', yr)
        files_to_check = file_list_for_given_year(yr, start_date=start_date)
        ds = xr.open_mfdataset(files_to_check, parallel=True).sel(time=yr).chunk(chunks={'time': 12})
        days, dsets = zip(*ds.groupby('time.dayofyear'))
        out_files = [out_name + dsets[d].time[0].item().strftime('%Y-%m-%d').replace(' ', '0') + '.nc' for d in range(len(days))]
        previously_processed = glob.glob(path_out + '*.nc')
        out_to_do = [out_files[d] for d in range(len(days)) if out_files[d] not in previously_processed]
        dsets_to_do = [dsets[d] for d in range(len(days)) if out_files[d] not in previously_processed]
        if out_to_do:
            print('First file: ' + out_to_do[0])
            print('Last file: ' + out_to_do[-1])
            xr.save_mfdataset(dsets_to_do, out_to_do)

def split_file_to_daily(fname):
    ds = xr.open_dataset(fname)
    days, dsets = zip(*ds.groupby('time.dayofyear'))
    out_files = [out_name + dsets[d].time[0].item().strftime('%Y-%m-%d').replace(' ', '0') + '.nc' for d in range(len(days))]
    previously_processed = glob.glob(path_out + '*.nc')
    out_to_do = [out_files[d] for d in range(len(days)) if out_files[d] not in previously_processed]
    dsets_to_do = [dsets[d] for d in range(len(days)) if out_files[d] not in previously_processed]
    if out_to_do:
        print('First file: ' + out_to_do[0])
        print('Last file: ' + out_to_do[-1])
        xr.save_mfdataset(dsets_to_do, out_to_do)

def main_alt(do_parallel=True, max_workers=8):
    """ Loads all files matching {path_in}/*{year}*.nc for year in specified list years
    and splits into daily files  {out_name}.yyyy-mm-dd.nc
    """
    for yr, start_date in zip(years, start_dates):
        print('Processing year ', yr)
        files_to_check = file_list_for_given_year(yr, start_date=start_date)
        if do_parallel:
            with ProcessPoolExecutor(max_workers=max_workers) as Executor:
                Executor.map(split_file_to_daily, files_to_check)
        else:  # only for testing
            for fname in files_to_check[:1]:
                split_file_to_daily(fname)

if __name__ == "__main__":
    # this version is much faster than using `main()` when it is known that each file 
    # contains full days (i.e., no day's output is split across multiple files)
    main_alt(do_parallel=True)

    