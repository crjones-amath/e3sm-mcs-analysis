"""Generate frames for animation of TMQ"""
import matplotlib
matplotlib.use('Agg')
import xarray as xr
import cartopy.crs as ccrs
from concurrent.futures import ProcessPoolExecutor
from e3sm_utils import cmclimate
import matplotlib.pyplot as plt
import pandas as pd
import glob

# dictionary of plot options by pvar
plot_opts = {'PRECT': {'vmin': 0, 'vmax': 25, 'figsize': (9, 3),
                       'cmap': cmclimate.cm.WhiteBlueGreenYellowRed,
                       'cbar_kwargs': {'orientation': 'horizontal', 'pad': 0.05, 'fraction': 0.05, 'aspect': 68, 'extend': 'max'}
                      },
             'TMQ': {'vmin': 0, 'vmax': 80,
                     'cbar_kwargs': {'orientation': 'horizontal', 'pad': 0.05, 'fraction': 0.05, 'aspect': 68, 'extend': 'neither'},
                    },
            }

# dictionary of dataArray transformation opts by pvar
da_opts = {'PRECT': {'rescale': 86400000, 'units': 'mm/day'}}

def load_dataset_from_date(date):
    topdir_e3sm = '/global/project/projectdirs/m3312/crjones/e3sm/early_science_e3sm/hourly_2d_hist/remap/daily/'    
    e3sm_case = 'earlyscience.FC5AV1C-H01A.ne120.E3SM.cam.h1'
    topdir_sp = '/global/project/projectdirs/m3312/crjones/e3sm/early_science/hourly_2d_hist/remap/daily/'
    sp_case = 'earlyscience.FC5AV1C-H01A.ne120.sp1_64x1_1000m.cam.h1'
    
    file_e3sm = topdir_e3sm + '.'.join([e3sm_case, date, 'nc'])
    file_sp = topdir_sp + '.'.join([sp_case, date, 'nc'])

    ds_e3sm = xr.open_dataset(file_e3sm)
    ds_sp = xr.open_dataset(file_sp)
    
    return xr.concat([ds_sp, ds_e3sm], dim=pd.Index(('MMF', 'E3SM'), name='model'))

def plot_frame(da, i, model_names=['MMF', 'E3SM'], **plot_kwargs):
    p = da.isel(time=i).plot(col='model', subplot_kws={'projection': ccrs.PlateCarree()},
                             transform=ccrs.PlateCarree(),
                             **plot_kwargs)
    p.fig.suptitle(str(da.time[i].item()))
    for j, ax in enumerate(p.axes.flat):
        ax.coastlines()
        ax.set_title(model_names[j])
        
def extract_da(plotvar, ds):
    """Extract da = ds[plotvar] and apply optional transformations"""
    da = ds[plotvar]
    opts = da_opts[plotvar] if plotvar in da_opts else dict()
    if 'rescale' in opts:
        da = da * opts['rescale']
    if 'units' in opts:
        da.attrs['units'] = opts['units']
    return da

def print_frames_from_date(date, plotvar='PRECT', out_prefix='es', topdir='/global/cscratch1/sd/crjones/figs/PRECT/'):
    """Plot hourly snapshots of plotvar for given date and save to file"""
    print('loading date ' + date)
    ds = load_dataset_from_date(date)
    da = extract_da(plotvar, ds)
    for i in range(len(ds.time)):
        out_name = '_'.join([out_prefix, plotvar, date, str(i).zfill(2)])
        plot_frame(da, i, **plot_opts[plotvar])
        plt.savefig(topdir + out_name + '.png', dpi=600, bbox_inches='tight')
        plt.close()

def main():
    # dates_to_process given as 'yyyy-mm-dd'
    files_e3sm = glob.glob('/global/project/projectdirs/m3312/crjones/e3sm/early_science_e3sm/hourly_2d_hist/remap/daily/*.nc')
    dates_to_process = sorted([f.split(sep='.')[-2] for f in files_e3sm])
    print(dates_to_process)
    
    # serial version:
    # for date in dates_to_process:
    #     print_frames_from_date(date)
    
    # loop over dates in parallel
    with ProcessPoolExecutor(max_workers=8) as Executor:
        Executor.map(print_frames_from_date, dates_to_process)


if __name__ == "__main__":
    """Note: can convert to animation using ffmpeg. E.g., 
    ffmpeg -pattern_type glob -i "*.png" -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p out.mp4
    """
    main()