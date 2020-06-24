"""Generate plots for each individual identified robust MCS track"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import xarray as xr
import os
from glob import glob
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from e3sm_utils import cmclimate
import pandas as pd
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from concurrent.futures import ProcessPoolExecutor

cmap_prect = cmclimate.cm.WhiteBlueGreenYellowRed

"""
# /global/cscratch1/sd/crjones/ECP/e3sm-mmf/statstb
# /global/cscratch1/sd/crjones/ECP/e3sm-mmf/mcstracking

stats_file = '/Users/jone003/tmp/ECP/e3sm-mmf/statstb/robust_mcs_tracks_20020301_20021031.nc'
pixel_files = sorted(glob('/Users/jone003/tmp/ECP/e3sm-mmf/mcstracking/20020301_20021031/mcstrack_*.nc'))
print(f'Found {len(pixel_files)} mcstrack files')

stats_ds = xr.open_dataset(stats_file)
"""

def link_track_to_pixel_files(this_track, pixel_files, pattern='%Y%m%d_%H%M'):
    # convert basetimes to string following pattern appearing in pixel_files names
    basetimes = [bt.strftime(pattern) for bt in pd.to_datetime(this_track.base_time.values) if str(bt) != 'NaT']
    file_list = []
    for bt in basetimes:
        # tricky way to append a file matching the pattern if it's there, None if not
        file_list.append(([p for p in pixel_files if bt in p] + [None])[0])
    return file_list

def process_track(this_track, pixel_files):
    """Drop's times where base_time is NaT from this_track and 
    returns truncated this_track and pixel (map) dataSet for this track"""
    this_track = this_track.where(~np.isnat(this_track.base_time), drop=True)
    pixel_file_list = link_track_to_pixel_files(this_track, pixel_files)
    pix_ds = xr.open_mfdataset(pixel_file_list)
    return this_track, pix_ds

# prepare figure canvas:
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')

def plot_area(this_track, ax=None, do_title=True, ax_title_text=None, **plot_kwargs):
    """Plot time-series of ccs_area and core_area for given track"""
    if ax is None:
        fig, ax = plt.subplots()
    track_id = this_track.tracks.item() + 1

    # prepare plot data
    ccs = this_track.ccs_area.copy()
    ccs.values = ccs.values * 1e-3
    ccs.attrs['units'] = '$10^{3}$ km$^2$'
    core_area = this_track.core_area.copy()
    core_area.values = core_area.values * 1e-3
    core_area.attrs['units'] = '$10^{3}$ km$^2$'
    da2 = this_track.majoraxislength
    
    # prepare twin plot axis
    ax2 = ax.twinx()
    color1 = 'tab:blue'
    color2 = 'tab:red'
    ccs.plot(ax=ax, **plot_kwargs, color=color1, label='ccs area')
    core_area.plot(ax=ax, **plot_kwargs, color=color1, linestyle='--', label='core area')
    da2.plot(ax=ax2, **plot_kwargs, color=color2, label='Semimajor axis')
    ax.set_ylabel(ax.get_ylabel(), color=color1)
    ax.tick_params(axis='y', labelcolor=color1)
    if do_title:
        trange = [t.strftime('%Y-%m-%d %H:%M') for t in pd.to_datetime(this_track.base_time.values[[1, -1]])]
        if ax_title_text is None:
            ax_title_text = 'MCS Track {}\n {} to {}'.format(track_id, *trange)
        ax.set_title(ax_title_text)
    ax.legend()
    ax2.set_title('')
    ax2.set_ylabel(ax2.get_ylabel(), color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    return ax

def plot_pf_area(this_track, ax=None, do_title=True, ax_title_text=None, **plot_kwargs):
    """Plot accumulated precipitation for this_track (sum of precipitation features)"""
    if ax is None:
        fig, ax = plt.subplots()
    track_id = this_track.tracks.item() + 1
    # rescale area and sum across prfs (note: keep nans not strictly necessary here, but done anyway)
    da = 1e-3 * this_track.pf_area.sum(dim='nmaxpf', min_count=1, keep_attrs=True)
    da.attrs['units'] = '$10^{3}$ km$^2$'
    da.plot(ax=ax, **plot_kwargs, label='all PFs')
    da2 = da.copy()
    da2.values = da2.values * this_track.pf_mcsstatus
    da2.plot(ax=ax, color='r', **plot_kwargs, label='PF MCS Status = 1')
    if do_title:
        trange = [t.strftime('%Y-%m-%d %H:%M') for t in pd.to_datetime(this_track.base_time.values[[1, -1]])]
        if ax_title_text is None:
            ax_title_text = 'MCS Track {}\n {} to {}'.format(track_id, *trange)
        ax.set_title(ax_title_text)
    ax.legend()
    return ax

def plot_accumulated_precip_from_mcs(this_track, pix_ds, ax=None, use_pf_mcsstatus=False, figsize=(8, 8),
                                     do_title=True, ax_title_text=None, plot_trajectory=True,
                                     **plot_kwargs):
    """Plot map of accumulated precipitation for this_track
    
    Return ax, accumulated precipitation dataArray, and pixel-level dataSet associated with this_track.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
    track_id = this_track.tracks.item() + 1
    # pix_ds = xr.open_mfdataset(process_track(this_track))
    # min_count=1 keeps nans from being replaced by zero
    accum_precip = pix_ds['precipitation'].where(pix_ds['cloudtracknumber'] == track_id).sum(dim='time', min_count=1)
    accum_precip.attrs['units'] = 'mm'
    accum_precip.attrs['long_name'] = 'accumulated MCS precipitation'
    
    # center figure on this MCS by cropping out rows/columns that are all nan
    nanprecip = np.isnan(accum_precip.values)
    lons = ~np.all(nanprecip, axis=0)
    lats = ~np.all(nanprecip, axis=1)
    accum_precip = accum_precip.sel(lat=lats, lon=lons)
    accum_precip.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap_prect, 
                      robust=True,
                      **plot_kwargs,
                     )
    
    if plot_trajectory:
        ax.plot(this_track.meanlon.values, this_track.meanlat.values,
                transform=ccrs.PlateCarree(), color='gray')
        # add arrow
        x = this_track.meanlon.values
        y = this_track.meanlat.values
        if not any(np.isnan([x[0], y[0], x[-1], y[-1]])):
            dx = x[-1] - x[0]
            dy = y[-1] - y[0]
            ax.arrow(x[0], y[0], dx, dy, transform=ccrs.PlateCarree(),
                     color='black', linestyle='--',
                     head_width=1, head_length=1, overhang=0.8)
    ax.add_feature(states_provinces, edgecolor='black', alpha=1)
    ax.coastlines()
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=1)
    ax.add_feature(cfeature.LAND)
    # ax.add_feature(cfeature.LAKES)

    # format gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gl.xlabels_top = False
    gl.xlocator = mticker.MaxNLocator(5)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    # add title
    if do_title:
        trange = [t.strftime('%Y-%m-%d %H:%M') for t in pd.to_datetime(this_track.base_time.values[[1, -1]])]
        if ax_title_text is None:
            ax_title_text = 'MCS Track {}\n {} to {}'.format(track_id, *trange)
        ax.set_title(ax_title_text)
    return ax, accum_precip

def plot_fig1(this_track, pix_ds, figsize=(12, 6)):
    """Analog of Figure 1 (c), (d), and (f) from Feng et al (2019) manuscript
    
    Note: work in progress; missing analogues for (a), (b), and (e)"""
    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot2grid((2, 2), (0, 1))  # core/ccs area
    ax2 = plt.subplot2grid((2, 2), (1, 1))  # pf_area
    ax3 = plt.subplot2grid((2, 2), (0, 0), rowspan=2, projection=ccrs.PlateCarree())  # map
    plt.subplots_adjust(wspace=0.4, hspace=0.3)

    plot_area(this_track, ax=ax1, ax_title_text='');
    plot_pf_area(this_track, ax=ax2, ax_title_text='');
    plot_accumulated_precip_from_mcs(this_track, pix_ds, ax=ax3, levels=10,
                                     cbar_kwargs={'orientation': 'horizontal', 'pad': 0.06, 'aspect': 50});
    return fig, ax1, ax2, ax3


def process_from_robust_stats(robust_mcs_filename, savefig=True, out_prefix='mcs_track',
                              topdir='/global/cscratch1/sd/crjones/ECP/e3sm', nmax=None):
    date_range = robust_mcs_filename[-20:-3]  # 200n0301_200n1031
    pixel_files = sorted(glob('{}/mcstracking/{}/mcstrack_*.nc'.format(topdir, date_range)))
    stats_ds = xr.open_dataset(robust_mcs_filename)
    if savefig:
        outdir = '{}/figs/{}/'.format(topdir, date_range)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    ntracks = len(stats_ds.tracks.values)
    nfill = len(str(ntracks))
    if nmax is not None:
        tracks_to_process = stats_ds.tracks.values[:nmax]
    else:
        tracks_to_process = stats_ds.tracks.values
    for track in tracks_to_process:
        this_track, pix_ds = process_track(stats_ds.sel(tracks=track), pixel_files)
        plot_fig1(this_track, pix_ds);
        out_name = '_'.join([out_prefix, str(track + 1).zfill(nfill)])
        plt.savefig(outdir + out_name + '.png', dpi=600, bbox_inches='tight')
        plt.close()


def main(do_parallel=False):
    # dates_to_process given as 'yyyy-mm-dd'
    robust_mcs_files = sorted(glob('/global/cscratch1/sd/crjones/ECP/e3sm/statstb/robust_mcs_tracks*.nc'))
    print(robust_mcs_files)
    
    # serial version:
    # for date in dates_to_process:
    #     print_frames_from_date(date)
    
    # loop over dates in parallel
    if do_parallel:
        with ProcessPoolExecutor(max_workers=len(robust_mcs_files)) as Executor:
            Executor.map(process_from_robust_stats, robust_mcs_files)
    else:
        # meant only for testing since parallel fails silently
        for filename in robust_mcs_files[1:2]:
            process_from_robust_stats(filename, nmax=25)


if __name__ == "__main__":
    """Note: can convert to animation using ffmpeg. E.g., 
    ffmpeg -pattern_type glob -i "*.png" -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p out.mp4
    """
    main(do_parallel=True)