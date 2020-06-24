"""Generate frames for animation of TMQ"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from concurrent.futures import ProcessPoolExecutor
from e3sm_utils import cmclimate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable  # to make reasonable colorbars
import matplotlib.gridspec as gridspec
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
import glob
import os

conus = [235, 295, 20, 50]  # [lon1, lon2, lat1, lat2]
# Create a feature for states
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')

# prepare the crm dataset
def rename_crm_vars(ds, squeeze=True, swap_xdim=True, crm_dx=1):
    """Strip suffix off coordinate and variable names
    Optionally apply squeeze() and swap crm_nx for 'x' dimension """
    # get the suffix
    suf = [d for d in ds.dims if 'ncol' in d][0][4:]
    
    if suf:
        # strip suffix off any dimension/variable that contains the suffix
        ds = ds.rename({v: v.replace(suf, '') for v in ds.dims if suf in v})
        ds = ds.rename({v: v.replace(suf, '') for v in ds if suf in v})
    if squeeze:
        ds = ds.squeeze()
    if swap_xdim:
        # add x dimension in place of crm_nx
        ds.coords['x'] = ('crm_nx', ds.crm_nx.values * crm_dx)
        ds['x'].attrs = {'units': 'km', 'long_name': 'width'}
        ds = ds.swap_dims({'crm_nx': 'x'})
    return ds

# need to order crms appropriately
def crm_to_subplot_order(crm_ds):
    lonsort = np.argsort(crm_ds.lon.values)
    latsort = np.argsort(crm_ds.lat.values)
    trial_rowmat = np.reshape(latsort, (3, 3))
    trial_colmat = np.reshape(lonsort, (3, 3))
    if np.all(trial_rowmat == trial_colmat.transpose()):
        # currently sorted bottom to top, so need to swap that
        return np.ravel(trial_rowmat[::-1, :])
    else:
        raise ValueError("need to do a better sort in longitude")
        
def crm_bounding_patch(crm_ds):
    """points in convex hull of crm array in crm_ds"""
    lons = crm_ds.lon.values
    lats = crm_ds.lat.values

    pts = np.stack([lons, lats], axis=1)
    hull = ConvexHull(pts)
    polygon_xy = np.stack([pts[hull.vertices,0], pts[hull.vertices, 1]], axis=1)
    return polygon_xy

# prepare figure canvas:
def fig_layout(extent=conus, polygon_xy=None, figsize=(36, 10)):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.15], figure=fig)
    # top: map with vertical colorbar on right
    ax = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, alpha=0.2)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=0.5)
    # ax.add_feature(cfeature.LAKES)
    ax.add_feature(states_provinces, edgecolor='black', alpha=0.2)
    ax.set_extent(extent, crs=ccrs.PlateCarree()) # CONUS
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    
    if polygon_xy is not None:
        # show region corresponding to CRM array
        ax.add_patch(Polygon(polygon_xy, facecolor='black', alpha=0.8, transform=ccrs.PlateCarree()))
    
    # middle: list of crm axes
    gs_crm = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[0, 1])
    crm_axes = []
    for ax1 in gs_crm:
        crm_axes.append(fig.add_subplot(ax1))
        
    # bottom: crm colorbars
    crm_cbars = []
    gs_cbars = gridspec.GridSpecFromSubplotSpec(4, 2, subplot_spec=gs[1, 1], hspace=0.25, wspace=0.25)
    for ax2 in [gs_cbars[i] for i in [0, 1, -1, -2]]:
        crm_cbars.append(fig.add_subplot(ax2))

    return fig, ax, cax, crm_axes, crm_cbars

def mask_dat(var, thresh, value=None):
    tmp = var.values.copy()
    tmp[tmp < thresh] = np.nan
    if value is not None:
        tmp[tmp >= thresh] = value
    return tmp

# function to plot crm level output:
def plot_crm_dat(ax1, crm_ds, edge_color=None, ylim=(0, 20),
                 ylabel='z (km)', xlabel='x (km)',
                 xtick_labels=True, ytick_labels=True,
                 cbar_axes=None,
                 cbar_kwargs={'orientation': 'horizontal', 'aspect': 50}
                ):
    # assume that crm_ds is already selected for a given time and ncol location
    if any(var in crm_ds.dims for var in ['ncol', 'time']):
        raise ValueError('crm_ds should not contain ncol or time dimensions')
    ax1.clear()
    tol = 1.0e-10
    lon = crm_ds.lon.values.item()
    lat = crm_ds.lat.values.item()

    # convert crm_nz to 'Z3' coordinate
    crm_ds0 = crm_ds.squeeze().copy()
    z3 = crm_ds['Z3'] / 1000.0 # in km
    zvals = z3.values[::-1] # reversed
    nz = len(crm_ds0['crm_nz'])
    crm_ds0.coords['z'] = ('crm_nz', zvals[:nz])
    crm_ds0['z'].attrs = z3.attrs
    crm_ds0 = crm_ds0.swap_dims({'crm_nz': 'z'})

    qc = crm_ds0['CRM_QC'] * 1000
    qi = crm_ds0['CRM_QI'] * 1000
    qpc = crm_ds0['CRM_QPC'] * 1000
    qpi = crm_ds0['CRM_QPI'] * 1000
    qc.attrs['units'] = 'g/kg'
    qi.attrs['units'] = 'g/kg'
    qpc.attrs['units'] = 'g/kg'
    qpi.attrs['units'] = 'g/kg'

    qc.values = mask_dat(qc, tol)
    qi.values = mask_dat(qi, tol)
    qpc.values = mask_dat(qpc, tol)
    qpi.values = mask_dat(qpi, tol)
    
    qc_cm = plt.cm.Greys
    qi_cm = plt.cm.Purples
    qpc_cm = plt.cm.Blues
    qpi_cm = plt.cm.Greens
    
    qc_max = 0.1
    qi_max = 0.1
    qpi_max = 0.4
    qpc_max = 0.4
    alphax = 0.6
    
    # add the (static) colorbars
    cb_kwargs = {'orientation': 'horizontal', 'extend': 'max'}
    norm_p = matplotlib.colors.Normalize(vmin=0, vmax=qpi_max)
    norm_c = matplotlib.colors.Normalize(vmin=0, vmax=qc_max)
    cb_qc = matplotlib.colorbar.ColorbarBase(cbar_axes[0], cmap=qc_cm,
                                             norm=norm_c, **cb_kwargs)
    cb_qc.set_label('CRM_QC (g/kg)')
    cb_qi = matplotlib.colorbar.ColorbarBase(cbar_axes[1], cmap=qi_cm,
                                             norm=norm_c, **cb_kwargs)
    cb_qi.set_label('CRM_QI (g/kg)')
    cb_qpc = matplotlib.colorbar.ColorbarBase(cbar_axes[2], cmap=qpc_cm,
                                             norm=norm_p, **cb_kwargs)
    cb_qpc.set_label('CRM_QPC (g/kg)')
    cb_qpi = matplotlib.colorbar.ColorbarBase(cbar_axes[3], cmap=qpi_cm,
                                             norm=norm_p, **cb_kwargs)
    cb_qpi.set_label('CRM_QPI (g/kg)')

    if not np.all(np.isnan(qc.values)):
        qc.plot(ax=ax1, vmin=0, vmax=qc_max, cmap=qc_cm, alpha=alphax, add_colorbar=False)
    if not np.all(np.isnan(qi.values)):
        qi.plot(ax=ax1, vmin=0, vmax=qi_max, cmap=qi_cm, alpha=alphax, add_colorbar=False)
    if not np.all(np.isnan(qpc.values)):
        qpc.plot(ax=ax1, vmin=0, vmax=qpc_max, cmap=qpc_cm, alpha=alphax, add_colorbar=False)
    if not np.all(np.isnan(qpi.values)):
        qpi.plot(ax=ax1, vmin=0, vmax=qpi_max, cmap=qpi_cm, alpha=alphax, add_colorbar=False)

    ax1.set_ylim(*ylim)
    ax1.set_xlim(0, qpc.x.values[-1])
    if edge_color is not None:
        ax1.spines['bottom'].set_color(edge_color)
        ax1.spines['top'].set_color(edge_color)
        ax1.spines['left'].set_color(edge_color)
        ax1.spines['right'].set_color(edge_color)
        ax1.set_title('2D CRM', color=edge_color)
        ax1.set_ylabel(ylabel, color=edge_color)
        ax1.set_xlabel(xlabel, color=edge_color)
        ax1.tick_params(axis='x', colors=edge_color)
        ax1.tick_params(axis='y', colors=edge_color)
    else:
        ax1.set_title('CRM at ({:.2f}, {:.2f})'.format(lon, lat))
        ax1.set_ylabel(ylabel)
        ax1.set_xlabel(xlabel)
    ax1.tick_params(labelbottom=xtick_labels, labelleft=ytick_labels)


def plot_da_with_crms(da, crm_ds, extent=conus, 
                      figsize=(36, 10),
                      da_kwargs={'cmap': cmclimate.cm.WhiteBlueGreenYellowRed, 'vmin': 0, 'vmax': 40}):
    fig, ax, cax, crm_axes, crm_cbars = fig_layout(extent=extent, polygon_xy=crm_bounding_patch(crm_ds),
                                                   figsize=figsize)
    da.plot(ax=ax, transform=ccrs.PlateCarree(), cbar_ax=cax, robust=True, **da_kwargs)
    for n, (ncol, crm_ax) in enumerate(zip(crm_to_subplot_order(crm_ds), crm_axes)):
        if n > 5:
            xlabel = 'x (km)'
            xtick_labels = True
        else:
            xlabel = ''
            xtick_labels = False
        if n in [0, 3, 6]:
            ylabel = 'z (km)'
            ytick_labels = True
        else:
            ylabel = ''
            ytick_labels = False
        plot_crm_dat(crm_ax, crm_ds.isel(ncol=ncol), edge_color=None,
                     ylim=(0, 20), xlabel=xlabel, ylabel=ylabel,
                     xtick_labels=xtick_labels, ytick_labels=ytick_labels,
                     cbar_axes=crm_cbars)

def match_date_subset(dates, year=None, month=None, days=None):
    """ Extract a subset of dates from the original list of dates
    """
    if year is None and month is None and days is None:
        return dates
    
    def sel(val, groups):
        return groups is None or val in groups
    
    return sorted(['-'.join([yr, mo, dy]) for yr, mo, dy in [d.split('-') for d in dates] if 
                   all([sel(yr, year), sel(mo, month), sel(dy, days)])])    

# dictionary of dataArray transformation opts by pvar
da_opts = {'PRECT': {'rescale': 86400000, 'units': 'mm/day'}}

def extract_da(plotvar, ds):
    """Extract da = ds[plotvar] and apply optional transformations"""
    da = ds[plotvar]
    opts = da_opts[plotvar] if plotvar in da_opts else dict()
    if 'rescale' in opts:
        da = da * opts['rescale']
    if 'units' in opts:
        da.attrs['units'] = opts['units']
    return da

def prepare_plot_data(date):
    topdir_sp = '/global/project/projectdirs/m3312/crjones/e3sm/early_science/hourly_2d_hist/remap/daily/'
    sp_case = 'earlyscience.FC5AV1C-H01A.ne120.sp1_64x1_1000m.cam.h1'    
    file_sp = topdir_sp + '.'.join([sp_case, date, 'nc'])
    ds_sp = xr.open_dataset(file_sp)
    
    topdir_crm = '/global/project/projectdirs/m3312/crjones/e3sm/early_science/hourly_crm_hist/daily/'
    crm_case = 'earlyscience.FC5AV1C-H01A.ne120.sp1_64x1_1000m.cam.h3'
    file_crm = topdir_crm + '.'.join([crm_case, date, 'nc'])
    crm_ds = rename_crm_vars(xr.open_dataset(file_crm))
    return extract_da('PRECT', ds_sp), crm_ds

def print_frames_from_date(date, out_prefix='conus_crm', topdir='/global/cscratch1/sd/crjones/figs/crm/'):
    """Plot hourly snapshots of plotvar for given date and save to file"""
    print('loading date ' + date)
    prect, crm_ds = prepare_plot_data(date)
    for i in range(len(crm_ds.time)):
        out_name = '_'.join([out_prefix, date, str(i).zfill(2)])
        plot_da_with_crms(prect.isel(time=i), crm_ds.isel(time=i), extent=conus)
        plt.savefig(topdir + out_name + '.png', dpi=600, bbox_inches='tight')
        plt.close()

def main(do_parallel=True):
    # dates_to_process given as 'yyyy-mm-dd'
    files_e3sm = glob.glob('/global/project/projectdirs/m3312/crjones/e3sm/early_science/hourly_2d_hist/remap/daily/*.nc')
    crm_files = glob.glob('/global/project/projectdirs/m3312/crjones/e3sm/early_science/hourly_crm_hist/*.nc')
    
    crm_ds = xr.open_mfdataset(crm_files)

    # set of dates in 'yyyy-mm-dd' format
    dates_in_crm_ds = set([t.strftime('%Y-%m-%d').replace(' ', '0') for t in crm_ds.time.values])
    dates_in_e3sm = sorted([f.split(sep='.')[-2] for f in files_e3sm])
    common_dates = [f for f in dates_in_e3sm if f in dates_in_crm_ds]

    # select May 0002
    dates_to_process = match_date_subset(common_dates, year='0002', month='05', days=None)
    print(dates_to_process)
    
    # crm_ds = rename_crm_vars(crm_ds)
    # serial version:
    if not do_parallel:
        for date in dates_to_process:
            print_frames_from_date(date)
    else:
        with ProcessPoolExecutor(max_workers=16) as Executor:
            Executor.map(print_frames_from_date, dates_to_process)

if __name__ == "__main__":
    """Note: can convert to animation using ffmpeg. E.g., 
    ffmpeg -pattern_type glob -i "*.png" -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p out.mp4 OR 
    ffmpeg -pattern_type glob -r 30 -i "*.png" -c:v libx264 -vf scale=2560:-2 -pix_fmt yuv420p conus_crm_0002-05.mp4
      options: -r framerate 
               -vf scale: height and width need to be divisible by 2
    """
    do_parallel = True
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    main(do_parallel)
