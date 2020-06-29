#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module contains useful definitions for MCS analysis

This module contains useful definitions and functions to aid in analyzing
mesoscale convective systems (MCS).

Created on Wed Oct 18 09:09:44 2017

@author: Christopher Jones (christopher.jones@pnnl.gov)
"""

import os
import xarray as xr
import pandas as pd
import glob
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import scipy.optimize as opt
#import cmocean

# host-dependent imports
if os.getenv('HOME') == '/ccs/home/crjones':
    # can't interactively plot on rhea
    import matplotlib
    matplotlib.use('Agg')
# finish import statements:
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# useful function definitions
def get_host():
    """Determine host where scripts are run (rhea or laptop)
    """
    home = os.getenv('HOME')
    if home == '/Users/jone003':
        return 'laptop'
    elif home == '/home/jone003':
        return 'we33999'
    elif home == '/ccs/home/crjones':
        return 'rhea'
    else:
        return None

    
def project_dir():
    host = get_host()
    if host == 'laptop':
        return '/Users/jone003/Dropbox/PNNL/ACME/runs'
    if host == 'we33999':
        return '/home/jone003/Dropbox/PNNL/ACME/runs'
    elif host == 'rhea':
        return '/lustre/atlas/proj-shared/csc249/crjones/ACME_ECP/hindcasts'
    else:
        return None


def ceres_dir():
    host = get_host()
    if host == 'laptop':
        return '/Users/jone003/Dropbox/PNNL/ACME/obs'
    elif host == 'rhea':
        return '/lustre/atlas/proj-shared/csc249/crjones/obs'
    else:
        return None


def load_case(case_name, pattern="/*.h2.*remap.nc", rootdir="."):
    """ Loads dataset from rootdir/case_name/ matching pattern
    """
    filepath = '/'.join([rootdir, case_name, pattern]).replace('//', '/')
    return xr.open_mfdataset(sorted(glob.glob(filepath)))


def fix_time(ds, time='time', round_to='min'):
    """Rounds time coordinate to nearest hour.
    Necessary for aligning the data, because not all times align by
    default due to roundoff (acme bug, or xarray bug?)
    """
    ds[time].values = pd.to_datetime(ds[time].values).round(round_to)


def parse_addbox(addbox):
    """ Parse addbox for map_us_simple to get xy, width, height

    addbox = {'xy': xy, 'width': width, 'height': height} OR
    addbox = {'lats': [latmin, latmax], 'lons': [lonmin, lonmax]} OR
    addbox = [lonmin, lonmax, latmin, latmax]
    """
    if isinstance(addbox, list) and len(addbox) == 4:
        addbox = {'lats': addbox[2:], 'lons': addbox[0:2]}
    if 'xy' in addbox:
        return addbox['xy'], addbox['width'], addbox['height']
    elif 'lats' in addbox:
        lon = addbox['lons']
        lat = addbox['lats']
        xy = [lon[0], lat[0]]
        width = lon[1] - lon[0]
        height = lat[1] - lat[0]
        return xy, width, height
    else:
        print("ERROR parse_addbox: addbox.keys() = ['xy', 'width', 'height']")
        print("                 or addbox.keys() = ['lats', 'lons']")
        raise KeyError('parse_addbox:addbox missing required keys')


def map_us_simple(ax=None, projection=ccrs.LambertConformal(), addbox=None,
                  extent=[-125, -66.5, 20, 50]):
    """Plot map of US and optionally add a lat-lon box
    """
    # Some references for future use:
    # http://scitools.org.uk/cartopy/docs/v0.14/matplotlib/feature_interface.html
    # https://uoftcoders.github.io/studyGroup/lessons/python/cartography/lesson/
    #
    # note to self: projection=ccrs.[...] is the projection used in the
    #               current map
    # ax.add_patch(xy, ..., transform=ccrs.PlateCarree()) says
    #    ccrs.PlateCarree() is the original coordinate system
    #    (which will be transformed to match projection)

    if ax is None:
        ax = plt.axes(projection=projection)
    ax.set_extent(extent)  # center on CONUS

    if addbox is not None:
        xy, width, height = parse_addbox(addbox)
        ax.add_patch(mpatches.Rectangle(xy=xy, width=width, height=height,
                                        facecolor='red',
                                        alpha=0.3,
                                        transform=ccrs.PlateCarree())
                     )
    ax.add_feature(cfeature.LAND, alpha=0.2)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle='-')
    ax.add_feature(cfeature.LAKES)
    ax.add_feature(cfeature.RIVERS)

    # Create a feature for states
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')
    ax.add_feature(states_provinces, edgecolor='gray')

    plt.title('Hovmoller Region')
    plt.show()
    return ax

# for fitting precipitation intensity and such to sine waves
def local_time(utc, lon):
    # assumes utc in hours, lon in degrees (-180, 180)
    return utc + lon / 15.0

# NOTE: The following functions are no longer used in my analysis
#       and will be removed in the future
def fit_sin(t, amp, phase, offset):
    """Sine curve with fixed period in hours"""
    period = 24.0  # assume only diurnal sine component here
    return amp * np.sin(2. * np.pi * t / period - phase) + offset


def fit_double_sin(t, amp1, amp2, phase1, phase2, offset):
    """Sine curve with fixed period in hours"""
    period1 = 24.0  # assume only diurnal sine component here
    period2 = 12.0  # include half-diurnal
    y = (amp1 * np.sin(2. * np.pi * t / period1 - phase1) + offset +
         amp2 * np.sin(2. * np.pi * t / period2 - phase2))
    return y


def fit_prec_latlon_to_fun(prec_data, **kwargs):
    nlat = len(prec_data.lat)
    nlon = len(prec_data.lon)
    t = prec_data.hour
    amp24h = np.empty([nlat, nlon])
    amp12h = np.empty([nlat, nlon])
    phase24h = np.empty([nlat, nlon])
    phase12h = np.empty([nlat, nlon])
    offset = np.empty([nlat, nlon])
    time_of_max_fit = np.empty([nlat, nlon])
    loc_time_of_max_fit = np.empty([nlat, nlon])
    intensity_max = np.empty([nlat, nlon])
    for ilat in range(nlat):
        for ilon in range(nlon):
            lon = prec_data.lon.isel(lon=ilon)
            popt, pcov = opt.curve_fit(fit_double_sin, t,
                                       prec_data.isel(lat=ilat, lon=ilon),
                                       **kwargs)
            amp24h[ilat, ilon] = popt[0]
            amp12h[ilat, ilon] = popt[1]
            phase24h[ilat, ilon] = popt[2]
            phase12h[ilat, ilon] = popt[3]
            offset[ilat, ilon] = popt[4]
            fun = lambda x: (-1.)*fit_double_sin(x, *popt)
            res = opt.minimize_scalar(fun, method='bounded', bounds=(0, 24))
            if res.success:
                time_of_max_fit[ilat, ilon] = res.x
                loc_time_of_max_fit[ilat, ilon] = np.mod(local_time(res.x, lon), 24)
                intensity_max[ilat, ilon] = -res.fun
            else:
                time_of_max_fit[ilat, ilon] = np.nan
    return xr.Dataset({'amp24h': (['lat', 'lon'], amp24h),
                       'amp12h': (['lat', 'lon'], amp12h),
                       'phase24h': (['lat', 'lon'], phase24h),
                       'phase12h': (['lat', 'lon'], phase12h),
                       'offset': (['lat', 'lon'], offset),
                       'utc_of_max_fit': (['lat', 'lon'], time_of_max_fit),
                       'loc_time_of_max_fit': (['lat', 'lon'], loc_time_of_max_fit),
                       'intensity_max_fit': (['lat', 'lon'], intensity_max)},
                      coords={'lat': prec_data.lat,
                              'lon': prec_data.lon})


def precip_plot(fit_ds, data_da, suptitle=None, rescale=True):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
    fit_ds['loc_time_of_max_fit'].plot(ax=ax[0], cmap=cmocean.cm.phase)
    fit_ds['intensity_max_fit'].plot(ax=ax[1])
    if rescale:
        (86400 * 1000 * data_da).plot(ax=ax[2])
    else:
        data_da.plot(ax=ax[2])
    if suptitle is not None:
        fig.suptitle(suptitle)
    ax[0].set_title('Local time of max')
    ax[1].set_title('Max (fit)')
    ax[2].set_title('Mean')
    ax[1].set_ylabel("")
    ax[2].set_ylabel("")
    return fig, ax

