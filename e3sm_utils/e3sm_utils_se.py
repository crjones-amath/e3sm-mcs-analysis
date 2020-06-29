#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of plotting routines / utilities for E3SM. This should be organized
later, but want to get this down to clean up my scripts.

Created on Wed Nov 21 12:28:15 2018

@author: christopher.jones@pnnl.gov
"""

# imports
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs   # map plots
from .map_utils import map_layout, map_axes_with_vertical_cb


#####################################
# Unstructured grid utilities
#####################################

# entries are (corner_max, edge_max)
gll_area_max_thresholds = {'ne30': (0.0001, 0.00025),
                           'ne120': (0.000006, 0.000015)}


def classify_gll_nodes(ds, res='ne30', add_mask_to_coords=True):
    """ Classify unstructured E3SM grid nodes based on area
    """
    max_corner, max_edge = gll_area_max_thresholds[res]
    area = ds.area.values  # convert to a numpy array
    corner = area <= max_corner
    edge = np.logical_and(area > max_corner, area <= max_edge)
    center = area > max_edge

    mask = 1 * corner + 2 * edge + 3 * center
    if add_mask_to_coords:
        ds.coords['mask'] = ('ncol', mask)
        ds.mask.attrs['description'] = 'mask identify gll node classification'
        ds.mask.attrs['long_name'] = 'gll_node_type'
        ds.mask.attrs['units'] = ' '
        ds.mask.attrs['flags'] = '1: corner node, 2: edge node, 3: center node'
    return mask


def awm(da, area, region):
    """return area-weighted mean of dataset over region

    Note: redundant with area_weighted_mean; need to find a better way to
    deal with (da, area) and (ds, variable) ways of specifying this."""
    if region is None:
        return (da * area).sum(dim='ncol') / area.sum(dim='ncol')
    else:
        num = (da * area).where(region).sum(dim='ncol')
        denom = area.where(region).sum(dim='ncol')
        return num / denom


def plot_profile_by_area(ds, variable, area, region, mask,
                         mask_vals=[1, 2, 3],
                         labels=['corner', 'edge', 'center'],
                         do_pert=True, figsize=(12, 6)):
    """plot profiles over region separated by corner/edge/center"""
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    da = ds[variable]
    if do_pert:
        # first calculate anomaly
        da = da - awm(da, area, region).mean(dim='time')
    awm(da, area, region).mean(dim='time').plot(ax=ax, linestyle='--',
                                                color='k', y='lev',
                                                label='all')
    for val, lab in zip(mask_vals, labels):
        reg = np.logical_and(region, mask == val)
        awm(da, area, reg).mean(dim='time').plot(ax=ax, y='lev', label=lab)
    ax.legend()
    ax.invert_yaxis()
    ax.set_xlabel(variable + ' (' + ds[variable].units + ')')
    return fig, ax


def area_weighted_mean(da=None, area=None, ds=None, var=None):
    """area-weighted mean of dataArray da given an 'area' dataArray

    Expected inputs either:
        da - xarray dataArray
        area - xarray dataArray with area entries corresponding to da
    or:
        ds - xarray dataSet containing data variables var and 'area'
        var - name of variable for which to calculate area-weighted mean
    """
    if ds is not None and var is not None:
        return area_weighted_mean(da=ds[var], area=ds['area'])
    dsum = [d for d in da.dims if d != 'time']
    return (da * area).sum(dim=dsum) / area.sum(dim=dsum)


def area_weighted_rmse(da, area):
    """area-weighted RMSE of anomaly da"""
    return np.sqrt(area_weighted_mean(da ** 2, area))


def toa_title(da, area, model_name, show_mean=True,
              show_rmse=False, units="", fmt='{:.3g}'):
    """Calculate mean and/or RMSE and return string to use as title in plots

    inputs:
        da - xarray dataarray with the data for spatial means
        model_name - model name to include in string
        show_mean - add global area-weighted mean to output string if true
        show_rmse - add global area-weighted rmse to output string if true
        units - optionally specify units
        fmt - optionally specify formatting string form mean and rmse
    output: string of form "(mean) (model_name) (rmse) (units)"
    """
    if show_mean:
        mean_val = area_weighted_mean(da, area).values.item()
        mn = ("Mean: " + fmt).format(mean_val)
    else:
        mn = ""
    if show_rmse:
        rmse_val = area_weighted_rmse(da, area).values.item()
        rmse = ("RMSE: " + fmt).format(rmse_val)
    else:
        rmse = ""
    return "{:12} {} {:>12}  ".format(mn, model_name, rmse) + units


lat_lon_memo = {}
def _get_plot_latlon(ds):
    """Extract lat and lon (remapped to [-180, 180]) from E3SM-SE ds
    """
    grid_size = ds.attrs['ne']
    if grid_size in lat_lon_memo:
        return lat_lon_memo[grid_size]
    # need to account for 'time' dimension if mfdataset:
    if 'time' in ds['lat'].dims:
        lat = ds['lat'].mean(dim='time').values
        lon = ds['lon'].mean(dim='time').values
    else:
        lat = ds['lat'].values
        lon = ds['lon'].values
    lon[lon > 180] = lon[lon > 180] - 360
    lat_lon_memo[grid_size] = (lat, lon)
    return lat, lon


def plot_da_on_ax(ax, lon, lat, da, plot_type='contourf', extent=None,
                  **kwargs):
    if plot_type == 'contourf':
        p = ax.tricontourf(lon, lat, da,
                           transform=ccrs.PlateCarree(), **kwargs)
    else:
        p = ax.tripcolor(lon, lat, da, transform=ccrs.PlateCarree(), **kwargs)
    map_layout(ax, extent=extent)
    return p


def plot_global(v, ds, time_slice=None, projection=ccrs.PlateCarree(),
                extent=None, rescale=1, units="", name='SP',
                mask_threshold=None, ilev=None, figsize=(8, 6),
                plot_type='contourf',
                ax=None, cax=None,
                **kwargs):
    """ 2D Map plot of time-mean of ds[v].

    Arguments:
        v - variable to plot
        ds - xarray dataset containing variable v
        time_slice - optionally specify slice for time-mean.
                     if None, do mean over all times.
        projection - cartopy projection to use for map
        extent - optionally specify region for plotting (global plot if None)
        rescale - factor to rescale output by, so plotted_v = v * rescale
        units - optional string to specify units on map titles
        name - name used in plot title
        mask_threshold - optionally mask out values below mask_threshold
        ilev - for 3D output, optionally specify level index to show
        figsize - size of figure
        plot_type - ['contourf'] | 'pcolor'
        ax, cax - optionally specify plot axis and colorbar axis
                  if None, create plot on new figure / ax / cax
        **kwargs - passed into plotting routine
    """
    # prepare data to plot:
    if time_slice is None and 'time' in ds:
        try:
            da = ds[v].mean(dim='time') * rescale
        except:
            da = ds[v] * rescale
    elif 'time' in ds:
        da = ds[v].sel(time=time_slice).mean(dim='time') * rescale
    else:
        da = ds[v] * rescale
    if ilev is not None:
        da = da.isel(lev=ilev)
    lat, lon = _get_plot_latlon(ds)
    # note: not sure I've done this right ...
    if mask_threshold is not None:
        da = da.where(da < mask_threshold)
        area = ds['area'].where(da < mask_threshold)
    else:
        area = ds['area']
    # (I think I've fixed an earlier problem toa_title choked on nans)
    ax_title = toa_title(da, area, model_name=name,
                         show_mean=True, show_rmse=False, units=units)
    # create axes if needed
    if ax is None:
        fig, ax, cax_tmp = map_axes_with_vertical_cb(figsize=figsize,
                                                     projection=projection)
        if cax is None:
            cax = cax_tmp
    p = plot_da_on_ax(ax, lon, lat, da, plot_type=plot_type, extent=extent,
                      **kwargs)
    # add colorbar
    if cax is not None:
        cb = plt.colorbar(p, cax=cax, label=v + " (" + units + ")")
    else:
        cb = None
    ax.set_title(ax_title)
    return da, ax, cax, cb
