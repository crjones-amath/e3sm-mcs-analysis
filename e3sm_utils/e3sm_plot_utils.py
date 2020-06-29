#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of plotting routines / utilities for E3SM. This should be organized
later, but want to get this down to clean up my scripts.

Created on Wed Nov 21 12:28:15 2018

@author: christopher.jones@pnnl.gov
"""

# imports
from . import e3sm_utils_se as ese
from . import e3sm_utils_fv as efv
from .map_utils import multi_map_axes_vert_cb
from matplotlib.pyplot import cm
import numpy as np


def plot_global(v, ds, *args, **kwargs):
    if 'ncol' in ds[v].dims:
        return ese.plot_global(v, ds, *args, **kwargs)
    else:
        return efv.plot_global(v, ds, *args, **kwargs)


def plot_map_comparison(v, ds1, ds2, *args, **kwargs):
    """Plot variable v in ds1 [top], ds2[middle], and ds2-ds1 (bottom)
    """
    # prepare axes
    kwax = {}
    if 'figsize' in kwargs:
        kwax['figsize'] = kwargs['figsize']
    if 'projection' in kwargs:
        kwax['projection'] = kwargs['projection']
    fig, ax, cax = multi_map_axes_vert_cb(nrows=3, ncols=1, **kwax)

    # plot ds1, ds2
    skip = ('cmap_div', 'vlim_div')
    kwglobal = {key: val for key, val in kwargs.items() if key not in skip}
    da1, *_ = plot_global(v, ds1, *args, ax=ax[0], cax=cax[0], **kwglobal)
    da2, *_ = plot_global(v, ds2, *args, ax=ax[1], cax=cax[1], **kwglobal)

    # plot difference
    skip = ('vmin', 'vmax', 'vlim_div', 'cmap', 'cmap_div', 'levels',
            'rescale')
    kwdelta = {key: val for key, val in kwargs.items() if key not in skip}
    kwdelta['cmap'] = cm.RdBu_r
    if 'cmap_div' in kwargs:
        kwdelta['cmap'] = kwargs['cmap_div']
    da = da2 - da1
    ds = da.to_dataset()
    ds.attrs['ne'] = ds1.attrs['ne']
    ds['area'] = ds1.area

    if 'vlim_div' in kwargs:
        vlim = kwargs['vlim_div']
    else:
        plot_data = da.values
        calc_data = np.ravel(plot_data[np.isfinite(plot_data)])
        vlim = max(abs(calc_data.max()), abs(calc_data.min()))
    kwdelta.update(vmin=-vlim, vmax=vlim)
    # print(vlim)
    # print(kwdelta)
    plot_global(v, ds, *args, ax=ax[2], cax=cax[2],
                **kwdelta)
    return ax, cax


#def multi_model_canvas(n, figsize=None):
#    """subplot array (nrows, ncols) = (n, n+1) with last column scaled 5% for colorbar
#    
#    Helper function called by multi_model_global_plot.
#    """
#    if figsize is None:
#        figsize = (8*n, 3*n)
#    width_ratios = [1] * n + [0.05]
#    fig = plt.figure(figsize=figsize)
#    gs = gridspec.GridSpec(n, n + 1, width_ratios=width_ratios)
#    return fig, gs
#
#
#def multi_model_global_plot(v, ds_dict, names, time_slice=None, isel_times=None,
#                            projection=ccrs.LambertCylindrical(), extent=None, 
#                            rescale=1, units="", cmap_div=plt.cm.RdBu_r, fmt='{:.3g}',
#                            vmin=None, vmax=None, cmap=None):
#    """Plot time-mean of variable v (and anomalies) on global map for datasets in ds_dict
#    
#    Each column corresponds to a model run. The top row plots the time-mean of the data.
#    The second row shows the anomaly of "v" relative to the data in the first column,
#    the 3rd row shows the anomaly of "v" relative to the data in the second column, 
#    and so on. Colorbars at the end of each row are common for the row.
#    
#    Inputs:
#        v - variable to plot
#        ds_dict - dictionary with (key, val) = (model_name, dataset)
#        names - list containing the names of models to show on plot
#                (expected to be keys of ds_dict)
#        time_slice - optionally specify slice for time-mean.
#                     if None, do mean over all times.
#        isel_times - optionally specify indices for time-mean
#                     only valid if time_slice is None
#        projection - cartopy projection to use for map
#        extent - optionally specify region for plotting (if None, do global plot)
#        rescale - factor to rescale output by, so plotted_v = v * rescale
#        units - optional string to specify units on map titles
#        cmap_div - optionally specify colormap for the "delta" plots
#        fmt - formatting string passed to toa_title
#        vmin, vmax - optionally specify range for colorbar
#        
#    Notes:
#      - names is emptied while this function is called
#      - I need to find a better way to set mins and maxes of the colorbars, but I want 
#        to make sure that the colorbar matches the colormaps for each subplot. I should
#        see if this approach here is really necessary.
#    """
#    
#    if cmap is None:
#        cmap = plt.cm.viridis  # default colormap
#    fig, gs = multi_model_canvas(len(names))
#    
#    # prep data_array
#    dat = {key: ds_dict[key][v] * rescale for key in names if key in ds_dict}
#    
#    # try to get units if none are supplied
#    if not units:
#        for key in dat:
#            try:
#                units = ds_dict[key][v].attrs['units']
#            except:
#                pass
#    
#    # subset in time and average
#    for key, val in dat.items():
#        if 'time' not in [d for d in val.dims]:
#            continue
#        if time_slice is None:
#            if isel_times is not None:
#                dat[key] = val.isel(time=isel_times).mean(dim="time")
#            else:
#                dat[key] = val.mean(dim="time")
#        else:
#            dat[key] = val.sel(time=time_slice).mean(dim="time")
#            
#    # top row: just plot the data
#    if vmin is None:
#        vmin = np.min([np.nanmin(v.values) for v in dat.values()])
#    if vmax is None:
#        vmax = np.max([np.nanmax(v.values) for v in dat.values()])
#    
#    cbax = fig.add_subplot(gs[0, -1])
#    for col, model in enumerate(names):
#        ax = fig.add_subplot(gs[0, col], projection=projection)
#        if model in dat:
#            dat[model].plot.contourf(ax=ax, transform=ccrs.PlateCarree(),
#                                     cbar_ax=cbax, vmin=vmin, vmax=vmax, cmap=cmap)
#            map_layout(ax, extent)
#            ax.set_title(toa_title(dat[model], model_name=model, show_mean=True, show_rmse=False, units=units, fmt=fmt))
#    
#    # for remaining rows, plot differences:
#    # note to self: someday find a better way to do this
#    row = 0
#    key = names.pop(0) # pop first element -- this will be the new "obs"
#    while names:
#        row = row + 1
#        d0 = dat.pop(key)
#
#        # update data with difference from d0
#        dat.update((k, val - d0) for k, val in dat.items())
#        vmin = np.min([np.nanmin(v.values) for v in dat.values()])
#        vmax = np.max([np.nanmax(v.values) for v in dat.values()])
#        
#        vmax = np.max([abs(vmin), abs(vmax)])
#        vmin = -vmax
#        cbax = fig.add_subplot(gs[row, -1])
#        
#        for col, model in enumerate(names):
#            ax = fig.add_subplot(gs[row, row + col], projection=projection)
#            if model in dat:
#                dat[model].plot.contourf(ax=ax, transform=ccrs.PlateCarree(),
#                                         cbar_ax=cbax, vmin=vmin, vmax=vmax, cmap=cmap_div, robust=True)
#                map_layout(ax, extent)
#                ax_name = ' - '.join([model, key])
#                ax.set_title(toa_title(dat[model], model_name=ax_name, show_mean=True, show_rmse=True, units=units, fmt=fmt))
#        key = names.pop(0)
