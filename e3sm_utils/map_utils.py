#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of plotting routines / utilities for E3SM. This should be organized
later, but want to get this down to clean up my scripts.

Created on Wed Nov 21 12:28:15 2018

@author: christopher.jones@pnnl.gov
"""

# imports
import matplotlib.pyplot as plt
import cartopy.crs as ccrs   # map plots
import cartopy.feature as cfeature   # needed for map features
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec


# Create a feature for states
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')


def map_layout(ax, *features, extent=None):
    """Add cartopy map to axis ax.

    sample features:
        cfeature.LAKES, cfeature.RIVERS
    """
    ax.add_feature(cfeature.LAND, alpha=0.2)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=0.5)
    ax.add_feature(states_provinces, edgecolor='black', alpha=0.2)
    for feat in features:
        ax.add_feature(feat)
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        ax.set_global()


def multi_model_canvas(n, figsize=None):
    """subplot array (nrows, ncols) = (n, n+1) with last column scaled 5%
    for colorbar

    Helper function called by multi_model_global_plot.
    """
    if figsize is None:
        figsize = (8*n, 3*n)
    width_ratios = [1] * n + [0.05]
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n, n + 1, width_ratios=width_ratios)
    return fig, gs


def map_axes_with_vertical_cb(figsize=(8, 6), projection=ccrs.PlateCarree()):
    """Prepare fig,ax,cax for map plot on ax with cb on cax"""

    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw={'projection': projection})

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05,
                              axes_class=plt.Axes)
    return fig, ax, cax


def multi_map_axes_vert_cb(nrows=3, ncols=1,
                           figsize=(12, 8), projection=ccrs.PlateCarree()):
    """Prepare fig,ax,cax for map plot on ax with cb on cax"""

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                           subplot_kw={'projection': projection})
    cax = [make_axes_locatable(a).append_axes('right', size='5%', pad=0.05,
                                              axes_class=plt.Axes) for a in ax]
    return fig, ax, cax
