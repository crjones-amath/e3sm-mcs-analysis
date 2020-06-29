#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functionaliy for connecting regions of spectral element e3sm grid

Defines ConnectedFeature class for labeling features based on a mask and
connecting them through space and time. This was written with the intention
of using it for MCS tracking purposes on the E3SM spectral element grid,
but the code written is somewhat more general than that.

Created on Mon Jun 11 15:28:41 2018

@author: christopher.jones@pnnl.gov
"""

import os
import numpy as np
import networkx
import xarray as xr
from itertools import chain
import json

home = os.getenv('HOME')
if home is None:
    home = "c:/Users/chris"

map_dir = home + '/Dropbox/globus/maps_for_regridding/'
mapping_files = {'ne30': map_dir + 'ne30np4_pentagons.091226.nc'}


def neighboring_cells(i1, i2, ds):
    """Determine if i1 and i2 share boundary points in xarray dataset ds."""

    # exclude self:
    if i1 == i2:
        return False

    # neighbors share a (grid_corner_lat, grid_corner_lon) pair
    pts1 = set(tuple(x) for x in
               zip(ds.isel(grid_size=i1).grid_corner_lat.values,
                   ds.isel(grid_size=i1).grid_corner_lon.values))
    pts2 = set(tuple(x) for x in
               zip(ds.isel(grid_size=i2).grid_corner_lat.values,
                   ds.isel(grid_size=i2).grid_corner_lon.values))
    return len(pts1 & pts2) > 0


def e3sm_variable_suffix(crm_ds):
    """Returns suffix added to e3sm dataset variables
    """
    crm_ncol = [x for x in crm_ds.dims if 'ncol' in x][0]
    return crm_ncol[4:]


def e3sm_master_ncol_index(subset_ds, master_ds, latvar='grid_center_lat',
                           lonvar='grid_center_lon'):
    """Map from ncol in subset_ds to ncol in master_ds.

    Returns: dictionary d where subset_ds[ncol=n] corresponds to
             naster_ds[ncol=d[n]]
    """

    mlats = np.around(master_ds[latvar].values, decimals=4)
    mlons = np.around(master_ds[lonvar].values, decimals=4)

    # e3sm output from subset of columns appends an identifying string after
    # all variables, inlcuding ncol, lat, and lon - need to account for that
    sub_suffix = e3sm_variable_suffix(subset_ds)
    sub_lats = np.around(subset_ds['lat' + sub_suffix].values, decimals=4)
    sub_lons = np.around(subset_ds['lon' + sub_suffix].values, decimals=4)

    sub_ncol_to_latlon = {n: val for n, val in enumerate(zip(sub_lats,
                                                             sub_lons))}
    master_latlon_to_ncol = {val: n for n, val in enumerate(zip(mlats, mlons))}

    # map from subset ncol index to master ncol index
    return {n: master_latlon_to_ncol[v] for n, v in sub_ncol_to_latlon.items()}


def neighborhood_to_search(subset_index, subset_ds, master_ds,
                           delta_lat_max=2, delta_lon_max=2):
    """Returns np array of indices in master_ds to check for neighors to
    subset_index in subset_ds.

    Note: only returns indices if they are also in subset_ds
    """
    ncol, lat, lon = (v + e3sm_variable_suffix(subset_ds) for v in
                      ['ncol', 'lat', 'lon'])
    delta_lat_condition = np.abs(master_ds.grid_center_lat -
                                 subset_ds[lat].isel(**{ncol: subset_index})
                                 ) < delta_lat_max
    delta_lon_condition = np.abs(master_ds.grid_center_lon -
                                 subset_ds[lon].isel(**{ncol: subset_index})
                                 ) < delta_lon_max
    return np.where(delta_lat_condition & delta_lon_condition)[0]


def persist_to_file(file_name):
    """Cache results of function evaluation to file_name."""
    def decorator(original_func):
        try:
            cache = json.load(open(file_name, 'r'))
        except (IOError, ValueError):
            cache = {}

        def new_func(*args):
            cache_arg = "_".join(args)
            if cache_arg not in cache:
                cache[cache_arg] = original_func(*args)
                json.dump(cache, open(file_name, 'w'))
            return {int(key): val for key, val in cache[cache_arg].items()}

        return new_func

    return decorator


@persist_to_file('cache_e3sm_neighborhoods.dat')
def populate_neighborhoods(e3sm_filename, ref_map_filename, debug=True,
                           **kwargs):
    """Use ref_map_ds to identify neighbors of each grid index in e3sm_ds

    Returns: neighbors = dict(key=crm_ds ncol index,
                              val=list of e3sm_ds ncol neighboring indices)
    """
    if debug:
        print('Debug is on')
    # need to pass in filename instead of ds for memoization to work
    e3sm_ds = xr.open_dataset(e3sm_filename).load()
    ref_map_ds = xr.open_dataset(ref_map_filename).load()
    
    if debug:
        print('Loaded e3sm_ds')
        print('Loaded ref_map_ds')
        print('indices up next')

    # key: e3sm_ds ncol index; val: ref_map_ds ncol index
    indices = e3sm_master_ncol_index(e3sm_ds, ref_map_ds, **kwargs)
    if debug:
        print('indices determined')
        print(len(indices))

    # key: ref_map_ds ncol index; val: e3sm_ds ncol index
    ref_ds_to_e3sm_indices = {val: key for key, val in indices.items()}
    if debug:
        print('ref_ds_to_e3sm_indices determined')

    neighbors = dict()
    if debug:
        print('populating neighborhoods')
    for ind_e3sm, ind_ref in indices.items():
        if debug:
            print(ind_e3sm, ind_ref)
        neighbors[ind_e3sm] = [ref_ds_to_e3sm_indices[idx]
                               for idx in
                               neighborhood_to_search(ind_e3sm, e3sm_ds,
                                                      ref_map_ds,
                                                      delta_lat_max=2,
                                                      delta_lon_max=2)
                               if (idx in ref_ds_to_e3sm_indices) and
                               neighboring_cells(ind_ref, idx, ref_map_ds)]
    return neighbors


# function for identifying connected regions
class ConnectedFeature:
    """ Class for labeling connected features in a 2D mask array

    Starting from E3SM output on native grid, identify features connected in
    space and time. Provides functionality for labeling connected features
    and all.

    Currently only supports native SE grid, but plan to extend to remapped
    grid in future.
    """
    def __init__(self, neighbors, unlabeled_mask=None,
                 time_axis=0, space_axis=1):
        self.Graph = networkx.from_dict_of_lists(neighbors)
        self.time_axis = time_axis
        self.space_axis = space_axis
        self._visited = list()
        self._axes_swapped = False
        self.add_unlabeled_feature(unlabeled_mask)
        self.time_slices = np.shape(self.unlabeled_feature)[time_axis]
        self._ntime = np.shape(self.unlabeled_feature)[time_axis]
        self._label = 0
        self._feature_splits_into = dict()
        self._feature_merged_from = dict()
        self._lifetimes = None
        self.feature_tree = None

    def __repn__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

    def add_unlabeled_feature(self, unlabeled_mask):
        self.unlabeled_feature = unlabeled_mask
        if self.time_axis != 0:
            self.unlabeled_feature = self.unlabeled_feature.swapaxes()
        self.labeled_feature = np.zeros_like(self.unlabeled_feature,
                                             dtype='int')

    def floodfill(self, ntime, n, label=0):
        if self.Graph.has_node(n) and (n not in self._visited) and self.unlabeled_feature[ntime, n]:
            self._visited.append(n)
            self.labeled_feature[ntime, n] = label
            for neighbor in self.Graph.neighbors(n):
                self.floodfill(ntime, neighbor, label)

    def label_connections_in_space(self):
        for nt in range(self.time_slices):
            self._visited = []
            # for n in self.Graph.nodes_iter():
            for n in self.Graph.nodes():
                if self.unlabeled_feature[nt, n] and (n not in self._visited):
                    self._label = self._label + 10
                    self.floodfill(nt, n, label=self._label)
        self._labels = self.get_labels()

    def connect_labels_in_time(self, do_clear=False):
        if do_clear:
            self._feature_splits_into = dict()
            self._feature_merged_from = dict()
        for nt in range(self._ntime - 1):
            children = dict()
            parents = dict()
            this_slice = self.labeled_feature[nt, :]
            next_slice = self.labeled_feature[nt + 1, :]
            for label in np.unique(this_slice[this_slice > 0]):
                # look at next layer to see which labels need to be replaced
                labels_to_replace = set(next_slice[n] for
                                        n in np.where(this_slice == label)[0]
                                        if next_slice[n] > this_slice[n])
                children[label] = labels_to_replace
                for child in labels_to_replace:
                    grow_dict_of_sets(parents, child, label)
            for par, kids in children.items():
                if len(kids) == 1:
                    kid = kids.pop()
                    if len(parents[kid]) == 1:
                        # Only 1 parent => same label
                        self.labeled_feature[self.labeled_feature == kid] = par
                    else:
                        # Multiple parents merge, retain labels
                        self._feature_merged_from[kid] = parents[kid]
                elif len(kids) > 1:
                    # One feature splits into multiple, retain labels
                    self._feature_splits_into[par] = kids
        self._labels = self.get_labels()

    def relabel(self):
        """Relabel features to integers
        """
        pairs = {old: -new for new, old
                 in enumerate(np.unique(self.labeled_feature))}
        for old, new in pairs.items():
            self.labeled_feature[self.labeled_feature == old] = new
        self.labeled_feature = -self.labeled_feature

        # relabel branching dictionaries as well:
        for d in [self._feature_merged_from, self._feature_splits_into]:
            for key, val in d.copy().items():
                for v in list(val):
                    val.remove(v)
                    val.add(pairs[v])
                val.update([-val.pop() for _ in range(len(val))])
                d[-pairs[key]] = d.pop(key)
        self._labels = self.get_labels()

    def construct_feature_tree(self, labels):
        """Identify linked features.
        Labeled events can split into events, or can form when
        two others merge into one. This function links those
        together.

        Inputs:
          labels: list containing tuples of mcs identifiers
          mcs_splits_into: dictionary of form {index: set}
          mcs_merged_from: dictionary of form {index: set}
        Output:
        """
        out = []
        visited = []
        while labels:
            label = labels.pop()
            if label[-1] in self._feature_splits_into:
                # feature splits -> append each branch and append to labels
                for track in self._feature_splits_into[label[-1]]:
                    labels.append(label + (track, ))
                    visited.append((track,))
            elif label[0] in self._feature_merged_from:
                # feature merged -> prepend each root and append to labels
                for track in self._feature_merged_from[label[0]]:
                    labels.append((track,) + label)
                    visited.append((track,))
            else:
                out.append(label)
        # return out
        self.feature_tree = sorted([o for o in out if o not in visited])

    def do_it_all(self):
        """Label connected features in space/time, construct feature tree
        Note to self: rename this
        """
        self.label_connections_in_space()
        self.connect_labels_in_time()
        self.relabel()
        labels = np.unique(self.labeled_feature[self.labeled_feature != 0])
        self.construct_feature_tree([(label, ) for label in labels])

    def get_labels(self):
        """Get unique set of non-zero feature labels"""
        return np.unique(self.labeled_feature[self.labeled_feature != 0])

    def calculate_lifetimes(self):
        def delta_t(array, label):
            return 1 + np.diff(np.where(array == label)[0][[0, -1]]).item()
        self._lifetimes = {lab: delta_t(self.labeled_feature, lab) for lab
                           in self._labels if lab != 0}
        for path in self.feature_tree:
            self._lifetimes[path] = sum(self._lifetimes[lab] for lab in path)

    def lifetime(self, label=None, update=False):
        if update or self._lifetimes is None:
            self.calculate_lifetimes()
        if label is None:
            return self._lifetimes
        else:
            return self._lifetimes[label]

    def calculate_extent(self):
        """Count extent as number of spatial nodes in feature"""
        def match(label):
            return np.where(self.labeled_feature == label)[0]
        self._extent = {lab: [match(lab).tolist().count(x) for x in
                              np.unique(match(lab))] for lab in self._labels}
        for path in self.feature_tree:
            self._extent[path] = list(chain.from_iterable(self._extent[lab]
                                                          for lab in path))
        self._max_extent = {k: np.max(v) for k, v in self._extent.items()}


def grow_dict_of_sets(d, key, val):
    """Add val to set s, such that d[key] = s"""
    if key not in d:
        d[key] = set((val,))
    else:
        d[key].add(val)


def embed_regional_array_in_grid(array, indices, shape_out, axis):
    """Map from regional grid to global grid, embed array in there

    E.g., for 2D array with axis = 1, this is equivalent to
        output = np.full(shape_out, np.nan)
        output[:, indices] = array
        return output
    """
    shape = np.shape(array)
    shape_check = [shape[i] == shape_out[i] for i in
                   range(len(shape_out)) if i != axis]
    if not np.all(shape_check):
        raise ValueError('Shape mismatch between array, shape_out, and axis')
    out = np.full(shape_out, np.nan)
    ind = [slice(None)] * len(shape_out)
    ind[axis] = indices
    out[ind] = array
    return out
