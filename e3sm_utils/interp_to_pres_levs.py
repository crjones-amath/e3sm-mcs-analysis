import xarray as xr
import numpy as np
from scipy.interpolate import interp1d
from itertools import product


def interp_to_plevs(ps, hyam, hybm, dat, levs, **interp_kwargs):
    """ Interpolate dataArray dat to pressure levels levs
    
    inputs: ps is surface pressure dataArray
            hyam, hybm: hybrid coeffients
            dat: dataArray of variable to interpolate
            levs: pressure levels to interpolate dat onto
            interp_kwargs: keyword arguments to pass to scipy.interpolate.interp1d
    
    Returns: out_data: numpy array of dat interpolated to levs.
    Note: out_data dimensions have same order as dat, except that the new levs 
          dimension is last
    """
    p = 1000 * hyam + ps * hybm / 100  # pressure levels (assumed to be of same dims as dat)
    logp = np.log(p)  # interpolate in log-space
    
    dims = [v for v in dat.dims if 'lev' not in v]  # loop over non-pressure dimensions
    limits = [len(dat[d]) for d in dims]   # length of each dimension in dims
    
    # out_data will have same order as dimensions, except lev in last dimension
    out_data = np.full([*limits, len(levs)], fill_value=np.nan) 
    loglevs = np.log(levs)
    
    for sel_dims in product(*[range(lim) for lim in limits]):
        sel_dict = {dim: val for dim, val in zip(dims, sel_dims)}
        interp_fun = interp1d(logp.isel(sel_dict), dat.isel(sel_dict), **interp_kwargs)
        for n, loglev in enumerate(loglevs):
            ind = tuple([*sel_dims] + [n])
            try:
                out_data[ind] = interp_fun(loglev)
            except ValueError:
                # Some interp1d methods raise ValueError if asked to extrapolate
                pass
    return out_data
