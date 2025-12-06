#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import matplotlib.pyplot as plt
import mplhep as hep
from coffea import hist
import numpy as np
hep.style.use("CMS")
import matplotlib.patches as patches


# In[9]:


import numpy as np
from typing import Any, Dict, List

def _get_edges(axis) -> np.ndarray:
    edges_attr = getattr(axis, "edges", None)
    return np.asarray(edges_attr() if callable(edges_attr) else edges_attr, dtype=float)

def _get_values(view) -> np.ndarray:
    for name in ("values", "view", "to_numpy"):
        if hasattr(view, name):
            try:
                return np.asarray(getattr(view, name)()).ravel()
            except Exception:
                pass
    return np.asarray(view).ravel()

def compute_ratio_arrays(
    h,
    numer_label: str,
    denom_label: str,
    sample_axis_name: str = "sample",
    axis_indices: List[int] | None = None,
    uncertainty_type: str = "efficiency",
) -> Dict[str, Any]:
    """
    Compute ratio arrays for hist views. Works for 1D and 2D numeric axes (and general N-D).

    Returns a dict containing:
      - edges: list of 1D numpy arrays (one per selected numeric axis)
      - centers: list of 1D numpy arrays (one per selected numeric axis)
      - shape: tuple of nbins per axis
      - ratio: ndarray shaped `shape` with num/den per bin (NaN where undefined)
      - ratio_uncert: ndarray shaped (2, *shape) with absolute uncertainties (down, up)
      - num, den: raw arrays reshaped to `shape`
    Parameters:
      - axis_indices: which numeric axes to use; default None -> use all remaining axes (typical for 1D/2D)
    """
    try:
        from hist.intervals import ratio_uncertainty
    except Exception as exc:
        raise RuntimeError("hist.intervals.ratio_uncertainty is required: %s" % exc)

    num_sel = {sample_axis_name: numer_label}
    den_sel = {sample_axis_name: denom_label}
    hnum = h[num_sel]
    hden = h[den_sel]

    # decide which numeric axes to include
    # hnum.axes contains the remaining axes after category selection
    total_axes = list(hnum.axes)
    if axis_indices is None:
        sel_axes = total_axes  # use all numeric axes (1D or 2D typically)
    else:
        sel_axes = [total_axes[i] for i in axis_indices]

    # edges and centers per axis
    edges_list = [_get_edges(ax) for ax in sel_axes]
    centers_list = [0.5 * (e[:-1] + e[1:]) for e in edges_list]
    nbins_per_axis = tuple(len(e) - 1 for e in edges_list)
    nbins_total = int(np.prod(nbins_per_axis)) if nbins_per_axis else 0

    # get raw flattened arrays and reshape to the multi-dim bin shape
    num_flat = _get_values(hnum)
    den_flat = _get_values(hden)

    # align sizes: many hist views include only the exact bins, others include over/underflow.
    if num_flat.size != nbins_total:
        num_flat = num_flat.ravel()[:nbins_total]
    if den_flat.size != nbins_total:
        den_flat = den_flat.ravel()[:nbins_total]

    if nbins_per_axis:
        shape = nbins_per_axis
        # reshape in the original axis order (hist views use C-order for axes)
        num = num_flat.reshape(shape)
        den = den_flat.reshape(shape)
    else:
        # no numeric axis selected -> empty
        num = num_flat
        den = den_flat
        shape = num.shape

    # compute ratio and uncertainties elementwise (ratio_uncertainty supports array inputs)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.true_divide(num, den)
    # keep NaN for undefined entries (0/0 or inf)
    ratio[~np.isfinite(ratio)] = np.nan

    ratio_uncert = ratio_uncertainty(num, den, uncertainty_type=uncertainty_type)
    # ratio_uncert shape -> (2, *shape)

    return {
        "edges": edges_list,
        "centers": centers_list,
        "shape": shape,
        "ratio": ratio,
        "ratio_uncert": ratio_uncert,
        "num": num,
        "den": den,
    }



import hepdata_lib


# In[54]:


import numpy as np
from typing import Any, Dict, List

def make_hepdata_table_from_arrays(
    arrays: Dict[str, Any],
    table_name: str,
    independent_names: List[str] | None = None,
    independent_units: List[str] | None = None,
    dependent_name: str = "ratio",
    dependent_units: str = "",
):
    """
    Build and return a hepdata_lib Table from the arrays produced by compute_ratio_arrays.

    Notes:
    - hepdata_lib.Variable expects `values` for dependent variables as an iterable of
      2-element sequences: (value, uncertainty). A common and simple form is to provide
      uncertainty as a (minus, plus) tuple for asymmetric errors.
    - This function therefore produces dependent `values` as:
        [(value, (minus, plus)), ...]
      where `value` is `None` for NaN entries and minus/plus are floats (or None).
    """
    try:
        from hepdata_lib import Table, Variable
    except Exception as exc:
        raise RuntimeError("hepdata_lib is required (pip install hepdata_lib): %s" % exc)

    edges_list = arrays["edges"]
    ratio = np.asarray(arrays["ratio"])
    ratio_uncert = np.asarray(arrays["ratio_uncert"])  # shape (2, *shape)
    shape = arrays["shape"]

    # defaults
    if independent_names is None:
        independent_names = [f"axis_{i}" for i in range(len(edges_list))]
    if independent_units is None:
        independent_units = ["" for _ in edges_list]

    # Prepare dependent values: flattened in C-order
    dep_values: List[tuple] = []
    # Prepare independent lists: for each independent var, a list of tuples (low, high) aligned with dep_values
    ind_values_per_axis: List[List[tuple]] = [[] for _ in edges_list]

    # iterate over all bin index combinations in C-order
    for idx in np.ndindex(*shape):
        # dependent value and asymmetric errors
        val = ratio[idx]
        if np.isnan(val):
            val_item = None
            down = None
            up = None
        else:
            val_item = float(val)
            down = float(ratio_uncert[(0,) + idx])
            up = float(ratio_uncert[(1,) + idx])
        # append as a 2-tuple: (value, (minus, plus))
        dep_values.append((val_item, (down, up)))

        # independent values: append tuple (low, high) for each independent axis
        for axis_i, bin_i in enumerate(idx):
            e = edges_list[axis_i]
            ind_values_per_axis[axis_i].append((float(e[bin_i]), float(e[bin_i + 1])))

    # Build table and Variables
    table = Table(table_name)

    for axis_i, e_vals in enumerate(ind_values_per_axis):
        var_name = independent_names[axis_i] if axis_i < len(independent_names) else f"axis_{axis_i}"
        unit = independent_units[axis_i] if axis_i < len(independent_units) else ""
        xvar = Variable(var_name, is_independent=True, units=unit, values=e_vals)
        table.add_variable(xvar)

    # dependent variable expects list of (value, uncertainty) pairs
    vals = [None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v, _ in dep_values]
    unc_pairs = [None if updown is None else (None if updown[0] is None else float(updown[0]),
                                             None if updown[1] is None else float(updown[1]))
                 for _, updown in dep_values]    
    yvar = Variable(dependent_name, is_independent=False, is_binned=False,units=dependent_units, values=vals)
    unc2 = Uncertainty("Stat uncertainty", is_symmetric=False)
    unc2.values = unc_pairs
    yvar.add_uncertainty(unc2)

    table.add_variable(yvar)

    return table


# Example usage:
# arrays = compute_ratio_arrays(h, "numer_hlt", "denom_hlt")  # from your earlier function
# table = make_hepdata_table_from_arrays(
#     arrays,
#     table_name="Fig 56",
#     independent_names=["MET"],
#     independent_units=["GeV"],
#     dependent_name="HLT efficiency"
# )
# table  # returns hepdata_lib.Table


# In[146]:


with open("EXO-23-016-MDS-hist.pkl",'rb') as f:
    histograms = pickle.load(f)


# # Fig 56

# In[180]:


def makeFig56table(histograms):
    arrays = compute_ratio_arrays(histograms['fig56_l'], "numer_hlt", "denom_hlt")
    table = make_hepdata_table_from_arrays(arrays,
                                           table_name ="Fig 56",
                                           independent_names = ["MET"],
                                           independent_units =["GeV"],
                                           dependent_name= 'HLT efficiency'
                                          )
    arrays = compute_ratio_arrays(histograms['fig56_r'], "numer_hlt", "denom_hlt")
    table_r = make_hepdata_table_from_arrays(arrays,
                                       table_name ="Fig 56r",
                                       independent_names = ["Cluster Size"],
                                       independent_units =[""],
                                       dependent_name= 'HLT efficiency'
                                      )
    table.add_variable(table_r.variable[1])
    return table


# # Fig 60

# In[176]:


def makeFig60table(histograms):
    result = histograms['results']
    table = Table("LLP Run 2, Run 3 acceptance comparison")
    mH=125
    mS=40
    data = np.array(sorted([[ctau/1000,v["MET200_csc"]/v['denom_csc']] for (mass,ctau),v in results.items() if v['denom_csc']>0 and mass==mS],key=lambda x: x[0]))
    
    ctau = Variable("LLP ctau", is_independent=True, is_binned=False, units="m", values=list(data[:,0]))
    run2_csc = Variable("Run 2 CSC - acceptance", is_independent=False,is_binned=False, units="", values=list(data[:,1]))
    
    data = np.array(sorted([[ctau/1000,v["HMTnominal_csc"]/v['denom_csc']] for (mass,ctau),v in results.items() if v['denom_csc']>0 and mass==mS],key=lambda x: x[0]))
    run3_csc_l1 = Variable("Run 3 CSC L1T - acceptance", is_independent=False, is_binned=False,units="", values=list(data[:,1]))
    
    data = np.array(sorted([[ctau/1000,v["CscLoose_csc"]/v['denom_csc']] for (mass,ctau),v in results.items() if v['denom_csc']>0 and mass==mS],key=lambda x: x[0]))
    run3_csc_hlt = Variable("Run 3 CSC L1T+HLT - acceptance", is_independent=False, is_binned=False,units="", values=list(data[:,1]))
    
    
    data = np.array(sorted([[ctau/1000,v["MET200_dt"]/v['denom_dt']] for (mass,ctau),v in results.items() if v['denom_dt']>100 and mass==mS],key=lambda x: x[0]))
    run2_dt = Variable("Run 2 DT - acceptance", is_independent=False,is_binned=False, units="", values=list(data[:,1]))
    
    data = np.array(sorted([[ctau/1000,v["METDT_dt"]/v['denom_dt']] for (mass,ctau),v in results.items() if v['denom_dt']>100 and mass==mS],key=lambda x: x[0]))
    run3_dt = Variable("Run 3 DT L1T+HLT - acceptance", is_independent=False,is_binned=False, units="", values=list(data[:,1]))
    
    table.add_variable(run2_csc)
    table.add_variable(run3_csc_l1)
    table.add_variable(run3_csc_hlt)
    table.add_variable(run2_dt)
    table.add_variable(run3_dt)
    return table


# In[178]:


makeFig60table(histograms)


# # Fig 61

# In[174]:


def makeFig61table(histograms):
    arrays = compute_ratio_arrays(histograms['fig61'], "numer_l1", "denom")
    table = make_hepdata_table_from_arrays(arrays,
                                       table_name ="Fig 61",
                                       independent_names = ["LLP decay z"],
                                       independent_units =["cm"],
                                       dependent_name= 'L1T Acceptance'
                                      )
    hlt_table = make_hepdata_table_from_arrays(arrays,
                                       table_name ="temp",
                                       independent_names = ["LLP decay z"],
                                       independent_units =["cm"],
                                       dependent_name= 'L1T+HLT Acceptance'
                                      )
    table.add_variable(hlt_table.variables[1])
    return table


# # Fig 62

# In[173]:


def makeFig62table(histograms):
    arrays = compute_ratio_arrays(histograms['fig62'], "numer_dt_L1MET_tight", "denom_dt_L1MET")
    table = make_hepdata_table_from_arrays(arrays,
                                       table_name ="Fig 62",
                                       independent_names = ["LLP decay R"],
                                       independent_units =["cm"],
                                       dependent_name= 'HLT Acceptance'
                                      )
    arrays = compute_ratio_arrays(histograms['fig62'], "numer_dt_L1MET_tight", "denom")
    hlt_table = make_hepdata_table_from_arrays(arrays,
                                           table_name ="temp",
                                           independent_names = ["LLP decay R"],
                                           independent_units =["cm"],
                                           dependent_name= 'L1T+HLT Acceptance'
                                          )
    table.add_variable(hlt_table.variables[1])
    return table


# # Fig 63

# In[172]:


def makeFig63table(histograms):
    arrays = compute_ratio_arrays(histograms['fig63'], "numer_l1", "denom")
    arrays_hlt = compute_ratio_arrays(histograms['fig63'], "numer", "denom")
    table = make_hepdata_table_from_arrays(arrays,
                                       table_name ="Fig 63",
                                       independent_names = ["LLP decay z","LLP decay R"],
                                       independent_units =["cm","cm"],
                                       dependent_name= 'L1T Acceptance'
                                      )
    hlt_table = make_hepdata_table_from_arrays(arrays_hlt,
                                       table_name ="Fig 63",
                                       independent_names = ["LLP decay z","LLP decay R"],
                                       independent_units =["cm","cm"],
                                       dependent_name= 'L1T+HLT Acceptance'
                                      )
    table.add_variable(hlt_table.variables[2])
    return table


# ## Fig 64

# In[171]:


def makeFig64table(histograms):
    arrays = compute_ratio_arrays(histograms['fig64'], "numer_dt_L1MET", "denom_dt_L1MET")
    table = make_hepdata_table_from_arrays(arrays,
                                       table_name ="Fig 64",
                                       independent_names = ["LLP decay z","LLP decay R"],
                                       independent_units =["cm","cm"],
                                       dependent_name= 'HLT Acceptance'
                                      )
    return table


# In[181]:


def makeMDStables(histograms):
    makeFig56table(histograms)
    makeFig60table(histograms)
    makeFig61table(histograms)
    makeFig62table(histograms)
    makeFig63table(histograms)
    makeFig64table(histograms)


# In[ ]:




