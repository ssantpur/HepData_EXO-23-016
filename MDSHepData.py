import os
import pickle
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
hep.style.use("CMS")
import matplotlib.patches as patches

import numpy as np
from typing import Any, Dict, List, Union

from hepdata_lib import Table, Variable, Uncertainty, Submission

with open("data_Martin/EXO-23-016-MDS-hist.pkl",'rb') as f:
    histograms = pickle.load(f)

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
    #axis_indices: List[int] | None = None,
    axis_indices: Union[List[int], None] = None,
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



def to_bins(edges):
    bin_tuples = np.empty(len(edges) - 1, dtype=object)
    bin_tuples[:] = list(zip(edges[:-1], edges[1:]))
    return bin_tuples


def make_hepdata_table_from_arrays(
        arrays: Dict[str, Any],
        table_name: str,
        table_description: str,
        table_location: str,
        table_image: str,
        independent_names: Union[List[str], None] = None,
        independent_units: Union[List[str], None] = None,
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
            val_item = 0 
            down = 0
            up = 1e-9
        else:
            val_item = float(val)
            down = float(ratio_uncert[(0,) + idx])
            up = float(ratio_uncert[(1,) + idx])
        # append as a 2-tuple: (value, (minus, plus))
        dep_values.append((val_item, (down, up)))


    # Build table and Variables
    table = Table(table_name)
    table.description = table_description
    table.location = table_location
    table.add_image(table_image)
    
    #format axis entries from arrays
    axes_entries = [to_bins(x) for x in edges_list]
    axes_entries= np.meshgrid(*axes_entries)

    for axis_i, x_edges in enumerate(edges_list):
        var_name = independent_names[axis_i] if axis_i < len(independent_names) else f"axis_{axis_i}"
        unit = independent_units[axis_i] if axis_i < len(independent_units) else ""
        e_vals = axes_entries[axis_i].flatten().tolist()
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


def makeFig56leftTable(histograms):
    arrays = compute_ratio_arrays(histograms['fig56_l'], "numer_hlt", "denom_hlt")
    table = make_hepdata_table_from_arrays(arrays,
                                           table_name ="HLT efficiency of DT MDS vs ptmiss",
                                           table_description = "The HLT efficiency of the DT MDS triggers as a function of $p_T^{miss}$, for simulated $H \\to S S \\to b\\bar{{b}}\\,b\\bar{{b}}$ events with $m_{H}=125$ GeV, $m_{S}=40$ GeV, and $c\\tau_{S}=1$ m, for 2023 conditions. Events are required to have at least one cluster with more than 50 hits.",
                                           table_location = "Data from Fig. 56 left",
                                           table_image = "data_Martin/MDS_DT_eff_v_MET.pdf",
                                           independent_names = ["$p_T^{miss}$"],
                                           independent_units =["GeV"],
                                           dependent_name= 'HLT efficiency'
                                           )
    return table

def makeFig56rightTable(histograms):    
    arrays = compute_ratio_arrays(histograms['fig56_r'], "numer_hlt", "denom_hlt")
    table = make_hepdata_table_from_arrays(arrays,
                                           table_name ="HLT efficiency of DT MDS vs cluser size",
                                           table_description = "The HLT efficiency of the DT MDS triggers as a function of cluster size, for simulated $H \\to S S \\to b\\bar{{b}}\\,b\\bar{{b}}$ events with $m_{H}=125$ GeV, $m_{S}=40$ GeV, and $c\\tau_{S}=1$ m, for 2023 conditions. Events are required to have $p_T^{miss}>250$ GeV.",
                                           table_location = "Data from Fig. 56 right",
                                           table_image = "data_Martin/MDS_DT_eff_v_cls.pdf",          
                                           independent_names = ["Cluster size"],
                                           independent_units =[""],
                                           dependent_name= 'HLT efficiency'
                                           )
    return table




def makeFig60table(histograms):
    results = histograms['results']
    table = Table("MDS Run 2, Run 3 acceptance comparison")
    table.description = "Comparison of the acceptances in Run 2 and Run 3 for the CSC (left) and DT (right) MDS triggers at the L1T and HLT as functions of the LLP lifetime, for $H \\to S S \\to b\\bar{{b}}\\,b\\bar{{b}}$ events with $m_{H}=125$ GeV and $m_{S}=40$ GeV, for 2023 conditions. The acceptance is defined as the fraction of events that pass the specified selection, given an LLP decay in the fiducial region of the CSCs (left) or DTs (right). The left plot compares the acceptance of the Run 2 strategy of triggering on $p_T^{miss}$ (blue circles), which corresponds to an offline requirement of $>200$ GeV, with that of the Run 3 strategy of triggering on the MDS signature in the CSCs, for both the L1T (L1T+HLT) acceptance is shown with orange squares (red triangles). The right plot compares the acceptance of the Run 2 strategy of triggering on $p_T^{miss}$ (blue circles) with the Run 3 strategy of triggering on the MDS signature in the DTs (red triangles), for L1T+HLT."
    table.location = "Data from Fig. 63"
    table.add_image("data_Martin/MDS_CSC_acc_v_ctau_mH-125_mS-40.pdf")
    table.add_image("data_Martin/MDS_DT_acc_v_ctau_mH-125_mS-40.pdf")

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
    
    table.add_variable(ctau)
    table.add_variable(run2_csc)
    table.add_variable(run3_csc_l1)
    table.add_variable(run3_csc_hlt)
    table.add_variable(run2_dt)
    table.add_variable(run3_dt)
    return table



def makeFig61table(histograms):
    arrays = compute_ratio_arrays(histograms['fig61'], "numer_l1", "denom")
    table = make_hepdata_table_from_arrays(arrays,
                                           table_name ="L1T and L1T+HLT acceptance for CSC MDS",
                                           table_description = "The L1T (blue circles) and L1T+HLT (orange squares) acceptances for the CSC MDS trigger as functions of the LLP decay positions in the $z$-direction, for $H \\to S S \\to b\\bar{{b}}\\,b\\bar{{b}}$ events with $m_{H}=350$ GeV, $m_{S}=80$ GeV, and $c\\tau_{S}=1$ m, for 2023 conditions.",
                                           table_location = "Data from Fig. 64",
                                           table_image = "data_Martin/MDS_CSC_acc_v_Z.pdf",
                                           independent_names = ["LLP decay Z"],
                                           independent_units =["cm"],
                                           dependent_name= 'L1T Acceptance'
                                           )
    arrays = compute_ratio_arrays(histograms['fig61'], "numer", "denom")    
    hlt_table = make_hepdata_table_from_arrays(arrays,
                                               table_name ="temp",
                                               table_description = "",
                                               table_location = "",
                                               table_image = "data_Martin/MDS_CSC_acc_v_Z.pdf",
                                               independent_names = ["LLP decay Z"],
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
                                           table_name ="HLT and L1T+HLT aceptance for DT MDS",
                                           table_description = "The HLT (blue circles) and L1T+HLT (orange squares) acceptances for the DT MDS trigger as functions of the LLP decay positions in the radial direction, for $H \\to S S \\to b\\bar{{b}}\\,b\\bar{{b}}$ events with $m_{H}=350$ GeV, $m_{S}=80$ GeV, and $c\\tau_{S}=1$ m, for 2023 conditions.",
                                           table_location = "Data from Fig. 65",
                                           table_image = "data_Martin/MDS_DT_acc_v_r.pdf",
                                           independent_names = ["LLP decay R"],
                                           independent_units =["cm"],
                                           dependent_name= 'HLT Acceptance'
                                           )
    arrays = compute_ratio_arrays(histograms['fig62'], "numer_dt_L1MET_tight", "denom")
    hlt_table = make_hepdata_table_from_arrays(arrays,
                                               table_name ="temp",
                                               table_description = "",
                                               table_location = "",
                                               table_image = "data_Martin/MDS_DT_acc_v_r.pdf",
                                               independent_names = ["LLP decay R"],
                                               independent_units =["cm"],
                                               dependent_name= 'L1T+HLT Acceptance'
                                               )
    table.add_variable(hlt_table.variables[1])
    return table


# # Fig 63

# In[172]:


def makeFig63leftTable(histograms):
    arrays = compute_ratio_arrays(histograms['fig63'], "numer_l1", "denom")
    table = make_hepdata_table_from_arrays(arrays,
                                           table_name ="2D L1T acceptance for CSC MDS",
                                           table_description = "The L1T acceptance for the CSC MDS trigger as functions of the LLP decay position, for $H \\to S S \\to b\\bar{{b}}\\,b\\bar{{b}}$ events with $m_{H}=350$ GeV, $m_{S}=80$ GeV, and $c\\tau_{S}=1$ m, for 2023 conditions.",
                                           table_location = "Data from Fig. 66 left",
                                           table_image = "data_Martin/MDS_CSC_2D_L1acc_v_rZ.pdf",
                                           independent_names = ["LLP decay Z","LLP decay R"],
                                           independent_units =["cm","cm"],
                                           dependent_name= 'L1T Acceptance'
                                           )
    return table

def makeFig63rightTable(histograms):    
    arrays_hlt = compute_ratio_arrays(histograms['fig63'], "numer", "denom")
    table = make_hepdata_table_from_arrays(arrays_hlt,
                                           table_name ="2D L1T+HLT acceptance for CSC MDS",
                                           table_description = "The L1T+HLT acceptance for the CSC MDS trigger as functions of the LLP decay position, for $H \\to S S \\to b\\bar{{b}}\\,b\\bar{{b}}$ events with $m_{H}=350$ GeV, $m_{S}=80$ GeV, and $c\\tau_{S}=1$ m, for 2023 conditions.",
                                           table_location = "Data from Fig. 66 right",
                                           table_image = "data_Martin/MDS_CSC_2D_HLTacc_v_rZ.pdf",
                                           independent_names = ["LLP decay Z","LLP decay R"],
                                           independent_units =["cm","cm"],
                                           dependent_name= 'L1T+HLT Acceptance'
                                           )
    return table


# ## Fig 64

# In[171]:


def makeFig64table(histograms):
    arrays = compute_ratio_arrays(histograms['fig64'], "numer_dt_L1MET", "denom_dt_L1MET")
    table = make_hepdata_table_from_arrays(arrays,
                                           table_name ="2D HLT acceptance for DT MDS",
                                           table_description = "The HLT acceptance for the DT MDS trigger as a function of the LLP decay position, for $H \\to S S \\to b\\bar{{b}}\\,b\\bar{{b}}$ events with $m_{H}=350$ GeV, $m_{S}=80$ GeV, and $c\\tau_{S}=1$ m, for 2023 conditions. The L1T acceptance that is based on the $p_T^{miss}$ trigger is not included.",
                                           table_location = "Data from Fig. 67",
                                           table_image = "data_Martin/MDS_DT_2D_HLTacc_v_rZ.pdf",
                                           independent_names = ["LLP decay Z","LLP decay R"],
                                           independent_units =["cm","cm"],
                                           dependent_name= 'HLT Acceptance'
                                           )
    return table


# In[181]:


def makeMDStables(histograms):
    makeFig56leftTable(histograms)
    makeFig56rightTable(histograms)
    makeFig60table(histograms)
    makeFig61table(histograms)
    makeFig62table(histograms)
    makeFig63leftTable(histograms)
    makeFig63rightTable(histograms)
    makeFig64table(histograms)

# Create the submission object                                                                                                              
#submission = Submission()

 
# Create output directory early                                                                                                             
#output_dir = "hepdataMartin_output"
#if not os.path.exists(output_dir):
#    os.makedirs(output_dir)

#submission.add_table(makeFig56leftTable(histograms))
#submission.add_table(makeFig56rightTable(histograms))
#submission.add_table(makeFig60table(histograms))
#submission.add_table(makeFig61table(histograms))
#submission.add_table(makeFig62table(histograms))
#submission.add_table(makeFig63leftTable(histograms))
#submission.add_table(makeFig63rightTable(histograms))
#submission.add_table(makeFig64table(histograms))

#submission.create_files(output_dir,remove_old=True)




