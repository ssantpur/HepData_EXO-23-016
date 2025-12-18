#!/usr/bin/env python3
"""
HEPData Generator for ROOT Files

This script extracts histograms from ROOT files and converts them to HEPData format.
It reads a ROOT file containing a TCanvas with multiple histograms and generates
the appropriate YAML files for HEPData submission.

For EXO-23-016

Usage:
    python3 createHepData_all.py

Output:
    - Creates 'hepdata_output/' directory with YAML files
    - Creates 'hepdata_submission.tar.gz' archive for submission

Dependencies:
    - ROOT (PyROOT)
    - hepdata_lib
    - readTest.py (for histogram extraction from canvas)
"""

from __future__ import print_function
import ROOT
import os
import sys
import yaml
import shutil
import subprocess

from hepdata_lib import RootFileReader, Submission, Variable, Uncertainty, Table
from hepdata_lib.root_utils import get_hist_1d_points

#import pickle
#import matplotlib.pyplot as plt
#import mplhep as hep
#import numpy as np
#hep.style.use("CMS")
#import matplotlib.patches as patches


#import numpy as np
#from typing import Any, Dict, List, Union

from MDSHepData import *

# Import the helper function from readTest.py
sys.path.append('.')
from readTest import collect_hists_from_canvas


def makeVariable(plot, label, is_independent, is_binned, is_symmetric, units, CME=13.6, uncertainty=True):
    var = Variable(label, is_independent=is_independent, is_binned=is_binned, units=units)
    var.values = plot["y"]
    if uncertainty:
        unc = Uncertainty("", is_symmetric=is_symmetric)
        unc.values = plot["dy"]
        var.add_uncertainty(unc)
    var.add_qualifier("SQRT(S)", CME, "TeV")
    #var.add_qualifier("HLT rate","2016")                                                                                                                                          
    return var

def check_imagemagick_available():
    """Check if ImageMagick convert command is available"""
    try:
        # Try to run ImageMagick convert command
        result = subprocess.run(['convert', '-version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False

def makeHT430EffTable():
    table = Table("Displaced jet HLT $H_{T}$ > 430 GeV efficiency")
    table.description = "the HLT efficiency for a given event passing the main displaced-jet trigger to satisfy HLT calorimeter $H_{\mathrm{T}}>430~\mathrm{GeV}$  as a function of the offline calorimeter $H_{\mathrm{T}}$. The measurements are performed in data collected in 2022 (green circles), in 2023 before an update of the HCAL gain values and energy response corrections (black squares), and in 2023 after the update (blue triangles)."
    image = "data_Jingyu/DisplacedJets_HT_performance_years.pdf"
    reader = RootFileReader("data_Jingyu/DisplacedJets_HT_performance_years.root")
    HT430_22 = "eff22_HT430"
    HT430_23 = "eff23_HT430"
    HT430_23late = "eff23late_HT430"
    table.location = "Data from Fig. 16 (left)"
    table.add_image(image)
    
    plot_HT430_22 = reader.read_teff(HT430_22)
    plot_HT430_23 = reader.read_teff(HT430_23)
    plot_HT430_23late = reader.read_teff(HT430_23late)
    
    xAxisVar = Variable("$H_{\mathrm{T}}$", is_independent=True, is_binned=False, units="$\mathrm{GeV}$")
    xAxisVar.values =plot_HT430_22["x"]
    table.add_variable(xAxisVar)
    table.add_variable(makeVariable(plot=plot_HT430_22, label="Data 2022", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot=plot_HT430_23, label="Data 2023 before HCAL conditions update", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot=plot_HT430_23late, label="Data 2023 after HCAL conditions update", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    
    return table


def makeHT390EffTable():
    table = Table("Displaced jet HLT $H_{T}$ > 390 GeV efficiency")
    table.description = "the HLT efficiency for a given event passing the main displaced-jet trigger to satisfy HLT calorimeter $H_{\mathrm{T}}>390~\mathrm{GeV}$  as a function of the offline calorimeter $H_{\mathrm{T}}$. The measurements are performed in data collected in 2022 (green circles), in 2023 before an update of the HCAL gain values and energy response corrections (black squares), and in 2023 after the update (blue triangles)."
    image = "data_Jingyu/DisplacedJets_HT_performance_years_HT390.pdf"
    reader = RootFileReader("data_Jingyu/DisplacedJets_HT_performance_years_HT390.root")
    HT390_23 = "eff23_HT390"
    HT390_23late = "eff23late_HT390"
    table.location = "Data from Fig. 16 (right)"
    table.add_image(image)
    
    plot_HT390_23 = reader.read_teff(HT390_23)
    plot_HT390_23late = reader.read_teff(HT390_23late)
    
    xAxisVar = Variable("$H_{\mathrm{T}}$", is_independent=True, is_binned=False, units="$\mathrm{GeV}$")
    xAxisVar.values =plot_HT390_23["x"]
    table.add_variable(xAxisVar)
    table.add_variable(makeVariable(plot=plot_HT390_23, label="Data 2023 before HCAL conditions update", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot=plot_HT390_23late, label="Data 2023 after HCAL conditions update", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    
    return table

def makePt40EffTable():
    table = Table("Displaced jet HLT $p_{T}$ > 40 GeV efficiency")
    table.description = "Efficiency of an offline calorimeter jet to pass the online $p_{\mathrm{T}}$ requirement in displaced-jet triggers, which require $p_{\mathrm{T}}>40\mathrm{GeV}$, in data collected in 2022 (green squares), in 2023 before an update of the HCAL gains and energy response corrections (black filled circles), and in 2023 after the update (blue open circles). The efficiencies measured with QCD multijet simulation are also shown, for 2022 (red triangles) and 2023 (purple triangles) conditions."
    image = "data_Jingyu/DisplacedJets_Run3_jetpt40_Eff_years.pdf"
    reader =RootFileReader("data_Jingyu/DisplacedJets_Run3_jetpt40_Eff_years.root")
    PT40_QCD22 = "eff_Pt40_QCD22"
    PT40_QCD23 = "eff_Pt40_QCD23"
    PT40_data22 = "eff_Pt40_data22"
    PT40_data23 = "eff_Pt40_data23"
    PT40_data23late = "eff_Pt40_data23late"
    table.location = "Data from Fig. 17 (left)"
    table.add_image(image)
    
    plot_PT40_QCD22 = reader.read_teff(PT40_QCD22)
    plot_PT40_QCD23 = reader.read_teff(PT40_QCD23)
    plot_PT40_data22 = reader.read_teff(PT40_data22)
    plot_PT40_data23 = reader.read_teff(PT40_data23)
    plot_PT40_data23late = reader.read_teff(PT40_data23late)
    xAxisVar = Variable("$p_{\mathrm{T}}$", is_independent=True, is_binned=False, units = "$\mathrm{GeV}$")
    xAxisVar.values = plot_PT40_QCD22["x"]
    table.add_variable(xAxisVar)
    table.add_variable(makeVariable(plot=plot_PT40_QCD22, label="QCD multijet (2022)", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot=plot_PT40_QCD23, label="QCD multijet (2023)", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot=plot_PT40_data22, label="Data 2022", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot=plot_PT40_data23, label="Data 2023 before HCAL conditions update", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot=plot_PT40_data23late, label="Data 2023 after HCAL conditions update", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    
    return table
    
def makePtrkEffTable():
    table = Table("Displaced jet HLT tracking requirement efficiency" )
    table.description = "Efficiency of an offline calorimeter jet to have at most one HLT prompt track for 2022 conditions, as a function of the number of offline prompt tracks, in simulated $\\mathrm{H} \\to \\mathrm{SS}$, $\mathrm{S} \\to \mathrm{b\overline{b}}$ signal events where $m_{\mathrm{H}} = 125~\mathrm{GeV}$ and $m_{\mathrm{S}}=40~\mathrm{GeV}$. Two proper decay lengths of the $\mathrm{S}$ particle are shown: $c\\tau=10~\mathrm{mm}$ (green circles) and $c\\tau = 100~\mathrm{mm}$ (blue squares)."
    image = "data_Jingyu/DisplacedJet_Trigger_Run3_signal_prompttrack_veto_eff.pdf"
    reader = RootFileReader("data_Jingyu/DisplacedJet_Trigger_Run3_signal_prompttrack_veto_eff.root")
    table.location = "Data from Fig. 17 (right)"
    table.add_image(image)
    plot10mm = reader.read_teff("eff_10mm")
    plot100mm = reader.read_teff("eff_100mm")
    
    xAxisVar = Variable("Number of offline prompt tracks",is_independent=True, is_binned=False, units="")
    xAxisVar.values = plot10mm["x"]
    table.add_variable(xAxisVar)
    table.add_variable(makeVariable(plot=plot10mm, label="$c\\tau_{0} = 10~\mathrm{mm}$", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot=plot100mm, label="$c\\tau_{0} = 100~\mathrm{mm}$", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    
    return table
    

    
def makeGenEffTable():
    table = Table("Displaced jet HLT tagging efficiency")
    table.description = "The per-parton (quark or lepton) HLT displaced-jet tagging efficiency as a function of the generator-level $L_{xy}$ of the parton is shown for displaced b quarks (blue circles), d quarks (purple triangels), and $\\tau$ leptons (green squares) with $p_{\mathrm{T}}>40~\mathrm{GeV}$ and $|\eta|<2.0$."
    image = "data_Jingyu/DisplacedJets_GenEff_cm.pdf"
    reader = RootFileReader("data_Jingyu/DisplacedJets_GenEff_cm.root")
    table.location = "Data from Fig. 18"
    table.add_image(image)
    plotbb = reader.read_teff("eff_dxy_bb")
    plotdd = reader.read_teff("eff_dxy_dd")
    plottau = reader.read_teff("eff_dxy_tau")
    
    xAxisVar = Variable("Gen.-level parton production vertex $L_{xy}$", is_independent=True, is_binned=False, units="cm")
    xAxisVar.values = plotbb["x"]
    table.add_variable(xAxisVar)
    table.add_variable(makeVariable(plot=plotbb, label="b quark", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot=plotdd, label="d quark", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot=plottau, label="$\\tau$ lepton", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    
    return table
    
def makeGainTable():
    table = Table("Displaced jet trigger efficiency Run 3 v.s. Run 2 ratio")
    table.description = "The ratio between the Run 3 displaced-jet trigger efficiency and the Run 2 displaced jet trigger efficiency as a function of LLP $c\\tau$, in simulated $\\mathrm{H}\\to\\mathrm{SS}$, $\\mathrm{S}\\to\\mathrm{b\overline{b}}$ signal events where $m_{\mathrm{H}}=125~\mathrm{GeV}$ and $m_{\mathrm{S}}=15$ (blue triangles), 40 (green squares), or 55 (red circles)$\mathrm{GeV}$. The Run 3 displaced trigger efficiencies are measured for 2022 conditions."
    image = "data_Jingyu/DisplacedJet_Trigger_Run3vsRun2_efficiency_gain.pdf"
    reader = RootFileReader("data_Jingyu/DisplacedJet_Trigger_Run3vsRun2_efficiency_gain.root")
    table.location = "Data from Fig. 19"
    table.add_image(image)
    plotM55 = reader.read_graph("gra_gain_M55")
    plotM40 = reader.read_graph("gra_gain_M40")
    plotM15 = reader.read_graph("gra_gain_M15")
    
    xAxisVar = Variable("$c\\tau$", is_independent=True, is_binned=False, units="cm")
    xAxisVar.values = plotM55["x"]
    table.add_variable(xAxisVar)
    table.add_variable(makeVariable(plot=plotM55, label="$m_{\mathrm{S}}=55~\mathrm{GeV}$", is_independent=False, is_binned=False, is_symmetric=False, units="", uncertainty=False))
    table.add_variable(makeVariable(plot=plotM40, label="$m_{\mathrm{S}}=40~\mathrm{GeV}$", is_independent=False, is_binned=False, is_symmetric=False, units="", uncertainty=False))
    table.add_variable(makeVariable(plot=plotM15, label="$m_{\mathrm{S}}=15~\mathrm{GeV}$", is_independent=False, is_binned=False, is_symmetric=False, units="", uncertainty=False))
    return table

def makeHcalTowerEffTable():
    table = Table("L1T HCAL delayed tower efficiency vs Timing shift [ns]")
    table.description = "The L1T HCAL trigger tower efficiency of the delayed timing towers in 2023 HCAL timing-scan data, with efficiencies split by trigger "\
        "towers centered at $\\eta\\approx 0$ (blue circles), 0.65 (red squares), 1.26 (black triangles), and with width $\\Delta \\eta = 0.087$. The sharp rise in "\
        "efficiency between timing delays of 0-6 ns is expected, as the prompt timing range includes pulses up to and including those recorded at a 6 ns "\
        "arrival time (reported in half-ns steps by the TDC), demonstrating the timing trigger performance. The delayed timing towers must have at least one delayed "\
        "cell, no prompt cells, and energy ${>} 4$ GeV. The efficiency is calculated relative to towers with any valid timing code, meaning the tower contains at least "\
        "one cell with energy ${>} 4$ GeV and a TDC code of prompt, slightly delayed, or very delayed. Multiple delayed or displaced towers are required for the "\
        "HCAL-based displaced- and delayed-jet L1T to pass, and this shows the efficiency at a per-tower level relative to incoming pulse timing."
    image = "data_Gillian/QIE_Tower_ieta_fg123_effs_diff_ieta_nopreliminary.pdf"
    reader = RootFileReader("data_Gillian/QIE_Tower_ieta_fg123_effs_diff_ieta.root")
    # Tefficiencies from the ROOT file
    eta0 = "time_flagged_eta_0;1"
    eta0pt65 = "time_flagged_eta_0.65;1"
    eta1pt26 = "time_flagged_eta_1.26;1"
    table.location = "Data from Fig. 21"
    table.add_image(image)

    plot_eta0 = reader.read_teff(eta0)
    plot_eta0pt65 = reader.read_teff(eta0pt65)
    plot_eta1pt26 = reader.read_teff(eta1pt26)

    xAxisVar = Variable("Timing shift", is_independent=True, is_binned=False, units="ns")
    xAxisVar.values = plot_eta0["x"]
    table.add_variable(xAxisVar)

    table.add_variable(makeVariable(plot=plot_eta0, label="Delayed tower, $\eta=0$", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot=plot_eta0pt65, label="Delayed tower, $\eta=0.65$", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot=plot_eta1pt26, label="Delayed tower, $\eta=1.26$", is_independent=False, is_binned=False, is_symmetric=False, units=""))

    return table

def makeHcalLLPflaggedJetEffTable():
    table = Table("L1T efficiency of LLP-flagged jets vs L1 jet ET [GeV]")
    table.description = "The L1T efficiency of the LLP jet trigger in 2023 HCAL timing-scan data. The HCAL LLP-flagged L1T trigger delayed jet fraction versus jet "\
        "$E_T$ during the 2023 HCAL phase scan demonstrates that the delayed jet fraction approaches unity as the timing shift increases. "\
        "The results are inclusive in pseudorapidity for the HCAL barrel, corresponding to $|{\\eta}| < 1.35$. The fraction of LLP-flagged L1 jets is compared to all L1 jets "\
        "from a data set of events enriched with jets or $p_T^{\\text{miss}}$. No explicit selection criterion is applied on the jet $E_T$, though the implicit requirement for a jet to "\
        "have at least two cells with $E_T > 4$ GeV shapes the resulting jet trigger efficiency curve."
    image = "data_Gillian/Jet_Et_all_delay_nopreliminary.pdf"
    reader = RootFileReader("data_Gillian/Jet_Et_all_delay.root")
    # Tefficiencies from the ROOT file
    timing_m4 = "timing_shift_-4ns;1"
    timing_m2 = "timing_shift_-2ns;1"
    timing_0 = "timing_shift_0ns;1"
    timing_2 = "timing_shift_2ns;1"
    timing_4 = "timing_shift_4ns;1"
    timing_6 = "timing_shift_6ns;1"
    timing_8 = "timing_shift_8ns;1"
    timing_10 = "timing_shift_10ns;1"
    table.location = "Data from Fig. 22"
    table.add_image(image)

    plot_timing_m4 = reader.read_teff(timing_m4)
    plot_timing_m2 = reader.read_teff(timing_m2)
    plot_timing_0 = reader.read_teff(timing_0)
    plot_timing_2 = reader.read_teff(timing_2)
    plot_timing_4 = reader.read_teff(timing_4)
    plot_timing_6 = reader.read_teff(timing_6)
    plot_timing_8 = reader.read_teff(timing_8)
    plot_timing_10 = reader.read_teff(timing_10)

    xAxisVar = Variable("L1 jet ET", is_independent=True, is_binned=False, units="GeV")
    xAxisVar.values = plot_timing_m4["x"]
    table.add_variable(xAxisVar)

    table.add_variable(makeVariable(plot=plot_timing_m4, label="Timing shift = -4 ns", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot=plot_timing_m2, label="Timing shift = -2 ns", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot=plot_timing_0, label="Timing shift = 0 ns", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot=plot_timing_2, label="Timing shift = 2 ns", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot=plot_timing_4, label="Timing shift = 4 ns", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot=plot_timing_6, label="Timing shift = 6 ns", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot=plot_timing_8, label="Timing shift = 8 ns", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot=plot_timing_10, label="Timing shift = 10 ns", is_independent=False, is_binned=False, is_symmetric=False, units=""))

    return table

def makeHcalL1JetHTEffTable(xvar):
    if xvar == "HT": 
        name = "event $H_T$"
        image = "data_Gillian/Plotefficiency_eventHT_log_HLT_v3_MC_eventHT_L1effs_noprelim.pdf"
        location = "left"
        reader = RootFileReader("data_Gillian/Figures_Plotefficiency_eventHT_log_HLT_v3_MC_eventHT_L1effs.root")
    elif xvar == "jet": 
        name = "jet $p_T$"
        image = "data_Gillian/Plotefficiency_perJet_Pt_log_HLT_v3_MC_L1effs_noprelim.pdf"
        location = "right"
        reader = RootFileReader("data_Gillian/Figures_Plotefficiency_perJet_Pt_log_HLT_v3_MC_L1effs.root")
    table = Table("L1T efficiency of HCAL based-LLP triggers vs. " + name)

    # Tefficiencies from the ROOT file
    if xvar == "HT": 
        LLP350 = "Plotefficiency_eventHT_log_HLT_v3_MC_eventHT_L1effs_350;1"
        LLP125 = "Plotefficiency_eventHT_log_HLT_v3_MC_eventHT_L1effs_125;1"
        table.description = "The L1T efficiency of the HCAL-based LLP jet triggers, as a function of event $H_T$, for $H \\to SS \\to "\
        "b\\bar{b}b\\bar{b}$ events with $m_{H}=350$ GeV, $m_{S}=80$ GeV, and $c\\tau_{S}=0.5$ m (light blue circles) and $m_{H}=125$ GeV, $m_{S}=50$ GeV, and "\
        "$c\\tau_{S}=3$ m (purple triangles), for 2023 conditions. The trigger efficiency is evaluated for LLPs decaying in HB depths 3 or 4, corresponding to "\
        "$214.2< R<295$ cm and $|{\\eta}|< 1.26$. These LLPs are also required to be matched to an offline jet in HB."

    elif xvar == "jet": 
        LLP350 = "Plotefficiency_perJet_Pt_log_HLT_v3_MC_L1effs_350;1"
        LLP125 = "Plotefficiency_perJet_Pt_log_HLT_v3_MC_L1effs_125;1"
        table.description = "The L1T efficiency of the HCAL-based LLP jet triggers, as a function of jet $p_T$, for $H \\to SS \\to "\
        "b\\bar{b}b\\bar{b}$ events with $m_{H}=350$ GeV, $m_{S}=80$ GeV, and $c\\tau_{S}=0.5$ m (light blue circles) and $m_{H}=125$ GeV, $m_{S}=50$ GeV, and "\
        "$c\\tau_{S}=3$ m (purple triangles), for 2023 conditions. The trigger efficiency is evaluated for LLPs decaying in HB depths 3 or 4, corresponding to "\
        "$214.2< R<295$ cm and $|{\\eta}|< 1.26$. These LLPs are also required to be matched to an offline jet in HB."

    table.location = "Data from Fig. 23 " + location
    table.add_image(image)
    plot_LLP350 = reader.read_teff(LLP350)
    plot_LLP125 = reader.read_teff(LLP125)

    if xvar == "jet": xAxisVar = Variable("Jet $pT$", is_independent=True, is_binned=False, units="GeV")
    elif xvar == "HT": xAxisVar = Variable("$Event HT$", is_independent=True, is_binned=False, units="GeV")
    xAxisVar.values = plot_LLP350["x"]
    table.add_variable(xAxisVar)

    table.add_variable(makeVariable(plot=plot_LLP350, label="$m_H = 350$ GeV, $m_S = 80$ GeV, $c\\tau = 0.5$ m", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot=plot_LLP125, label="$m_H = 125$ GeV, $m_S = 50$ GeV, $c\\tau = 3$ m", is_independent=False, is_binned=False, is_symmetric=False, units=""))

    return table

def makeHcalL1DecayREffTable():
    table = Table("L1T efficiency of HCAL bsaed-LLP triggers vs. LLP decay R")
    table.description = "The L1T efficiency of the HCAL-based LLP jet triggers as a function of LLP decay radial position $R$ for $H \\to SS \\to b\\bar{b}b\\bar{b}$ events with "\
        "$m_{H}=350$ GeV, $m_{S}=80$ GeV, and $c\\tau_{S}=0.5$ m (light blue circles) and $m_{H}=125$ GeV, $m_{S}=50$ GeV, and $c\\tau_{S}=3$ m (purple "\
        "triangles), for 2023 conditions. The trigger efficiency is evaluated for LLPs within $|{\\eta}| <1.26$ where either the LLP or its decay products are matched to an "\
        "offline jet in HB with $p_T>100$ GeV."
    image = "data_Gillian/Plotefficiency_perJet_MatchedLLP_DecayR_log_HLT_v3_MC_jetE100_L1effs_noprelim.pdf"
    reader = RootFileReader("data_Gillian/Figures_Plotefficiency_perJet_MatchedLLP_DecayR_log_HLT_v3_MC_jetE100_L1effs.root")
    # Tefficiencies from the ROOT file
    LLP350 = "Plotefficiency_perJet_MatchedLLP_DecayR_log_HLT_v3_MC_jetE100_L1effs_lin;1"
    LLP125 = "Plotefficiency_perJet_MatchedLLP_DecayR_log_HLT_v3_MC_jetE100_L1effs_lin;2"
    table.location = "Data from Fig. 24"
    table.add_image(image)

    plot_LLP350 = reader.read_teff(LLP350)
    plot_LLP125 = reader.read_teff(LLP125)

    xAxisVar = Variable("LLP Decay R", is_independent=True, is_binned=False, units="cm")
    xAxisVar.values = plot_LLP350["x"]
    table.add_variable(xAxisVar)

    table.add_variable(makeVariable(plot=plot_LLP350, label="$m_H = 350$ GeV, $m_S = 80$ GeV, $c\\tau = 0.5$ m", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot=plot_LLP125, label="$m_H = 125$ GeV, $m_S = 50$ GeV, $c\\tau = 3$ m", is_independent=False, is_binned=False, is_symmetric=False, units=""))

    return table

def makeCalRatioJetEffTable():
    table = Table("HLT efficiency of CalRatio trigger vs. leading jet NHEF")
    table.description = "The HLT efficiency of the CalRatio trigger as a function of the leading PF jet NHEF in 2024 data, measured with respect to a logical " \
        "OR of the HCAL-based LLP L1 jet triggers (left). Events are required to have $H_{\mathrm{T}} > 200 \\,\mathrm{GeV}$ and the leading jet is required to have $p_{\mathrm{T}} > 60 \\,\mathrm{GeV}$ and $|\eta| < 1.5$, which are " \
        "equivalent to the respective HLT jet object selections. The signal distributions additionally require the leading jet to be matched to an LLP decaying " \
        "anywhere inside the barrel calorimeter volume ($129 < R < 295 \\,\mathrm{cm}$)."
    image = "data_Kiley/calratio_efficiency.pdf"
    table.add_image(image)
    table.location = "Data from Fig. 25 left"

    # Tefficiencies from the ROOT file
    reader = RootFileReader("data_Kiley/calratio_efficiency.root")
    plot_efficiency = reader.read_teff("data2024;1")

    xAxisVar = Variable("Leading jet neutral hadron energy fraction", is_independent=True, is_binned=False, units="")
    xAxisVar.values = plot_efficiency["x"]
    table.add_variable(xAxisVar)

    table.add_variable(makeVariable(plot=plot_efficiency, label="Data", is_independent=False, is_binned=False, is_symmetric=False, units=""))

    return table

def makeCalRatioJetDistributionTable():
    table = Table("Distribution of leading jet neutral hadron energy fraction")
    table.description = "Distribution of the leading PF jet NHEF (right) in 2024 data (black circles), W$\\to l\\nu$ " \
        "background simulation for 2024 conditions (red squares), and $H \\to SS \\to b\\bar{b}b\\bar{b}$ signal simulation for 2023 conditions (blue and purple " \
        "triangles). Events are required to have $H_{\mathrm{T}} > 200\\,\mathrm{GeV}$ and the leading jet is required to have $p_{\mathrm{T}} > 60\\,\mathrm{GeV}$ and $|\eta| < 1.5$, which are " \
        "equivalent to the respective HLT jet object selections. The signal distributions additionally require the leading jet to be matched to an LLP decaying " \
        "anywhere inside the barrel calorimeter volume ($129 < R < 295\\,\mathrm{cm}$). The clear separation between the displaced signal and the prompt background " \
        "in the plot motivates the development of the CalRatio trigger."
    image = "data_Kiley/calratio_distribution.pdf"
    table.add_image(image)
    table.location = "Data from Fig. 25 right"

    # THists from the ROOT file
    reader = RootFileReader("data_Kiley/calratio_distribution.root")
    data   = reader.read_hist_1d("data2024;1")
    bkg    = reader.read_hist_1d("background_wplusjets;1")
    sig350 = reader.read_hist_1d("signal_HToSSTo4b_350_80_0p5m;1")
    sig125 = reader.read_hist_1d("signal_HToSSTo4b_125_50_3m;1")

    xAxisVar = Variable("Leading jet neutral hadron energy fraction", is_independent=True, is_binned=True, units="")
    xAxisVar.values = data["x_edges"]
    table.add_variable(xAxisVar)

    table.add_variable(makeVariable(plot=data, label="Data", is_independent=False, is_binned=False, is_symmetric=True, units=""))
    table.add_variable(makeVariable(plot=bkg, label="$\mathrm{W} \\to l\\nu+\mathrm{jets}$", is_independent=False, is_binned=False, is_symmetric=True, units=""))
    table.add_variable(makeVariable(plot=sig350, label="$\mathrm{H} \\to \mathrm{SS} \\to 4\mathrm{b}$ (350 GeV, 80 GeV, 0.5 m)", is_independent=False, is_binned=False, is_symmetric=True, units=""))
    table.add_variable(makeVariable(plot=sig125, label="$\mathrm{H} \\to \mathrm{SS} \\to 4\mathrm{b}$ (125 GeV, 50 GeV, 3 m)", is_independent=False, is_binned=False, is_symmetric=True, units=""))

    return table

def makeDisplacedMuonL1EffTable(xvar):
    table_title = xvar
    table = Table("L1T efficiency vs displaced muon $\mathrm{d_{0}}$ in "+table_title)

    if xvar=="BMTF":
        location = "upper left"
        table.description = "The BMTF L1T efficiencies for beamspot-constrained and beamspot-unconstrained $\mathrm{p_{T}}$ assignment " \
            "algorithms for L1T $\mathrm{p_{T}} > 10\mathrm{GeV}$ with respect to generator-level muon track $\mathrm{d_{0}}$, obtained " \
            "using a sample which produces LLPs that decay to dimuons. The L1T algorithms and data-taking conditions correspond to 2024. A selection " \
            "on the generator-level muon track $\mathrm{p_{T}} > 15\mathrm{GeV}$ is applied to show the performance at the efficiency plateau. The " \
            "generator-level muon tracks are extrapolated to the second muon station to determine the $\eta^{\mathrm{gen}}_{\mathrm{st2}}$ values that " \
            "are used in the plot. The solid markers show the new vertex-unconstrained algorithm performance, while the hollow markers show the default " \
            "beamspot-constrained algorithm performance."
        image = "data_Efe/Figure39_upper_left.pdf"
        reader = RootFileReader("data_Efe/Figure39_upper_left.root")
        prompt = "eff_dxy_BMTF;1"
        disp = "eff_dxy_NN_BMTF;1"
        label_prompt = "BMTF beamspot-constrained efficiency"
        label_disp = "BMTF beamspot-unconstrained efficiency"
    elif xvar=="OMTF":
        location = "upper right"
        table.description = "The OMTF L1T efficiencies for beamspot-constrained and beamspot-unconstrained $\mathrm{p_{T}}$ assignment " \
            "algorithms for L1T $\mathrm{p_{T}} > 10\mathrm{GeV}$ with respect to generator-level muon track $\mathrm{d_{0}}$, obtained " \
            "using a sample which produces LLPs that decay to dimuons. The L1T algorithms and data-taking conditions correspond to 2024. A selection " \
            "on the generator-level muon track $\mathrm{p_{T}} > 15\mathrm{GeV}$ is applied to show the performance at the efficiency plateau. The " \
            "generator-level muon tracks are extrapolated to the second muon station to determine the $\eta^{\mathrm{gen}}_{\mathrm{st2}}$ values that " \
            "are used in the plot. The solid markers show the new vertex-unconstrained algorithm performance, while the hollow markers show the default " \
            "beamspot-constrained algorithm performance."
        image = "data_Efe/Figure39_upper_right.pdf"
        reader = RootFileReader("data_Efe/Figure39_upper_right.root")
        prompt = "eff_dxy_OMTF;1"
        disp = "eff_dxy_NN_OMTF;1"
        label_prompt = "OMTF beamspot-constrained efficiency"
        label_disp = "OMTF beamspot-unconstrained efficiency"
    elif xvar=="EMTF":
        location = "lower"
        table.description = "The EMTF L1T efficiencies for beamspot-constrained and beamspot-unconstrained $\mathrm{p_{T}}$ assignment " \
            "algorithms for L1T $\mathrm{p_{T}} > 10\mathrm{GeV}$ with respect to generator-level muon track $\mathrm{d_{0}}$, obtained " \
            "using a sample which produces LLPs that decay to dimuons. The L1T algorithms and data-taking conditions correspond to 2024. A selection " \
            "on the generator-level muon track $\mathrm{p_{T}} > 15\mathrm{GeV}$ is applied to show the performance at the efficiency plateau. The " \
            "generator-level muon tracks are extrapolated to the second muon station to determine the $\eta^{\mathrm{gen}}_{\mathrm{st2}}$ values that " \
            "are used in the plot. The solid markers show the new vertex-unconstrained algorithm performance, while the hollow markers show the default " \
            "beamspot-constrained algorithm performance. In the EMTF plot, the different colors show different $|\eta|$ " \
            "regions: $1.24 < \eta^{\mathrm{gen}}_{\mathrm{st2}} < 1.6$ (blue), $1.6 < \eta^{\mathrm{gen}}_{\mathrm{st2}} < 2.0$ (red)."
        image = "data_Efe/Figure39_lower.pdf"
        reader = RootFileReader("data_Efe/Figure39_lower.root")
        prompt = "eff_dxy_EMTF1;1"
        disp = "eff_dxy_NN_EMTF1;1"
        prompt2 = "eff_dxy_EMTF2;1"
        disp2 = "eff_dxy_NN_EMTF2;1"
        label_prompt = "EMTF ($1.24 < \eta^{\mathrm{gen}}_{\mathrm{st2}} < 1.6$) beamspot-constrained efficiency"
        label_disp = "EMTF ($1.24 < \eta^{\mathrm{gen}}_{\mathrm{st2}} < 1.6$) beamspot-unconstrained efficiency"
        label_prompt2 = "EMTF ($1.6 < \eta^{\mathrm{gen}}_{\mathrm{st2}} < 2.0$) beamspot-constrained efficiency"
        label_disp2 = "EMTF ($1.6 < \eta^{\mathrm{gen}}_{\mathrm{st2}} < 2.0$) beamspot-unconstrained efficiency"
    else:
        raise ValueError("Unexpected input to function makeDelayedDiPhotonHistTable()")

    table.location = "Data from Fig. 39 "+location
    table.add_image(image)

    if xvar=="EMTF":
        plot_prompt = reader.read_graph(prompt)
        plot_disp = reader.read_graph(disp)
        plot_prompt2 = reader.read_graph(prompt2)
        plot_disp2 = reader.read_graph(disp2)
    else:
        plot_prompt = reader.read_graph(prompt)
        plot_disp = reader.read_graph(disp)

    xAxisVar = Variable("Gen.-level muon track d_{0}", is_independent=True, is_binned=False, units="cm")
    xAxisVar.values = plot_prompt["x"]
    table.add_variable(xAxisVar)

    table.add_variable(makeVariable(plot=plot_prompt, label=label_prompt, is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot=plot_disp, label=label_disp, is_independent=False, is_binned=False, is_symmetric=False, units=""))
    if xvar=="EMTF":
        table.add_variable(makeVariable(plot=plot_prompt2, label=label_prompt2, is_independent=False, is_binned=False, is_symmetric=False, units=""))
        table.add_variable(makeVariable(plot=plot_disp2, label=label_disp2, is_independent=False, is_binned=False, is_symmetric=False, units=""))

    return table


def makeDelayedDiPhotonHistTable(xvar):
    table_title = 'barrel' if xvar=='eb' else 'endcap'
    table = Table("ECAL crystal seed time delay for LLP signature in "+table_title)

    if xvar=="eb":
        location = "left"
        table.description = "The ECAL time delay of the $\\mathrm{e/\\gamma}$ L1 seeds in the barrel. The distributions are "\
            "shown for $\\mathrm{Z\\ \\rightarrow\\ ee}$ simulation and $\\mathrm{\\chi^{0}\\ c\\tau}$ values of "\
            "$\\mathrm{3\\ cm,\\ 30\\ cm}$ and $\\mathrm{3\\ m}$, "\
            "assuming the singlet-triplet Higgs dark portal model ($\\mathrm{\\chi^{\\pm}\\ \\rightarrow\\ \\chi^{0} \\ell^{\\pm} \\nu}$, "\
            "where the $\\mathrm{\\chi^{\\pm}}$ has a mass of $\\mathrm{220\\ GeV}$ and the $\\mathrm{\\chi^{0}}$ has a mass of "\
            "$\\mathrm{200\\ GeV}$), for 2023 "\
            "conditions. The distributions are normalized to unity."
    elif xvar=="ee":
        location = "right"
        table.description = "The ECAL time delay of the $\\mathrm{e/\\gamma}$ L1 seeds in the endcap. The distributions are "\
            "shown for $\\mathrm{Z\\ \\rightarrow\\ ee}$ simulation and $\\mathrm{\\chi^{0}\\ c\\tau}$ values of "\
            "$\\mathrm{3\\ cm,\\ 30\\ cm}$ and $\\mathrm{3\\ m}$, "\
            "assuming the singlet-triplet Higgs dark portal model ($\\mathrm{\\chi^{\\pm}\\ \\rightarrow\\ \\chi^{0} \\ell^{\\pm} \\nu}$, "\
            "where the $\\mathrm{\\chi^{\\pm}}$ has a mass of $\\mathrm{220\\ GeV}$ GeV and the $\\mathrm{\\chi^{0}}$ has a mass of "\
            "$\\mathrm{200\\ GeV}$ GeV), for 2023 "\
            "conditions. The distributions are normalized to unity."
    else:
        raise ValueError("Unexpected input to function makeDelayedDiPhotonHistTable()")
    
    table.location = "Data from Fig. 31 "+location
    table.add_image(f"data_DelayedDiPhoton_SahasransuAR/tdelay_ecal_w_d0_{xvar}.pdf")

    reader = RootFileReader(f"data_DelayedDiPhoton_SahasransuAR/tdelay_ecal_w_d0_{xvar}.root")
    dymc = reader.read_hist_1d(f"genmatched_ph_seedtime_{xvar}_rebinned;1")
    llp3cm = reader.read_hist_1d(f"genmatched_ph_seedtime_{xvar}_rebinned;2")
    llp30cm = reader.read_hist_1d(f"genmatched_ph_seedtime_{xvar}_rebinned;3")
    llp3m = reader.read_hist_1d(f"genmatched_ph_seedtime_{xvar}_rebinned;4")

    xAxisVar = Variable("seed time", is_independent=True, is_binned=True, units="ns")
    xAxisVar.values = dymc["x_edges"]
    table.add_variable(xAxisVar)

    table.add_variable(makeVariable(plot=dymc, label="Z $\\rightarrow$ ee", is_independent=False, is_binned=False, is_symmetric=True, units=""))
    table.add_variable(makeVariable(plot=llp3cm, label="$\\mathrm{\\chi^{\\pm}\\ \\rightarrow\\ \\chi^{0} \\ell^{\\pm} \\nu},\\ c\\tau\\ =\\ $3 cm", is_independent=False, is_binned=False, is_symmetric=True, units=""))
    table.add_variable(makeVariable(plot=llp30cm, label="$\\mathrm{\\chi^{\\pm}\\ \\rightarrow\\ \\chi^{0} \\ell^{\\pm} \\nu},\\ c\\tau\\ =\\ $30 cm", is_independent=False, is_binned=False, is_symmetric=True, units=""))
    table.add_variable(makeVariable(plot=llp3m, label="$\\mathrm{\\chi^{\\pm}\\ \\rightarrow\\ \\chi^{0} \\ell^{\\pm} \\nu},\\ c\\tau\\ =\\ $3 m", is_independent=False, is_binned=False, is_symmetric=True, units=""))

    return table

def makeDelayedDiPhotonDataRateTable():
    table = Table("Delayed Di-Photon HLT rate. with intergated luminosity")

    reader = RootFileReader("data_DelayedDiPhoton_SahasransuAR/ratewintlumi.root")
    table.description = "The HLT rate (blue points) of the delayed-diphoton trigger for a few representative runs in the first data collected in 2023, "\
        "corresponding to an integrated luminosity of $\\mathrm{4.2\\ fb^{-1}}$, compared with the PU during the same data-taking period (red points), "\
        "as a function of integrated luminosity. The rate decreases nonlinearly during a single fill as a result of the increasing crystal "\
        "opacity. It recovers by the start of the next fill with $\\mathrm{<\\ 1\\%}$ reduction in rate between the fills. The rate generally "\
        "increased throughout the year because of periodic online calibrations to mitigate the loss in trigger efficiency, which was produced "\
        "as a result of the ECAL crystal radiation damage."

    table.location = "Data from Fig. 32 left"
    table.add_image("data_DelayedDiPhoton_SahasransuAR/ratewintlumi.pdf")

    rate = reader.read_graph("rate;1")
    lumi = reader.read_graph("intlumi;1")

    xAxisVar = Variable("Integrated luminosity", is_independent=True, is_binned=False, units="$fb^{-1}$")
    xAxisVar.values = rate["x"]

    table.add_variable(xAxisVar)
    table.add_variable(makeVariable(plot = rate, label = "HLT rate", is_independent=False, is_binned=False, is_symmetric=True, units="Hz"))
    table.add_variable(makeVariable(plot = lumi, label = "PU / 9.04", is_independent=False, is_binned=False, is_symmetric=True, units=""))

    return table

def makeDelayedDiPhotonDataEffTable(xvar):
    table = Table("Delayed Di-Photon L1+HLT eff. with "+xvar)

    if xvar=="seed time ($\mathrm{e_{2}}$)":
        units = "ns"
        location = "Data from Fig. 33"
        data = "tid_elprobe_inZwindow_seedtime_rebinned_clone;1"
        image = "data_DelayedDiPhoton_SahasransuAR/hltdipho10t1ns_eff_seedtime.pdf"
        reader = RootFileReader("data_DelayedDiPhoton_SahasransuAR/hltdipho10t1ns_eff_seedtime.root")
        table.description = "The L1T+HLT efficiency of the delayed-diphoton trigger as a function of the subleading probe electron "\
            "($\\mathrm{e_2}$) supercluster seed time, measured with data collected in 2023. At the HLT, the subleading "\
            "$\\mathrm{e/\\gamma}$ supercluster ($\\mathrm{e/\\gamma_2}$) is required to have $\\mathrm{E_T\\ >\\ 12\\ GeV}$, "\
            "$\\mathrm{|\\eta|\\ <\\ 2.1}$, and a seed time $\\mathrm{>\\ 1\\ ns}$. The trigger is fully efficient above $\\mathrm{1\\ ns}$."

    elif xvar=="$p_{T}$ ($\mathrm{e_{2}}$)":
        units = "GeV"
        location = "Data from Fig. 34 left"
        data = "tid1ns_elprobe_inZwindow_pt_rebinned_clone;1"
        image = "data_DelayedDiPhoton_SahasransuAR/hltdipho10t1ns_eff_pt.pdf"
        reader = RootFileReader("data_DelayedDiPhoton_SahasransuAR/hltdipho10t1ns_eff_pt.root")
        table.description = "The L1T+HLT efficiency of the delayed-diphoton trigger as a function of subleading probe electron "\
        "($\\mathrm{e_2}$) \\mathrm{p_T}, measured with data collected in 2023. At the HLT, the subleading $\\mathrm{e/\\gamma}$ "\
        "supercluster ($\\mathrm{e/\\gamma_2}$) is required to have $\\mathrm{E_T\\ >\\ 12\\ GeV}$, $\\mathrm{|\\eta|\\ <\\ 2.1}$, and a "\
        "seed time $\\mathrm{>\\ 1\\ ns}$. The efficiency rises sharply for $\\mathrm{p_T\\ >\\ 12\\ GeV}$ and plateaus for "\
        "$\\mathrm{p_T\\ >\\ 35\\ GeV}$. The slow rise in between is from additional L1 $\\mathrm{H_T}$ requirements."

    elif xvar=="$\eta$ ($\mathrm{e_{2}}$)":
        units = ""
        location = "Data from Fig. 34 right"
        data = "tid1ns_elprobe_inZwindow_eta_rebinned_clone;1"
        image = "data_DelayedDiPhoton_SahasransuAR/hltdipho10t1ns_eff_eta.pdf"
        reader = RootFileReader("data_DelayedDiPhoton_SahasransuAR/hltdipho10t1ns_eff_eta.root")
        table.description = "The L1T+HLT efficiency of the delayed-diphoton trigger as a function of subleading probe electron "\
        "($\\mathrm{e_2}$) $\\mathrm{\eta}$, measured with data collected in 2023. At the HLT, the subleading $\\mathrm{e/\\gamma}$ "\
        "supercluster ($\\mathrm{e/\\gamma_2}$) is required to have $\\mathrm{E_T\\ >\\ 12\\ GeV}$, $\\mathrm{|\\eta|\\ <\\ 2.1}$, "\
        "and a seed time $\\mathrm{>\\ 1\\ ns}$. The trigger is efficient in the region $\\mathrm{|\\eta|\\ <\\ 2.1}$."
        
    else:
        raise ValueError("Unexpected input to function makeDelayedDiPhotonDataEffTable()")

    table.location = location
    table.add_image(image)

    data = reader.read_teff(data)

    xAxisVar = Variable(xvar, is_independent=True, is_binned=False, units=units)
    xAxisVar.values = data["x"]

    table.add_variable(xAxisVar)
    table.add_variable(makeVariable(plot = data, label = "Data (2023)", is_independent=False, is_binned=False, is_symmetric=False, units=""))

    return table

def extract_histogram_data(hist):
    """
    Extract data from a ROOT histogram including bin centers, values, and uncertainties
    """
    n_bins = hist.GetNbinsX()
    
    # Extract bin centers (x values)
    x_values = []
    x_low = []
    x_high = []
    
    for i in range(1, n_bins + 1):
        bin_center = hist.GetBinCenter(i)
        bin_low = hist.GetBinLowEdge(i)
        bin_high = hist.GetBinLowEdge(i) + hist.GetBinWidth(i)
        
        x_values.append(bin_center)
        x_low.append(bin_low)
        x_high.append(bin_high)
    
    # Extract y values and uncertainties
    y_values = []
    y_errors = []
    
    for i in range(1, n_bins + 1):
        y_val = hist.GetBinContent(i)
        y_err = hist.GetBinError(i)
        
        y_values.append(y_val)
        y_errors.append(y_err)
    
    return {
        'x_centers': x_values,
        'x_low': x_low,
        'x_high': x_high,
        'y_values': y_values,
        'y_errors': y_errors
    }

def create_binned_variable(name, x_low, x_high, units="", is_independent=True):
    """
    Create a binned variable for HEPData
    """
    var = Variable(name, is_independent=is_independent, is_binned=True, units=units)
    
    # Create bin edges from low and high values
    bin_edges = []
    for i in range(len(x_low)):
        bin_edges.append([x_low[i], x_high[i]])
    
    var.values = bin_edges
    return var

def create_dependent_variable(name, y_values, y_errors, units="", uncertainty_label="Statistical uncertainty", CME=13.6):
    """
    Create a dependent variable with uncertainties for HEPData
    """
    var = Variable(name, is_independent=False, is_binned=False, units=units)
    var.values = y_values
    
    # Add statistical uncertainties
    if y_errors and any(err > 0 for err in y_errors):
        unc = Uncertainty(uncertainty_label, is_symmetric=True)
        unc.values = y_errors
        var.add_uncertainty(unc)

    var.add_qualifier("SQRT(S)", CME, "TeV")
    return var

def get_figure_metadata(figure_name):
    """
    Get metadata for each figure based on the paper content
    """
    metadata = {
        "Figure41a": {
            "description": "The L1T+HLT efficiency of the Run 3 (2022, L3) triggers in 2022 data (black), 2023 data (red), and simulation (green) as a function of min($p_T$) of the two muons forming TMS-TMS dimuons in events enriched in J/ψ → μμ events. The efficiency in data is the fraction of J/ψ → μμ events recorded by the triggers based on the information from jets and $p_T^{miss}$ that also satisfy the requirements of the Run 3 (2022, L3) triggers. It is compared to the efficiency of the Run 3 (2022, L3) triggers in a combination of simulated samples of J/ψ → μμ events produced in various b hadron decays. The lower panels show the ratio of the data to simulated events.",
            "location": "Data from Figure 41 (upper left)."
        },
        "Figure41b": {
            "description": "The L1T+HLT efficiency of the Run 3 (2022, L3) triggers in 2022 data (black), 2023 data (red), and simulation (green) as a function of max($p_T$) of the two muons forming TMS-TMS dimuons in events enriched in J/ψ → μμ events. Efficiency in data is the fraction of J/ψ → μμ events recorded by the triggers based on the information from jets and $p_T^{miss}$ that also satisfy the requirements of the Run 3 (2022, L3) triggers. It is compared to the efficiency of the Run 3 (2022, L3) triggers in a combination of simulated samples of J/ψ → μμ events produced in various b hadron decays. The lower panels show the ratio of the data to simulated events.",
            "location": "Data from Figure 41 (upper right)."
        },
        "Figure41c": {
            "description": "The L1T+HLT efficiency of the Run 3 (2022, L3) triggers in 2022 data (black), 2023 data (red), and simulation (green) as a function of min($d_0$) of the two muons forming TMS-TMS dimuons in events enriched in J/ψ → μμ events. Efficiency in data is the fraction of J/ψ → μμ events recorded by the triggers based on the information from jets and $p_T^{miss}$ that also satisfy the requirements of the Run 3 (2022, L3) triggers. It is compared to the efficiency of the Run 3 (2022, L3) triggers in a combination of simulated samples of J/ψ → μμ events produced in various b hadron decays. The lower panels show the ratio of the data to simulated events.",
            "location": "Data from Figure 41 (lower)."
        },
        "Figure42a": {
            "description": "The HLT efficiency, defined as the fraction of events recorded by the Run 2 (2018) triggers that also satisfied the requirements of the Run 3 (2022, L3) triggers, as a function of offline-reconstructed min($d_0$) of the two muons forming TMS-TMS dimuons in events enriched in J/ψ → μμ. The data represent efficiencies during the 2022 and 2023 data-taking periods. For dimuons with offline min($d_0$) > 0.012 cm, the combined efficiency of the L3 muon reconstruction and the online min($d_0$) requirement is larger than 90% in all data-taking periods.",
            "location": "Data from Figure 42 (upper left)."
        },
        "Figure42b": {
            "description": "The HLT efficiency of the Run 3 (2022, L3) triggers and the Run 3 (2022, L3 dTks) triggers for J/ψ → μμ events in the 2022 and 2023 data set as a function of offline-reconstructed min($d_0$) of the two muons forming TMS-TMS dimuons in events enriched in J/ψ → μμ.",
            "location": "Data from Figure 42 (upper right)."
        },
        "Figure42c": {
            "description": "Invariant mass distribution for TMS-TMS dimuons in events recorded by the Run 2 (2018) triggers in the combined 2022 and 2023 data set, and in the subset of events also selected by the Run 3 (2022, L3) trigger and Run 3 (2022, L3 dTks) trigger, illustrating the prompt muon rejection of the L3 triggers.",
            "location": "Data from Figure 42 (lower)."
        },
        "Figure43a": {
            "description": "The HLT efficiency, defined as the fraction of events recorded by the Run 2 (2018) triggers that also satisfied the requirements of the Run 3 (2022, L2) triggers, as a function of offline-reconstructed min($d_0$) of the two muons forming STA-STA dimuons in events enriched in cosmic ray muons. The data represent efficiencies during the 2022 and 2023 data-taking periods. For displaced muons, the efficiency of the online min($d_0$) requirement is larger than 95% in all data-taking periods.",
            "location": "Data from Figure 43 (left)."
        },
        "Figure43b": {
            "description": "The invariant mass distribution for TMS-TMS dimuons in events recorded by the Run 2 (2018) triggers in the combined 2022 and 2023 data set, and in the subset of events also selected by the Run 3 (2022, L2) triggers, illustrating the prompt muon rejection of the Run 3 (2022, L2) triggers.",
            "location": "Data from Figure 43 (right)."
        },
        "Figure40": {
            "description": "The L1T+HLT efficiencies of the various displaced-dimuon triggers and their logical OR as a function of $c\\tau$ for the HAHM signal events with $m_H = 125\ GeV$ and $m_{Z_D} = 20\ GeV$, for 2022 conditions. The efficiency is defined as the fraction of simulated events that satisfy the detector acceptance and the requirements of the following sets of triggers: the Run 2 (2018) triggers (dashed black); the Run 3 (2022, L3) triggers (blue); the Run 3 (2022, L2) triggers (red); and the logical OR of all these triggers (Run 3 (2022), solid black). The lower panel shows the ratio of the overall Run 3 (2022) efficiency to the Run 2 (2018) efficiency.",
            "location": "Data from Figure 40."
        }
    }
    return metadata.get(figure_name, {
        "description": f"PLACEHOLDER: Description for {figure_name}",
        "location": f"PLACEHOLDER: Location for {figure_name}"
    })

def process_single_figure(submission, root_file_path, figure_name):
    """
    Process a single ROOT file and add its data to the submission
    """
    print(f"\n=== Processing {figure_name} ===")
    
    # Open ROOT file and extract histograms
    root_file = ROOT.TFile.Open(root_file_path)
    if not root_file or root_file.IsZombie():
        print(f"Error: Could not open ROOT file {root_file_path}")
        return False
    
    # Get the canvas
    canvas = root_file.Get("c")
    if not canvas:
        print("Error: Could not find canvas 'c' in ROOT file")
        root_file.Close()
        return False
    
    # Extract histograms from canvas
    histograms = collect_hists_from_canvas(canvas)
    
    if not histograms:
        print("Error: No histograms found in canvas")
        root_file.Close()
        return False
    
    print(f"Found {len(histograms)} histograms")
    
    # Get figure metadata
    metadata = get_figure_metadata(figure_name)
    
    # Create a single table for all histograms in the figure
    table = Table(figure_name)
    table.description = metadata["description"]
    table.location = metadata["location"]
    
    # Add keywords
    table.keywords["reactions"] = ["P P --> X"]
    table.keywords["observables"] = ["EFF"]
    table.keywords["phrases"] = ["CMS", "LLP", "long-lived particles", "trigger", "efficiency", "displaced muons"]
    
    # Add figure image if PDF exists and ImageMagick is available
    pdf_path = f"data_Alejandro/{figure_name}.pdf"
    if os.path.exists(pdf_path) and check_imagemagick_available():
        try:
            table.add_image(pdf_path)
            print(f"Added image: {pdf_path}")
        except Exception as e:
            print(f"Warning: Could not add image {pdf_path}: {e}")
    elif os.path.exists(pdf_path):
        print(f"ImageMagick not available - skipping image: {pdf_path}")
    else:
        print(f"Image not found: {pdf_path}")
    
    # Process histograms and filter out duplicates and ratio panels
    processed_names = set()
    unique_histograms = []
    
    for hist in histograms:
        hist_name = hist.GetName()
        y_axis_title = hist.GetYaxis().GetTitle()
        
        # Skip ratio histograms (from ratio panels)
        if "ratio" in hist_name.lower() or "Data/MC" in y_axis_title:
            print(f"Skipping ratio histogram: {hist_name}")
            continue
            
        # Skip duplicate histograms (like _copy versions)
        base_name = hist_name.replace("_copy", "")
        if base_name in processed_names:
            print(f"Skipping duplicate histogram: {hist_name}")
            continue
            
        processed_names.add(base_name)
        unique_histograms.append(hist)
        print(f"Including histogram: {hist_name}")
    
    # Add x-axis variable (same for all histograms)
    if unique_histograms:
        first_hist = unique_histograms[0]
        first_hist_data = extract_histogram_data(first_hist)
        x_axis_title = first_hist.GetXaxis().GetTitle()
        
        # Get proper x-axis name and units
        x_name, x_units = get_x_axis_info(x_axis_title, figure_name)
        
        # Create independent variable (x-axis) - shared by all histograms
        x_var = create_binned_variable(
            name=x_name,
            x_low=first_hist_data['x_low'],
            x_high=first_hist_data['x_high'],
            units=x_units,
            is_independent=True
        )
        table.add_variable(x_var)
    
    # Add each histogram as a dependent variable
    for hist in unique_histograms:
        hist_name = hist.GetName()
        hist_data = extract_histogram_data(hist)
        y_axis_title = hist.GetYaxis().GetTitle()
        
        # Create descriptive variable name based on histogram name and content
        var_name = create_variable_name(hist_name, figure_name)
        
        # Determine units for y-axis
        if "Efficiency" in y_axis_title:
            y_units = ""
        elif "Data/MC" in y_axis_title:
            y_units = ""
        else:
            y_units = ""
        
        # Create dependent variable (y-axis)
        y_var = create_dependent_variable(
            name=var_name,
            y_values=hist_data['y_values'],
            y_errors=hist_data['y_errors'],
            units=y_units,
            uncertainty_label="Statistical uncertainty"
        )
        
        # Add variable to table
        table.add_variable(y_var)
        
        print(f"  - Added {var_name} with {len(hist_data['y_values'])} data points")
    
    # Add table to submission
    submission.add_table(table)
    
    # Close ROOT file
    root_file.Close()
    return True

def get_x_axis_info(x_axis_title, figure_name):
    """
    Get proper x-axis variable name and units based on figure and axis title
    """
    # Clean up the title and extract units
    title = x_axis_title.strip()
    
    # Handle specific figure naming
    if figure_name == "Figure41a":
        return "min($p_T$)", "GeV"
    elif figure_name == "Figure41b":
        return "max($p_T$)", "GeV"
    elif figure_name == "Figure41c":
        return "min($d_0$)", "cm"
    elif figure_name in ["Figure42a", "Figure42b"]:
        return "min($d_0$)", "cm"
    elif figure_name in ["Figure42c", "Figure43b"]:
        return "$m_{\\mu\\mu}$", "GeV"
    elif figure_name == "Figure43a":
        return "min($d_0$)", "cm"
    else:
        # Generic handling - extract units from title
        if "[GeV]" in title:
            clean_title = title.replace("[GeV]", "").strip()
            return clean_title, "GeV"
        elif "[cm]" in title:
            clean_title = title.replace("[cm]", "").strip()
            return clean_title, "cm"
        else:
            return title if title else "Variable", ""

def create_variable_name(hist_name, figure_name):
    """
    Create descriptive variable names based on histogram name and figure
    """
    # Special handling for specific figures with event counts instead of efficiencies
    if figure_name == "Figure42c":
        if "MuonRun3HLTRun2" in hist_name:
            return "Run 2 (2018) observed events"
        elif "DisplacedL3" in hist_name and "dTks" not in hist_name:
            return "Run 3 (2022, L3) observed events"
        elif "dTks" in hist_name:
            return "Run 3 (2022, L3 dTks) observed events"
    elif figure_name == "Figure43b":
        if "MuonRun3HLTRun2" in hist_name:
            return "Run 2 (2018) observed events"
        elif "L3VetoOR" in hist_name:
            return "Run 3 (2022, L2) observed events"
    elif figure_name == "Figure42a":
        # Fix specific naming for Figure42a
        if "HData2022" in hist_name:
            return "Run 3 (2022, L3) trigger efficiency - 2022 data"
        elif "2023" in hist_name:
            return "Run 3 (2022, L3) trigger efficiency - 2023 data"
    
    # Handle different histogram types based on naming patterns (for efficiency plots)
    if "dTks" in hist_name:
        return "Run 3 (2022, L3 dTks) trigger efficiency"
    elif "DisplacedL3" in hist_name and "Muon" in hist_name:
        return "Run 3 (2022, L3) trigger efficiency"
    elif "DisplacedL3" in hist_name and "JetMET" in hist_name:
        if "2022" in hist_name:
            return "Run 3 (2022, L3) trigger efficiency - 2022 data"
        elif "2023" in hist_name:
            return "Run 3 (2022, L3) trigger efficiency - 2023 data"
        else:
            return "Run 3 (2022, L3) trigger efficiency"
    elif "BtoJPsi" in hist_name:
        return "Run 3 (2022, L3) trigger efficiency - simulation"
    elif "COSMI" in hist_name:
        if "2022" in hist_name:
            return "Run 3 (2022, L2) trigger efficiency - 2022 data"
        elif "2023" in hist_name:
            return "Run 3 (2022, L2) trigger efficiency - 2023 data"
        else:
            return "Run 3 (2022, L2) trigger efficiency"
    elif "ratio" in hist_name:
        return "Data/MC ratio"
    else:
        # Generic fallback
        return f"Efficiency ({hist_name})"

def process_existing_yaml_file(submission, yaml_file_path, figure_name):
    """
    Process an existing YAML file and add it to the submission with proper metadata
    """
    print(f"\n=== Processing existing YAML: {figure_name} ===")
    
    if not os.path.exists(yaml_file_path):
        print(f"Error: YAML file not found: {yaml_file_path}")
        return False
    
    try:
        # Read the existing YAML file
        with open(yaml_file_path, 'r') as f:
            yaml_content = yaml.safe_load(f)
        
        # Get figure metadata
        metadata = get_figure_metadata(figure_name)
        
        # Create a new table with proper metadata
        table = Table(figure_name)
        table.description = metadata["description"]
        table.location = metadata["location"]
        
        # Add keywords
        table.keywords["reactions"] = ["P P --> X"]
        table.keywords["observables"] = ["EFF"]
        table.keywords["phrases"] = ["CMS", "LLP", "long-lived particles", "trigger", "efficiency", "displaced muons", "proper decay length"]
        
        # Add figure image if PDF exists and ImageMagick is available
        pdf_source_path = f"data_Alejandro/fromDisplacedDimuons/{figure_name}.pdf"
        if os.path.exists(pdf_source_path) and check_imagemagick_available():
            try:
                # Use relative path to the PDF in fromDisplacedDimuons directory
                table.add_image(pdf_source_path)
                print(f"Added image: {pdf_source_path}")
            except Exception as e:
                print(f"Warning: Could not add image {pdf_source_path}: {e}")
        elif os.path.exists(pdf_source_path):
            print(f"ImageMagick not available - skipping image: {pdf_source_path}")
        else:
            print(f"Image not found: {pdf_source_path}")
        
        # Extract and add independent variables
        if 'independent_variables' in yaml_content:
            for indep_var in yaml_content['independent_variables']:
                var_name = indep_var['header']['name']
                var_units = indep_var['header'].get('units', '')
                var_values = indep_var['values']
                
                # Create Variable object
                var = Variable(var_name, is_independent=True, is_binned=False, units=var_units)
                var.values = [v['value'] for v in var_values]
                table.add_variable(var)
                print(f"  - Added independent variable: {var_name} with {len(var.values)} points")
        
        # Extract and add dependent variables
        if 'dependent_variables' in yaml_content:
            for dep_var in yaml_content['dependent_variables']:
                var_name = dep_var['header']['name']
                var_values = dep_var['values']
                qualifiers = dep_var.get('qualifiers', [])
                
                # Create Variable object
                var = Variable(var_name, is_independent=False, is_binned=False, units="")
                var.values = [v['value'] for v in var_values]
                
                # Add qualifiers
                for qualifier in qualifiers:
                    var.add_qualifier(qualifier['name'], qualifier['value'])
                
                table.add_variable(var)
                print(f"  - Added dependent variable: {var_name} with {len(var.values)} points")
        
        # Add table to submission
        submission.add_table(table)
        return True
        
    except Exception as e:
        print(f"Error processing YAML file {yaml_file_path}: {e}")
        return False


def makeDoubleDispL3MuonSigEffTable():
    table = Table("Double displaced L3 muon signal eff vs min($\mathrm{p_{T}}$)")
    table.description = "The L1T+HLT efficiency of the double displaced L3 muon trigger as a function of min($\mathrm{p_{T}}$) of the two global or tracker muons in the event. The efficiency is plotted for HAHM signal events for 2022 conditions with $m_{Z_D} = 50$ GeV and $\epsilon = 4 \\times 10^{-9}$ (black triangles), $m_{Z_D} = 60$ GeV and $\epsilon = 2 \\times 10^{-9}$ (red triangles), and $m_H=125$ GeV in both cases. The events are required to have at least two good global or tracker muons with $\mathrm{p_T}>23$ GeV."
    table.location = "Data from Fig. 45"
    table.add_image("data_Juliette/pat_2022_minpt_REP_PRESEL_PT23_add_HLTDoubleMu43.pdf")

    reader = RootFileReader("data_Juliette/DoubleDispL3MuonSigEff.root")
    g_125_50_4 = reader.read_teff("HHTo2ZdTo2Mu2X_125_50_4e-09preHLTDoubleMu43_clone;1")
    g_125_60_2 = reader.read_teff("HHTo2ZdTo2Mu2X_125_60_2e-09preHLTDoubleMu43_clone;1")

    minpT = Variable("min($\mathrm{p_{T}}$)", is_independent=True, is_binned=False, units="GeV")
    minpT.values = g_125_50_4["x"]

    ### add variables and add table to submission
    table.add_variable(minpT)
    table.add_variable(makeVariable(plot = g_125_50_4, label = "$\mathrm{H} \\to 2\mathrm{Z_D} \\to 2\\mu 2\mathrm{X}$ (125, 50, 4e-09)", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot = g_125_60_2, label = "$\mathrm{H} \\to 2\mathrm{Z_D} \\to 2\\mu 2\mathrm{X}$ (125, 60, 2e-09)", is_independent=False, is_binned=False, is_symmetric=False, units=""))

    return table

def makeDoubleDispL3MuonDataMCEffTable(xvar):
    table = Table("Double disp. L3mu data&bkg eff vs "+xvar)

    if(xvar=="min($\mathrm{d_{0}}$)"):
        units = "cm"
        location = "left"
        image = "data_Juliette/pat_2022_mind0pv_REP_PT45_JPSIMASS_add_HLTDoubleMu43_Jpsi_eff.pdf"
        reader = RootFileReader("data_Juliette/DoubleDispL3MuonDataMCEff_mind0.root")
        table.description = "The L1T+HLT efficiency of the double displaced L3 muon trigger in 2022, as a function of "+xvar+" of the two global or tracker muons in the event. The efficiency is plotted for MC-simulated $J/\\psi \to \\mu\\mu$ events produced in various b hadron decays (green squares) and data enriched in $J/\\psi \to \\mu\\mu$  events recorded by jet- and $\mathrm{p_T^miss}$-based triggers (black points). The events are required to have at least two good global or tracker muons compatible with the $J/\\psi$ meson mass and with $\mathrm{p_T}>45$ GeV."
    elif(xvar=="min($\mathrm{p_{T}}$)"):
        units = "GeV"
        location = "right"
        image = "data_Juliette/pat_2022_minpt_REP_JPSIMASS_PT23_add_HLTDoubleMu43_Jpsi_eff.pdf"
        reader = RootFileReader("data_Juliette/DoubleDispL3MuonDataMCEff_minpT.root")
        table.description = "The L1T+HLT efficiency of the double displaced L3 muon trigger in 2022, as a function of "+xvar+" of the two global or tracker muons in the event. The efficiency is plotted for MC-simulated $J/\\psi \to \\mu\\mu$ events produced in various b hadron decays (green squares) and data enriched in $J/\\psi \to \\mu\\mu$  events recorded by jet- and $\mathrm{p_T^miss}$-based triggers (black points). The events are required to have at least two good global or tracker muons compatible with the $J/\\psi$ meson mass and with $\mathrm{p_T}>23$ GeV."

    table.location = "Data from Fig. 46 "+location
    table.add_image(image)

    MC = reader.read_teff("HBtoJPsipreHLTDoubleMu43_clone;1")
    data = reader.read_teff("HJetMETData2022preHLTDoubleMu43_clone;1")

    xAxisVar = Variable(xvar, is_independent=True, is_binned=False, units=units)
    xAxisVar.values = MC["x"]

    ### add variables and add table to submission
    table.add_variable(xAxisVar)
    table.add_variable(makeVariable(plot = MC, label = "Nonprompt $J/\\psi \\to \\mu\\mu$", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot = data, label = "Data (2022)", is_independent=False, is_binned=False, is_symmetric=False, units=""))

    return table

def makeMuonNoBPTXRateVsNBunchesTable(year):
    if(year=="2016" or year=="2017" or year=="2018"):
        CME="13"
    elif(year=="2022" or year=="2023" or year=="2024"):
        CME="13.6"
    table = Table("Muon NoBPTX HLT rate vs number of colliding bunches ("+year+")")
    table.description = "Rate of the main muon No-BPTX HLT path as a function of the number of colliding bunches, for "+year+"."
    table.location = "Data from Fig. 58"
    table.add_image("data_Juliette/NoBPTXL2Mu40_RateVsNCollidingBunches.pdf")

    reader = RootFileReader("data_Juliette/juliette.root")
    g = reader.read_graph("g_"+year+";1")

    nBunches = Variable("Number of colliding bunches", is_independent=True, is_binned=False, units="")
    nBunches.values = g["x"]

    ### add variables and add table to submission
    table.add_variable(nBunches)
    table.add_variable(makeVariable(plot = g, label = "HLT rate in "+year, is_independent=False, is_binned=False, is_symmetric=True, units="Hz", CME=CME))

    return table

def convertHLTMuResoToYaml(rootfile, label, xtitle, ytitle, units):
    # Table to store and output all data
    table = Table(label)

    # Read in graphs from the root file
    reader = RootFileReader(rootfile)

    graph_L2 = reader.read_graph("c/main/L2ptres")
    graph_L3 = reader.read_graph("c/main/L3ptres")

    # Form and fill the x-variable using one of the graphs
    x = Variable(xtitle, is_independent=True, is_binned=True, units=units)
    x.values = [
        (
            # Lower
            graph_L2["x"][i] - graph_L2["dx"][i],
            # Upper
            graph_L2["x"][i] + graph_L2["dx"][i],
        ) for i,_ in enumerate(graph_L2["x"])
    ]

    # Form and fill the y-variables from the graphs
    # L2
    y_L2 = Variable(ytitle + "  (L2 muons)", is_independent=False, is_binned=False)
    y_L2.values = graph_L2["y"]
    y_L2.values = [val if val != 0 else "-" for val in y_L2.values]
    yunc_L2 = Uncertainty("", is_symmetric=True)
    yunc_L2.values = graph_L2["dy"]
    y_L2.uncertainties.append(yunc_L2)
    y_L2.add_qualifier("CHANNEL", "$\\text{H} \\rightarrow \\text{Z}_\\text{D}\\text{Z}_\\text{D}, \\text{Z}_\\text{D} \\rightarrow \\text{μ}\\text{μ}$")
    y_L2.add_qualifier("SQRT(S)", "13.6", "TeV")

    # L3
    y_L3 = Variable(ytitle + "  (L3 muons)", is_independent=False, is_binned=False)
    y_L3.values = graph_L3["y"]
    y_L3.values = [val if val != 0 else "-" for val in y_L3.values]
    yunc_L3 = Uncertainty("", is_symmetric=True)
    yunc_L3.values = graph_L3["dy"]
    y_L3.uncertainties.append(yunc_L3)
    y_L3.add_qualifier("CHANNEL", "$\\text{H} \\rightarrow \\text{Z}_\\text{D}\\text{Z}_\\text{D}, \\text{Z}_\\text{D} \\rightarrow \\text{μ}\\text{μ}$")
    y_L3.add_qualifier("SQRT(S)", "13.6", "TeV")

    # Add variables to the table
    table.add_variable(x)
    table.add_variable(y_L2)
    table.add_variable(y_L3)

    return table

def makeHLTMuResoTable(xvar):
    if xvar == "genpt":
        table = convertHLTMuResoToYaml("data_Ansar/HLTMuonsRes_vs_genpt.root", "HLT muon pt resolution vs pt", "$p_\\text{T}^\\text{gen}$", "$\\text{Fitted }\\sigma : (1/p_\\text{T}^\\text{HLT} - 1/p_\\text{T}^\\text{gen}) / 1/p_\\text{T}^\\text{gen}$", "GeV")
        table.description = "Inverse HLT muon $p_\\text{T}$ resolution ($(1/p_\\text{T}^\\text{HLT}-1/p_\\text{T}^\\text{gen})/(1/p_\\text{T}^\\text{gen})$) as a function of the generator-level muon $p_\\text{T}$, for simulated HAHM signal events, where the dark Higgs boson ($\\text{H}_\\text{D}$) mixes with the SM Higgs boson ($\\text{H}$) and decays to a pair of long-lived dark photons ($\\text{Z}_\\text{D}$), for various values of $m_{\\text{Z}_\\text{D}}$ and $\\epsilon$. Conditions for 2022 data-taking are shown. The muons must have $p_\\text{T}>10\\text{ GeV}$, and the L2 and L3 muons are geometrically matched to the generator-level muons."
        table.location = "Data from Figure 70 left"
        table.add_image("data_Ansar/HLTMuonsRes_vs_genpt.png")
        table.keywords["cmenergies"] = [13600.0]
        table.keywords["reactions"] = ["P P --> H --> ZD ZD --> MU MU + ANYTHING"]
        table.keywords["phrases"] = ["pT resolution", "L2 muons", "L3 muons", "Tracker muon", "Displaced standalone muon", "pT"]

    elif xvar == "genlxy":
        table = convertHLTMuResoToYaml("data_Ansar/HLTMuonsRes_vs_genlxy.root", "HLT muon pt resolution vs Lxy", "$L_{xy}^\\text{gen}$", "$\\text{Fitted }\\sigma : (1/p_\\text{T}^\\text{HLT} - 1/p_\\text{T}^\\text{gen}) / 1/p_\\text{T}^\\text{gen}$", "cm")
        table.description = "Inverse HLT muon $p_\\text{T}$ resolution ($(1/p_\\text{T}^\\text{HLT}-1/p_\\text{T}^\\text{gen})/(1/p_\\text{T}^\\text{gen})$) as a function of the generator-level $L_{xy}$, for simulated HAHM signal events, where the dark Higgs boson ($\\text{H}_\\text{D}$) mixes with the SM Higgs boson ($\\text{H}$) and decays to a pair of long-lived dark photons ($\\text{Z}_\\text{D}$), for various values of $m_{\\text{Z}_\\text{D}}$ and $\\epsilon$. Conditions for 2022 data-taking are shown. The muons must have $p_\\text{T}>10\\text{ GeV}$, and the L2 and L3 muons are geometrically matched to the generator-level muons. The dashed vertical lines indicate the radial positions of the layers of the tracking detectors, with BPX, TIB, and TOB denoting the barrel pixel, tracker inner barrel, and tracker outer barrel, respectively."
        table.location = "Data from Figure 70 right"
        table.add_image("data_Ansar/HLTMuonsRes_vs_genlxy.png")
        table.keywords["cmenergies"] = [13600.0]
        table.keywords["reactions"] = ["P P --> H --> ZD ZD --> MU MU + ANYTHING"]
        table.keywords["phrases"] = ["pT resolution", "L2 muons", "L3 muons", "Tracker muon", "Displaced standalone muon", "Transverse decay length"]

    return table

def makeDisplacedTauEffTable(var):

    var_label = '$\mathrm{p_{T}^{miss}}$'
    position = 'right'
    if var == 'd0':
        var_label = '$d_{0}$'
        position = 'left'
    
    table = Table(f"Displaced tau trigger efficiency vs {var_label}")
    table.description = "The L1T+HLT efficiency of the displaced $\\tau_\mathrm{h}$ trigger, for simulated $\mathrm{p}\mathrm{p} \\to \\tilde{\\tau}\\tilde{\\tau},(\\tilde{\\tau} \\to \\tau\\tilde{\\chi}^{0}_{1})$ events, \
where the $\\tilde{\\tau}$ has $c\\tau = 10$ cm and each $\\tau$ decays hadronically. The efficiency is shown for the displaced di-$\\tau_\mathrm{h}$ trigger path (blue filled triangles), \
the previously available $\mathrm{p_{T}^{miss}}$-based paths (orange open circles), the previously available prompt di-$\\tau_\mathrm{h}$ paths (purple open squares), \
the combination of the $\mathrm{p_{T}^{miss}}$-based and prompt di-$\\tau_\mathrm{h}$ paths (gray open triangles), and the combination of the $\mathrm{p_{T}^{miss}}$-based, \
prompt di-$\\tau_\mathrm{h}$, and displaced di-$\\tau_\mathrm{h}$ paths (red filled circles), using 2022 data-taking conditions. \
The efficiency is evaluated with respect to generator-level quantities. \
Efficiency of the highest $\mathrm{p_{T}}$ $\\tau$ lepton in the event as a function of the $d_{0}$ (left). \
Efficiency as a function of $\mathrm{p_{T}^{miss}}$ (right). \
A selection on the visible component of the generator-level $\\tau$ lepton $\mathrm{p_{T}} >$ 30 GeV and its pseudorapidity $|{\\eta}| < 2.1$ is applied. \
The lower panels show the ratio (improvement in \%) of the trigger efficiency given by the combination of the displaced di-$\\tau_\mathrm{h}$ trigger path with the $\mathrm{p_{T}^{miss}}$-based \
and prompt di-$\\tau_\mathrm{h}$ paths to that of the combination of the previously available $\mathrm{p_{T}^{miss}}$-based and prompt di-$\\tau_\mathrm{h}$ paths."
    table.location = f"Data from Fig. 13 ({position})"
    
    if var == 'd0':
        table.add_image("data_Sara/efficiency_tau_dxy_ditau_perEvt_GENpT30_Tau32_M100ctau100_officialsummer22EE.pdf")
    else:
        table.add_image("data_Sara/efficiency_met_ditau_perEvt_GENpT30_Tau32_M100ctau100_officialsummer22EE.pdf")
    
    reader = RootFileReader("data_Sara/Figs13_gen_met.root")
    if var == 'd0':
        reader = RootFileReader("data_Sara/Figs13_tau_gen_dxy.root")
    g_ditau = reader.read_teff("eff_ditau;1")
    g_ptmiss = reader.read_teff("eff_ptmiss;1")
    g_prompt = reader.read_teff("eff_prompttau;1")
    g_old_OR = reader.read_teff("eff_old_OR;1")
    g_new_OR = reader.read_teff("eff_new_OR;1")

    xvar = Variable("Gen.-level $\mathrm{p_{T}^{miss}}$", is_independent=True, is_binned=False, units="GeV") ### to be changed
    if var == 'd0':
        xvar = Variable("Gen.-level $\\tau$ $\mathrm{d_0}$", is_independent=True, is_binned=False, units="cm") ### to be changed
    xvar.values = g_ditau["x"]

    table.add_variable(xvar)
    table.add_variable(makeVariable(plot = g_ditau, label = "displaced di-$\\tau_{h}$ paths", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot = g_ptmiss, label = "$\mathrm{p_{T}^{miss}}$ paths", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot = g_prompt, label = "prompt di-$\\tau_{h}$ paths", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot = g_old_OR, label = "$\mathrm{p_{T}^{miss}}$ OR prompt di-$\\tau_{h}$ paths", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot = g_new_OR, label = "displaced di-$\\tau_{h}$ OR $\mathrm{p_{T}^{miss}}$ OR prompt di-$\\tau_{h}$ paths", is_independent=False, is_binned=False, is_symmetric=False, units=""))

    return table


def makeDisplacedTauAccTable():

    table = Table("Displaced tau trigger acceptance vs decay vertex radial positon")
    table.description = "The L1T+HLT acceptance of the displaced $\\tau_\mathrm{h}$ trigger, for simulated \
$\mathrm{p}\mathrm{p} \\to \\tilde{\\tau}\\tilde{\\tau},(\\tilde{\\tau} \\to \\tau\\tilde{\\chi}^{0}_{1})$ events,\
where each $\\tau$ decays hadronically and the $\\tilde{\\tau}$ has a simulated $c\\tau$ of 10 cm. \
The acceptance is shown for the displaced di-$\\tau_\mathrm{h}$ trigger path for 2022 data-taking conditions and is \
plotted with respect to the generator-level $\\tau$ lepton decay vertex radial position. \
Selections on the visible component of the generator-level $\\tau$ lepton $\mathrm{p_{T}}$ ($\mathrm{p_{T}}(\\tau) > 30$ GeV), \
its pseudorapidity ($|\\eta(\\tau)| <$ 2.1), and its decay vertex radial position ($R < $115 cm) are applied."
    
    table.location = f"Data from Fig. 68"
    table.add_image("data_Sara/efficiency_tau_lxy_ditau_perEvt_GENpT30_Tau32_M100ctau100_officialsummer22EE_radiusWithin115.pdf")
    
    reader = RootFileReader("data_Sara/Fig70_tau_gen_lxy.root")
    g_ditau = reader.read_teff("eff_ditau;1")

    xvar = Variable("Gen.-level $\\tau$ decay vertex radial positon", is_independent=True, is_binned=False, units="cm")
    xvar.values = g_ditau["x"]

    table.add_variable(xvar)
    table.add_variable(makeVariable(plot = g_ditau, label = "displaced di-$\\tau_{h}$ paths", is_independent=False, is_binned=False, is_symmetric=False, units=""))

    return table


def makeDiTauRateTable(year):
    
    table = Table(f"Displaced tau trigger rate vs pileup in {year}")
    table.location = f"Data from Fig. 14 (left)"
    if year == '2023':
        table.location = f"Data from Fig. 14 (right)"
        table.add_image("data_Sara/rate_vs_pileup_2023D_ditau.pdf")
    else:    
        table.add_image("data_Sara/rate_vs_pileup_2022EFG_ditau.pdf")
    table.description = "Total rate of the displaced $\\tau_\mathrm{h}$ trigger for a few representative runs in 2022 (left) and 2023 (right) data, as a function of PU."
    with open(f"data_Sara/hepdata_ditau_rate_{year}.yaml") as f:
        data = yaml.safe_load(f)
    pu_vals = [v["value"] for v in data["independent_variables"][0]["values"]]
    pileup = Variable("Pileup", is_independent=True, is_binned=False, units="")
    pileup.values = pu_vals

    rate_vals = [v["value"] for v in data["dependent_variables"][0]["values"]]
    rate = Variable("rate", is_independent=False, is_binned=False, units="Hz")
    rate.values = rate_vals

    table.add_variable(pileup)
    table.add_variable(rate)

    return table

def makeDiphotonRateTable():
    
    table = Table(f"Delayed-diphoton trigger rate vs pileup in 2024")
    table.add_image("data_Sara/rate_vs_pileup_new_perFill_lastFour_2024C_diphoton_ps1p8E34.pdf")
    table.location = f"Data from Fig. 32 (right)"
    table.description = "The delayed-diphoton trigger rate is shown as a function of PU for selected fills in 2024 data, \
at an instantaneous luminosity of approximately $1.8\\times10^{34}~cm^{-2}~s^{-1}$. \
The trigger rate displays a linear dependency on PU."
    with open(f"data_Sara/hepdata_diphoton_rate_2024.yaml") as f:
        data = yaml.safe_load(f)
    pu_vals = [v["value"] for v in data["independent_variables"][0]["values"]]
    pileup = Variable("Pileup", is_independent=True, is_binned=False, units="")
    pileup.values = pu_vals

    rate_vals = [v["value"] for v in data["dependent_variables"][0]["values"]]
    rate = Variable("rate", is_independent=False, is_binned=False, units="Hz")
    rate.values = rate_vals

    fill_vals = [v["value"] for v in data["dependent_variables"][1]["values"]]
    fill = Variable("Fill", is_independent=False, is_binned=False, units="")
    fill.values = fill_vals

    table.add_variable(pileup)
    table.add_variable(rate)
    table.add_variable(fill)

    return table


def makeDisplPhotonRateTable(year):
    
    table = Table(f"Displaced photon plus HT trigger rate vs pileup in {year}")
    if year == '2023':
        table.add_image("data_Sara/rate_vs_pileup_new_2023BCD_displphoton_ps2p0E34.pdf")
    elif year == '2022':
        table.add_image("data_Sara/rate_vs_pileup_new_2022BCDEFG_displphoton_ps1p8E34.pdf")

    position = 'left'    
    if year == '2023':
        position = 'right'    
    table.location = f"Data from Fig. 37 ({position})"

    table.description = "Total rate of the displaced-photon + $H_\mathrm{T}$ HLT path for a few representative \
runs in 2022 data (left), at an instantaneous luminosity of approximately \
$1.8\\times10^{34}~cm^{-2}~s^{-1}$, and 2023 data (right), at an instantaneous luminosity \
of approximately $2.0\\times10^{34}~cm^{-2}~s^{-1}$, as a function of PU. \
The rate vs PU behavior was nonlinear in 2022 and fixed in time for 2023 data taking."

    with open(f"data_Sara/hepdata_displphoton_rate_{year}.yaml") as f:
        data = yaml.safe_load(f)
    pu_vals = [v["value"] for v in data["independent_variables"][0]["values"]]
    pileup = Variable("Pileup", is_independent=True, is_binned=False, units="")
    pileup.values = pu_vals

    rate_vals = [v["value"] for v in data["dependent_variables"][0]["values"]]
    rate = Variable("rate", is_independent=False, is_binned=False, units="Hz")
    rate.values = rate_vals

    table.add_variable(pileup)
    table.add_variable(rate)

    return table

#Figure 27
def makeDelayedJetEfficiencyTable():
    reader = RootFileReader("data_Neha/Efficiency_comparison_delayedjets_Run2HTtrigger.root")

    h_incl = reader.read_hist_1d("SingleInclusive")
    h_trk  = reader.read_hist_1d("SingleTrackless")
    h_ht   = reader.read_hist_1d("Run2HT1050")

    table = Table("L1T+HLT eff vs HT (mH = 1000, mX = 450 GeV, ctau=10m)")
    table.location = "Figure 27"
    table.description = ("The L1T+HLT efficiency of the inclusive and trackless delayed-jet triggers introduced in Run 3, shown as red squares and blue triangles, for 2022 conditions, and of the $H_T$ trigger (black circles), which was the most appropriate path available in Run 2, for a $H \\to X X \\to b\\bar{b}\\,b\\bar{b}$ signal with $m_H$ = 1000 GeV, $m_X$ = 450 GeV, and $c\\tau = 10$ m. The addition of these delayed-jet triggers results in a significant improvement in the efficiency of the signal for $430 < H_T < 1050 GeV$.")
    table.add_image("data_Neha/Efficiency_comparison_delayedjets_Run2HTtrigger.pdf")

    x = Variable("$H_T$", is_independent=True, is_binned=True, units="GeV")
    x.values = h_incl["x_edges"]
    table.add_variable(x)

    def add_dep(label, hist):
        y = Variable(label, is_independent=False, is_binned=False)
        y.values = hist["y"]
        if "dy" in hist:
            u = Uncertainty("stat", is_symmetric=True)
            u.values = hist["dy"]
            y.add_uncertainty(u)
        y.add_qualifier("SQRT(S)", 13.6, "TeV")
        y.add_qualifier("Final state", "$H \\to X X \\to b\\bar{b}\\,b\\bar{b}$")
        table.add_variable(y)

    add_dep("Inclusive delayed-jet trigger (Run 3)", h_incl)
    add_dep("Trackless delayed-jet trigger (Run 3)", h_trk)
    add_dep("$H_T > 1050$ GeV trigger (Run 2)", h_ht)

    return table

#Figure 28
def makeDelayedHTTauTable(rootfile, final_state_label, final_state_tex, location):
    reader = RootFileReader(rootfile)

    h_incl_ht  = reader.read_hist_1d("Inclusiveht")
    h_trk_ht   = reader.read_hist_1d("Tracklessht")
    h_trk_tau  = reader.read_hist_1d("Tracklesstau")
    h_incl_tau = reader.read_hist_1d("Inclusivetau")

    title = f"Delayed jet trigger efficiency vs HT, ({final_state_label} final state)"
    table = Table(title)
    table.location = "Data from Figure 28 " + location
    table.description = ("The L1T+HLT efficiency of the $H_T$-seeded delayed jet trigger, the $H_T$-seeded delayed trackless jet trigger, the tau-seeded delayed jet trigger, and the tau-seeded delayed trackless jet trigger, as functions of $H_T$, for a $H \\to X X \\to " + final_state_tex + "$ signal. The addition of the delayed-jet triggers results in a significant improvement in the efficiency of the signal in the intermediate $H_T$ range.")
    table.add_image(rootfile.replace(".root", ".pdf"))

    x = Variable("$H_T$", is_independent=True, is_binned=True, units="GeV")
    x.values = h_incl_ht["x_edges"]
    table.add_variable(x)

    def add_dep(label, hist):
        y = Variable(label, is_independent=False, is_binned=False)
        y.values = hist["y"]
        if "dy" in hist:
            u = Uncertainty("stat", is_symmetric=True)
            u.values = hist["dy"]
            y.add_uncertainty(u)
        y.add_qualifier("SQRT(S)", 13.6, "TeV")
        y.add_qualifier("Final state", f"$H \\to X X \\to {final_state_tex}$")
        table.add_variable(y)

    add_dep("$H_T$-seeded delayed jet trigger", h_incl_ht)
    add_dep("$H_T$-seeded delayed trackless jet trigger", h_trk_ht)
    add_dep("Tau-seeded delayed jet trigger", h_incl_tau)
    add_dep("Tau-seeded delayed trackless jet trigger", h_trk_tau)

    return table

#Figure 29
def makeDelayedJetTimeTable(rootfile, table_title, hlt_text, location):
    reader = RootFileReader(rootfile)

    g2023 = reader.read_graph("2023")
    g2022 = reader.read_graph("2022")

    table = Table(table_title)
    table.location = "Data from Figure 29 " + location
    table.description = ("Data from Figure 28: The L1T+HLT efficiency of the delayed-jet triggers as a function of jet timing for 2022 and 2023 data-taking periods. A clear rise in efficiency is evident around the threshold values. The plots include events that pass the $E_T^{\\text{miss}} > 200\\,\\mathrm{GeV}$ trigger and have at least one barrel jet with $p_T > 50\\,\\mathrm{GeV}$, number of ECAL cells $> 8$, and ECAL energy $ > 25\\,\\mathrm{GeV}$. The $H_T$ is calculated using the scalar sum of jets with offline $p_T > 40 GeV$, and this is different from the $H_T$ calculation used at the HLT level, which can cause trigger inefficiencies. The maximum jet time accepted by the trigger is $12.5\\,\\mathrm{ns}$.")
    table.add_image(rootfile.replace(".root", ".pdf"))

    x = Variable("Jet time", is_independent=True, is_binned=False, units="ns")
    x.values = g2023["x"]
    table.add_variable(x)

    def add_curve(graph, label):
        y = Variable(label, is_independent=False, is_binned=False)
        y.values = graph["y"]
        u = Uncertainty("stat", is_symmetric=False)
        u.values = graph["dy"]
        y.add_uncertainty(u)
        y.add_qualifier("SQRT(S)", 13.6, "TeV")
        table.add_variable(y)

    add_curve(g2023, "L1T+HLT efficiency (2023)")
    add_curve(g2022, "L1T+HLT efficiency (2022)")

    return table

#old Figures 65 and 66
#new Figures 61 and 62
def makeAcceptanceTables(mH, mX, coord):
    tag = f"{mH}_{mX}"
    if coord == "R":
        fname = f"data_Neha/overlay_acceptance_{tag}_CTau-1000mm.root"
        pdfbase = f"data_Neha/overlay_acceptance_{tag}_CTau-1000mm"
        axis_label = "LLP decay R"
        position_phrase = "LLP decay radial position"
    else:
        fname = f"data_Neha/overlay_acceptance_z_{tag}_CTau-1000mm.root"
        pdfbase = f"data_Neha/overlay_acceptance_z_{tag}_CTau-1000mm"
        axis_label = "LLP decay Z"
        position_phrase = "LLP decay position along the beam line"

    if tag == "1000_200":
        location = "(upper left)"
    elif tag == "350_160":
        location = "(lower left)"
    elif tag == "350_80":
        location = "(upper right)"
    elif tag == "125_25":
        location = "(lower right)"
    else:
        location = "null"
        
    reader = RootFileReader(fname)

    curves = [
        ("g_trk",  "Displaced-jet triggers using the tracker ($c\\tau = 0.1\\,\\mathrm{m}$)", "Tracker displaced-jet", "0.1 m"),
        ("g_ecal", "Delayed-jet triggers using ECAL timing ($c\\tau = 1\\,\\mathrm{m}$)",      "ECAL delayed-jet",     "1 m"),
        ("g_hcal", "Displaced-jet triggers using the HCAL ($c\\tau = 1\\,\\mathrm{m}$)",       "HCAL displaced-jet",   "1 m"),
        ("g_dt",   "Muon Detector Showers with the DTs ($c\\tau = 1\\,\\mathrm{m}$)",          "DT MDS",               "1 m"),
        ("g_csc",  "Muon Detector Showers with the CSCs ($c\\tau = 1\\,\\mathrm{m}$)",         "CSC MDS",              "1 m"),
    ]

    tables = []

    for graph_name, label, short, ctau in curves:
        g = reader.read_graph(graph_name)

        if coord == "R":
            title = f"{short} acceptance vs R (mH={mH}, mX={mX})"
        else:
            title = f"{short} acceptance vs Z (mH={mH}, mX={mX})"
        table = Table(title)
        if coord == "R":
            table.location = "Data from Fig. 61 " + location
        else:
            table.location = "Data from Fig. 62 " + location
        table.description = ("The L1T+HLT acceptances for various LLP triggers using different subdetectors, as functions of the " + position_phrase + f", for $H \\to X X \\to b\\bar{{b}}\\,b\\bar{{b}}$ events for 2023 conditions with $m_H={mH}\\,\\mathrm{{GeV}}$ and $m_X={mX}\\,\\mathrm{{GeV}}$. The $c\\tau$ is 0.1\\,m for the displaced-jet triggers using the tracker and 1\\,m for the other triggers. The acceptance is shown for the displaced-jet triggers using the tracker (cyan points), for the delayed-jet triggers using ECAL timing (red circles), for the displaced-jet triggers using the HCAL (blue squares), for the MDS triggers with the DTs (green triangles), and for the MDS triggers with the CSCs (pink points). The boundaries of the tracker, ECAL, HCAL, DTs, and CSCs are also shown.")
        table.add_image(pdfbase + ".pdf")

        x = Variable(axis_label, is_independent=True, is_binned=False, units="cm")
        x.values = g["x"]
        table.add_variable(x)

        y = Variable(label, is_independent=False, is_binned=False)
        y.values = g["y"]
        u = Uncertainty("stat", is_symmetric=False)
        u.values = g["dy"]
        y.add_uncertainty(u)
        y.add_qualifier("SQRT(S)", 13.6, "TeV")
        y.add_qualifier("Final state", "$H \\to X X \\to b\\bar{b}\\,b\\bar{b}$")
        table.add_variable(y)

        tables.append(table)
    return tables

def makeTrackingEfficiencyTable(trigger_or_offline):
    if trigger_or_offline == 'HLT':
        table = Table("HLT Tracking efficiency vs simulated radial position")
        table.add_image("data_Enrico/ttbar_Run3_HLT_efficiency_r_logx.pdf")
        table.location = "Data from Fig. 2 (right)"
        table.description = "Overall standard tracking efficiency at the HLT during Run~3, as a function of the simulated radial position of the track production vertex. In the figure, \\ttbar simulation for 2025 conditions and an average PU of 62 is used, and the tracks are required to have $\pt>0.9\GeV$ and $\\abs{\eta}<2.5$. The tracking efficiency is defined as the ratio of the simulated tracks (with the aforementioned selection requirements) geometrically matched to a reconstructed track, divided by the total simulated tracks passing the selections."

        reader = RootFileReader("data_Enrico/ttbar_Run3_HLT_efficiency_r_logx.root")
        h_trkeff = reader.read_hist_1d("effic_vs_vertpos;1")

        xvar = Variable("Radial vertex position ", is_independent=True, is_binned=False, units="cm")
        xvar.values = h_trkeff["x"]

        table.add_variable(xvar)
        table.add_variable(makeVariable(plot = h_trkeff, label = "HLT tracking efficiency", is_independent=False, is_binned=False, is_symmetric=True, units=""))

        return table
    elif trigger_or_offline == 'Offline':
        table = Table("Offline Tracking efficiency vs simulated radial position")
        table.add_image("data_Enrico/ttbar_Run3_efficiency_r_cum_logx.pdf")
        table.location = "Data from Fig. 2 (left)"
        table.description = "Offline standard tracking efficiency during Run~3 for different tracking iterations, as a function of simulated radial position of the track production vertex (left). In the figure, \\ttbar simulation for 2025 conditions and an average PU of 62 is used, and the tracks are required to have $\pt>0.9\GeV$ and $\\abs{\eta}<2.5$. The tracking efficiency is defined as the ratio of the simulated tracks (with the aforementioned selection requirements) geometrically matched to a reconstructed track, divided by the total simulated tracks passing the selections."

        reader = RootFileReader("data_Enrico/ttbar_Run3_efficiency_r_cum_logx.root")
        h_trkeff = reader.retrieve_object("effic_vs_vertpos;1")

        total_eff = None
        for h in h_trkeff.GetHists():
            trk_eff_by_iter = get_hist_1d_points(h)
            table.add_variable(makeVariable(plot = trk_eff_by_iter, label = f"{h.GetName()} iteration tracking efficiency", is_independent=False, is_binned=False, is_symmetric=True, units=""))
            xvar_values = trk_eff_by_iter['x']
            if total_eff is None:
                total_eff = trk_eff_by_iter['y']
            else:
                total_eff = [total_eff[i] + trk_eff_by_iter['y'][i] for i in range(len(total_eff))]
        total_y = Variable("Total offline tracking efficiency", is_independent=False, is_binned=False, units="")
        total_y.values = total_eff
        table.add_variable(total_y)


        xvar = Variable("Radial vertex position ", is_independent=True, is_binned=False, units="cm")
        xvar.values = xvar_values
        table.add_variable(xvar)
        return table
    else:
        raise ValueError("Must pass either HLT or Offline")
# Figure 48
def makeScoutingMuonDataEffVSPtTable():
    table = Table("Scouting dimuon trigger eff vs pt in data")
    table.description = "L1T+HLT efficiency of the dimuon scouting trigger as a function of the subleading muon $p_{T}$, for 2024 data. The efficiency of the L1T dimuon seeds (pink squares) and the HLT dimuon scouting trigger with the vertex-unconstrained reconstruction algorithm (blue triangles) is shown. The events in the denominator are required to have at least two vertex-unconstrained muons ($N_{\\mu(\\text{no-vtx})} > 2$) and additionally have $\\chi^2/N_{\\text{dof}} < 3$ and $\\Delta R > 0.1$."

    table.location = "Data from Fig. 48"
    table.add_image("data_Celia/Figure_048.pdf")
    reader = RootFileReader("data_Celia/Scouting_DoubleMuonEfficiency_data_pt.root")
    graph_L1 = reader.read_teff("efficiency_pt_L1;1")
    graph_HLT = reader.read_teff("efficiency_pt_HLT;1")

    subpT = Variable("sub $\mathrm{p_{T}}$", is_independent=True, is_binned=False, units="GeV")
    subpT.values = graph_L1["x"]

    ### add variables and add table to submission
    table.add_variable(subpT)
    table.add_variable(makeVariable(plot = graph_L1, label = "Dimuon Level-1 seeds", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot = graph_HLT, label = "Dimuon HLT scouting trigger", is_independent=False, is_binned=False, is_symmetric=False, units=""))

    return table


# Figure 49
def makeScoutingMuonSigEffVSLxyTable():
    table = Table("Scouting dimuon trigger eff vs Lxy")
    table.description = "L1T+HLT efficiency of the dimuon scouting trigger as a function of the generator-level $L_{xy}$, for HAHM signal events, for 2024 conditions. The efficiency is shown for $m_{Z_D} = 14$ GeV and $c\\tau = 100$ mm (pink squares) and $m_{Z_D} = 2.5$ GeV and $c\\tau = 100$ mm (blue triangles). The muons are required to have $p_{T} > 15$ GeV and $|\\eta| < 2.4$ at the generator level."

    table.location = "Data from Fig. 49"
    table.add_image("data_Celia/Figure_049.pdf")
    reader = RootFileReader("data_Celia/Scouting_signalEfficiency_lxy.root")
    graph_2p5 = reader.read_teff("efficiency_minlxy_DoubleMuonNoVtx_2p5_100mm;1")
    graph_14 = reader.read_teff("efficiency_minlxy_DoubleMuonNoVtx_14_100mm;1")

    lxy = Variable("gen $\mathrm{L_{xy}}$", is_independent=True, is_binned=False, units="cm")
    lxy.values = graph_2p5["x"]

    ### add variables and add table to submission
    table.add_variable(lxy)
    table.add_variable(makeVariable(plot = graph_2p5, label = "$m_{Z_D} = 2.5$ GeV, $c\\tau = 100$ mm", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot = graph_14, label = "$m_{Z_D} = 14$ GeV, $c\\tau = 100$ mm", is_independent=False, is_binned=False, is_symmetric=False, units=""))

    return table

# Figure 50
def makeScoutingMuonSigEffVSPtTable(mass):
    table = Table("Scouting dimuon trigger eff vs sub $\mathrm{p_{T}}$ for m = " + mass)
    table.description = "L1T+HLT efficiency of the dimuon scouting trigger as a function of the generator-level subleading muon $\mathrm{p_{T}}$, for HAHM signal events for 2024 conditions. The efficiency is shown for $m_{Z_D}$ masses of 2.5 and 14 GeV, and $c\\tau$ values of 1 (purple squares), 10 (blue triangles), and 100 mm (pink circles). The muons are required to have $|\eta|<2.4$ at the generator level."
    
    if mass=="2.5":
        table.location = "Data from Fig. 50 left"
        table.add_image("data_Celia/Figure_050-a.pdf")
        reader = RootFileReader("data_Celia/Scouting_signalEfficiency_pt_mZD-2p5GeV.root")
        graph_1mm = reader.read_teff("efficiency_minpt_DoubleMuonNoVtx_2p5_1mm;1")
        graph_10mm = reader.read_teff("efficiency_minpt_DoubleMuonNoVtx_2p5_10mm;1")
        graph_100mm = reader.read_teff("efficiency_minpt_DoubleMuonNoVtx_2p5_100mm;1")
    if mass=="14":
        table.location = "Data from Fig. 50 right"
        table.add_image("data_Celia/Figure_050-b.pdf")
        reader = RootFileReader("data_Celia/Scouting_signalEfficiency_pt_mZD-14GeV.root")
        graph_1mm = reader.read_teff("efficiency_minpt_DoubleMuonNoVtx_14_1mm;1")
        graph_10mm = reader.read_teff("efficiency_minpt_DoubleMuonNoVtx_14_10mm;1")
        graph_100mm = reader.read_teff("efficiency_minpt_DoubleMuonNoVtx_14_100mm;1")

    subpT = Variable("sub $\mathrm{p_{T}}$", is_independent=True, is_binned=False, units="GeV")
    subpT.values = graph_1mm["x"]

    ### add variables and add table to submission
    table.add_variable(subpT)
    table.add_variable(makeVariable(plot = graph_1mm, label = "$m_{Z_D} = %s$ GeV, $c\\tau = 1$ mm"%(mass), is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot = graph_10mm, label = "$m_{Z_D} = %s$ GeV, $c\\tau = 10$ mm"%(mass), is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot = graph_100mm, label = "$m_{Z_D} = %s$ GeV, $c\\tau = 100$ mm"%(mass), is_independent=False, is_binned=False, is_symmetric=False, units=""))

    return table

# Figure 51
def makeScoutingMuonRecoEffVSLxyTable(mass):
    table = Table("Scouting reconstruction eff vs Lxy for m = " + mass)
    table.description = "Scouting muon reconstruction efficiency of the vertex-constrained (pink circles) and vertex-unconstrained (blue triangles) algorithms as a function of the generator-level $L_{xy}$, for HAHM signal events for 2024 conditions. This efficiency is representative of the reconstruction efficiency of the L2 and L3 HLT muon reconstruction employed in scouting data. The efficiency is shown for $m_{Z_D} = 2.5$ GeV and $c\\tau = 100$ mm and $m_{Z_D} = 14$ GeV and $c\\tau = 100$ mm. The muons are required to have $p_{T} > 15$ GeV and $|\\eta| < 2.4$ at the generator level."
    
    if mass=="2.5":
        table.location = "Data from Fig. 51 left"
        table.add_image("data_Celia/Figure_051-a.pdf")
        reader = RootFileReader("data_Celia/Scouting_signalEfficiency_lxy_2p5GeV_100mm.root")
        graph_vtx = reader.read_teff("efficiency_lxy_vtx_2p5_100mm;1")
        graph_novtx = reader.read_teff("efficiency_lxy_novtx_2p5_100mm;1")
    if mass=="14":
        table.location = "Data from Fig. 51 right"
        table.add_image("data_Celia/Figure_051-b.pdf")
        reader = RootFileReader("data_Celia/Scouting_signalEfficiency_lxy_14GeV_100mm.root")
        graph_vtx = reader.read_teff("efficiency_lxy_vtx_14_100mm;1")
        graph_novtx = reader.read_teff("efficiency_lxy_novtx_14_100mm;1")

    lxy = Variable("gen $\mathrm{L_{xy}}$", is_independent=True, is_binned=False, units="cm")
    lxy.values = graph_vtx["x"]

    ### add variables and add table to submission
    table.add_variable(lxy)
    table.add_variable(makeVariable(plot = graph_vtx, label = "Vertex-constrained reconstruction", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot = graph_novtx, label = "Vertex-unconstrained reconstruction", is_independent=False, is_binned=False, is_symmetric=False, units=""))

    return table

# Figure 52
def makeScoutingMuonResolutionTable():
    table = Table("Scouting resolution vs pt")
    table.description = "The $p_{T}$ resolution of scouting muons with respect to offline muons, as a function of the scouting muon $p_{T}$, for 2024 data events. The root mean square (RMS) of the difference of the scouting muon $p_{T}$ and the offline muon $p_{T}$, divided by the offline muon $p_{T}$, is plotted. The dimuon $\\Delta R$ is required to be greater than 0.2, and the scouting muon $p_{T}$ is required to be greater than 3 GeV. The resolution is shown for muons in the barrel (blue filled points) and the endcaps (purple filled triangles) that are reconstructed with both the vertex-unconstrained reconstruction algorithm, as well as for muons in the barrel (red filled squares) and the endcaps (orange unfilled squares) that are reconstructed with the vertex-constrained reconstruction algorithm. A special monitoring data set is used that collects events triggered by a mixture of HLT paths (both scouting and standard triggers) with a very high prescale, in which all information about the muon objects is stored from the offline and scouting reconstruction."

    table.location = "Data from Fig. 52"
    table.add_image("data_Celia/LLP-Paper_bothReco_ptres_graph_BE_2024.pdf")

    reader_noVtxMu = RootFileReader("data_Celia/TGraph_ptres_noVtxMu_v2.root")
    graph_noVtxMu_B = reader_noVtxMu.read_graph("noVtxMu_B;1")
    graph_noVtxMu_E = reader_noVtxMu.read_graph("noVtxMu_E;1")

    reader_vtxMu = RootFileReader("data_Celia/TGraph_ptres_vtxMu_v2.root")
    graph_vtxMu_B = reader_vtxMu.read_graph("vtxMu_B;1")
    graph_vtxMu_E = reader_vtxMu.read_graph("vtxMu_E;1")

    pT = Variable("$\mathrm{p_{T}}$", is_independent=True, is_binned=False, units="GeV")
    pT.values = graph_noVtxMu_B["x"]

    ### add variables and add table to submission
    table.add_variable(pT)
    table.add_variable(makeVariable(plot = graph_noVtxMu_B, label = "Barrel: vertex-unconstrained reconstruction", is_independent=False, is_binned=False, is_symmetric=True, units=""))
    table.add_variable(makeVariable(plot = graph_noVtxMu_E, label = "Endcap: vertex-unconstrained reconstruction", is_independent=False, is_binned=False, is_symmetric=True, units=""))
    table.add_variable(makeVariable(plot = graph_vtxMu_B, label = "Barrel: vertex-constrained reconstruction", is_independent=False, is_binned=False, is_symmetric=True, units=""))
    table.add_variable(makeVariable(plot = graph_vtxMu_E, label = "Encap: vertex-constrained reconstruction", is_independent=False, is_binned=False, is_symmetric=True, units=""))

    return table


def makeMETIsoTrkHLTPathEffTable(tableName,fileName,isSignal,xAxisLabel,xAxisUnits):
    table = Table(tableName)
    if isSignal: table.description = "L1T+HLT efficiency of the MET+IsoTrk trigger as a function of the number of tracker layers with valid measurements of the track that pass the offline requirements, in $\\tilde{\\chi}_{1}^{\\pm} \\rightarrow \\tilde{\\chi}_{1}^{0}$+X simulated events for 2022 conditions, where $m_{\\tilde{\\chi}_{1}^{\\pm}}=900$ GeV and $\\tilde{\\chi}_{1}^{0}$ is nearly mass-degenerate with $\\tilde{\\chi}_{1}^{\\pm}$. The efficiency is shown for LLPs with $c\\tau=$ 10, 100, and 1000 cm in black, blue, and red, respectively."
    else: table.description = "Comparison of L1T+HLT efficiencies of the MET+IsoTrk trigger calculated with 2022 data (black), 2023 data (blue), and $W \\rightarrow l \\nu$ simulation (red), as a function of offline reconstructed PF $p_{T}^{miss, \\mu \\hspace{-0.15cm} /}$ (right). The data follow the typical rise in efficiency, but the efficiency does not reach 100% because of the isolated track leg of the algorithm."

    image = "data_Breno/" + fileName + ".pdf"
    reader = RootFileReader("data_Breno/" + fileName + ".root")
    complement = ''
    if isSignal: complement = 'left'
    else: complement = 'right'
    table.location = "Data from Fig. 10 " + complement
    table.add_image(image)

    plot_1 = ""
    plot_2 = ""
    plot_3 = ""

    if isSignal:
        plot_1 = reader.read_hist_1d("overallEff_HLT_MET105_IsoTrk50_10cm")
        plot_2 = reader.read_hist_1d("overallEff_HLT_MET105_IsoTrk50_100cm")
        plot_3 = reader.read_hist_1d("overallEff_HLT_MET105_IsoTrk50_1000cm")
    else:
        plot_1 = reader.read_graph("HLT_MET105_IsoTrk50_2022Data")
        plot_2 = reader.read_graph("HLT_MET105_IsoTrk50_2023Data")
        plot_3 = reader.read_graph("HLT_MET105_IsoTrk50_MC")

    xAxisVar = Variable(xAxisLabel, is_independent=True, is_binned=False, units=xAxisUnits)
    if isSignal: xAxisVar.values = plot_1["x_labels"]
    else: xAxisVar.values = plot_1["x"]

    table.add_variable(xAxisVar)

    label_1 = ""
    label_2 = ""
    label_3 = ""

    if isSignal:
        label_1 = "$m_{\\tilde{\chi}_{1}^{\pm}} = 900$ GeV, $c \\tau = 10$ cm"
        label_2 = "$m_{\\tilde{\chi}_{1}^{\pm}} = 900$ GeV, $c \\tau = 100$ cm"
        label_3 = "$m_{\\tilde{\chi}_{1}^{\pm}} = 900$ GeV, $c \\tau = 1000$ cm"
    else:
        label_1 = "2022 data"
        label_2 = "2023 data"
        label_3 = "$W \\rightarrow l \\nu$"

    if isSignal:
        table.add_variable(makeVariable(plot=plot_1, label=label_1, is_independent=False, is_binned=False, is_symmetric=True, units=""))
        table.add_variable(makeVariable(plot=plot_2, label=label_2, is_independent=False, is_binned=False, is_symmetric=True, units=""))
        table.add_variable(makeVariable(plot=plot_3, label=label_3, is_independent=False, is_binned=False, is_symmetric=True, units=""))
    else:
        table.add_variable(makeVariable(plot=plot_1, label=label_1, is_independent=False, is_binned=False, is_symmetric=False, units=""))
        table.add_variable(makeVariable(plot=plot_2, label=label_2, is_independent=False, is_binned=False, is_symmetric=False, units=""))
        table.add_variable(makeVariable(plot=plot_3, label=label_3, is_independent=False, is_binned=False, is_symmetric=False, units=""))

    return table

def makeMETAndIsoTrkHLTPathEffTable(tableName,fileName,isMET,xAxisLabel):
    table = Table(tableName)

    if isMET: table.description = "Efficiency of the L1T+HLT $p_{T}^{miss}$ leg as a function of offline reconstructed PF $p_{T}^{miss, \\mu \\hspace{-0.15cm} /}$ in 2022 data (black), 2023 data (blue), and $W \\rightarrow l \\nu$ simulation (red)."
    else: table.description = "Efficiency of the full HLT path, taking into account only events that already passed through the $p_{T}^{miss}$ leg, as a function of the selected muon $p_{T}$ in 2022 data (black), 2023 data (blue), and $W \\rightarrow l \\nu$ simulation (red)."

    image = "data_Breno/" + fileName + ".pdf"
    reader = RootFileReader("data_Breno/" + fileName + ".root")
    complement = ''
    if isMET: complement = 'left'
    else: complement = 'right'
    table.location = "Data from Fig. 11 " + complement
    table.add_image(image)

    plot_1 = ""
    plot_2 = ""
    plot_3 = ""

    if isMET:
        plot_1 = reader.read_graph("filterMET105_2022Data")
        plot_2 = reader.read_graph("filterMET105_2023Data")
        plot_3 = reader.read_graph("filterMET105_MC")
    else:
        plot_1 = reader.read_graph("filterIsoTrk50_2022Data")
        plot_2 = reader.read_graph("filterIsoTrk50_2023Data")
        plot_3 = reader.read_graph("filterIsoTrk50_MC")

    xAxisVar = Variable(xAxisLabel, is_independent=True, is_binned=False, units="GeV")
    xAxisVar.values = plot_1["x"]
    table.add_variable(xAxisVar)

    table.add_variable(makeVariable(plot=plot_1, label="2022 data", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot=plot_2, label="2023 data", is_independent=False, is_binned=False, is_symmetric=False, units=""))
    table.add_variable(makeVariable(plot=plot_3, label="$W \\rightarrow l\\nu$", is_independent=False, is_binned=False, is_symmetric=False, units=""))

    return table

def main():
    # Check if ImageMagick is available for image processing
    has_imagemagick = check_imagemagick_available()
    if has_imagemagick:
        print("✓ ImageMagick available - images will be processed")
    else:
        print("⚠ ImageMagick not available - skipping image processing")
    
    # Create the submission object
    submission = Submission()
    
    # Add general submission metadata
    #submission.comment = "Strategy and performance of CMS long-lived particle triggers in proton-proton collisions at sqrt(s) = 13.6 TeV. This submission contains trigger efficiency measurements for displaced muon triggers during CMS Run 3 (2022-2024)."

    submission.read_abstract("data_Juliette/abstract.txt")
    #submission.add_link("Webpage with all figures and tables", "https://cms-results.web.cern.ch/cms-results/public-results/publications/EXO-23-016/")
    #submission.add_link("arXiv", "http://arxiv.org/abs/arXiv:")
    #submission.add_record_id(1940976, "inspire")

    # Create output directory early
    output_dir = "hepdata_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    successful_figures = 0

    # Figure 2
    submission.add_table(makeTrackingEfficiencyTable("Offline"))
    submission.add_table(makeTrackingEfficiencyTable("HLT"))

    # Figure 10
    submission.add_table(makeMETIsoTrkHLTPathEffTable("MET+IsoTrk efficiency vs Tracker layers with measurement","900_10vs100vs1000_HLT_MET105_IsoTrk50",True,"Number of tracker layers with measurement",""))
    submission.add_table(makeMETIsoTrkHLTPathEffTable("MET+IsoTrk efficiency vs PF missing transverse momentum","moreLogNoExponent_2022vs2023vMC_HLT_MET105_IsoTrk50_v",False,"PF $p_{T}^{miss, \\mu \\hspace{-0.15cm} /}$","GeV"))

    # Figure 11
    submission.add_table(makeMETAndIsoTrkHLTPathEffTable("MET filter efficiency vs PF missing transverse momentum","moreLogNoExponent_2022vs2023vMC_filterMET105",True,"PF $p_{T}^{miss, \\mu \\hspace{-0.15cm} /}$"))
    submission.add_table(makeMETAndIsoTrkHLTPathEffTable("IsoTrk filter efficiency vs PF missing transverse momentum","moreLogNoExponent_2022vs2023vMC_filterIsoTrk50",False,"Muon $p_{T}$"))

    # Figure 13
    submission.add_table(makeDisplacedTauEffTable('MET'))
    submission.add_table(makeDisplacedTauEffTable('d0'))
    
    # Figure 14
    submission.add_table(makeDiTauRateTable('2022'))
    submission.add_table(makeDiTauRateTable('2023'))

    # Figure 16
    submission.add_table(makeHT430EffTable())
    submission.add_table(makeHT390EffTable())
    
    #Figure 17
    submission.add_table(makePt40EffTable())
    submission.add_table(makePtrkEffTable())
    
    #Figure 18
    submission.add_table(makeGenEffTable())
    
    #Figure 19
    submission.add_table(makeGainTable())

    # Figure 21
    submission.add_table(makeHcalTowerEffTable())

    # Figure 22
    submission.add_table(makeHcalLLPflaggedJetEffTable())

    # Figure 23
    submission.add_table(makeHcalL1JetHTEffTable("HT"))
    submission.add_table(makeHcalL1JetHTEffTable("jet"))

    # Figure 24
    submission.add_table(makeHcalL1DecayREffTable())

    # Figure 25
    submission.add_table(makeCalRatioJetEffTable())
    submission.add_table(makeCalRatioJetDistributionTable())

    # Figure 27
    submission.add_table(makeDelayedJetEfficiencyTable())

    # Figure 28
    submission.add_table(makeDelayedHTTauTable("data_Neha/Signal_efficiency_HT430vsL1Tau_HtoXXto4b.root", "4b", "4b", "(left)"))
    submission.add_table(makeDelayedHTTauTable("data_Neha/Signal_efficiency_HT430vsL1Tau_HtoXXto4tau.root", "4tau", "4\\tau", "(right)"))

    # Figure 29
    submission.add_table(makeDelayedJetTimeTable("data_Neha/HLT_HTg430_Delayedjet2nsthreshold_efficiency.root","$H_{T}$-seeded delayed jet trigger efficiency vs jet time"," HLT selection: $H_{T}> 430 GeV$, $\\geq 1$ jet with $p_{T} > 40 GeV$ and $t > 2 ns$.","(left)"))
    submission.add_table(makeDelayedJetTimeTable("data_Neha/HLT_L1Tau_Delayedjet3p5nsthreshold_efficiency.root","L1Tau-seeded delayed jet trigger efficiency vs jet time"," HLT selection: L1Tau, $\\geq 1$ jet with $p_{T} > 40 GeV$ and $t > 3.5 ns$.","(right)"))
    
    # Figure 31
    submission.add_table(makeDelayedDiPhotonHistTable("eb"))
    submission.add_table(makeDelayedDiPhotonHistTable("ee"))

    # Figure 32
    submission.add_table(makeDelayedDiPhotonDataRateTable())
    submission.add_table(makeDiphotonRateTable())

    # Figure 33
    submission.add_table(makeDelayedDiPhotonDataEffTable("seed time ($\mathrm{e_{2}}$)"))
    
    # Figure 34
    submission.add_table(makeDelayedDiPhotonDataEffTable("$p_{T}$ ($\mathrm{e_{2}}$)"))
    submission.add_table(makeDelayedDiPhotonDataEffTable("$\eta$ ($\mathrm{e_{2}}$)"))

    # Figure 37
    submission.add_table(makeDisplPhotonRateTable('2022'))
    submission.add_table(makeDisplPhotonRateTable('2023'))

    #Figure 39
    submission.add_table(makeDisplacedMuonL1EffTable("BMTF"))
    submission.add_table(makeDisplacedMuonL1EffTable("OMTF"))
    submission.add_table(makeDisplacedMuonL1EffTable("EMTF"))
    
    # Process existing YAML files from fromDisplacedDimuons directory FIRST (for Figure 40)
    yaml_dir = "data_Alejandro/fromDisplacedDimuons"
    if os.path.exists(yaml_dir):
        yaml_files = [f for f in os.listdir(yaml_dir) if f.endswith('.yaml')]
        yaml_files.sort()  # Process in alphabetical order
        for yaml_file in yaml_files:
            figure_name = yaml_file.replace('.yaml', '')
            yaml_file_path = os.path.join(yaml_dir, yaml_file)
            
            if process_existing_yaml_file(submission, yaml_file_path, figure_name):
                successful_figures += 1
            else:
                print(f"Failed to process existing YAML {figure_name}")
    
    # Process ROOT files (Figures 41-43)
    data_dir = "data_Alejandro"
    root_files = [f for f in os.listdir(data_dir) if f.endswith('.root')]
    root_files.sort()  # Process in alphabetical order
    
    for root_file in root_files:
        figure_name = root_file.replace('.root', '')
        root_file_path = os.path.join(data_dir, root_file)
        
        if process_single_figure(submission, root_file_path, figure_name):
            successful_figures += 1
        else:
            print(f"Failed to process {figure_name}")

    print(f"\nSuccessfully processed {successful_figures} figures")

    #Figure 45
    submission.add_table(makeDoubleDispL3MuonSigEffTable())

    #Figure 46
    submission.add_table(makeDoubleDispL3MuonDataMCEffTable("min($\mathrm{d_{0}}$)"))
    submission.add_table(makeDoubleDispL3MuonDataMCEffTable("min($\mathrm{p_{T}}$)"))

    #Figure 48
    submission.add_table(makeScoutingMuonDataEffVSPtTable())

    #Figure 49
    submission.add_table(makeScoutingMuonSigEffVSLxyTable())
    
    #Figure 50
    submission.add_table(makeScoutingMuonSigEffVSPtTable("2.5")) # left
    submission.add_table(makeScoutingMuonSigEffVSPtTable("14")) # right
    
    #Figure 51
    submission.add_table(makeScoutingMuonRecoEffVSLxyTable("2.5")) # left
    submission.add_table(makeScoutingMuonRecoEffVSLxyTable("14")) # right

    #Figure 52
    submission.add_table(makeScoutingMuonResolutionTable())

    #Figure 56
    submission.add_table(makeFig56leftTable(histograms))
    submission.add_table(makeFig56rightTable(histograms))
    
    #Figure 58
    submission.add_table(makeMuonNoBPTXRateVsNBunchesTable("2016"))
    submission.add_table(makeMuonNoBPTXRateVsNBunchesTable("2017"))
    submission.add_table(makeMuonNoBPTXRateVsNBunchesTable("2018"))
    submission.add_table(makeMuonNoBPTXRateVsNBunchesTable("2022"))
    submission.add_table(makeMuonNoBPTXRateVsNBunchesTable("2023"))
    submission.add_table(makeMuonNoBPTXRateVsNBunchesTable("2024"))

    #old Figure 65
    #new Figure 61
    mass_points = [(125,25), (350,80), (350,160), (1000,200)]
    for mH, mX in mass_points:
        for t in makeAcceptanceTables(mH, mX, "R"):
            submission.add_table(t)
    #old Figure 66
    #new Figure 62
    for mH, mX in mass_points:
        for t in makeAcceptanceTables(mH, mX, "Z"):
            submission.add_table(t)

    #old Figures 60-64
    #new Figures 63-67
    submission.add_table(makeFig60table(histograms))
    submission.add_table(makeFig61table(histograms))
    submission.add_table(makeFig62table(histograms))
    submission.add_table(makeFig63leftTable(histograms))
    submission.add_table(makeFig63rightTable(histograms))
    submission.add_table(makeFig64table(histograms))

    #old Figure 70
    #new Figure 68
    submission.add_table(makeDisplacedTauAccTable())

    #old Figure 69
    #new Figure 70
    submission.add_table(makeHLTMuResoTable("genpt"))
    submission.add_table(makeHLTMuResoTable("genlxy"))
    
    for table in submission.tables:
        table.keywords["cmenergies"] = [13000,13600]
    
    # Generate HEPData files
    try:
        submission.create_files(output_dir,remove_old=True)
        print(f"\nHEPData files created in '{output_dir}' directory")
    except Exception as e:
        print(f"Error creating HEPData files: {e}")
        if "ImageMagick" in str(e) or "convert" in str(e) or "does not exist" in str(e):
            print("This appears to be an ImageMagick-related error.")
            print("Attempting to create submission without images...")
            
            # Remove all images from tables and try again
            for table in submission.tables:
                if hasattr(table, 'image_files'):
                    table.image_files = []

            try:
                submission.create_files(output_dir,remove_old=True)
                print(f"\nHEPData files created in '{output_dir}' directory (without images)")
            except Exception as e2:
                print(f"Failed to create submission even without images: {e2}")
                raise
        else:
            raise
    print("Files generated:")
    for file in sorted(os.listdir(output_dir)):
        print(f"  - {file}")
    
    # Create tar.gz archive (hepdata_lib creates submission.tar.gz automatically)
    
    print(f"\nProcessing complete! Generated HEPData submission with {successful_figures} figures.")
    print("\nSubmission includes:")
    print("  - Trigger efficiency measurements from CMS Run 3 (2022-2024)")
    print("  - Figures 40, 41a-c, 42a-c, and 43a-b from EXO-23-016")
    print("  - Complete metadata and figure descriptions from the paper")
    print("  - Statistical uncertainties for all measurements")
    print("  - Proper decay length (cτ) dependence studies for displaced dimuons")


    #submission.add_additional_resource("Signal generation", "data_Juliette/signalGeneration.tar.gz", copy_file=True)                                                                            
    #submission.create_files("hepdataRecord",remove_old=True)
    
    
if __name__ == "__main__":
    main()
