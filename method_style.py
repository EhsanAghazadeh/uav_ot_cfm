"""Shared method colors and display labels for the plotting tools.

Both plot_pareto.py and plot_curves.py import from here so figures stay
visually consistent. Add new methods to METHOD_COLOR / METHOD_LABEL in
this file only — never duplicate the dicts in the plot scripts.
"""
from __future__ import annotations


METHOD_COLOR = {
    # baselines
    "toes":           "#1f77b4",
    "nonuniform_erg": "#2ca02c",
    "opt_transport":  "#d62728",
    "sac":            "#9467bd",
    # OT-CFM family
    "otcfm":          "#ff7f0e",
    # legacy short tags (kept so older result dirs still render correctly)
    "otcfm_nfz":          "#8c564b",
    "otcfm_energy":       "#17becf",
    "otcfm_energy_nfz":   "#bcbd22",
    # current winner-tagged dirs (one per loss combo)
    "otcfm_nfz_l5":                              "#8c564b",
    "otcfm_energy_lambda1e-2":                   "#17becf",
    "otcfm_energy_lambda1e-3_nfz_l500":          "#bcbd22",
    "otcfm_energy_la1e-1_nfz_l500_le1e-3":       "#e377c2",
}
DEFAULT_COLOR = "#666666"


METHOD_LABEL = {
    # baselines
    "toes":           "TOES",
    "nonuniform_erg": "NonUniformErg",
    "opt_transport":  "OT",
    "sac":            "SAC",
    # OT-CFM family
    "otcfm":          "OT-CFM",
    # legacy short tags
    "otcfm_nfz":          "OT-CFM +NFZ",
    "otcfm_energy":       "OT-CFM +E",
    "otcfm_energy_nfz":   "OT-CFM +NFZ+E",
    # current winner-tagged dirs
    "otcfm_nfz_l5":                              "OT-CFM +NFZ",
    "otcfm_energy_lambda1e-2":                   "OT-CFM +E",
    "otcfm_energy_lambda1e-3_nfz_l500":          "OT-CFM +NFZ+E",
    "otcfm_energy_la1e-1_nfz_l500_le1e-3":       "OT-CFM +NEA",
}


def color(method: str) -> str:
    """Color hex for a method tag, falling back to DEFAULT_COLOR."""
    return METHOD_COLOR.get(method, DEFAULT_COLOR)


def label(method: str) -> str:
    """Display label for a method tag, falling back to the tag itself."""
    return METHOD_LABEL.get(method, method)
