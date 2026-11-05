"""Shared method colors and display labels for the plotting tools.

Both plot_pareto.py and plot_curves.py import from here so figures stay
visually consistent. Add new methods to METHOD_COLOR / METHOD_LABEL in
this file only -- never duplicate the dicts in the plot scripts.

Methods that aren't in the explicit dicts get a *deterministic* color
from a fallback palette (md5(method) -> palette index), so every plot
that includes the same unknown method shows it in the same color across
runs. Labels for unknown methods fall back to the dirname with
underscores replaced by spaces.
"""
from __future__ import annotations
import hashlib


METHOD_COLOR = {
    # baselines
    "toes":           "#1f77b4",
    "nonuniform_erg": "#2ca02c",   # Time-warp
    "opt_transport":  "#d62728",
    "sac":            "#9467bd",
    # SAC kinematic-regulariser sweep (run_sac_velpen.sh)
    "sac_velpen00":   "#7f7f7f",   # beta=0 -- sanity / no penalty
    "sac_velpen010":  "#9467bd",   # beta=0.1
    "sac_velpen020":  "#5e2f8c",   # beta=0.2 -- darker purple
    # FMEC baselines
    "fmec_stein":     "#e6550d",
    "fmec_sinkhorn":  "#fdae6b",
    # OT-CFM family
    "otcfm":          "#ff7f0e",
    # legacy short tags (kept so older result dirs still render correctly)
    "otcfm_nfz":          "#8c564b",
    "otcfm_energy":       "#17becf",
    "otcfm_energy_nfz":   "#bcbd22",
    # current winner-tagged dirs (one per loss combo)
    "otcfm_nfz_l5":                              "#8c564b",
    "otcfm_energy_lambda1e-2":                   "#17becf",
    "otcfm_energy_la1e-4":                       "#c5b0d5",
    "otcfm_energy_lambda1e-3_nfz_l500":          "#bcbd22",
    "otcfm_energy_la1e-1_nfz_l500_le1e-3":       "#e377c2",
    # NFZ-masked ablations
    "otcfm_mask_only":                           "#1f9e89",
    "otcfm_mask_plus_penalty":                   "#26828e",
    # velocity-penalty variants
    "otcfm_mask_vel":                            "#f768a1",
}
DEFAULT_COLOR = "#666666"


# Fallback palette: 16 visually distinct colors used for *unknown* methods.
# Picked to avoid collisions with the explicit METHOD_COLOR entries above
# while still printing legibly on white. Method -> color is determined by
# md5(method) % len(palette), so the same dirname always gets the same
# color across runs.
_FALLBACK_PALETTE = [
    "#393b79", "#637939", "#8c6d31", "#843c39",
    "#7b4173", "#5254a3", "#9c9ede", "#cedb9c",
    "#e7969c", "#bd9e39", "#ad494a", "#a55194",
    "#6b6ecf", "#b5cf6b", "#e7ba52", "#ce6dbd",
]


def _hash_color(method: str) -> str:
    h = int(hashlib.md5(method.encode("utf-8")).hexdigest(), 16)
    return _FALLBACK_PALETTE[h % len(_FALLBACK_PALETTE)]


def _prettify_label(method: str) -> str:
    """Best-effort human-readable label from an arbitrary dirname.

    Recognises a few common suffix patterns, otherwise just replaces
    underscores with spaces.

      otcfm_mask_only_center_0_6_0_6    -> 'OT-CFM (mask only) @ (0.6, 0.6)'
      otcfm_mask_plus_penalty_center... -> 'OT-CFM (mask+penalty) @ (...)'
      nonuniform_erg_nfz_center_-0_6... -> 'Time-warp @ (-0.6, -0.6)'
    """
    import re
    suffix = ""
    m = re.search(
        r"_(?:nfz_)?center_(-?\d+)_(\d+)_(-?\d+)_(\d+)$", method)
    if m:
        cx_int, cx_frac, cy_int, cy_frac = m.groups()
        cx = float(f"{cx_int}.{cx_frac}")
        cy = float(f"{cy_int}.{cy_frac}")
        suffix = f" @ ({cx:.1f}, {cy:.1f})"
        method = method[:m.start()]
    # Map known prefixes to nicer base names
    prefix_map = [
        ("otcfm_mask_vel",         "OT-CFM (mask+vel)"),
        ("otcfm_mask_only",        "OT-CFM (mask only)"),
        ("otcfm_mask_plus_penalty","OT-CFM (mask+penalty)"),
        ("otcfm_energy",           "OT-CFM +E"),
        ("otcfm_nfz",              "OT-CFM +NFZ"),
        ("otcfm",                  "OT-CFM"),
        ("nonuniform_erg",         "Time-warp"),
        ("opt_transport",          "OT"),
        ("fmec_stein",             "FMEC-Stein"),
        ("fmec_sinkhorn",          "FMEC-Sinkhorn"),
        ("sac_velpen",             "SAC velpen"),
        ("sac",                    "SAC"),
        ("toes",                   "TOES"),
    ]
    for prefix, name in prefix_map:
        if method == prefix or method.startswith(prefix + "_"):
            tail = method[len(prefix):].lstrip("_").replace("_", " ")
            return name + (f" {tail}" if tail else "") + suffix
    return method.replace("_", " ") + suffix


METHOD_LABEL = {
    # baselines
    "toes":           "TOES",
    "nonuniform_erg": "Time-warp",
    "opt_transport":  "OT",
    "sac":            "SAC",
    # SAC velocity-penalty sweep
    "sac_velpen00":   r"SAC ($\beta{=}0$)",
    "sac_velpen010":  r"SAC ($\beta{=}0.1$)",
    "sac_velpen020":  r"SAC ($\beta{=}0.2$)",
    # FMEC baselines
    "fmec_stein":     "FMEC-Stein",
    "fmec_sinkhorn":  "FMEC-Sinkhorn",
    # OT-CFM family
    "otcfm":          "OT-CFM",
    # legacy short tags
    "otcfm_nfz":          "OT-CFM +NFZ",
    "otcfm_energy":       "OT-CFM +E",
    "otcfm_energy_nfz":   "OT-CFM +NFZ+E",
    # current winner-tagged dirs
    "otcfm_nfz_l5":                              "OT-CFM +NFZ",
    "otcfm_energy_lambda1e-2":                   "OT-CFM +E",
    "otcfm_energy_la1e-4":                       "OT-CFM +A",
    "otcfm_energy_lambda1e-3_nfz_l500":          "OT-CFM +NFZ+E",
    "otcfm_energy_la1e-1_nfz_l500_le1e-3":       "OT-CFM +NEA",
    # NFZ-masked ablations
    "otcfm_mask_only":                           "OT-CFM (mask only)",
    "otcfm_mask_plus_penalty":                   "OT-CFM (mask+penalty)",
    # velocity-penalty variants
    "otcfm_mask_vel":                            "OT-CFM (mask+vel)",
}


def color(method: str) -> str:
    """Color hex for a method tag.

    Explicit entries in METHOD_COLOR take precedence; unknown methods get
    a deterministic distinct color from a hashed fallback palette so each
    method shows up in its own consistent color across all plots.
    """
    if method in METHOD_COLOR:
        return METHOD_COLOR[method]
    return _hash_color(method)


def label(method: str) -> str:
    """Display label for a method tag.

    Explicit entries in METHOD_LABEL take precedence; unknown methods get
    a best-effort prettified label (known prefix + parsed center suffix
    when present, otherwise just dirname-with-spaces).
    """
    if method in METHOD_LABEL:
        return METHOD_LABEL[method]
    return _prettify_label(method)
