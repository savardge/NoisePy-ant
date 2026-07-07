"""Shared YAML-configuration helper for the mode-separation picking workflow.

One YAML file per network (template: param_files/modesep_params.yaml) drives the whole
chain: vsg_modesep.py -> pick_reference_ridges.py -> dispersion_batch_modesep.py ->
network_station_qc.py / network_consistency_analysis.py / network_pick_maps.py /
lateral_structure_map.py.

Precedence everywhere: explicit CLI flag > YAML value > built-in default.

Note: dispersion_batch_modesep.py also still honours the deprecated DISP_NET /
DISP_REF_DIR / DISP_LIMIT environment variables, but only when run WITHOUT --config.
"""
import glob
import os
import numpy as np
import yaml


def load_config(path):
    """Read the YAML and fill the derived path defaults.

    Derived (when null/absent):
      paths.dispersion_dir = {paths.project_dir}/dispersion_V6
      paths.ref_dir        = {paths.project_dir}/vsg_modesep
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for sec in ("network", "paths", "vsg_modesep", "pick_windows", "batch", "analysis"):
        cfg.setdefault(sec, {})
    p = cfg["paths"]
    proj = p.get("project_dir")
    if not p.get("dispersion_dir") and proj:
        p["dispersion_dir"] = os.path.join(proj, "dispersion_V6")
    if not p.get("ref_dir") and proj:
        p["ref_dir"] = os.path.join(proj, "vsg_modesep")
    return cfg


def apply_overrides(cfg, args, mapping):
    """Overwrite cfg values with non-None argparse values.

    mapping: {argparse_attr: (section, key)}, e.g. {"sign": ("vsg_modesep", "sign")}.
    Returns cfg (modified in place) for convenience.
    """
    for attr, (sec, key) in mapping.items():
        val = getattr(args, attr, None)
        if val is not None:
            cfg.setdefault(sec, {})[key] = val
    return cfg


def ref_curve_paths(cfg):
    """{'fundamental': ..., 'overtone': ...} reference-curve files under paths.ref_dir."""
    ref_dir = cfg["paths"]["ref_dir"]
    return {b: os.path.join(ref_dir, f"ref_{b}_phase.txt")
            for b in ("fundamental", "overtone")}


def vsg_station_coords(vsg_dir):
    """Harvest station lon/lat from the per-virtual-source npz files (ZZ/sources/*.npz).

    Every source file stores its own coordinates plus those of all its receivers, so one
    component suffices to cover the network. Returns {station_code: (lon, lat)}.
    """
    coords = {}
    for f in glob.glob(os.path.join(vsg_dir, "ZZ", "sources", "*.npz")):
        z = np.load(f, allow_pickle=True)
        if np.isfinite(z["src_lon"]) and np.isfinite(z["src_lat"]):
            coords[str(z["src"])] = (float(z["src_lon"]), float(z["src_lat"]))
        for c, lo, la in zip(z["rx_codes"], z["rx_lons"], z["rx_lats"]):
            if np.isfinite(lo) and np.isfinite(la):
                coords[str(c)] = (float(lo), float(la))
    return coords
