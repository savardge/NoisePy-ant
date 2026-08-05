#!/usr/bin/env python3
"""Rebuild the three multi-period overview figures from an existing production run.

Reads <run>/production/<wave>/map_T*.npz and writes into <run>/figures/<wave>/:

    maps_<wave>.png          absolute velocity, each panel on its own p5-p95 scale
    maps_<wave>_common.png   absolute velocity, one shared p2-p98 scale over all periods
    maps_<wave>_anomaly.png  % departure from THAT map's own mean, one shared scale

Same figures run_production.py emits, but decoupled from the inversion, so restyling costs
seconds instead of a re-run. Useful for runs produced before the figure code changed.

Colour choices: magma for absolute velocity (perceptually uniform, no false midpoint) and
RdBu_r for the anomaly (a relative departure DOES have a midpoint at zero; blue slow, red
fast). The anomaly limit is the median over periods of each map's p98 |anomaly| -- a pooled
percentile lets one pathological period set the scale and flatten everything else.

dx is read from the run directory name (..._dx0.2); station markers are drawn when
<run>/cache/stations_in_grid.csv is present, and skipped otherwise.

Usage:
  python plot_map_overviews.py <run_dir> [<run_dir> ...] [--waves fund overtone love]
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CMAP = "magma"
ALPHA_HIDDEN = 0.28   # plausibility-flagged cells: shown, de-emphasised
CMAP_ANOM = "RdBu"


def load_wave(run, wave):
    """[(T, Vm)] sorted by period, from the run's npz maps."""
    out = []
    for f in glob.glob(os.path.join(run, "production", wave, "map_T*.npz")):
        try:
            z = np.load(f)
            # --vplaus veil DISABLED (2026-08-04): inversion output is not masked on a
            # histogram of the input picks. Runs made with it on keep the veiled values in
            # `vel_hidden`, so merge them straight back into `vel` and draw nothing
            # separately. Vh is kept as an all-NaN array so the downstream overlay is a no-op.
            v = z["vel"]
            if "vel_hidden" in z.files:
                vh = z["vel_hidden"]
                v = np.where(np.isfinite(v), v, np.where(np.isfinite(vh), vh, np.nan))
            out.append((float(z["period"]), v, np.full_like(v, np.nan)))
        except Exception:
            continue
    return sorted(out, key=lambda t: t[0])


def load_picks(run, wave):
    """[(T, gv)] from the run's own cache -- the velocities that actually entered the solve.
    Period comes from the file name; the cache itself does not store it."""
    out = []
    for f in glob.glob(os.path.join(run, "cache", "%s_T*_std*_c*.npz" % wave)):
        m = re.search(r"_T([\d.]+)_std", os.path.basename(f))
        if not m:
            continue
        try:
            out.append((float(m.group(1)), np.asarray(np.load(f)["gv"], float)))
        except Exception:
            continue
    return sorted(out, key=lambda t: t[0])


def draw_vdist(run, wave, maps, picks, label):
    """Picks (data space) beside map cells (model space), on shared axes."""
    figdir = os.path.join(run, "figures", wave)
    os.makedirs(figdir, exist_ok=True)
    rungs = np.array([t for t, _, _ in maps], float)
    if len(rungs) < 3:
        return None
    mid = np.sqrt(rungs[1:] * rungs[:-1])
    tedges = np.concatenate(([rungs[0] ** 2 / mid[0]], mid, [rungs[-1] ** 2 / mid[-1]]))
    allv = np.concatenate([v[np.isfinite(v)] for _, v in picks]
                          + [m[np.isfinite(m)].ravel() for _, m, _ in maps])
    vhi = np.nanpercentile(allv, 99.5) if allv.size else 4.0
    vedges = np.arange(0.195, vhi + 0.1, 0.02)
    f2, a2 = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    for ax, (src, data) in zip(a2, (("picks (station pairs)", picks),
                                    ("map cells (inversion)",
                                     [(t, m) for t, m, _ in maps]))):
        Tl, Vl = [], []
        for t, v in data:
            v = np.asarray(v, float).ravel(); v = v[np.isfinite(v)]
            Tl.append(np.full(v.size, t)); Vl.append(v)
        if not Tl:
            continue
        Tl = np.concatenate(Tl); Vl = np.concatenate(Vl)
        H, _, _ = np.histogram2d(Tl, Vl, bins=[tedges, vedges])
        pos = H[H > 0]
        pc = ax.pcolormesh(tedges, vedges, np.where(H > 0, H, np.nan).T, cmap=CMAP,
                           vmin=0, vmax=np.percentile(pos, 99) if pos.size else 1,
                           shading="flat")
        plt.colorbar(pc, ax=ax, shrink=0.85,
                     label="%s per cell (p99 clip)"
                           % ("picks" if "picks" in src else "grid cells"))
        med = [np.median(np.asarray(v, float).ravel()[
                   np.isfinite(np.asarray(v, float).ravel())]) for _, v in data]
        ax.plot([t for t, _ in data], med, "-", color="deepskyblue", lw=1.8, label="median")
        ax.set_xlabel("period [s]  (CWT scale rungs)")
        ax.set_title("%s  --  %s" % (src, wave), fontsize=10)
        ax.legend(fontsize=8)
    a2[0].set_ylabel("%s [km/s]" % label)
    f2.suptitle("%s %s: velocity distribution vs period, data space vs model space"
                % (os.path.basename(run), wave), y=1.0)
    f2.tight_layout()
    out = os.path.join(figdir, "vdist_%s.png" % wave)
    f2.savefig(out, dpi=130); plt.close(f2)
    return out


def draw(run, wave, maps, dx, sta):
    figdir = os.path.join(run, "figures", wave)
    os.makedirs(figdir, exist_ok=True)
    ny, nx = maps[0][1].T.shape
    xc = (np.arange(maps[0][1].shape[0]) + 0.5) * dx
    yc = (np.arange(maps[0][1].shape[1]) + 0.5) * dx

    allv = np.concatenate([m[np.isfinite(m)].ravel() for _, m, _ in maps if np.isfinite(m).any()])
    glo, ghi = (np.nanpercentile(allv, [2, 98]) if allv.size else (None, None))
    per_map = []
    for _, m, _ in maps:
        mu = np.nanmean(m)
        if np.isfinite(mu) and mu:
            a = np.abs(100.0 * (m[np.isfinite(m)] - mu) / mu)
            if a.size:
                per_map.append(np.nanpercentile(a, 98))
    alim = max(float(np.nanmedian(per_map)) if per_map else 1.0, 1.0)

    label = "phase-velocity" if "phase" in os.path.basename(run).lower() else "group-velocity"
    for scaling in ("perperiod", "common", "anomaly"):
        n = len(maps); nc = min(5, n); nr = int(np.ceil(n / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(3.4 * nc, 3.2 * nr), squeeze=False)
        pc = None
        for ax, (T, Vm, Vh) in zip(axs.ravel(), maps):
            if scaling == "anomaly":
                mu = np.nanmean(Vm)
                Z = 100.0 * (Vm - mu) / mu if np.isfinite(mu) and mu else Vm * np.nan
                pc = ax.pcolormesh(xc, yc, Z.T, cmap=CMAP_ANOM, vmin=-alim, vmax=alim,
                                   shading="auto")
                if np.isfinite(Vh).any():
                    Zh = 100.0 * (Vh - mu) / mu if np.isfinite(mu) and mu else Vh * np.nan
                    ax.pcolormesh(xc, yc, Zh.T, cmap=CMAP_ANOM, vmin=-alim, vmax=alim,
                                  shading="auto", alpha=ALPHA_HIDDEN)
            else:
                vlo, vhi = (glo, ghi) if scaling == "common" \
                    else np.nanpercentile(Vm, [5, 95])
                pc = ax.pcolormesh(xc, yc, Vm.T, cmap=CMAP, vmin=vlo, vmax=vhi,
                                   shading="auto")
                if np.isfinite(Vh).any():
                    ax.pcolormesh(xc, yc, Vh.T, cmap=CMAP, vmin=vlo, vmax=vhi,
                                  shading="auto", alpha=ALPHA_HIDDEN)
                if scaling == "perperiod":
                    plt.colorbar(pc, ax=ax, shrink=0.8)
            if sta is not None:
                ax.plot(sta[0], sta[1], "^", ms=1.5, mfc="k", mec="none")
            ax.set_aspect("equal"); ax.set_title(f"T={T:g}s", fontsize=9)
        for ax in axs.ravel()[len(maps):]:
            ax.axis("off")
        txt = {"common": f"COMMON colour scale {glo:.2f}-{ghi:.2f} km/s (p2-p98 over all periods)",
               "anomaly": (f"RELATIVE anomaly vs each map's OWN mean, common scale "
                           f"+/-{alim:.1f}% (median over periods of each map's p98 "
                           f"|anomaly|; outlier periods saturate)")}.get(
                   scaling, "per-panel colour scale (p5-p95 of that period)")
        fig.suptitle(f"{os.path.basename(run)} {wave}: {label} maps\n{txt}", y=1.0)
        if scaling in ("common", "anomaly"):
            fig.subplots_adjust(right=0.90)
            cax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
            fig.colorbar(pc, cax=cax,
                         label=(f"d{label}/mean [%]" if scaling == "anomaly"
                                else f"{label} [km/s]"))
        else:
            fig.tight_layout()
        suffix = {"perperiod": "", "common": "_common", "anomaly": "_anomaly"}[scaling]
        out = os.path.join(figdir, f"maps_{wave}{suffix}.png")
        fig.savefig(out, dpi=120)
        plt.close(fig)
    return figdir


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="production run directories")
    ap.add_argument("--waves", nargs="+", default=["fund", "overtone", "love"])
    a = ap.parse_args()
    for run in a.runs:
        run = os.path.normpath(run)
        m = re.search(r"_dx([\d.]+)", os.path.basename(run))
        dx = float(m.group(1)) if m else 1.0
        sta = None
        sf = os.path.join(run, "cache", "stations_in_grid.csv")
        if os.path.exists(sf):
            import pandas as pd
            d = pd.read_csv(sf)
            if {"xstat", "ystat"} <= set(d.columns):
                sta = (d["xstat"].values, d["ystat"].values)
        for wave in a.waves:
            maps = load_wave(run, wave)
            if not maps:
                print("  %-46s %-9s no maps" % (os.path.basename(run), wave))
                continue
            draw(run, wave, maps, dx, sta)
            label = ("phase-velocity" if "phase" in os.path.basename(run).lower()
                     else "group-velocity")
            picks = load_picks(run, wave)
            nfig = 3
            if picks:
                if draw_vdist(run, wave, maps, picks, label):
                    nfig = 4
            print("  %-46s %-9s %d periods -> %d figures%s"
                  % (os.path.basename(run), wave, len(maps), nfig,
                     "" if picks else "   (no cache -> no vdist)"))


if __name__ == "__main__":
    main()
