"""
How FTAN amplitude normalization affects PICKING (test on real pairs).

Per-period normalization makes every period's max = 1, so the group-Viterbi ridge (emission =
-amplitude) treats all periods equally and tracks the brightest velocity even where the signal
is genuinely weak. Global normalization ("max energy of the signal", Esteve/Shirzad 2025) keeps
the true amplitude, so the ridge is energy-weighted and a simple amplitude threshold drops the
low-energy (unreliable) part of the curve.

Panels: (1) per-period image + its ridge; (2) global image + per-period ridge (dashed) vs
energy-weighted ridge (cyan) + amplitude-threshold-kept picks (green); (3) genuine amplitude
along the per-period ridge vs period, with the threshold.

Usage: python compare_norm_picking.py <stack.h5> [comp=ZZ] [stack=Allstack_pws] [outdir=/tmp]
"""
import sys
import os
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from noisepy import dispersion as d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_picks_image as cp

logging.getLogger("findpeaks").setLevel(logging.ERROR)
AMP_THRESH = 0.10   # keep picks where genuine (global-normalised) energy >= this fraction of peak


def _ridge(amp, per, vel, coi, Tmax):
    rp, rv = d.extract_dispersion_viterbi(amp, per, vel, smooth_weight=cp.GROUP_SMOOTH,
                                          max_step=cp.GROUP_MAXSTEP, short_priority=cp.SHORT_PRIORITY,
                                          coi=coi)
    cat = np.array([coi[np.argmin(np.abs(vel - v))] for v in rv])
    m = (rp <= cat) & (rp <= Tmax)
    return rp[m], rv[m]


def make_figure(h5path, comp="ZZ", stack="Allstack_pws", outdir="/tmp"):
    pair = os.path.basename(h5path).replace(".h5", "")
    wave = cp.WAVE_OF.get(comp, "Rayleigh")
    sym, dist, dt = cp.read_sym(h5path, comp, stack)
    Tmin, dT, vmin, vmax, dvel, vave = cp.Tmin, cp.dT, cp.vmin, cp.vmax, cp.dvel, cp.vave
    Tmax = dist / vave

    cwt = d.compute_cwt(sym, dist, dt, Tmin=Tmin, vmin=vmin, vmax=vmax, vave=vave)
    imk = dict(Tmin=Tmin, dT=dT, vmin=vmin, vmax=vmax, dvel=dvel, vave=vave)
    amp_pp, per, vel, coi = d.disp_image_from_cwt(cwt, dist, normalize='per_period', **imk)
    amp_gl, _, _, _ = d.disp_image_from_cwt(cwt, dist, normalize='global', **imk)

    rp_pp, rv_pp = _ridge(amp_pp, per, vel, coi, Tmax)     # amplitude-blind ridge
    rp_gl, rv_gl = _ridge(amp_gl, per, vel, coi, Tmax)     # energy-weighted ridge

    # genuine energy along the per-period ridge (sampled from the global image)
    eint = lambda P, V: amp_gl[np.argmin(np.abs(per - P)), np.argmin(np.abs(vel - V))]
    e_pp = np.array([eint(P, V) for P, V in zip(rp_pp, rv_pp)])
    keep = e_pp >= AMP_THRESH

    ext = [per[0], per[-1], vel[0], vel[-1]]
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))

    def overlay(a):
        a.plot(coi, vel, "k--", lw=1)
        a.set_xlim(Tmin, Tmax); a.set_ylim(vmin, vmax)
        a.set_xlabel("period [s]")

    ax[0].imshow(amp_pp.T, cmap="RdBu_r", extent=ext, aspect="auto", origin="lower", vmin=0, vmax=1)
    ax[0].plot(rp_pp, rv_pp, "w-", lw=2, label="ridge (per-period)")
    overlay(ax[0]); ax[0].set(title="PER-PERIOD norm — amplitude-blind ridge", ylabel="group vel [km/s]")
    ax[0].legend(loc="upper right", fontsize=8)

    ax[1].imshow(np.sqrt(amp_gl).T, cmap="RdBu_r", extent=ext, aspect="auto", origin="lower", vmin=0, vmax=1)
    ax[1].plot(rp_pp, rv_pp, "w--", lw=1.5, label="ridge (per-period)")
    ax[1].plot(rp_gl, rv_gl, "c-", lw=1.5, label="ridge (energy-weighted)")
    if keep.any():
        ax[1].plot(rp_pp[keep], rv_pp[keep], "g.", ms=8, label=f"kept (energy≥{AMP_THRESH:g})")
    overlay(ax[1]); ax[1].set(title="GLOBAL norm — energy-weighted ridge + amplitude threshold")
    ax[1].legend(loc="upper right", fontsize=8)

    ax[2].plot(rp_pp, e_pp, "k.-", label="energy along per-period ridge")
    ax[2].axhline(AMP_THRESH, color="r", ls="--", label=f"threshold {AMP_THRESH:g}")
    ax[2].fill_between(rp_pp, 0, e_pp, where=~keep, color="red", alpha=0.15, label="below threshold")
    ax[2].set(title="Genuine energy along the pick", xlabel="period [s]",
              ylabel="global-norm energy", xlim=(Tmin, Tmax), ylim=(0, 1.02))
    ax[2].grid(alpha=0.3); ax[2].legend(loc="upper right", fontsize=8)

    nkept = int(keep.sum())
    fig.suptitle(f"{pair}  {comp} ({wave})  dist={dist:.1f} km   normalization effect on picking "
                 f"(kept {nkept}/{len(rp_pp)} picks above energy threshold)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(outdir, f"norm_pick_{pair}_{comp}.png")
    fig.savefig(out, dpi=115); plt.close(fig)
    return out, dist


if __name__ == "__main__":
    h5 = sys.argv[1]
    comp = sys.argv[2] if len(sys.argv) > 2 else "ZZ"
    stack = sys.argv[3] if len(sys.argv) > 3 else "Allstack_pws"
    outdir = sys.argv[4] if len(sys.argv) > 4 else "/tmp"
    out, dist = make_figure(h5, comp, stack, outdir)
    print(f"dist={dist:.1f} km -> {out}")
