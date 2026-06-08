"""
FTAN group image WITHOUT per-period amplitude normalization.

Following Esteve et al. (2025) Geothermics Fig. 2 and Shirzad et al. (2025) Geophys. Prospect.
Fig. 2: the dispersion image is "normalized by the maximum energy of the signal" (a single
GLOBAL max) instead of "normalized at each frequency" (per-period). The global version preserves
the relative amplitude across periods, so genuinely high-energy parts of the image stand out (red)
and weak periods fade (blue) -- telling you where the measurement is actually reliable.

Left panel = global-normalized (energy), right = per-period (every frequency equally bright).
Picked group curve overlaid at nominal period (blue) and Shapiro-corrected period (green).

Usage: python show_unnormalized_ftan.py <stack.h5> [comp=ZZ] [stack=Allstack_pws] [outdir=/tmp]
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
CMAP = "RdBu_r"          # blue (low energy) -> white -> red (high), as in the reference papers


def make_figure(h5path, comp="ZZ", stack="Allstack_pws", outdir="/tmp"):
    pair = os.path.basename(h5path).replace(".h5", "")
    wave = cp.WAVE_OF.get(comp, "Rayleigh")
    sym, dist, dt = cp.read_sym(h5path, comp, stack)
    Tmin, dT, vmin, vmax, dvel, vave = cp.Tmin, cp.dT, cp.vmin, cp.vmax, cp.dvel, cp.vave
    Tmax = dist / vave

    cwt = d.compute_cwt(sym, dist, dt, Tmin=Tmin, vmin=vmin, vmax=vmax, vave=vave)
    imk = dict(Tmin=Tmin, dT=dT, vmin=vmin, vmax=vmax, dvel=dvel, vave=vave)
    amp_g, per, vel, coi = d.disp_image_from_cwt(cwt, dist, normalize='global', **imk)
    amp_p, _, _, _ = d.disp_image_from_cwt(cwt, dist, normalize='per_period', **imk)
    # display amplitude (sqrt of energy) so the strong band reads broadly, as in the papers
    img_g = np.sqrt(amp_g)

    # pick the ridge on the per-period image (robust), then the Shapiro/Levshin period correction
    rp, rv = d.extract_dispersion_viterbi(amp_p, per, vel, smooth_weight=cp.GROUP_SMOOTH,
                                          max_step=cp.GROUP_MAXSTEP, short_priority=cp.SHORT_PRIORITY,
                                          coi=coi)
    cat = np.array([coi[np.argmin(np.abs(vel - v))] for v in rv])
    m = (rp <= cat) & (rp <= Tmax)
    rp, rv = rp[m], rv[m]
    corr = d.measure_corrections_and_phase(cwt, rp, rv, dist, c_ref=None)
    Tcen = corr["T_centroid"]

    ext = [per[0], per[-1], vel[0], vel[-1]]
    v_cut = dist / (3.0 * per)            # 3-wavelength usable boundary
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

    def overlay(a):
        a.plot(coi, vel, "k--", lw=1, label="COI")
        a.plot(per, v_cut, "0.4", ls=":", lw=1.2, label="3λ")
        a.plot(rp, rv, "o", mfc="none", mec="b", ms=5, label="nominal period")
        a.plot(Tcen, rv, "g.", ms=6, label="centroid-corrected (Shapiro)")
        a.set_xlim(Tmin, Tmax); a.set_ylim(vmin, vmax)

    im0 = ax[0].imshow(img_g.T, cmap=CMAP, extent=ext, aspect="auto", origin="lower", vmin=0, vmax=1)
    overlay(ax[0]); ax[0].set(title="GLOBAL norm — energy of the signal (genuine amplitude)",
                              xlabel="period [s]", ylabel="group vel [km/s]")
    fig.colorbar(im0, ax=ax[0], shrink=0.8, label="norm. amplitude")
    ax[0].legend(loc="upper right", fontsize=8)

    im1 = ax[1].imshow(amp_p.T, cmap=CMAP, extent=ext, aspect="auto", origin="lower", vmin=0, vmax=1)
    overlay(ax[1]); ax[1].set(title="PER-PERIOD norm (every frequency equally bright)",
                              xlabel="period [s]")
    fig.colorbar(im1, ax=ax[1], shrink=0.8, label="norm. amplitude")

    fig.suptitle(f"{pair}  {comp} ({wave})  dist={dist:.1f} km   FTAN normalization: global vs per-period",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(outdir, f"ftan_energy_{pair}_{comp}.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    return out, dist


if __name__ == "__main__":
    h5 = sys.argv[1]
    comp = sys.argv[2] if len(sys.argv) > 2 else "ZZ"
    stack = sys.argv[3] if len(sys.argv) > 3 else "Allstack_pws"
    outdir = sys.argv[4] if len(sys.argv) > 4 else "/tmp"
    out, dist = make_figure(h5, comp, stack, outdir)
    print(f"dist={dist:.1f} km -> {out}")
