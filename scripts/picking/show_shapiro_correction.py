"""
Visualise what the Shapiro & Singh (1999) period correction does to the group-dispersion output.

FTAN assigns each group-velocity measurement to the filter's NOMINAL centre period; when the
spectral amplitude varies across the (Morlet) filter band the correct period is the spectral
CENTROID period (Shapiro eq. 6), or the instantaneous period (Levshin 5.92). The correction is a
horizontal (period) shift of each (T, U) pick. This script overlays the nominal vs corrected
dispersion curves, the per-period shift, and the CC amplitude spectrum that drives it.

Usage: python show_shapiro_correction.py <stack.h5> [comp=ZZ] [stack=Allstack_pws] [outdir=/tmp]
"""
import sys
import os
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import fft
from noisepy import dispersion as d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_picks_image as cp

logging.getLogger("findpeaks").setLevel(logging.ERROR)


def make_figure(h5path, comp="ZZ", stack="Allstack_pws", outdir="/tmp"):
    pair = os.path.basename(h5path).replace(".h5", "")
    sym, dist, dt = cp.read_sym(h5path, comp, stack)
    Tmin, dT, vmin, vmax, dvel, vave = cp.Tmin, cp.dT, cp.vmin, cp.vmax, cp.dvel, cp.vave
    Tmax = dist / vave

    cwt = d.compute_cwt(sym, dist, dt, Tmin=Tmin, vmin=vmin, vmax=vmax, vave=vave)
    amp, per, vel, coi = d.disp_image_from_cwt(cwt, dist, Tmin=Tmin, dT=dT, vmin=vmin,
                                               vmax=vmax, dvel=dvel, vave=vave)
    rp, rv = d.extract_dispersion_viterbi(amp, per, vel, smooth_weight=cp.GROUP_SMOOTH,
                                          max_step=cp.GROUP_MAXSTEP, short_priority=cp.SHORT_PRIORITY,
                                          coi=coi)
    cat = np.array([coi[np.argmin(np.abs(vel - v))] for v in rv])
    m = (rp <= cat) & (rp <= Tmax)
    rp, rv = rp[m], rv[m]

    corr = d.measure_corrections_and_phase(cwt, rp, rv, dist, c_ref=None)
    Tcen, Tinst = corr["T_centroid"], corr["T_inst"]
    sh_cen = 100 * (Tcen - rp) / rp           # % period shift (centroid)
    sh_ins = 100 * (Tinst - rp) / rp          # % period shift (instantaneous)

    # CC amplitude spectrum (drives the correction)
    X = np.abs(fft.rfft(sym))
    fr = fft.rfftfreq(len(sym), dt)
    X = X / np.max(X[fr > 0])

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    # (0,0) group image with nominal vs corrected ridge
    ext = [per[0], per[-1], vel[0], vel[-1]]
    ax[0, 0].imshow(amp.T, cmap=cp.GROUP_CMAP, extent=ext, aspect="auto", origin="lower")
    ax[0, 0].plot(rp, rv, "w.-", ms=5, lw=1, label="nominal T")
    ax[0, 0].plot(Tcen, rv, "c.-", ms=5, lw=1, label="centroid T (Shapiro)")
    ax[0, 0].plot(Tinst, rv, ".-", c="orange", ms=4, lw=1, label="instantaneous T (Levshin)")
    ax[0, 0].set(title="Group ridge: nominal vs corrected period", xlabel="period [s]",
                 ylabel="group vel [km/s]", xlim=(Tmin, Tmax), ylim=(vmin, vmax))
    ax[0, 0].legend(fontsize=8)

    # (0,1) U(T) curves (line view of the same shift)
    ax[0, 1].plot(rp, rv, "k.-", label="U(T nominal)")
    ax[0, 1].plot(Tcen, rv, "c.-", label="U(T centroid)")
    ax[0, 1].plot(Tinst, rv, ".-", c="orange", label="U(T instantaneous)")
    ax[0, 1].set(title="Dispersion curve shift", xlabel="period [s]", ylabel="group vel [km/s]")
    ax[0, 1].grid(alpha=0.3); ax[0, 1].legend(fontsize=8)

    # (1,0) per-period shift
    ax[1, 0].axhline(0, color="k", lw=0.8)
    ax[1, 0].plot(rp, sh_cen, "c.-", label="centroid (Shapiro)")
    ax[1, 0].plot(rp, sh_ins, ".-", c="orange", label="instantaneous (Levshin)")
    ax[1, 0].set(title="Period correction", xlabel="nominal period [s]",
                 ylabel="(T_corr − T_nom)/T_nom  [%]")
    ax[1, 0].grid(alpha=0.3); ax[1, 0].legend(fontsize=8)

    # (1,1) CC amplitude spectrum driving the correction
    ax[1, 1].plot(fr, X, "0.3", lw=1)
    ax[1, 1].axvspan(1 / Tmax, 1 / Tmin, color="gold", alpha=0.25, label="measurement band")
    ax[1, 1].set(title="CC amplitude spectrum (falloff drives the correction)",
                 xlabel="frequency [Hz]", ylabel="norm. amplitude", xlim=(0, 1.5 / Tmin))
    ax[1, 1].legend(fontsize=8)

    fig.suptitle(f"{pair}  {comp}  dist={dist:.1f} km   Shapiro/Levshin period correction "
                 f"(median |shift| centroid={np.nanmedian(np.abs(sh_cen)):.1f}%)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(outdir, f"shapiro_{pair}_{comp}.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    return out, dist, np.nanmedian(np.abs(sh_cen))


if __name__ == "__main__":
    h5 = sys.argv[1]
    comp = sys.argv[2] if len(sys.argv) > 2 else "ZZ"
    stack = sys.argv[3] if len(sys.argv) > 3 else "Allstack_pws"
    outdir = sys.argv[4] if len(sys.argv) > 4 else "/tmp"
    out, dist, sh = make_figure(h5, comp, stack, outdir)
    print(f"dist={dist:.1f} km  median|centroid shift|={sh:.1f}%  -> {out}")
