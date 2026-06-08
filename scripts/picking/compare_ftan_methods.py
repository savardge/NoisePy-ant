"""
Compare the two FTAN implementations on one station pair:
  * CWT (Morlet wavelet)  -- compute_cwt
  * classic multiple-filter narrowband Gaussian (Levshin 1989 / Bensen 2007 eqs. 3-6)
    -- compute_narrowband

Both produce a complex analytic field A(t,w0) e^{i phi}, so the SAME image builders and picks
are applied to each. Figure (2x2): group images (top) and phase-velocity images (bottom),
CWT (left) vs narrowband (right). The CWT group ridge + phase picks + forward-modelled reference
are overlaid identically on both, so differences are purely the method/filter.

Usage: python compare_ftan_methods.py <stack.h5> [comp=ZZ] [stack=Allstack_pws] [alpha=18] [outdir=/tmp]
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
import compare_picks_image as cp   # reuse config, reference, read_sym

logging.getLogger("findpeaks").setLevel(logging.ERROR)


def make_figure(h5path, comp="ZZ", stack="Allstack_pws", alpha=18.0, outdir="/tmp"):
    pair = os.path.basename(h5path).replace(".h5", "")
    wave = cp.WAVE_OF.get(comp, "Rayleigh")
    ps = cp.PHASE_SHIFT_OF[wave]
    sym, dist, dt = cp.read_sym(h5path, comp, stack)
    Tmin, dT, vmin, vmax, dvel, vave = cp.Tmin, cp.dT, cp.vmin, cp.vmax, cp.dvel, cp.vave
    Tmax = dist / vave

    kw = dict(Tmin=Tmin, vmin=vmin, vmax=vmax, vave=vave)
    cwt = d.compute_cwt(sym, dist, dt, **kw)
    nb = d.compute_narrowband(sym, dist, dt, alpha=alpha, **kw)

    imk = dict(Tmin=Tmin, dT=dT, vmin=vmin, vmax=vmax, dvel=dvel, vave=vave)
    amp_c, per, vel, coi_c = d.disp_image_from_cwt(cwt, dist, **imk)
    amp_n, _, _, _ = d.disp_image_from_cwt(nb, dist, **imk)

    # group ridge from the CWT image (data-driven), reused on both panels
    rp, rv = d.extract_dispersion_viterbi(amp_c, per, vel, smooth_weight=cp.GROUP_SMOOTH,
                                          max_step=cp.GROUP_MAXSTEP, short_priority=cp.SHORT_PRIORITY,
                                          coi=coi_c)
    cat = np.array([coi_c[np.argmin(np.abs(vel - v))] for v in rv])
    m = (rp <= cat) & (rp <= Tmax)
    rp, rv = rp[m], rv[m]

    # phase images built from the SAME ridge (so picks land on crests in each)
    pimg_c, _, _ = d.phase_velocity_image(cwt, dist, phase_shift=ps, phase_offset=cp.PHASE_OFFSET,
                                          group_per=rp, group_vel=rv, **imk)
    pimg_n, _, _ = d.phase_velocity_image(nb, dist, phase_shift=ps, phase_offset=cp.PHASE_OFFSET,
                                          group_per=rp, group_vel=rv, **imk)

    wkey = wave.lower()
    rp0, rv0 = cp.REF_ALL[(wkey, "phase", 0)]
    REF = d.load_reference_curve((rp0, rv0)) if len(rp0) else None
    cj = (d.measure_corrections_and_phase(cwt, rp, rv, dist, c_ref=REF, phase_shift=ps,
                                          phase_offset=cp.PHASE_OFFSET, joint=True,
                                          smooth_weight=cp.PHASE_SMOOTH,
                                          phase_max_step=cp.PHASE_MAXSTEP)["phase_velocity"]
          if REF is not None else np.full(len(rp), np.nan))

    # --- figure ---
    ext = [per[0], per[-1], vel[0], vel[-1]]
    fig, ax = plt.subplots(2, 2, figsize=(16, 11), sharex=True, sharey=True)

    def refov(a, kind):
        for mode, ls in [(0, "-"), (1, "--")]:
            p, v = cp.REF_ALL[(wkey, kind, mode)]
            if len(p):
                a.plot(p, v, color=cp.REF_COLORS[mode], ls=ls, lw=1.5)

    def setup(a, img, cmap, **kw2):
        a.imshow(img.T, cmap=cmap, extent=ext, aspect="auto", origin="lower", **kw2)
        a.plot(coi_c, vel, "k--", lw=1)
        a.set_xlim(Tmin, Tmax); a.set_ylim(vmin, vmax)

    setup(ax[0, 0], amp_c, cp.GROUP_CMAP); ax[0, 0].plot(rp, rv, "w-", lw=2); refov(ax[0, 0], "group")
    ax[0, 0].set(title="GROUP — CWT (Morlet)", ylabel="group vel [km/s]")
    setup(ax[0, 1], amp_n, cp.GROUP_CMAP); ax[0, 1].plot(rp, rv, "w-", lw=2, label="CWT ridge")
    refov(ax[0, 1], "group")
    ax[0, 1].set(title=f"GROUP — narrowband Gaussian (α={alpha:g})")

    setup(ax[1, 0], pimg_c, "RdBu", vmin=-1, vmax=1); ax[1, 0].plot(rp, cj, "k.", ms=5)
    refov(ax[1, 0], "phase")
    ax[1, 0].set(title="PHASE — CWT (Morlet)", xlabel="period [s]", ylabel="phase vel [km/s]")
    setup(ax[1, 1], pimg_n, "RdBu", vmin=-1, vmax=1); ax[1, 1].plot(rp, cj, "k.", ms=5, label="phase pick (CWT)")
    refov(ax[1, 1], "phase")
    ax[1, 1].set(title=f"PHASE — narrowband Gaussian (α={alpha:g})", xlabel="period [s]")

    ax[0, 1].legend(loc="upper right", fontsize=8)
    ax[1, 1].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{pair}  {comp} ({wave})  dist={dist:.1f} km   CWT vs classic narrowband FTAN",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(outdir, f"ftan_compare_{pair}_{comp}.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    return out, dist


if __name__ == "__main__":
    h5 = sys.argv[1]
    comp = sys.argv[2] if len(sys.argv) > 2 else "ZZ"
    stack = sys.argv[3] if len(sys.argv) > 3 else "Allstack_pws"
    alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 18.0
    outdir = sys.argv[5] if len(sys.argv) > 5 else "/tmp"
    out, dist = make_figure(h5, comp, stack, alpha, outdir)
    print(f"dist={dist:.1f} km -> {out}")
