"""
Visual comparison of group/phase dispersion images and the picks produced by the different
approaches, for ONE station-pair stack (S2_stacking.py output). Reads ASDF via h5py.

Top row  = GROUP image (|CWT|^2):
   left  : argmax picks; those the group-Viterbi ridge selects are filled, others are x.
   right : topology peaks coloured by persistence score; the ones Viterbi selects are ringed.
   both  : the group-Viterbi ridge (white line) and U predicted from the picked PHASE curve
           (magenta dashed, Bensen eq.7) -- should overlie the ridge if self-consistent.
Bottom row = PHASE-velocity image (cos image whose positive crests are the c(N) branches):
   phase curve extracted along the Viterbi group ridge, joint-Viterbi-N (left) vs per-period-N
   (right), with the proxy reference.

Period axis runs to the 1-wavelength period; the N_LAMBDA (default 3) usable boundary is drawn.

NOTE: with no FK curve supplied, the phase 2*pi*N is resolved against a PROXY reference
(smoothed group velocity) so the comparison can run -- swap REF for your FK curve for real work.

Usage (single):  python compare_picks_image.py <stack.h5> [comp=ZZ] [stack=Allstack_pws] [outdir=/tmp]
"""
import sys
import os
import logging
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from noisepy import dispersion as d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_model import reference_curves   # noqa: E402

logging.getLogger("findpeaks").setLevel(logging.ERROR)

REF_ALL = reference_curves()                    # forward-modelled reference (computed once)
REF_COLORS = {0: "cyan", 1: "deepskyblue"}

# ----- parameters -----
Tmin, dT = 0.2, 0.1
vmin, vmax, dvel = 0.5, 4.0, 0.01
vave = 3.0                       # typical velocity used only to set the displayed Tmax (1λ)
N_LAMBDA = 3                     # wavelengths required for a reliable measurement (Bensen)
maxgap = int(0.2 / dvel)
min_score = 0.6                  # topology persistence threshold (used set)
PHASE_OFFSET = 0.0
PHASE_SMOOTH = 5.0               # phase-curve continuity weight (joint N resolver)
PHASE_MAXSTEP = 0.15             # hard cap on |Δc| per period -> no branch hops in phase curve
GROUP_CMAP = "magma"             # perceptually uniform (was jet)
# wave type per component + stationary-phase term (Love sign less settled -- verify vs FK)
WAVE_OF = {"ZZ": "Rayleigh", "RR": "Rayleigh", "RZ": "Rayleigh", "ZR": "Rayleigh", "TT": "Love"}
PHASE_SHIFT_OF = {"Rayleigh": -np.pi / 4.0, "Love": -np.pi / 4.0}
# Group-Viterbi: continuity is enforced by the HARD max_step cap, so the smooth weight is kept
# small -- otherwise it over-smooths the ridge off the genuine energy. short_priority=0 (the
# tracker follows the energy; reliability is handled separately by the global-energy threshold).
GROUP_SMOOTH = 0.3               # group-Viterbi velocity-jump weight [cost per km/s] (small!)
GROUP_MAXSTEP = 0.1              # hard cap on |Δv| per period step [km/s] (prevents gaps)
SHORT_PRIORITY = 0.0             # no short-period bias (it pulled the ridge off the energy)
SEL_TOL = 0.06                   # km/s tolerance for "argmax pick lies on the Viterbi ridge"


def read_sym(h5path, comp, stack="Allstack_pws"):
    with h5py.File(h5path, "r") as h:
        node = h["AuxiliaryData"][stack][comp]
        data = node[:]
        dist = float(node.attrs["dist"]); dt = float(node.attrs["dt"])
    i = len(data) // 2
    return 0.5 * (data[i:] + np.flip(data[:i + 1])), dist, dt


def make_figure(h5path, comp="ZZ", stack="Allstack_pws", outdir="/tmp"):
    pair = os.path.basename(h5path).replace(".h5", "")
    wave = WAVE_OF.get(comp, "Rayleigh")
    phase_shift = PHASE_SHIFT_OF[wave]
    sym, dist, dt = read_sym(h5path, comp, stack)
    Tmax = dist / vave

    cwt = d.compute_cwt(sym, dist, dt, Tmin=Tmin, vmin=vmin, vmax=vmax, vave=vave)
    amp, per, vel, coi = d.disp_image_from_cwt(cwt, dist, Tmin=Tmin, dT=dT, vmin=vmin,
                                               vmax=vmax, dvel=dvel, vave=vave)

    # --- candidate group picks ---
    a_per, a_vel, _ = d.remove_picks_coi(*d.extract_dispersion(
        amp, per, vel, dist, vmax=vmax, maxgap=maxgap, minlambda=1.0), vel, coi)
    ta_per, ta_vel, ta_sc = d.remove_picks_coi(*map(np.asarray, d.extract_curves_topology(
        amp, per, vel, limit=0.02)), vel, coi)

    # --- group-Viterbi ridge over the image; restrict to COI-valid, <=1λ region ---
    r_per, r_vel = d.extract_dispersion_viterbi(amp, per, vel, smooth_weight=GROUP_SMOOTH,
                                                max_step=GROUP_MAXSTEP, short_priority=SHORT_PRIORITY,
                                                coi=coi)
    coi_at = np.array([coi[np.argmin(np.abs(vel - v))] for v in r_vel])
    valid = (r_per <= coi_at) & (r_per <= Tmax)
    rp, rv = r_per[valid], r_vel[valid]

    # phase image built from the SAME ridge the picks use -> picks land on its crests
    pimg, _, _ = d.phase_velocity_image(cwt, dist, Tmin=Tmin, dT=dT, vmin=vmin, vmax=vmax,
                                        dvel=dvel, vave=vave, phase_shift=phase_shift,
                                        phase_offset=PHASE_OFFSET, group_per=rp, group_vel=rv)

    # which candidates does Viterbi select?
    ridge_v = np.interp(a_per, rp, rv, left=np.nan, right=np.nan)
    a_sel = np.abs(a_vel - ridge_v) <= SEL_TOL                       # argmax on the ridge
    t_sel, _, _ = d.viterbi_select_candidates(ta_per, ta_vel, ta_sc, smooth_weight=GROUP_SMOOTH)

    # --- reference = forward-modelled FUNDAMENTAL phase curve for this wave type ---
    wkey = wave.lower()
    rp0, rv0 = REF_ALL[(wkey, "phase", 0)]
    REF = (d.load_reference_curve((rp0, rv0)) if len(rp0)
           else d.load_reference_curve((np.array([Tmin, Tmax]), np.array([2., 3.]))))
    if len(rp) >= 4:
        mj = d.measure_corrections_and_phase(cwt, rp, rv, dist, c_ref=REF, phase_shift=phase_shift,
                                             phase_offset=PHASE_OFFSET, joint=True,
                                             smooth_weight=PHASE_SMOOTH, phase_max_step=PHASE_MAXSTEP)
        mp = d.measure_corrections_and_phase(cwt, rp, rv, dist, c_ref=REF, phase_shift=phase_shift,
                                             phase_offset=PHASE_OFFSET, joint=False)
        cj, Nj = mj["phase_velocity"], mj["N_ambiguity"]
        cp, Np = mp["phase_velocity"], mp["N_ambiguity"]
        U_from_phase = d.group_from_phase(rp, cj)                    # Bensen eq.7 prediction
    else:
        cj = cp = U_from_phase = np.full(len(rp), np.nan)
        Nj = Np = np.zeros(len(rp), dtype=int)
    ndiff_mask = (Nj != Np) & np.isfinite(cj) & np.isfinite(cp)

    def ref_overlay(a, kind):
        for mode, ls, lab in [(0, "-", "fund"), (1, "--", "1st ovt")]:
            p, v = REF_ALL[(wkey, kind, mode)]
            if len(p):
                a.plot(p, v, color=REF_COLORS[mode], ls=ls, lw=1.6,
                       label=f"ref {lab} ({wave} {kind})")

    # --- figure ---
    ext = [per[0], per[-1], vel[0], vel[-1]]
    v_cut1, v_cutN = dist / per, dist / (N_LAMBDA * per)
    Tmax_use = dist / (N_LAMBDA * vave)
    fig, ax = plt.subplots(2, 2, figsize=(16, 11), sharex=True, sharey=True)

    def base(a, img, cmap, **kw):
        a.imshow(img.T, cmap=cmap, extent=ext, aspect="auto", origin="lower", **kw)
        a.plot(coi, vel, "k--", lw=1, label="COI")
        a.plot(per, v_cut1, color="0.5", ls=":", lw=1, label="1λ")
        a.plot(per, v_cutN, "m-", lw=1.2, label=f"{N_LAMBDA}λ usable")
        a.axvline(Tmax_use, color="m", ls="--", lw=1)
        a.set_xlim(Tmin, Tmax); a.set_ylim(vmin, vmax)

    def group_overlays(a):
        a.plot(rp, rv, "w-", lw=2, label="group Viterbi ridge")
        a.plot(rp, U_from_phase, "m:", lw=2, label="U from phase (eq.7)")
        ref_overlay(a, "group")

    # (0,0) argmax + Viterbi selection
    base(ax[0, 0], amp, GROUP_CMAP)
    if a_sel.any():
        ax[0, 0].plot(a_per[a_sel], a_vel[a_sel], "o", mfc="lime", mec="k", ms=6, label="argmax (selected)")
    if (~a_sel).any():
        ax[0, 0].plot(a_per[~a_sel], a_vel[~a_sel], "x", c="r", ms=6, label="argmax (rejected)")
    group_overlays(ax[0, 0])
    ax[0, 0].set(title="GROUP — argmax vs Viterbi", ylabel="group vel [km/s]")

    # (0,1) topology coloured by score + Viterbi selection
    base(ax[0, 1], amp, GROUP_CMAP)
    scat = ax[0, 1].scatter(ta_per, ta_vel, c=ta_sc, cmap="viridis", vmin=0.5, vmax=1.0,
                            s=22, edgecolors="k", lw=0.3)
    if t_sel.any():
        ax[0, 1].scatter(ta_per[t_sel], ta_vel[t_sel], s=85, facecolors="none",
                         edgecolors="red", lw=1.4, label="Viterbi-selected")
    fig.colorbar(scat, ax=ax[0, 1], label="persistence score", shrink=0.8)
    group_overlays(ax[0, 1])
    ax[0, 1].set(title=f"GROUP — topology (score) vs Viterbi (used limit={min_score})")

    # (1,0) phase — joint vs per-period overlaid, disagreements flagged
    base(ax[1, 0], pimg, "RdBu", vmin=-1, vmax=1)
    ref_overlay(ax[1, 0], "phase")
    ax[1, 0].plot(rp, cp, "o", mfc="none", mec="orange", ms=10, label="per-period N")
    ax[1, 0].plot(rp, cj, "k.", ms=6, label="joint Viterbi N")
    if ndiff_mask.any():
        ax[1, 0].plot(rp[ndiff_mask], cj[ndiff_mask], "r*", ms=14, label="N differs")
    ax[1, 0].set(title=f"PHASE — joint vs per-period N (differ at {int(ndiff_mask.sum())} periods)",
                 xlabel="period [s]", ylabel="phase vel [km/s]")

    # (1,1) phase vs group: phase curve sits on + crests and lies above the group ridge
    base(ax[1, 1], pimg, "RdBu", vmin=-1, vmax=1)
    ref_overlay(ax[1, 1], "phase")
    ax[1, 1].plot(rp, cj, "k.-", ms=5, lw=1, label="phase c(T)")
    ax[1, 1].plot(rp, rv, "w-", lw=2, label="group ridge U(T)")
    ax[1, 1].set(title="PHASE vs GROUP  (phase on + crests, ≥ group)", xlabel="period [s]")

    for a in ax.ravel():
        a.legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.suptitle(f"{pair}   {comp} ({wave})   dist={dist:.1f} km   Tmax(1λ)={Tmax:.1f}s   {N_LAMBDA}λ={Tmax_use:.1f}s",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(outdir, f"compare_{pair}_{comp}.png")
    fig.savefig(out, dpi=110); plt.close(fig)
    return out, dist


if __name__ == "__main__":
    h5 = sys.argv[1]
    comp = sys.argv[2] if len(sys.argv) > 2 else "ZZ"
    stack = sys.argv[3] if len(sys.argv) > 3 else "Allstack_pws"
    outdir = sys.argv[4] if len(sys.argv) > 4 else "/tmp"
    out, dist = make_figure(h5, comp, stack, outdir)
    print(f"dist={dist:.1f} km -> {out}")
