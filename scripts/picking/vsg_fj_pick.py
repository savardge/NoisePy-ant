#!/usr/bin/env python3
"""Re-pick the network reference dispersion curves on the F-J images, keeping MULTIPLE ridges.

The original ref_*_phase.txt curves were picked on the phase-shift slant-stack images. The F-J
transform resolves several branches where the slant stack showed one smeared blob, and the
old picks visibly do not follow the F-J ridges. This re-picks on F-J with:

  1. per-frequency PEAK DETECTION by topological persistence (findpeaks, method='topology' --
     the same statistic the Haute-Sorne manuscript's picker used, score threshold 0.7). Every
     column can yield SEVERAL peaks, so two ridges present at one frequency are both kept.
  2. BRANCH LINKING across frequency: peaks are chained into branches by nearest-neighbour in
     phase velocity between adjacent frequency columns (max jump `--max-dc` km/s), starting new
     branches when a peak has no continuation. Branches shorter than `--min-len` columns are
     dropped. This is what turns "two peaks in a column" into "two ridges", instead of one
     zig-zagging curve that hops between them.
  3. LABELLING: R0 = the LONGEST continuous branch (most frequency columns), NOT the slowest.
     "Slowest" fails: at every network F-J shows a flat ~2.1 km/s low-frequency sidelobe running
     BELOW the fundamental at 0.18-0.4 Hz, which would steal the R0 label. The fundamental is
     the dispersion ridge that spans the widest band. Other branches are numbered by increasing
     mean velocity (B1, B2, ...). No branch is called "R1" here -- that identification needs
     the forward comparison against the Vs(halfspace) ceiling (vsg_overtone_mode_check.py); the
     picker only reports what the image contains.

Outputs (in Projects/<net>/vsg_modesep/):
   fj_picks_<comp>.csv                one row per (branch, frequency, c, score)
   fj_picks_<comp>.png                F-J image + old picks (thin) + new branches (bold)
   ref_fundamental_phase_FJ.txt       R0 as (period, c) -- candidate replacement reference;
   ref_branchN_phase_FJ.txt           the other branches, one file each.
The old ref_*_phase.txt files are NOT overwritten.

Usage:
  python vsg_fj_pick.py --net aargau
  python vsg_fj_pick.py --net riehen --score 0.6 --max-dc 0.15
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from findpeaks import findpeaks

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"


def column_peaks(col, fp, score_min):
    """(indices, scores) of topologically persistent peaks in one normalised column."""
    r = fp.fit(col)
    d = r["df"]
    d = d[(d["peak"]) & (d["score"] >= score_min)]
    return d["x"].values.astype(int), d["score"].values


def link_branches(F, c, peaks, max_dc, min_len):
    """Chain per-column peaks into branches by nearest-neighbour continuation in c."""
    branches = []                     # each: list of (j, ic, score)
    active = []                       # branches still open at the previous column
    for j in range(len(F)):
        idx, sc = peaks[j]
        cands = list(zip(idx, sc))
        new_active = []
        used = set()
        # extend active branches greedily by closest c
        for b in active:
            jl, il, _ = b[-1]
            best, bd = None, max_dc
            for k, (ic, s) in enumerate(cands):
                if k in used:
                    continue
                d = abs(c[ic] - c[il])
                if d < bd:
                    best, bd = k, d
            if best is not None:
                ic, s = cands[best]; used.add(best)
                b.append((j, ic, s)); new_active.append(b)
            else:
                branches.append(b)          # closed
        for k, (ic, s) in enumerate(cands):
            if k not in used:
                new_active.append([(j, ic, s)])
        active = new_active
    branches += active
    return [b for b in branches if len(b) >= min_len]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", required=True)
    ap.add_argument("--comp", default="ZZ")
    ap.add_argument("--score", type=float, default=0.7, help="min topological persistence")
    ap.add_argument("--max-dc", type=float, default=0.12, help="km/s; max c jump between columns")
    ap.add_argument("--min-len", type=int, default=25, help="min columns for a branch to survive")
    ap.add_argument("--fmin", type=float, default=0.18)
    ap.add_argument("--fmax", type=float, default=2.5)
    ap.add_argument("--cmax", type=float, default=5.0)
    a = ap.parse_args()

    d = f"{EHM}/{a.net}/vsg_modesep"
    z = np.load(f"{d}/vsg_fj_{a.comp}_sign+1.npz")
    F, c, A = z["f"], z["vel"], np.asarray(z["FJ"], float)
    fm = (F >= a.fmin) & (F <= a.fmax); cm = c <= a.cmax
    F, A = F[fm], A[cm][:, fm]; c = c[cm]
    fp = findpeaks(method="topology", lookahead=1, verbose=0)
    peaks = []
    for j in range(len(F)):
        col = A[:, j]; col = col / col.max() if col.max() > 0 else col
        idx, sc = column_peaks(col, fp, a.score)
        peaks.append((idx, sc))
    branches = link_branches(F, c, peaks, a.max_dc, a.min_len)
    if not branches:
        raise SystemExit("no branches survived; lower --score or --min-len")
    # R0 = the fundamental. Two rules failed in practice, both are documented:
    #   * "slowest branch"  -> picks the flat ~2 km/s LOW-frequency sidelobe under the fundamental
    #   * "longest branch"  -> at Riehen the sidelobe (76 cols) beat the fundamental, which the
    #     linker had split in two (0.63-1.08 Hz and 1.16-1.55 Hz), and the inversion of that
    #     sidelobe returned a 1.68 km/s half-space.
    # Robust rule: the fundamental is the branch that (a) reaches the HIGHEST frequency of the
    # band and (b) among branches doing so, has the most columns; then MERGE onto it any other
    # branch that continues it in frequency within max_dc (the linker breaks the fundamental
    # wherever a stronger ridge steals its nearest neighbour for one column, or its persistence
    # dips under the floor for a column or two -- Riehen: a 7-column gap at dc=0.06 km/s).
    def fmax_of(b): return F[max(j for j, _, _ in b)]
    fmax_all = max(fmax_of(b) for b in branches)
    cands = [b for b in branches if fmax_of(b) >= fmax_all - 0.15 * fmax_all]
    r0 = max(cands, key=len)
    branches.remove(r0)
    merged = True
    while merged:
        merged = False
        for b in list(branches):
            j0, i0, _ = r0[0]; j1, i1, _ = r0[-1]
            jb0, ib0, _ = b[0]; jb1, ib1, _ = b[-1]
            # b ends just before r0 starts, or starts just after r0 ends, and c is continuous
            if jb1 < j0 and (j0 - jb1) <= 12 and abs(c[ib1] - c[i0]) <= 2 * a.max_dc:
                r0 = b + r0; branches.remove(b); merged = True
            elif jb0 > j1 and (jb0 - j1) <= 12 and abs(c[ib0] - c[i1]) <= 2 * a.max_dc:
                r0 = r0 + b; branches.remove(b); merged = True
    # SPLIT R0 at a kink: a physical dispersion branch has |dc/df| that varies smoothly. At
    # Riehen the leaking ridge (4.2 km/s at 0.4 Hz) descends continuously INTO the fundamental
    # at ~0.63 Hz, so continuity alone joins two different modes. Cut where c jumps by more than
    # KINK between adjacent columns and keep the piece that reaches the highest frequency (the
    # fundamental); the detached piece becomes its own branch.
    KINK = 0.15
    cs = np.array([c[i] for _, i, _ in r0]); jumps = np.flatnonzero(np.abs(np.diff(cs)) > KINK)
    if jumps.size:
        k = int(jumps[-1]) + 1                     # last kink before the high-frequency end
        detached, r0 = r0[:k], r0[k:]
        if len(detached) >= a.min_len:
            branches.append(detached)
    branches.sort(key=lambda b: np.mean([c[i] for _, i, _ in b]))
    branches = [r0] + branches
    names = ["R0"] + [f"B{k}" for k in range(1, len(branches))]

    rows = []
    for name, b in zip(names, branches):
        for j, ic, s in b:
            rows.append(dict(branch=name, freq=F[j], period=1.0 / F[j], c=c[ic], score=s))
    D = pd.DataFrame(rows)
    D.to_csv(f"{d}/fj_picks_{a.comp}.csv", index=False)
    for name in names:
        s = D[D.branch == name].sort_values("period")
        fn = ("ref_fundamental_phase_FJ.txt" if name == "R0"
              else f"ref_{name.lower()}_phase_FJ.txt")
        np.savetxt(f"{d}/{fn}", s[["period", "c"]].values, fmt="%.4f %.4f",
                   header=f"period[s]  phase_velocity[km/s]  ({name}, F-J topology pick, "
                          f"score>={a.score}, {a.comp}, {a.net})")

    # figure
    fig, ax = plt.subplots(figsize=(11.5, 7))
    ax.pcolormesh(F, c, np.clip(A / np.percentile(A, 99.5), 0, 1), cmap="gray_r", shading="auto")
    for old, fn, col in (("old R0", "ref_fundamental_phase.txt", "cyan"),
                         ("old ‘R1’", "ref_overtone_phase.txt", "magenta")):
        p = f"{d}/{fn}"
        if os.path.exists(p):
            r = np.loadtxt(p); ax.plot(1 / r[:, 0], r[:, 1], "--", color=col, lw=1.2, alpha=0.8, label=old)
    cols = plt.cm.tab10(np.linspace(0, 0.9, max(len(names), 3)))
    for k, name in enumerate(names):
        s = D[D.branch == name].sort_values("freq")
        ax.scatter(s.freq, s.c, s=9 + 25 * s.score, color=cols[k], edgecolor="k", lw=0.3,
                   label=f"{name}  (n={len(s)}, {s.freq.min():.2f}-{s.freq.max():.2f} Hz)", zorder=5)
    ax.set_xscale("log"); ax.set_xlim(a.fmin, a.fmax); ax.set_ylim(c.min(), a.cmax)
    ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("phase velocity [km/s]")
    ax.set_title(f"{a.net} — F-J image ({a.comp}, {int(z['n_sources'])} sources) with topology "
                 f"picks (score ≥ {a.score}); marker size ∝ persistence", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{d}/fj_picks_{a.comp}.png", dpi=140, bbox_inches="tight")
    print(f"{a.net}: {len(branches)} branches")
    for name in names:
        s = D[D.branch == name]
        print(f"   {name:<4} n={len(s):>4}  f {s.freq.min():.2f}-{s.freq.max():.2f} Hz  "
              f"c {s.c.min():.2f}-{s.c.max():.2f}  mean score {s.score.mean():.2f}")
    print(f"wrote {d}/fj_picks_{a.comp}.png")


if __name__ == "__main__":
    main()
