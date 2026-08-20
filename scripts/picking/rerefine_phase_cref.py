#!/usr/bin/env python3
"""Re-resolve the 2*pi*N branch of the Rayleigh-fundamental PHASE picks against a new reference
curve (c_ref) WITHOUT re-picking, and evaluate how the phase-pick distribution changes.

Why this is exact. The production picker resolves phase in `unwrap` mode
(dispersion.resolve_phase_curve_unwrap): each curve segment is frequency-unwrapped from the
measured phases (independent of c_ref), and c_ref enters ONLY through one global integer M per
segment, chosen to minimise the median relative distance to c_ref. Hence for every stored pick

    K_i = w_i * dist / c_old_i          (= kr_i + 2*pi*M_old, fully determined by the stored c)
    c_i(dM) = w_i * dist / (K_i + 2*pi*dM)

and re-anchoring to a new reference is just re-choosing dM per segment. Segmentation follows the
picker (|dU| > 0.25 km/s or a period gap > 2.5x the median step) on the per-scale representative
picks; the picker used the refined U (dist/t_peak) there, we use the stored picked U -- the
`--validate` pass (re-anchor to the OLD reference, expect dM = 0) measures the effect.

QC re-flagging: for rayleigh/fundamental phase rows the only phase-killing gates are snr,
band_edge (both measures, c-independent), vbounds and phase_phys (c-dependent) -- all computable
from stored columns, so phase_ok is recomputed exactly under the run's qc_params.

Outputs (own directory, never touching qc_current):
  <picks_tree>/<qc_label>_cref<TAG>/picks_unified_QCd.csv   full table, phase columns patched
  <picks_tree>/<qc_label>_cref<TAG>/{README.txt, qc_params_used.yaml, phase_cref_eval.png,
                                      phase_cref_eval.csv, summary.txt}

Usage:
  python rerefine_phase_cref.py --net aargau --ref-new ref_fundamental_phase_FJmerged.txt --tag FJ
"""
import argparse, os, shutil, sys, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
TREE = "dispersion_unified_vmin0.2"
KEYS = ["pair", "component", "lag", "stack_method", "pick_method"]
NEED = KEYS + ["nominal_period", "group_velocity", "phase_velocity", "N_ambiguity", "distance", "score",
               "scale_j", "T_scale", "snr_nbG", "flagged_sta", "phase_ok", "phase_killer",
               "wave_type", "mode", "group_ok"]
N_SEARCH = 16
PICKER_NSEARCH = 8      # resolve_phase_curve_unwrap n_search: |M| <= 8 (absolute)


def load_ref(path):
    a = np.loadtxt(path); o = np.argsort(a[:, 0])
    T, c = a[o, 0], a[o, 1]
    return lambda t: np.interp(t, T, c, left=np.nan, right=np.nan)


def reanchor(df, cref, seg_dU=0.25, seg_gap=2.5):
    """Return (c_new, dM) aligned with df rows (fund phase rows with finite c_old)."""
    c_new = np.full(len(df), np.nan); dM_out = np.zeros(len(df), dtype=int)
    T = df["nominal_period"].to_numpy(float); U = df["group_velocity"].to_numpy(float)
    C = df["phase_velocity"].to_numpy(float); D = df["distance"].to_numpy(float)
    S = df["scale_j"].to_numpy(int); NA = df["N_ambiguity"].to_numpy(int)
    gid = df.groupby(KEYS, sort=False).ngroup().to_numpy()
    order = np.lexsort((T, gid))               # by curve, then T ascending = the picker's emission order
    gid_s = gid[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(gid_s)) + 1, len(order)]
    dMs = np.arange(-N_SEARCH, N_SEARCH + 1)
    for a, b in zip(bounds[:-1], bounds[1:]):
        ii = order[a:b]
        # one representative per CWT scale (identical measurement): the picker takes the FIRST pick
        # in emission order (= shortest nominal period) -- its w and U are what entered the unwrap
        _, first = np.unique(S[ii], return_index=True)
        rep = ii[first]
        rep = rep[np.argsort(-T[rep])]                 # then w ascending (T descending) as in the unwrap
        Tr, Ur, Cr, dist = T[rep], U[rep], C[rep], D[rep[0]]
        w = 2 * np.pi / Tr
        # segmentation as in resolve_phase_curve_unwrap
        brk = np.zeros(len(rep), bool)
        if len(rep) > 1:
            dTmed = np.median(np.abs(np.diff(Tr)))
            brk[1:] = (np.abs(np.diff(Ur)) > seg_dU) | ((dTmed > 0) & (np.abs(np.diff(Tr)) > seg_gap * dTmed))
        seg = np.cumsum(brk)
        K = w * dist / Cr
        cr = cref(Tr)
        best = np.zeros(len(rep), int); cn = np.full(len(rep), np.nan)
        for s in np.unique(seg):
            m = seg == s
            # the picker searched the ABSOLUTE integer M in [-8, 8]; the first pick of a segment has
            # unwrap step 0, so its stored N_ambiguity is M_old -> restrict M_old + dM to that range
            M_old = NA[rep[m][0]]
            allowed = np.abs(M_old + dMs) <= PICKER_NSEARCH
            with np.errstate(divide="ignore", invalid="ignore"):
                den = K[m][None, :] + 2 * np.pi * dMs[:, None]
                cc = np.where(den > 0, (w[m] * dist)[None, :] / den, np.nan)
                cost = np.nanmedian(np.abs(cc - cr[m][None, :]) / cr[m][None, :], axis=1)
            cost[~allowed] = np.nan
            if np.all(~np.isfinite(cost)):
                continue                                   # no reference coverage: keep old
            k = int(np.nanargmin(cost)); best[m] = dMs[k]; cn[m] = cc[k]
        # broadcast to every row sharing the scale within the curve
        by_scale = dict(zip(S[rep], zip(cn, best)))
        for i in ii:
            v = by_scale[S[i]]; c_new[i] = v[0]; dM_out[i] = v[1]
    return c_new, dM_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--ref-new", required=True, help="filename under <net>/vsg_modesep/")
    ap.add_argument("--ref-old", default="ref_fundamental_phase.txt")
    ap.add_argument("--tag", default="FJ")
    ap.add_argument("--no-write", action="store_true", help="evaluate only, do not write the patched table")
    a = ap.parse_args()
    tree = f"{P}/{a.net}/{TREE}"; qc = os.path.realpath(f"{tree}/qc_current"); label = os.path.basename(qc)
    out = f"{tree}/{label}_cref{a.tag}"; os.makedirs(out, exist_ok=True)
    src = f"{qc}/picks_unified_QCd.csv"
    ref_old = load_ref(f"{P}/{a.net}/vsg_modesep/{a.ref_old}")
    ref_new = load_ref(f"{P}/{a.net}/vsg_modesep/{a.ref_new}")
    log = open(f"{out}/summary.txt", "w")
    def say(*s):
        print(*s); print(*s, file=log); log.flush()
    say(f"# {a.net}: re-anchor Rayleigh-fundamental phase to {a.ref_new} (old: {a.ref_old}); source {src}")

    # ---- pass 1: collect fund phase rows with their absolute row numbers
    t0 = time.time(); parts = []; off = 0
    for ch in pd.read_csv(src, usecols=NEED, chunksize=2_000_000):
        m = (ch["wave_type"] == "rayleigh") & (ch["mode"] == "fundamental") & ch["phase_velocity"].notna()
        sub = ch.loc[m].copy(); sub["row"] = sub.index.to_numpy(); parts.append(sub); off += len(ch)   # chunk index is GLOBAL
    df = pd.concat(parts, ignore_index=True); n_total = off
    say(f"rows total {n_total:,}; rayleigh-fund rows with phase {len(df):,}; curves {df.groupby(KEYS).ngroups:,}  ({time.time()-t0:.0f}s)")

    # ---- validation: re-anchor to the OLD reference -> expect no change
    c_val, dM_val = reanchor(df, ref_old)
    same = np.isclose(c_val, df["phase_velocity"], rtol=0, atol=1e-6)
    say(f"VALIDATION (old ref -> old ref): reproduced {100*same.mean():.3f}% of rows; dM!=0 on {100*(dM_val!=0).mean():.3f}%")

    # ---- new reference
    c_new_raw, dM_raw = reanchor(df, ref_new)
    # Ship ONLY the reference-driven change: shift the stored production value by the branch
    # difference between the new-ref and old-ref anchoring of the same machinery, so rows where the
    # reference makes no difference keep their production value bit-for-bit (the 3-6% machinery
    # residual -- refined vs stored U in the segmentation -- never enters the table).
    # Rows where this machinery already disagrees with production under the OLD reference (near-tie
    # segments decided by the refined-vs-stored U) cannot separate a reference effect from a
    # machinery effect: keep production unchanged there. (Applying dM_raw - dM_val instead pushed
    # near-reference picks one whole branch up -> a spurious 3.3-4.5 km/s band at 1.5-3.5 s.)
    dM = np.where(dM_val == 0, dM_raw, 0)
    say(f"rows where machinery != production under OLD ref (kept unchanged): {100*(dM_val!=0).mean():.2f}%")
    w_ = 2 * np.pi / df["nominal_period"].to_numpy(float); K_ = w_ * df["distance"].to_numpy(float) / df["phase_velocity"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        c_new = np.where(dM == 0, df["phase_velocity"].to_numpy(float), w_ * df["distance"].to_numpy(float) / (K_ + 2 * np.pi * dM))
    c_new = np.where(np.isfinite(c_new) & (c_new > 0), c_new, np.nan)
    df["c_new"] = c_new; df["dM"] = dM
    # QC re-flag (rayleigh fund phase): snr, band_edge, vbounds(fund), phase_phys, station(if used)
    import yaml
    prm = yaml.safe_load(open(f"{qc}/qc_params_used.yaml"))["resolved"]
    lo, hi = [float(x) for x in str(prm["vbounds_fund"]).split(",")]
    RUNG = 2.0 ** (1.0 / 12.0)
    killers = set(df["phase_killer"].fillna("").unique())

    def reflag(c):
        """phase_ok under this label's qc params for a phase-velocity column c (rayleigh fund rows):
        snr, band_edge (both measures), vbounds + phase_phys (phase), station if used, then the
        phase scale_dedupe (one survivor per (pair, component, lag, mode, scale_j): argmax first,
        nominal period closest to T_scale, highest score) -- exactly qc_unified_picks.py gate 12."""
        ok = (df["snr_nbG"] >= float(prm["snr_min"])) & np.isfinite(c) & c.between(lo, hi) & (c > df["group_velocity"])
        if int(prm.get("band_edge_rungs", 0)) > 0:
            ok &= ~(df["T_scale"].notna() & (df["T_scale"] * RUNG ** int(prm["band_edge_rungs"]) > df["distance"] / float(prm["vave"])))
        if "station" in killers:
            ok &= ~(df["flagged_sta"] > 0)
        if "scale_dedupe" in killers:
            ph = df.loc[ok.to_numpy() & (df["scale_j"] >= 0).to_numpy(), ["pair", "component", "lag", "mode", "scale_j", "pick_method", "nominal_period", "T_scale", "score"]].copy()
            ph["pm_rank"] = (ph["pick_method"] != "argmax").astype(int)
            ph["dT_scale"] = (ph["nominal_period"] - ph["T_scale"]).abs()
            keep = (ph.sort_values(["pm_rank", "dT_scale", "score"], ascending=[True, True, False])
                      .drop_duplicates(subset=["pair", "component", "lag", "mode", "scale_j"]).index)
            kill = df.index.isin(ph.index.difference(keep))
            ok = ok & ~kill
        return ok.to_numpy()

    df["phase_ok_new"] = reflag(df["c_new"])
    df["c_base"] = df["phase_velocity"].to_numpy(); df["phase_ok_base"] = df["phase_ok"].to_numpy(bool)   # production = baseline
    # sanity: the same rule applied to the OLD c must reproduce the stored phase_ok
    ok_old = reflag(df["phase_velocity"])
    say(f"QC re-flag check: recomputed phase_ok agrees with stored on {100*(ok_old==df['phase_ok'].to_numpy(bool)).mean():.3f}% of fund-phase rows "
        f"(stored ok {int(df['phase_ok'].sum()):,} vs recomputed {int(ok_old.sum()):,})")

    # ---- evaluation
    okO = df["phase_ok"].to_numpy(bool); okN = df["phase_ok_new"].to_numpy()
    say(f"phase_ok picks: old {okO.sum():,}  new {okN.sum():,}  ({100*(okN.sum()/okO.sum()-1):+.2f}%)")
    chg = (df["dM"] != 0).to_numpy()
    relchg = np.abs(df["c_new"].to_numpy() / df["phase_velocity"].to_numpy() - 1)
    material = chg & (relchg > 0.05)
    say(f"branch changed (dM!=0): {100*chg.mean():.2f}% of all fund-phase rows; {100*chg[okO].mean():.2f}% of old-phase_ok rows")
    say(f"  of which MATERIAL (|dc|/c > 5%): {100*material[okO].mean():.2f}% of old-phase_ok rows; "
        f"noise-level (adjacent branch at large d/lambda): {100*(chg & ~material)[okO].mean():.2f}%")
    # baseline for a like-with-like comparison: this machinery with the OLD reference
    chg_b = chg
    crn = ref_new(df["nominal_period"].to_numpy(float)); cro = ref_old(df["nominal_period"].to_numpy(float))
    m_ = chg & okO & np.isfinite(crn)
    say(f"  cost check on changed accepted rows: median |c-cref_NEW|/cref  production branch {np.nanmedian(np.abs(df['phase_velocity'].to_numpy()[m_]-crn[m_])/crn[m_]):.3f} -> new branch {np.nanmedian(np.abs(df['c_new'].to_numpy()[m_]-crn[m_])/crn[m_]):.3f};"
        f"  vs cref_OLD: production {np.nanmedian(np.abs(df['phase_velocity'].to_numpy()[m_]-cro[m_])/cro[m_]):.3f} -> new {np.nanmedian(np.abs(df['c_new'].to_numpy()[m_]-cro[m_])/cro[m_]):.3f}")
    say(f"  changed accepted rows moving UP in c: {100*(df['c_new'].to_numpy()[chg&okO] > df['phase_velocity'].to_numpy()[chg&okO]).mean():.1f}%; median T of changed rows {np.median(df['nominal_period'].to_numpy()[chg&okO]):.2f} s")
    Tb = np.arange(0.15, 6.05, 0.1); Tc = 0.5 * (Tb[:-1] + Tb[1:])
    T = df["nominal_period"].to_numpy(float)
    rows = []
    for t0_, t1_ in zip(Tb[:-1], Tb[1:]):
        m = (T >= t0_) & (T < t1_)
        if not m.any(): continue
        mo, mn = m & okO, m & okN
        rows.append(dict(T=round(0.5*(t0_+t1_), 2), n_old=int(mo.sum()), n_new=int(mn.sum()),
                         frac_changed_vs_stored=round(float(chg[mo].mean()) if mo.any() else np.nan, 4),
                         frac_changed_vs_baseline=round(float(chg_b[mo].mean()) if mo.any() else np.nan, 4),
                         c_med_old=round(float(np.median(df["phase_velocity"].to_numpy()[mo])) if mo.any() else np.nan, 3),
                         c_med_new=round(float(np.median(df["c_new"].to_numpy()[mn])) if mn.any() else np.nan, 3),
                         ref_old=round(float(ref_old(0.5*(t0_+t1_))), 3), ref_new=round(float(ref_new(0.5*(t0_+t1_))), 3)))
    ev = pd.DataFrame(rows); ev.to_csv(f"{out}/phase_cref_eval.csv", index=False)
    say(ev.to_string(index=False))

    # figure: 2-D histograms old / new / difference, refs overlaid, plus F-J image on the period axis
    vb = np.arange(0.5, 4.5, 0.02)
    Ho = np.histogram2d(T[okO], df["phase_velocity"].to_numpy()[okO], bins=[Tb, vb])[0]
    Hn = np.histogram2d(T[okN], df["c_new"].to_numpy()[okN], bins=[Tb, vb])[0]
    fig, axs = plt.subplots(1, 4, figsize=(24, 6.2))
    vmax = np.percentile(np.r_[Ho[Ho > 0], Hn[Hn > 0]], 99)
    for ax, H, ttl in ((axs[0], Ho, f"OLD c_ref (slant-stack): {okO.sum():,} phase_ok picks"),
                       (axs[1], Hn, f"NEW c_ref ({a.tag}): {okN.sum():,} phase_ok picks")):
        ax.pcolormesh(Tb, vb, H.T, cmap="viridis", vmin=0, vmax=vmax, shading="flat")
        ax.set_title(ttl, fontsize=10)
    okB = df["phase_ok_base"].to_numpy(bool)
    Hb = np.histogram2d(T[okB], df["c_base"].to_numpy()[okB], bins=[Tb, vb])[0]
    d = Hn - Hb; lim = np.percentile(np.abs(d[d != 0]), 99) if (d != 0).any() else 1
    axs[2].pcolormesh(Tb, vb, d.T, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="flat")
    axs[2].set_title("NEW − production counts (reference-driven change only; red = gained)", fontsize=10)
    fj = np.load(f"{P}/{a.net}/vsg_modesep/vsg_fj_ZZ_sign+1.npz"); Fq, cq, A = fj["f"], fj["vel"], fj["FJ"]
    axs[3].pcolormesh(1 / Fq, cq, np.clip(A / np.percentile(A, 99.5), 0, 1), cmap="gray_r", shading="auto")
    axs[3].plot(ev["T"], ev["c_med_old"], "o-", ms=3, color="tab:red", label="median pick, OLD c_ref")
    axs[3].plot(ev["T"], ev["c_med_new"], "s-", ms=3, color="tab:blue", label="median pick, NEW c_ref")
    axs[3].set_title("F-J image (period axis) with per-period pick medians", fontsize=10)
    for ax in axs:
        Tg = np.arange(0.5, 6, 0.05)
        ax.plot(Tg, ref_old(Tg), "--", color="orange", lw=1.6, label="c_ref OLD (slant-stack)")
        ax.plot(Tg, ref_new(Tg), "-", color="lime", lw=1.6, label="c_ref NEW (F-J)")
        ax.set_xlim(0.2, 6); ax.set_ylim(0.5, 4.5); ax.set_xlabel("period [s]"); ax.set_ylabel("phase velocity [km/s]")
        ax.grid(alpha=.25); ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{a.net}: Rayleigh-fundamental phase picks re-anchored to the F-J reference "
                 f"({100*chg[okO].mean():.1f}% of accepted picks change branch)", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{out}/phase_cref_eval.png", dpi=130, bbox_inches="tight")
    say("wrote", f"{out}/phase_cref_eval.png")

    if a.no_write:
        return
    # ---- pass 2: stream the full table, patch phase columns for these rows
    patch_c = np.full(n_total, np.nan); patch_ok = np.zeros(n_total, bool); patch_dM = np.zeros(n_total, int)
    hit = np.zeros(n_total, bool)
    r = df["row"].to_numpy(); patch_c[r] = df["c_new"]; patch_ok[r] = df["phase_ok_new"]; patch_dM[r] = df["dM"]; hit[r] = True
    dst = f"{out}/picks_unified_QCd.csv"; off = 0; first = True
    for ch in pd.read_csv(src, chunksize=2_000_000):
        idx = ch.index.to_numpy(); h = hit[idx]                       # global row index
        if h.any():
            sel = ch.index[h]
            ch.loc[sel, "phase_velocity"] = patch_c[idx[h]]
            ch.loc[sel, "N_ambiguity"] = ch.loc[sel, "N_ambiguity"].to_numpy() + patch_dM[idx[h]]
            ch.loc[sel, "phase_ok"] = patch_ok[idx[h]]
            # killer bookkeeping: rows that flip status
            newly_bad = sel[~patch_ok[idx[h]] & ch.loc[sel, "phase_killer"].eq("").to_numpy()]
            ch.loc[newly_bad, "phase_killer"] = "cref_requal"
            revived = sel[patch_ok[idx[h]] & ch.loc[sel, "phase_killer"].isin(["vbounds", "phase_phys"]).to_numpy()]
            ch.loc[revived, "phase_killer"] = ""
            # U_from_phase would need the whole curve; mark stale where c changed
            ch.loc[sel[patch_dM[idx[h]] != 0], "U_from_phase"] = np.nan
        ch.to_csv(dst, index=False, mode="w" if first else "a", header=first); first = False; off += len(ch)
    shutil.copy(f"{qc}/qc_params_used.yaml", f"{out}/qc_params_used.yaml")
    with open(f"{out}/README.txt", "w") as fh:
        fh.write(f"Derived from {qc} by rerefine_phase_cref.py: Rayleigh-fundamental PHASE picks re-anchored\n"
                 f"to {a.ref_new} (2*pi*N re-choice per curve segment; group picks, Love and overtone untouched).\n"
                 f"phase_ok recomputed under the same qc params. U_from_phase set NaN where the branch changed.\n"
                 f"See summary.txt / phase_cref_eval.png. qc_current NOT repointed.\n")
    say("wrote", dst)


if __name__ == "__main__":
    main()
