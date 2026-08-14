#!/usr/bin/env python3
"""Build an INTERACTIVE period-range explorer: pick vs cell 2D histograms + draggable bounds.

The static counterpart is `pick_vs_cell_hist2d.py`. This packs the same two-column comparison
(left = input picks, right = output map cells, one row per Cd) for EVERY
(network, measure, wave, k) into one self-contained HTML file with:

  * zoom / pan on shared period-velocity axes (all panels stay locked together),
  * two draggable period handles per (net, measure, wave) that snap to the CWT rungs,
  * a diagnostics strip (rays, var_red, anomaly amplitude, geology eta2) on the same x-axis,
  * export of the choices straight back into `period_ranges_DECISIONS.csv` format.

Nothing is decided here -- it is the same evidence as the static figures, made steerable.

BINNING is identical to the static script, and deliberately so: the period axis is the CWT
scale ladder (geometric midpoints between the rungs actually present in the file) and the
velocity axis is 0.02 km/s with edges offset half a node (0.195, not 0.20). `group_velocity`
steps by exactly 0.01 km/s and edges landing ON a node push picks into the bin below.

NORMALISATION is also identical: each period column sums to 1, independently for picks and
cells, so the comparison is not swamped by the pick/cell count ratio or by the strong period
dependence of both.

ENCODING. Column-normalised histograms are stored as uint8 with a SQRT transform
(q = 255*sqrt(H/Hmax)), which keeps resolution in the low-density tails that a linear uint8
would flatten to zero; the browser undoes it as (q/255)^2 before applying the display clip.
Velocity rows outside the occupied band are trimmed per (net, measure, wave) to keep the file
small. Typical size is a few MB, entirely offline.

Usage:
  python build_period_explorer.py                    # every net x measure x wave, k3 + k2
  python build_period_explorer.py --k k3 --net riehen
  open /Users/genevievesavard/Codes/extract_higher_modes/Projects/_period_validity/period_explorer.html
"""
import argparse
import base64
import csv
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_exported_picks_hist2d import V_EDGES, rung_edges      # noqa: E402

EHM = "/Users/genevievesavard/Codes/extract_higher_modes/Projects"
CMP = f"{EHM}/_inversion_comparison"
OUT = f"{EHM}/_period_validity"
DX = {"riehen": "0.2", "aargau": "0.5", "hautesorne": "0.5"}
WAVES = ("fund", "overtone", "love")
MEASURES = ("group", "phase")
CDS = ("blanket", "measured", "scaled")
# recommended Cd per measure -- see each net's 1_velocity_maps/README.md
CD_REC = {"group": "scaled", "phase": "blanket"}
TITLE = {"fund": "Rayleigh fundamental", "overtone": "Rayleigh overtone",
         "love": "Love fundamental"}


def col_normalise(H):
    """Each period column -> sums to 1; empty columns stay 0 (JS treats 0 as 'no data')."""
    s = H.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        Hn = np.where(s > 0, H / np.where(s == 0, 1.0, s), 0.0)
    return np.nan_to_num(Hn, nan=0.0)


def median_curve(H, centres):
    """Per-period median of a column-normalised histogram (NaN where the column is empty)."""
    out = []
    for j in range(H.shape[0]):
        col = H[j]
        good = col > 0
        if not good.any():
            out.append(np.nan)
            continue
        c = np.cumsum(col[good]) / col[good].sum()
        out.append(float(centres[good][np.searchsorted(c, 0.5)]))
    return np.asarray(out, float)


def quantise(H, vlo, vhi):
    """uint8 sqrt-encoded histogram over velocity rows [vlo, vhi), plus its scale."""
    sub = H[:, vlo:vhi]
    m = float(sub.max()) if sub.size and sub.max() > 0 else 1.0
    q = np.clip(np.sqrt(sub / m) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return base64.b64encode(q.tobytes()).decode("ascii"), m


def jnum(a, nd=4):
    """Compact JSON-safe float list; NaN -> None (JSON has no NaN)."""
    return [None if not np.isfinite(v) else round(float(v), nd) for v in np.asarray(a, float)]


def inferno_lut():
    cm = matplotlib.colormaps["inferno"]
    return [[int(round(255 * c)) for c in cm(i / 255.0)[:3]] for i in range(256)]


def read_decisions(path):
    """Existing T_valid_min/max, so re-running the builder never discards choices."""
    if not os.path.exists(path):
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        lo = (r.get("T_valid_min") or "").strip()
        hi = (r.get("T_valid_max") or "").strip()
        why = (r.get("reason") or "").strip()
        if lo or hi or why:
            out["%s|%s|%s" % (r["net"], r["measure"], r["wave"])] = {
                "lo": float(lo) if lo else None, "hi": float(hi) if hi else None,
                "reason": why}
    return out


def build_combo(net, meas, wave, k, met):
    """One (net, measure, wave, k) payload: pick hist, per-Cd cell hists, diagnostics."""
    V = f"{EHM}/{net}/tomo/1_velocity_maps"
    suf = "_phase" if meas == "phase" else ""
    pf = f"{V}/0_inputs/culled_picks_vbounds/{k}/picks_{wave}_uni{suf}.csv"
    if not os.path.exists(pf):
        return None
    d = pd.read_csv(pf, usecols=["inst_period", "group_velocity"])
    d = d[np.isfinite(d.group_velocity)]
    if len(d) < 500:
        return None

    te = rung_edges(d["inst_period"].values)
    tc = np.sqrt(te[:-1] * te[1:])
    rungs = np.sort(np.unique(np.round(d["inst_period"].values, 4)))
    vc = 0.5 * (V_EDGES[:-1] + V_EDGES[1:])

    Hp_raw, _, _ = np.histogram2d(d.inst_period, d.group_velocity, bins=[te, V_EDGES])
    npick = Hp_raw.sum(axis=1)
    Hp = col_normalise(Hp_raw)

    cells, occ = {}, (Hp_raw.sum(axis=0) > 0)
    for cd in CDS:
        root = f"{V}/1_production/tspws_{meas}_{cd}_dx{DX[net]}_prod3_{k}/production/{wave}"
        files = glob.glob(f"{root}/map_T*.npz")
        if not files:
            continue
        Hc = np.zeros((len(te) - 1, len(V_EDGES) - 1))
        ncell = np.zeros(len(te) - 1)
        nray = np.full(len(te) - 1, np.nan)
        vred = np.full(len(te) - 1, np.nan)
        # the map's OWN period value per column. It is a rounded version of the pick rung
        # (rung 0.2016 -> map 0.20, rung 1.0763 -> map 1.08), and `restrict_periods` compares
        # the bound against THIS number -- so the bounds must snap to it, never to the rung.
        mper = np.full(len(te) - 1, np.nan)
        for f in files:
            z = np.load(f)
            T = float(z["period"])
            v = z["vel"][np.isfinite(z["vel"])]
            if v.size < 50:
                continue
            j = int(np.clip(np.searchsorted(te, T) - 1, 0, len(te) - 2))
            Hc[j] += np.histogram(v, bins=V_EDGES)[0]
            ncell[j] = v.size
            nray[j] = float(z["N"])
            vred[j] = float(z["var_red"])
            mper[j] = T
        if not np.isfinite(nray).any():
            continue
        occ |= (Hc.sum(axis=0) > 0)
        # amplitude / geology correlation come from the comparison table, not the maps
        m = met[(met.net == net) & (met.measure == meas) & (met.cd == cd) &
                (met.wave == wave) & (met.k == k)]
        amp = np.full(len(te) - 1, np.nan)
        eta = np.full(len(te) - 1, np.nan)
        if len(m):
            for _, r in m.iterrows():
                j = int(np.clip(np.searchsorted(te, r["T"]) - 1, 0, len(te) - 2))
                amp[j] = r["amp_std"]
                eta[j] = r["eta2"]
        cells[cd] = dict(H=col_normalise(Hc), ncell=ncell, nray=nray, vred=vred,
                         amp=amp, eta=eta, mper=mper)
    if not cells:
        return None

    # The Vs inversion reads ONE production root per measure (the recommended Cd), so the
    # snap targets and the "available periods" are that root's map periods -- not the pick
    # rungs, which run past the last map (Haute-Sorne picks reach 10.8 s, maps stop at 8.6).
    snap_cd = CD_REC[meas] if CD_REC[meas] in cells else sorted(cells)[0]
    tsnap = np.sort(cells[snap_cd]["mper"][np.isfinite(cells[snap_cd]["mper"])])

    # trim the velocity axis to the occupied band (+5 bins of air) -- pure file-size saving
    idx = np.flatnonzero(occ)
    vlo = max(0, int(idx[0]) - 5)
    vhi = min(len(V_EDGES) - 1, int(idx[-1]) + 6)

    pq, _ = quantise(Hp, vlo, vhi)
    out = dict(
        te=jnum(te, 4), tc=jnum(tc, 4), rungs=jnum(rungs, 3),
        tsnap=jnum(tsnap, 4), snap_cd=snap_cd,
        v0=round(float(V_EDGES[vlo]), 4), dv=round(float(V_EDGES[1] - V_EDGES[0]), 4),
        nv=int(vhi - vlo), nt=int(len(te) - 1),
        picks=dict(q=pq, med=jnum(median_curve(Hp, vc), 3), n=jnum(npick, 0),
                   total=int(len(d))),
        cells={}, cd_rec=CD_REC[meas])
    for cd, c in cells.items():
        q, _ = quantise(c["H"], vlo, vhi)
        out["cells"][cd] = dict(q=q, med=jnum(median_curve(c["H"], vc), 3),
                                n=jnum(c["ncell"], 0), nray=jnum(c["nray"], 0),
                                vred=jnum(c["vred"], 3), amp=jnum(c["amp"], 2),
                                eta=jnum(c["eta"], 3), mper=jnum(c["mper"], 4))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", default=None, help="k3 or k2; default both")
    ap.add_argument("--net", default=None)
    ap.add_argument("--out", default=f"{OUT}/period_explorer.html")
    ap.add_argument("--decisions", default=None,
                    help="decision sheet to preload the handles from; default is the most "
                         "recently modified period_ranges_DECISIONS*.csv in --out")
    a = ap.parse_args()
    if a.decisions is None:
        found = sorted(glob.glob(f"{a.out and os.path.dirname(a.out) or OUT}"
                                 "/period_ranges_DECISIONS*.csv"),
                       key=os.path.getmtime, reverse=True)
        a.decisions = found[0] if found else f"{OUT}/period_ranges_DECISIONS.csv"
        print("decisions sheet: %s" % a.decisions)   # never pick a version silently
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    met = pd.read_csv(f"{CMP}/per_period_metrics.csv")
    ks = (a.k,) if a.k else ("k3", "k2")
    nets = (a.net,) if a.net else tuple(DX)

    combos, avail = {}, {}
    for net in nets:
        for meas in MEASURES:
            for wave in WAVES:
                for k in ks:
                    c = build_combo(net, meas, wave, k, met)
                    if c is None:
                        print("  skip %s %s %s %s" % (net, meas, wave, k))
                        continue
                    key = "%s|%s|%s|%s" % (net, meas, wave, k)
                    # Stable per-histogram identity for the browser's offscreen cache. It
                    # MUST NOT be derived from the encoded bytes: picks and cells share nt,
                    # nv and a run of leading zero bins, so a content-prefix key collides
                    # and one panel silently renders the other's histogram.
                    c["picks"]["id"] = key + "|picks"
                    for cd in c["cells"]:
                        c["cells"][cd]["id"] = key + "|" + cd
                    combos[key] = c
                    avail.setdefault(net, {}).setdefault(meas, {}).setdefault(wave, []).append(k)
                    print("  %-38s %2d rungs x %3d vbins, Cd %s"
                          % (key, c["nt"], c["nv"], ",".join(sorted(c["cells"]))))
    if not combos:
        raise SystemExit("no combos built -- check the paths in DX / EHM")

    payload = dict(combos=combos, avail=avail, lut=inferno_lut(),
                   decisions=read_decisions(a.decisions), cds=list(CDS),
                   waves=list(WAVES), measures=list(MEASURES), title=TITLE,
                   dx=DX, ks=list(ks), source=os.path.abspath(a.decisions))
    js = json.dumps(payload, separators=(",", ":"))
    html = TEMPLATE.replace("/*__PAYLOAD__*/null", js)
    with open(a.out, "w") as f:
        f.write(html)
    print("\nwrote %s  (%.1f MB, %d panels)"
          % (a.out, os.path.getsize(a.out) / 1e6, len(combos)))


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Period-range explorer — picks vs map cells</title>
<style>
:root{
  --bg:#12131a; --panel:#1b1d27; --line:#2f3340; --fg:#e8eaf2; --dim:#9aa0b4;
  --accent:#5ec8ff; --warn:#ffb454; --bad:#ff6b6b; --ok:#39ff88;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
header{padding:10px 16px;border-bottom:1px solid var(--line);background:var(--panel);
       position:sticky;top:0;z-index:20}
h1{font-size:15px;margin:0 0 8px;font-weight:600}
h1 span{color:var(--dim);font-weight:400}
.row{display:flex;flex-wrap:wrap;gap:14px;align-items:center}
.grp{display:flex;gap:5px;align-items:center}
.grp>label{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
button,select,input[type=text],input[type=number]{
  background:#252836;color:var(--fg);border:1px solid var(--line);border-radius:5px;
  padding:4px 9px;font:inherit;font-size:12px;cursor:pointer}
button:hover{border-color:var(--accent)}
button.on{background:var(--accent);color:#06121a;border-color:var(--accent);font-weight:600}
input[type=text],input[type=number]{cursor:text}
input[type=range]{width:110px;accent-color:var(--accent)}
main{padding:12px 16px 60px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.cell{background:var(--panel);border:1px solid var(--line);border-radius:7px;overflow:hidden}
.cap{padding:5px 9px;font-size:11.5px;color:var(--dim);border-bottom:1px solid var(--line);
     display:flex;justify-content:space-between;gap:8px}
.cap b{color:var(--fg);font-weight:600}
canvas{display:block;width:100%;cursor:crosshair}
.strip{margin-top:10px;background:var(--panel);border:1px solid var(--line);border-radius:7px}
.bar{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:10px 0 4px;
     padding:8px 12px;background:var(--panel);border:1px solid var(--line);border-radius:7px}
.pill{padding:2px 7px;border-radius:20px;font-size:11px;background:#252836;border:1px solid var(--line)}
.pill.bad{color:var(--bad);border-color:#5a2b2b}
.pill.warn{color:var(--warn);border-color:#5a452b}
.pill.ok{color:var(--ok);border-color:#2b5a3d}
#tip{position:fixed;pointer-events:none;background:#0c0d12ee;border:1px solid var(--line);
     border-radius:5px;padding:6px 9px;font-size:11.5px;white-space:pre;z-index:50;display:none;
     box-shadow:0 4px 16px #0008}
#help{color:var(--dim);font-size:11.5px;margin-top:10px;line-height:1.7}
kbd{background:#252836;border:1px solid var(--line);border-radius:3px;padding:0 4px;font-size:11px}
table{border-collapse:collapse;width:100%;font-size:11.5px;margin-top:6px}
th,td{text-align:left;padding:3px 8px;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:500}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.done{color:var(--ok)}
dialog{background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:9px;
       max-width:min(920px,92vw);padding:16px}
dialog::backdrop{background:#000a}
textarea{width:100%;height:300px;background:#0f1017;color:var(--fg);border:1px solid var(--line);
         border-radius:5px;font:11.5px/1.5 ui-monospace,Menlo,monospace;padding:8px}
</style>
</head>
<body>
<header>
  <h1>Period-range explorer <span id="sub"></span></h1>
  <div class="row">
    <div class="grp"><label>net</label><select id="net"></select></div>
    <div class="grp"><label>measure</label><select id="meas"></select></div>
    <div class="grp"><label>wave</label><select id="wave"></select></div>
    <div class="grp"><label>k</label><span id="kbtns"></span></div>
    <div class="grp"><label>Cd</label><span id="cdbtns"></span></div>
    <div class="grp"><label>contrast</label><input type="range" id="gam" min="0.15" max="1" step="0.05" value="0.45"></div>
    <div class="grp">
      <button id="reset">reset zoom</button>
      <button id="fit">fit velocity</button>
      <button id="export">export CSV…</button>
    </div>
  </div>
</header>
<main>
  <div class="bar">
    <div class="grp"><label>T&nbsp;min</label><input type="number" id="tlo" step="0.01" style="width:82px"></div>
    <div class="grp"><label>T&nbsp;max</label><input type="number" id="thi" step="0.01" style="width:82px"></div>
    <div class="grp"><label>reason</label><input type="text" id="why" placeholder="why this range" style="width:300px"></div>
    <button id="clear">clear this range</button>
    <button id="all">apply to all waves of this net+measure</button>
    <span id="stat"></span>
  </div>
  <div class="grid">
    <div class="cell"><div class="cap"><span>INPUT PICKS <b id="capL"></b></span><span id="nL"></span></div>
      <canvas id="cL"></canvas></div>
    <div class="cell"><div class="cap"><span>MAP CELLS <b id="capR"></b></span><span id="nR"></span></div>
      <canvas id="cR"></canvas></div>
  </div>
  <div class="strip"><canvas id="cS"></canvas></div>
  <div id="help">
    <b>Zoom</b> scroll over any panel (zooms both axes at the cursor) · <kbd>shift</kbd>+scroll = period only ·
    <kbd>alt</kbd>+scroll = velocity only · <b>pan</b> drag with the middle/right button or <kbd>space</kbd>+drag ·
    <b>double-click</b> resets. Panels and the diagnostics strip share one period axis.<br>
    <b>Bounds</b> drag the two orange handles, or type into T&nbsp;min / T&nbsp;max. They snap to the
    <b>map periods</b>, which is the number <code>restrict_periods()</code> compares against — the
    map value is a rounded version of the pick rung (rung 0.2016 → map 0.20), so a bound placed on
    the rung would drop the very period you meant to keep. Dark shading is what the Vs inversion
    discards; <span style="color:#ff6b6b">red columns</span> in the right panel are periods with
    picks but no usable map. Choices are kept in this browser as you switch panels;
    <b>export CSV</b> writes them back in <code>period_ranges_DECISIONS.csv</code> format.
  </div>
  <table id="tbl"></table>
</main>
<div id="tip"></div>
<dialog id="dlg">
  <h3 style="margin:0 0 8px">period_ranges_DECISIONS.csv</h3>
  <p style="color:var(--dim);margin:0 0 8px">Save over <code id="dst"></code>, or copy the text.</p>
  <textarea id="csv" readonly></textarea>
  <div class="row" style="margin-top:10px">
    <button id="dl">download</button><button id="copy">copy</button>
    <button id="close" style="margin-left:auto">close</button>
  </div>
</dialog>
<script>
const D = /*__PAYLOAD__*/null;
const LUT = D.lut;
const $ = id => document.getElementById(id);
const S = {net:null, meas:null, wave:null, k:null, cd:null, gam:0.45,
           view:null, drag:null, dec:{}};

/* ---------- persistence: browser copy of the decision sheet ---------- */
const LSKEY = "period_explorer_decisions_v1";
S.dec = Object.assign({}, D.decisions, JSON.parse(localStorage.getItem(LSKEY) || "{}"));
const saveDec = () => localStorage.setItem(LSKEY, JSON.stringify(S.dec));
const dkey = () => S.net+"|"+S.meas+"|"+S.wave;          // k is NOT part of the decision
const cur  = () => D.combos[S.net+"|"+S.meas+"|"+S.wave+"|"+S.k];

/* ---------- decode a base64 uint8 histogram ---------- */
function unb64(s){
  const bin = atob(s), a = new Uint8Array(bin.length);
  for (let i=0;i<bin.length;i++) a[i] = bin.charCodeAt(i);
  return a;
}
const CACHE = new Map();
/* offscreen canvas nt x nv, one pixel per (period rung, velocity bin) */
function offscreen(id, q, nt, nv, gam){
  const key = id+"|"+gam;            // id is assigned per histogram at build time, never
  if (CACHE.has(key)) return CACHE.get(key);   // derived from the bytes (they collide)
  const a = unb64(q), cv = document.createElement("canvas");
  cv.width = nt; cv.height = nv;
  const im = new ImageData(nt, nv);
  for (let j=0;j<nt;j++) for (let i=0;i<nv;i++){
    const v = a[j*nv+i]/255;
    // undo the sqrt encoding, then apply the display gamma as a contrast control
    const h = v*v;
    const p = ((nv-1-i)*nt + j)*4;                 // canvas y grows downward
    if (h <= 0){ im.data[p+3] = 0; continue; }
    const c = LUT[Math.min(255, Math.round(255*Math.pow(Math.min(1,h/gam), 0.75)))];
    im.data[p]=c[0]; im.data[p+1]=c[1]; im.data[p+2]=c[2]; im.data[p+3]=255;
  }
  cv.getContext("2d").putImageData(im, 0, 0);
  if (CACHE.size > 60) CACHE.clear();
  CACHE.set(key, cv);
  return cv;
}

/* ---------- view (shared period + velocity window) ---------- */
function fullView(c){
  const v1 = c.v0 + c.nv*c.dv;
  return {t0:c.te[0], t1:c.te[c.te.length-1], v0:c.v0, v1:v1};
}
function ensureView(){
  const c = cur();
  if (!c) return;
  if (!S.view) S.view = fullView(c);
  const f = fullView(c);
  // keep the window inside the new panel's data when switching combos
  S.view.t0 = Math.max(f.t0, Math.min(S.view.t0, f.t1));
  S.view.t1 = Math.min(f.t1, Math.max(S.view.t1, f.t0));
  if (S.view.t1 - S.view.t0 < 1e-6) { S.view.t0 = f.t0; S.view.t1 = f.t1; }
  S.view.v0 = Math.max(f.v0, Math.min(S.view.v0, f.v1));
  S.view.v1 = Math.min(f.v1, Math.max(S.view.v1, f.v0));
  if (S.view.v1 - S.view.v0 < 1e-6) { S.view.v0 = f.v0; S.view.v1 = f.v1; }
}

const M = {l:58, r:12, t:8, b:34};
const PLOT_H = 340, STRIP_H = 300;
function rect(cv){ return {x:M.l, y:M.t, w:cv.width/DPR - M.l - M.r, h:cv.height/DPR - M.t - M.b}; }
const DPR = window.devicePixelRatio || 1;
function sizeCanvas(cv, h){
  // a container laid out at zero width (pane collapsed, tab hidden, print) would give a
  // 0-px canvas that stays blank until something forces a resize -- keep a usable floor
  const w = Math.max(cv.parentElement.clientWidth, 320);
  cv.width = w*DPR; cv.height = h*DPR; cv.style.height = h+"px";
  const g = cv.getContext("2d");
  g.setTransform(DPR,0,0,DPR,0,0);
  return g;
}

function axes(g, r, xlab, ylab, yfmt){
  g.strokeStyle = "#3a3f50"; g.fillStyle = "#9aa0b4"; g.lineWidth = 1;
  g.font = "11px -apple-system,Helvetica,Arial";
  g.strokeRect(r.x+.5, r.y+.5, r.w, r.h);
  const V = S.view;
  for (const t of ticks(V.t0, V.t1)){
    const x = r.x + (t-V.t0)/(V.t1-V.t0)*r.w;
    if (x < r.x-1 || x > r.x+r.w+1) continue;
    g.strokeStyle = "#2a2e3b"; g.beginPath(); g.moveTo(x,r.y); g.lineTo(x,r.y+r.h); g.stroke();
    g.strokeStyle = "#5a6070"; g.beginPath(); g.moveTo(x,r.y+r.h); g.lineTo(x,r.y+r.h+4); g.stroke();
    g.textAlign = "center"; g.fillText(fmt(t), x, r.y+r.h+15);
  }
  if (ylab !== null){
    for (const v of ticks(V.v0, V.v1)){
      const y = r.y + r.h - (v-V.v0)/(V.v1-V.v0)*r.h;
      if (y < r.y-1 || y > r.y+r.h+1) continue;
      g.strokeStyle = "#2a2e3b"; g.beginPath(); g.moveTo(r.x,y); g.lineTo(r.x+r.w,y); g.stroke();
      g.textAlign = "right"; g.fillText((yfmt||fmt)(v), r.x-5, y+3.5);
    }
    g.save(); g.translate(13, r.y+r.h/2); g.rotate(-Math.PI/2);
    g.textAlign = "center"; g.fillText(ylab, 0, 0); g.restore();
  }
  if (xlab){ g.textAlign="center"; g.fillText(xlab, r.x+r.w/2, r.y+r.h+29); }
}
function fmt(v){ return Math.abs(v) >= 100 ? v.toFixed(0)
                      : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2); }
function ticks(a, b){
  const span = b-a, raw = span/7, p = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw/p, step = p*(n<1.5?1:n<3?2:n<7?5:10);
  const out = []; for (let v=Math.ceil(a/step)*step; v<=b+1e-9; v+=step) out.push(+v.toFixed(10));
  return out;
}

/* ---------- one histogram panel ---------- */
function drawPanel(cv, c, src, med, other, isPicks, mper){
  const g = sizeCanvas(cv, PLOT_H), r = rect(cv), V = S.view;
  g.clearRect(0,0,cv.width,cv.height);
  const X = t => r.x + (t-V.t0)/(V.t1-V.t0)*r.w;
  const Y = v => r.y + r.h - (v-V.v0)/(V.v1-V.v0)*r.h;
  if (!src){ axes(g,r,"period [s]","velocity [km/s]"); return; }
  const off = offscreen(src.id, src.q, c.nt, c.nv, S.gam);
  const vTop = c.v0 + c.nv*c.dv;
  g.save(); g.beginPath(); g.rect(r.x,r.y,r.w,r.h); g.clip();
  g.imageSmoothingEnabled = false;
  // one drawImage per period rung: the period axis is NOT uniform (CWT ladder), so each
  // column gets its own x extent -- the same reason the static figure needs pcolormesh
  for (let j=0;j<c.nt;j++){
    const xa = X(c.te[j]), xb = X(c.te[j+1]);
    if (xb < r.x-1 || xa > r.x+r.w+1) continue;
    g.drawImage(off, j, 0, 1, c.nv, xa, Y(vTop), Math.max(xb-xa, .6), Y(c.v0)-Y(vTop));
  }
  const line = (m, col, dash) => {
    g.strokeStyle = col; g.lineWidth = dash ? 1.4 : 1.9; g.setLineDash(dash||[]);
    g.beginPath(); let up = false;
    for (let j=0;j<c.nt;j++){
      if (m[j] == null){ up = false; continue; }
      const x = X(c.tc[j]), y = Y(m[j]);
      up ? g.lineTo(x,y) : g.moveTo(x,y); up = true;
    }
    g.stroke(); g.setLineDash([]);
  };
  // periods where picks exist but the tomography produced no usable map: nothing for the
  // Vs inversion to load there, however dense the picks look on the left
  if (mper){
    g.fillStyle = "#ff6b6b38";
    for (let j=0;j<c.nt;j++){
      if (mper[j] != null) continue;
      const xa = X(c.te[j]), xb = X(c.te[j+1]);
      if (xb < r.x || xa > r.x+r.w) continue;
      g.fillRect(xa, r.y, Math.max(xb-xa, 1), r.h);
    }
    // label each contiguous run of map-less rungs, so an empty column is never read as
    // "the tomography found nothing here" when in fact no map was ever produced
    for (let j=0;j<c.nt;j++){
      if (mper[j] != null) continue;
      let e = j; while (e+1 < c.nt && mper[e+1] == null) e++;
      const xa = Math.max(X(c.te[j]), r.x), xb = Math.min(X(c.te[e+1]), r.x+r.w);
      if (xb - xa > 46){
        g.fillStyle = "#ffb0b0"; g.font = "10.5px Helvetica"; g.textAlign = "center";
        g.fillText("no map", (xa+xb)/2, r.y+14);
        g.fillStyle = "#ff6b6b38";
      }
      j = e;
    }
  }
  if (other) line(other, "#5ec8ff", [5,3]);
  if (med)   line(med,   "#39ff88", null);
  // excluded period bands
  const d = S.dec[dkey()] || {};
  if (d.lo != null && X(d.lo) > r.x) excluded(g, r, r.x, Math.min(X(d.lo), r.x+r.w));
  if (d.hi != null && X(d.hi) < r.x+r.w) excluded(g, r, Math.max(X(d.hi), r.x), r.x+r.w);
  g.restore();
  axes(g, r, "period [s]  (CWT scale rungs)", (isPicks?"pick":"cell")+" velocity [km/s]");
  handles(g, r, X);
}

/* A discarded period band: dimmed and hatched, so "excluded" reads instantly against a
   colormap that is itself mostly dark. Caller must already have clipped to the plot rect. */
function excluded(g, r, xa, xb){
  if (xb <= xa) return;
  g.fillStyle = "#0a0b10e0";
  g.fillRect(xa, r.y, xb-xa, r.h);
  g.save();
  g.beginPath(); g.rect(xa, r.y, xb-xa, r.h); g.clip();
  g.strokeStyle = "#ffb45426"; g.lineWidth = 1;
  for (let x = xa - r.h; x < xb + r.h; x += 9){
    g.beginPath(); g.moveTo(x, r.y+r.h); g.lineTo(x + r.h, r.y); g.stroke();
  }
  g.restore();
}

function handles(g, r, X){
  const d = S.dec[dkey()] || {};
  for (const [side, t] of [["lo", d.lo], ["hi", d.hi]]){
    if (t == null) continue;
    const x = X(t);
    if (x < r.x-8 || x > r.x+r.w+8) continue;
    g.strokeStyle = "#ffb454"; g.lineWidth = 2;
    g.beginPath(); g.moveTo(x, r.y); g.lineTo(x, r.y+r.h); g.stroke();
    g.fillStyle = "#ffb454";
    g.beginPath();
    g.moveTo(x, r.y+r.h); g.lineTo(x-6, r.y+r.h+9); g.lineTo(x+6, r.y+r.h+9);
    g.closePath(); g.fill();
    g.fillStyle = "#06121a"; g.font = "bold 9px Helvetica"; g.textAlign = "center";
    g.fillText(side === "lo" ? "▶" : "◀", x, r.y+r.h+8);
  }
}

/* ---------- diagnostics strip: 4 mini-plots on the shared period axis ---------- */
const METRICS = [
  {k:"nray", lab:"rays / period map", log:true,  col:"#ff6b6b", ref:300},
  {k:"vred", lab:"var_red",           log:false, col:"#5ec8ff", ref:0},
  {k:"amp",  lab:"anomaly amp [%]",   log:false, col:"#39ff88", ref:3},
  {k:"eta",  lab:"eta² geology",      log:false, col:"#c49bff", ref:null},
];
function drawStrip(){
  const cv = $("cS"), c = cur();
  const g = sizeCanvas(cv, STRIP_H);
  g.clearRect(0,0,cv.width,cv.height);
  if (!c) return;
  const cd = c.cells[S.cd] ? S.cd : Object.keys(c.cells)[0];
  const src = c.cells[cd], V = S.view;
  const hh = (STRIP_H - M.t - M.b) / METRICS.length;
  const X = t => M.l + (t-V.t0)/(V.t1-V.t0)*(cv.width/DPR - M.l - M.r);
  const w = cv.width/DPR - M.l - M.r;
  METRICS.forEach((mt, i) => {
    const r = {x:M.l, y:M.t + i*hh, w:w, h:hh-8};
    const raw = src[mt.k] || [];
    const vals = raw.filter(v => v != null && (!mt.log || v > 0));
    let lo = vals.length ? Math.min(...vals) : 0, hi = vals.length ? Math.max(...vals) : 1;
    if (mt.ref != null){ lo = Math.min(lo, mt.ref); hi = Math.max(hi, mt.ref); }
    if (mt.log){ lo = Math.log10(Math.max(lo, 1)); hi = Math.log10(Math.max(hi, 10)); }
    if (hi - lo < 1e-9) hi = lo + 1;
    const pad = (hi-lo)*0.12; lo -= pad; hi += pad;
    const Y = v => { const u = mt.log ? Math.log10(Math.max(v,1)) : v;
                     return r.y + r.h - (u-lo)/(hi-lo)*r.h; };
    g.strokeStyle = "#3a3f50"; g.strokeRect(r.x+.5, r.y+.5, r.w, r.h);
    g.save(); g.beginPath(); g.rect(r.x, r.y, r.w, r.h); g.clip();
    // excluded bands, same as the panels
    const d = S.dec[dkey()] || {};
    if (d.lo != null) excluded(g, r, r.x, Math.min(X(d.lo), r.x+r.w));
    if (d.hi != null) excluded(g, r, Math.max(X(d.hi), r.x), r.x+r.w);
    if (mt.ref != null){
      const y = Y(mt.ref);
      g.strokeStyle = "#6b7185"; g.setLineDash([4,3]);
      g.beginPath(); g.moveTo(r.x,y); g.lineTo(r.x+r.w,y); g.stroke(); g.setLineDash([]);
    }
    g.strokeStyle = mt.col; g.fillStyle = mt.col; g.lineWidth = 1.6;
    g.beginPath(); let up = false;
    for (let j=0;j<c.nt;j++){
      const v = raw[j];
      if (v == null || (mt.log && v <= 0)){ up = false; continue; }
      const x = X(c.tc[j]), y = Y(v);
      up ? g.lineTo(x,y) : g.moveTo(x,y); up = true;
    }
    g.stroke();
    for (let j=0;j<c.nt;j++){
      const v = raw[j];
      if (v == null || (mt.log && v <= 0)) continue;
      const bad = (mt.k==="vred" && v<=0) || (mt.k==="nray" && v<300) || (mt.k==="amp" && v<3);
      g.fillStyle = bad ? "#ff6b6b" : mt.col;
      g.beginPath(); g.arc(X(c.tc[j]), Y(v), bad?3:2, 0, 7); g.fill();
    }
    g.restore();
    g.fillStyle = "#9aa0b4"; g.font = "10.5px Helvetica"; g.textAlign = "left";
    g.fillText(mt.lab + (mt.log ? "  (log)" : ""), r.x+6, r.y+12);
    g.textAlign = "right";
    g.fillText(fmt(mt.log?Math.pow(10,hi):hi), r.x-4, r.y+9);
    g.fillText(fmt(mt.log?Math.pow(10,lo):lo), r.x-4, r.y+r.h);
  });
  const r = {x:M.l, y:M.t, w:w, h:STRIP_H-M.t-M.b};
  g.fillStyle = "#9aa0b4"; g.font = "11px Helvetica";
  for (const t of ticks(V.t0, V.t1)){
    const x = X(t);
    if (x < r.x-1 || x > r.x+r.w+1) continue;
    g.textAlign = "center"; g.fillText(fmt(t), x, STRIP_H-M.b+18);
  }
  g.fillText("period [s]   —   diagnostics for Cd = "+cd, r.x+r.w/2, STRIP_H-M.b+32);
}

/* ---------- redraw everything ---------- */
function draw(){
  ensureView();
  const c = cur();
  if (!c){ $("stat").textContent = "no data for this combination"; return; }
  const cd = c.cells[S.cd] ? S.cd : Object.keys(c.cells)[0];
  const cc = c.cells[cd];
  $("capL").textContent = "("+S.k+", "+S.meas+")";
  $("capR").textContent = "(Cd = "+cd+(cd===c.cd_rec?", recommended":"")+")";
  $("nL").textContent = (c.picks.total).toLocaleString()+" rows, "
                      + c.rungs.length+" rungs "+c.rungs[0].toFixed(2)+"–"
                      + c.rungs[c.rungs.length-1].toFixed(2)+" s";
  const nm = cc.n.filter(v=>v>0), nmap = cc.mper.filter(v=>v!=null).length;
  $("nR").innerHTML = (nm.length ? Math.round(nm.reduce((a,b)=>a+b,0)/nm.length).toLocaleString()+" cells/period · " : "")
    + (nmap < c.rungs.length ? '<span style="color:#ff6b6b">'+nmap+" maps only</span>" : nmap+" maps");
  drawPanel($("cL"), c, c.picks, c.picks.med, cc.med,      true);
  drawPanel($("cR"), c, cc,      cc.med,      c.picks.med, false, cc.mper);
  drawStrip();
  syncInputs();
  drawTable();
}

/* ---------- bounds ---------- */
/* Snap to the MAP periods, not the pick rungs. restrict_periods() filters the curve with
   T >= tmin / T <= tmax against the value stored in the map npz, which is a rounded version
   of the rung (rung 0.2016 -> map 0.20, rung 1.0763 -> map 1.08). A bound sitting on the rung
   would silently drop the very period the user meant to keep. */
function snap(c, t){
  let best = c.tsnap[0], bd = 1e9;
  for (const r of c.tsnap){ const d = Math.abs(r-t); if (d < bd){ bd = d; best = r; } }
  return best;
}
const keptPeriods = (c, d) =>
  c.tsnap.filter(t => (d.lo==null||t>=d.lo-1e-9) && (d.hi==null||t<=d.hi+1e-9));
function setBound(side, t, doSnap){
  const c = cur(); if (!c) return;
  const d = S.dec[dkey()] || (S.dec[dkey()] = {lo:null, hi:null, reason:""});
  d[side] = t == null ? null : (doSnap ? snap(c, t) : +t);
  if (d.lo != null && d.hi != null && d.lo > d.hi){ const x = d.lo; d.lo = d.hi; d.hi = x; }
  saveDec(); draw();
}
function syncInputs(){
  const c = cur(), d = S.dec[dkey()] || {};
  $("tlo").value = d.lo != null ? d.lo : "";
  $("thi").value = d.hi != null ? d.hi : "";
  $("why").value = d.reason || "";
  const kept = c ? keptPeriods(c, d) : [];
  const cd = c && (c.cells[S.cd] || c.cells[Object.keys(c.cells)[0]]);
  let bad = 0, thin = 0;
  if (c && cd) for (let j=0;j<c.nt;j++){
    const t = cd.mper[j];
    if (t == null) continue;                       // rung with no map: nothing to keep
    if ((d.lo!=null && t<d.lo-1e-9) || (d.hi!=null && t>d.hi+1e-9)) continue;
    if (cd.vred[j] != null && cd.vred[j] <= 0) bad++;
    if (cd.nray[j] != null && cd.nray[j] < 300) thin++;
  }
  $("stat").innerHTML =
    '<span class="pill">'+kept.length+" / "+(c?c.tsnap.length:0)+" period maps kept</span> " +
    (bad ? '<span class="pill bad">'+bad+" with var_red ≤ 0</span> " : "") +
    (thin ? '<span class="pill warn">'+thin+" with &lt;300 rays</span> " : "") +
    (!bad && !thin && kept.length ? '<span class="pill ok">no flagged period in range</span>' : "");
}

/* ---------- summary table of all decisions ---------- */
function drawTable(){
  let h = "<tr><th>net</th><th>measure</th><th>wave</th><th class='num'>available</th>"
        + "<th class='num'>T min</th><th class='num'>T max</th><th class='num'>kept</th><th>reason</th></tr>";
  for (const net of Object.keys(D.avail))
    for (const m of Object.keys(D.avail[net]))
      for (const w of Object.keys(D.avail[net][m])){
        const c = D.combos[net+"|"+m+"|"+w+"|"+D.avail[net][m][w][0]];
        const d = S.dec[net+"|"+m+"|"+w] || {};
        const kept = keptPeriods(c, d);
        const set = (d.lo != null || d.hi != null);
        const sel = (net===S.net && m===S.meas && w===S.wave);
        h += "<tr"+(sel?' style="background:#252836"':'')+"><td>"+net+"</td><td>"+m+"</td><td>"+w+"</td>"
           + "<td class='num'>"+c.tsnap[0].toFixed(2)+"–"+c.tsnap[c.tsnap.length-1].toFixed(2)+"</td>"
           + "<td class='num"+(set?" done":"")+"'>"+(d.lo!=null?d.lo.toFixed(2):"—")+"</td>"
           + "<td class='num"+(set?" done":"")+"'>"+(d.hi!=null?d.hi.toFixed(2):"—")+"</td>"
           + "<td class='num'>"+(set?kept.length:"—")+"</td><td>"+(d.reason||"")+"</td></tr>";
      }
  $("tbl").innerHTML = h;
}

/* ---------- interaction ---------- */
function px2t(cv, ev){
  const b = cv.getBoundingClientRect(), r = rect(cv), V = S.view;
  return V.t0 + (ev.clientX - b.left - r.x)/r.w*(V.t1-V.t0);
}
function px2v(cv, ev){
  const b = cv.getBoundingClientRect(), r = rect(cv), V = S.view;
  return V.v0 + (r.y + r.h - (ev.clientY - b.top))/r.h*(V.v1-V.v0);
}
let SPACE = false;
addEventListener("keydown", e => { if (e.code === "Space"){ SPACE = true; e.preventDefault(); } });
addEventListener("keyup",   e => { if (e.code === "Space") SPACE = false; });

for (const id of ["cL","cR","cS"]){
  const cv = $(id), isStrip = (id === "cS");
  cv.addEventListener("wheel", ev => {
    ev.preventDefault();
    const f = Math.exp(ev.deltaY * 0.0016), V = S.view;
    if (!ev.altKey){
      const t = px2t(cv, ev);
      S.view.t0 = t - (t-V.t0)*f; S.view.t1 = t + (V.t1-t)*f;
    }
    if (!ev.shiftKey && !isStrip){
      const v = px2v(cv, ev);
      S.view.v0 = v - (v-V.v0)*f; S.view.v1 = v + (V.v1-v)*f;
    }
    draw();
  }, {passive:false});

  cv.addEventListener("mousedown", ev => {
    const c = cur(); if (!c) return;
    const r = rect(cv), b = cv.getBoundingClientRect();
    const t = px2t(cv, ev), d = S.dec[dkey()] || {};
    const X = tt => r.x + (tt-S.view.t0)/(S.view.t1-S.view.t0)*r.w;
    const mx = ev.clientX - b.left;
    if (ev.button === 0 && !SPACE){
      // grab a handle if the cursor is near it, else start a new range by dragging
      for (const side of ["lo","hi"])
        if (d[side] != null && Math.abs(X(d[side]) - mx) < 7){
          S.drag = {mode:"handle", side, cv}; return;
        }
      S.drag = {mode:"new", t0:t, cv};
      setBound("lo", t, true); setBound("hi", t, true);
      return;
    }
    S.drag = {mode:"pan", cv, x:ev.clientX, y:ev.clientY, V:Object.assign({}, S.view), isStrip};
    ev.preventDefault();
  });
  cv.addEventListener("contextmenu", e => e.preventDefault());
  cv.addEventListener("dblclick", () => { S.view = fullView(cur()); draw(); });

  cv.addEventListener("mousemove", ev => {
    if (S.drag && S.drag.cv === cv){
      const t = px2t(cv, ev);
      if (S.drag.mode === "handle") setBound(S.drag.side, t, true);
      else if (S.drag.mode === "new"){
        setBound(t < S.drag.t0 ? "lo" : "hi", t, true);
        setBound(t < S.drag.t0 ? "hi" : "lo", S.drag.t0, true);
      } else {
        const r = rect(cv), V = S.drag.V;
        const dt = (ev.clientX - S.drag.x)/r.w*(V.t1-V.t0);
        S.view.t0 = V.t0 - dt; S.view.t1 = V.t1 - dt;
        if (!S.drag.isStrip){
          const dv = (ev.clientY - S.drag.y)/r.h*(V.v1-V.v0);
          S.view.v0 = V.v0 + dv; S.view.v1 = V.v1 + dv;
        }
        draw();
      }
      return;
    }
    tooltip(cv, ev, id);
  });
  cv.addEventListener("mouseleave", () => { $("tip").style.display = "none"; });
}
addEventListener("mouseup", () => { S.drag = null; });

function tooltip(cv, ev, id){
  const c = cur(); if (!c) return;
  const tip = $("tip"), t = px2t(cv, ev);
  let j = -1;
  for (let i=0;i<c.nt;i++) if (t >= c.te[i] && t < c.te[i+1]){ j = i; break; }
  if (j < 0){ tip.style.display = "none"; return; }
  const cd = c.cells[S.cd] ? S.cd : Object.keys(c.cells)[0], cc = c.cells[cd];
  const L = ["rung "+(j+1)+"/"+c.nt+"   pick T = "+c.tc[j].toFixed(3)+" s",
             "map T       "+(cc.mper[j] != null ? cc.mper[j]+" s   ← bounds snap to this"
                                                : "no map at this period")];
  if (id !== "cS") L.push("V = "+px2v(cv, ev).toFixed(2)+" km/s");
  L.push("picks       "+(c.picks.n[j]||0).toLocaleString()+"   median "+(c.picks.med[j]??"—"));
  L.push("cells       "+(cc.n[j]||0).toLocaleString()+"   median "+(cc.med[j]??"—"));
  L.push("rays        "+(cc.nray[j]??"—")+(cc.nray[j]!=null&&cc.nray[j]<300?"  ← thin":""));
  L.push("var_red     "+(cc.vred[j]??"—")+(cc.vred[j]!=null&&cc.vred[j]<=0?"  ← unusable":""));
  L.push("amp / eta²  "+(cc.amp[j]??"—")+" %  /  "+(cc.eta[j]??"—"));
  if (c.picks.med[j] != null)
    L.push("λ ≈ "+(c.picks.med[j]*c.tc[j]).toFixed(2)+" km   λ/3 ≈ "
           +(c.picks.med[j]*c.tc[j]/3).toFixed(2)+" km");
  tip.textContent = L.join("\n");
  tip.style.display = "block";
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.min(ev.clientX+14, innerWidth-w-8)+"px";
  tip.style.top  = Math.min(ev.clientY+14, innerHeight-h-8)+"px";
}

/* ---------- controls ---------- */
function fillSelect(el, vals, val){
  el.innerHTML = vals.map(v => "<option"+(v===val?" selected":"")+">"+v+"</option>").join("");
}
function btns(el, vals, val, cb){
  el.innerHTML = "";
  vals.forEach(v => {
    const b = document.createElement("button");
    b.textContent = v; if (v === val) b.className = "on";
    b.onclick = () => cb(v);
    el.appendChild(b);
  });
}
function refreshControls(){
  const nets = Object.keys(D.avail);
  if (!nets.includes(S.net)) S.net = nets[0];
  const ms = Object.keys(D.avail[S.net]);
  if (!ms.includes(S.meas)) S.meas = ms[0];
  const ws = Object.keys(D.avail[S.net][S.meas]);
  if (!ws.includes(S.wave)) S.wave = ws[0];
  const ks = D.avail[S.net][S.meas][S.wave];
  if (!ks.includes(S.k)) S.k = ks[0];
  fillSelect($("net"), nets, S.net);
  fillSelect($("meas"), ms, S.meas);
  fillSelect($("wave"), ws, S.wave);
  btns($("kbtns"), ks, S.k, v => { S.k = v; refreshControls(); });
  const c = cur(), cds = c ? Object.keys(c.cells) : [];
  if (!cds.includes(S.cd)) S.cd = (c && cds.includes(c.cd_rec)) ? c.cd_rec : cds[0];
  btns($("cdbtns"), cds, S.cd, v => { S.cd = v; draw(); });
  $("sub").textContent = "— " + (D.title[S.wave]||S.wave) + ", " + S.meas
                       + " velocity, dx " + D.dx[S.net] + " km, prod3 " + S.k;
  draw();
}
$("net").onchange  = e => { S.net = e.target.value;  S.view = null; refreshControls(); };
$("meas").onchange = e => { S.meas = e.target.value; refreshControls(); };
$("wave").onchange = e => { S.wave = e.target.value; S.view = null; refreshControls(); };
$("gam").oninput   = e => { S.gam = +e.target.value; draw(); };
$("reset").onclick = () => { S.view = fullView(cur()); draw(); };
$("fit").onclick   = () => {
  // fit velocity to the pick distribution inside the current period window
  const c = cur(), a = unb64(c.picks.q); let lo = 1e9, hi = -1e9;
  for (let j=0;j<c.nt;j++){
    if (c.tc[j] < S.view.t0 || c.tc[j] > S.view.t1) continue;
    for (let i=0;i<c.nv;i++) if (a[j*c.nv+i] > 8){
      const v = c.v0 + (i+0.5)*c.dv; lo = Math.min(lo,v); hi = Math.max(hi,v);
    }
  }
  if (hi > lo){ S.view.v0 = lo-0.15; S.view.v1 = hi+0.15; draw(); }
};
$("tlo").onchange = e => setBound("lo", e.target.value === "" ? null : +e.target.value, true);
$("thi").onchange = e => setBound("hi", e.target.value === "" ? null : +e.target.value, true);
$("why").onchange = e => {
  const d = S.dec[dkey()] || (S.dec[dkey()] = {lo:null,hi:null,reason:""});
  d.reason = e.target.value; saveDec(); drawTable();
};
$("clear").onclick = () => { delete S.dec[dkey()]; saveDec(); draw(); };
$("all").onclick = () => {
  const d = S.dec[dkey()]; if (!d) return;
  for (const w of Object.keys(D.avail[S.net][S.meas]))
    S.dec[S.net+"|"+S.meas+"|"+w] = Object.assign({}, d);
  saveDec(); draw();
};

/* ---------- export ---------- */
function csvText(){
  const head = ["net","measure","wave","T_available_min","T_available_max","n_periods",
                "T_valid_min","T_valid_max","reason"];
  const rows = [head.join(",")];
  for (const net of Object.keys(D.avail))
    for (const m of Object.keys(D.avail[net]))
      for (const w of Object.keys(D.avail[net][m])){
        const c = D.combos[net+"|"+m+"|"+w+"|"+D.avail[net][m][w][0]];
        const d = S.dec[net+"|"+m+"|"+w] || {};
        const q = s => /[",\n]/.test(s) ? '"'+s.replace(/"/g,'""')+'"' : s;
        // available range = the MAP periods of the recommended Cd, i.e. what the Vs
        // inversion can actually load; the pick tables run past the last map.
        rows.push([net, m, w, c.tsnap[0], c.tsnap[c.tsnap.length-1], c.tsnap.length,
                   d.lo != null ? d.lo : "", d.hi != null ? d.hi : "",
                   q(d.reason || "")].join(","));
      }
  return rows.join("\n") + "\n";
}
$("export").onclick = () => {
  $("csv").value = csvText(); $("dst").textContent = D.source; $("dlg").showModal();
};
$("dl").onclick = () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csvText()], {type:"text/csv"}));
  a.download = "period_ranges_DECISIONS.csv"; a.click();
};
$("copy").onclick = () => navigator.clipboard.writeText(csvText());
$("close").onclick = () => $("dlg").close();
addEventListener("resize", draw);
// redraw when the container is actually given a width (first paint in a collapsed pane)
if (window.ResizeObserver){
  let last = 0;
  new ResizeObserver(es => {
    const w = es[0].contentRect.width;
    if (w > 0 && Math.abs(w - last) > 2){ last = w; draw(); }
  }).observe($("cL").parentElement);
}
refreshControls();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
