#!/usr/bin/env python3
"""Build a local HTML index for browsing a network's velocity-map runs.

A production tree holds one directory per (measure, Cd model, lc scale) and, inside each,
three overview figures plus a data-vs-model comparison per wave -- 12 figures per run, and
several hundred per-period maps. Comparing Cd models means opening the same figure from
several directories at once, which Finder makes tedious.

This writes ONE self-contained page per network: rows are the run variants, columns the
figure types, thumbnails link to the full-size PNG, and each cell notes the per-period
gallery. Relative paths only, no external CSS or JS, so it opens straight from disk.

Usage:
  python make_map_index.py --production-root <.../1_velocity_maps/production> --net riehen
  open <.../production/index_<net>.html
"""
import argparse
import glob
import os
import re

WAVES = ("fund", "overtone", "love")
KINDS = (("maps_%s.png", "per-panel scale"),
         ("maps_%s_common.png", "common scale"),
         ("maps_%s_anomaly.png", "anomaly %"),
         ("vdist_%s.png", "picks vs model"))

CSS = """
body{font:13px/1.45 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;
     background:#fafafa;color:#222}
h1{font-size:20px;margin:0 0 4px} h2{font-size:16px;margin:28px 0 8px;
   border-bottom:2px solid #ddd;padding-bottom:4px}
.note{color:#666;margin:0 0 18px}
table{border-collapse:collapse;margin-bottom:8px;background:#fff}
th,td{border:1px solid #ddd;padding:6px;vertical-align:top;text-align:center}
th{background:#f0f0f0;font-weight:600;font-size:12px}
td.run{text-align:left;font:12px ui-monospace,Menlo,monospace;white-space:nowrap;
       background:#fbfbfb}
img{width:230px;height:auto;display:block;border:1px solid #eee}
a{color:#06c;text-decoration:none} a:hover{text-decoration:underline}
.miss{color:#bbb;font-style:italic}
.gal{font-size:11px;color:#666}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--production-root", required=True)
    ap.add_argument("--net", required=True)
    a = ap.parse_args()

    root = os.path.abspath(a.production_root)
    runs = sorted(d for d in glob.glob(os.path.join(root, "tspws_*")) if os.path.isdir(d))
    if not runs:
        raise SystemExit("no tspws_* run directories under %s" % root)

    h = ["<!doctype html><meta charset=utf-8><title>%s velocity maps</title><style>%s</style>"
         % (a.net, CSS),
         "<h1>%s &mdash; ts-PWS velocity map runs</h1>" % a.net,
         "<p class=note>%d run variants. Thumbnails link to the full-size PNG; each row's "
         "per-period maps are in the same folder as <code>map_T*.png</code>.</p>" % len(runs)]

    for wave in WAVES:
        h.append("<h2>%s</h2><table><tr><th>run</th>%s</tr>"
                 % (wave, "".join("<th>%s</th>" % lab for _, lab in KINDS)))
        for run in runs:
            rel = os.path.relpath(run, root)
            npz = len(glob.glob(os.path.join(run, "production", wave, "map_T*.npz")))
            png = len(glob.glob(os.path.join(run, "figures", wave, "map_T*.png")))
            h.append("<tr><td class=run>%s<br><span class=gal>%d maps, %d per-period png"
                     "</span></td>" % (rel, npz, png))
            for pat, _ in KINDS:
                f = os.path.join(run, "figures", wave, pat % wave)
                if os.path.exists(f):
                    r = os.path.relpath(f, root)
                    h.append('<td><a href="%s"><img src="%s" loading="lazy"></a></td>' % (r, r))
                else:
                    h.append('<td class=miss>&mdash;</td>')
            h.append("</tr>")
        h.append("</table>")

    out = os.path.join(root, "index_%s.html" % a.net)
    open(out, "w").write("\n".join(h))
    print("wrote %s  (%d runs x %d waves)" % (out, len(runs), len(WAVES)))


if __name__ == "__main__":
    main()
