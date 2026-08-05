#!/usr/bin/env python3
"""Regenerate tomo/1_velocity_maps/README.md for each network from a shared template.

Lives in the repo on purpose: the previous generator sat in a session scratchpad and was
lost twice when the scratchpad was cleared, leaving the READMEs regenerable only by hand.
The template is `vmaps_readme_template.md` beside this file; it carries the workflow layout,
the Cd-by-measure recommendation, the evaluation caveats, the supporting-evidence index and
the bug ledger, with @@NET@@, @@DX@@ and @@V@@ substituted per network. Token substitution rather
than str.format: the markdown contains literal brace globs like {k2,k3}.

The 2_superseded/ and 0_inputs/ READMEs are NOT written here -- they are static and already
in place; this only owns the top-level one.

Usage:
  python write_vmaps_readme.py [--projects .../Projects] [--dry-run]
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "vmaps_readme_template.md")
NETS = {"riehen": "0.2", "aargau": "0.5", "hautesorne": "0.5"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--projects",
                    default="/Users/genevievesavard/Codes/extract_higher_modes/Projects")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    tpl = open(TEMPLATE).read()
    for net, dx in NETS.items():
        V = os.path.join(a.projects, net, "tomo", "1_velocity_maps")
        if not os.path.isdir(V):
            print("skip %s" % net)
            continue
        txt = (tpl.replace("@@NET@@", net).replace("@@DX@@", dx)
                  .replace("@@V@@", V))
        p = os.path.join(V, "README.md")
        if a.dry_run:
            cur = open(p).read() if os.path.exists(p) else ""
            print("%-11s %s" % (net, "unchanged" if cur == txt else "WOULD CHANGE"))
            continue
        open(p, "w").write(txt)
        print("wrote %s" % p)


if __name__ == "__main__":
    main()
