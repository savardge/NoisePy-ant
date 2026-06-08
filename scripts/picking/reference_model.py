"""
Layered Vs model digitised from the well-log figure (red 'Vs model' layer-cake) and its
forward-modelled surface-wave dispersion (Rayleigh + Love, phase + group, fundamental + 1st
overtone) via disba. Used as the reference for picking and for overlay on the comparison images.

vs_model() returns the layered model; reference_curves() returns a dict keyed by
(wave, kind, mode) -> (periods[s], velocity[km/s]).  wave in {rayleigh,love}, kind in
{phase,group}, mode in {0,1}.
"""
import numpy as np
from disba import PhaseDispersion, GroupDispersion

# Digitised red curve: layer thickness [km] and Vs [km/s] (last = half-space).
# Tertiary (0-0.5 km) ~0.7-1.6 ; Mesozoic (0.5-1.65) ~2.0-2.3 ; Paleozoic (1.65-2.6) ~2.4-2.65 ;
# Basement transition ~3.0 ; basement half-space ~3.4 km/s.
_THICK = np.array([0.15, 0.20, 0.15, 0.20, 0.60, 0.35, 0.65, 0.30, 0.45, 5.00])
_VS = np.array([0.70, 1.00, 1.55, 2.00, 2.15, 2.30, 2.40, 2.65, 3.00, 3.40])


def vs_model():
    """Return (thickness[km], vp[km/s], vs[km/s], density[g/cm3]) using Brocher (2005)."""
    vs = _VS
    vp = 0.9409 + 2.0947 * vs - 0.8206 * vs**2 + 0.2683 * vs**3 - 0.0251 * vs**4
    rho = (1.6612 * vp - 0.4721 * vp**2 + 0.0671 * vp**3
           - 0.0043 * vp**4 + 0.000106 * vp**5)
    return _THICK, vp, vs, rho


def reference_curves(periods=None):
    """
    Forward-model the reference dispersion. Returns dict[(wave, kind, mode)] = (periods, vel).
    Modes that do not exist over (part of) the band simply return the sub-range disba found.
    """
    if periods is None:
        periods = np.geomspace(0.2, 12.0, 90)
    th, vp, vs, rho = vs_model()
    out = {}
    for wave in ("rayleigh", "love"):
        for kind, Disp in (("phase", PhaseDispersion), ("group", GroupDispersion)):
            disp = Disp(th, vp, vs, rho)
            for mode in (0, 1):
                try:
                    cp = disp(periods, mode=mode, wave=wave)
                    out[(wave, kind, mode)] = (np.asarray(cp.period), np.asarray(cp.velocity))
                except Exception:
                    out[(wave, kind, mode)] = (np.array([]), np.array([]))
    return out


if __name__ == "__main__":
    th, vp, vs, rho = vs_model()
    print("layered model (thick km, vp, vs, rho):")
    for t, a, b, r in zip(th, vp, vs, rho):
        print(f"  {t:5.2f}  {a:5.2f}  {b:5.2f}  {r:5.2f}")
    ref = reference_curves()
    print("\nreference dispersion (km/s) sampled:")
    for (wave, kind, mode), (p, v) in ref.items():
        if len(p):
            print(f"  {wave:8} {kind:5} mode{mode}:  T {p.min():.2f}-{p.max():.2f}s   "
                  f"v {v.min():.2f}-{v.max():.2f}  (n={len(p)})")
        else:
            print(f"  {wave:8} {kind:5} mode{mode}:  (none found)")
