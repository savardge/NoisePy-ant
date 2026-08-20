"""
GEOPSY / DINVER TARGET BUILDING -- the ONE place NoisePy-ant units meet Geopsy units.

The pipeline works in (period s, velocity km/s, sigma km/s); Geopsy's .target wants
(frequency Hz, slowness s/m, log-slowness stddev). Every conversion lives here so a factor of
1000 or a slowness inversion can only be wrong in one file, and the round-trip test at the
bottom pins it down.

Why not swprepost.TargetSet.to_file? Because swprepost hard-codes `<slowness>Phase</slowness>`
per ModalCurve (targetset.py) and Dinver DOES accept `Group` there -- QGpCoreWave's
DispersionFactory dispatches Mode::Group to a group-slowness forward (DispersionFactory.cpp,
Mode.h `enum Slowness {Phase=0, Group=1}`). Our targets are GROUP velocity from the FTAN
picks, so we need the group tag. `write_target` below re-emits swprepost's own 3.4.2 XML block
verbatim except that one tag per curve, and reuses its container recipe (gzip tar of a
UTF-16-LE `contents.xml` with BOM), so what Dinver receives is byte-for-byte swprepost's format.

Depends on swprepost (~/Codes/swprepost on PYTHONPATH, or pip-installed): ModalTarget gives
setmincov / easy_resample / slowness / logstd exactly as Vantassel & Cox (2021) define them.
"""
from __future__ import annotations
import io
import tarfile
import warnings

import numpy as np

# wave key -> (Geopsy polarization, mode index).  Mirrors vs_inversion.WAVEDEF; the base wave
# key ("fund", "overtone", "love", "love_ot") is what CellData / curve_key use.
POLARIZATION = {"fund": ("Rayleigh", 0), "overtone": ("Rayleigh", 1),
                "love": ("Love", 0), "love_ot": ("Love", 1)}
KM = 1000.0                          # km -> m


def curve_to_hz_ms(T, V, S):
    """(period s, velocity km/s, sigma km/s) -> (frequency Hz, velocity m/s, sigma m/s).

    Sorted by frequency ascending (ModalTarget re-sorts anyway, but keep the arrays paired)."""
    T = np.asarray(T, float); V = np.asarray(V, float); S = np.asarray(S, float)
    f = 1.0 / T
    o = np.argsort(f)
    return f[o], V[o] * KM, S[o] * KM


def curve_from_hz_ms(f, v_ms, s_ms):
    """Inverse of curve_to_hz_ms -> (period s, velocity km/s, sigma km/s), period ascending."""
    f = np.asarray(f, float)
    T = 1.0 / f
    o = np.argsort(T)
    return T[o], np.asarray(v_ms, float)[o] / KM, np.asarray(s_ms, float)[o] / KM


def to_modal_target(T, V, S, wave, min_cov=0.05, n_resample=30, wavelength_range=None):
    """Build a swprepost.ModalTarget (Hz, m/s) from a pipeline curve, SWinvert-conditioned.

    wave      : base wave key -> polarization + mode via POLARIZATION.
    min_cov   : per-point COV floor (Vantassel & Cox 2021 recommend 0.05-0.10 where site
                uncertainty is unquantified; our tomographic sigma exists but underestimates,
                so the floor is applied on top of it, not instead of it). None = no floor.
    n_resample: resample count (paper: 20-30+, log-WAVELENGTH preferred, log-frequency second,
                linear-frequency discouraged). None/0 = keep the raw points.
    wavelength_range : (lmin_m, lmax_m) to resample over; default the curve's own extent.
                Ignored when the curve falls back to the frequency domain (see below).

    Domain choice -- READ THIS. swprepost's wavelength-domain resample interpolates velocity
    against lambda = V*T and assumes lambda is monotone in T. That holds for well-behaved
    PHASE curves, but NOT for group curves through an Airy-phase minimum (Love group at
    Riehen: 3 reversals) nor for wiggly phase curves (Riehen fund_phase: 3 reversals). On a
    non-monotone lambda the interpolation is garbage -- measured: frequencies scrambled and
    even NEGATIVE for Love group, and up to 0.46 km/s (2-3 sigma) of spurious velocity on
    fund_phase, which silently poisoned every phase/joint run made before 2026-08-18. So:
    log-wavelength only when lambda(T) is strictly monotone; otherwise log-FREQUENCY over the
    curve's own band (the paper's second choice), with a warning. The output is validated
    (finite, positive, strictly ordered) and refused otherwise -- never trust it silently.
    """
    import swprepost
    pol, mode = POLARIZATION[wave]
    f, v, s = curve_to_hz_ms(T, V, S)
    tgt = swprepost.ModalTarget(f, v, s, description=((pol.lower(), mode),))
    if min_cov is not None and min_cov > 0:
        tgt.setmincov(float(min_cov))
    if n_resample:
        n = int(min(n_resample, len(f)))          # never invent more points than we measured
        # ModalTarget sorts by frequency; lambda must then be strictly DEcreasing.
        lam = np.asarray(tgt.wavelength, float)

        def _valid(t):
            fq, vv = np.asarray(t.frequency, float), np.asarray(t.velocity, float)
            return (np.all(np.isfinite(fq)) and np.all(np.isfinite(vv)) and np.all(fq > 0)
                    and np.all(vv > 0) and (np.all(np.diff(fq) > 0) or np.all(np.diff(fq) < 0)))

        import copy
        done = False
        if np.all(np.diff(lam) < 0):
            cand = copy.deepcopy(tgt)
            if wavelength_range is None:
                lmin, lmax = float(lam.min()), float(lam.max())
            else:
                lmin, lmax = map(float, wavelength_range)
            cand.easy_resample(pmin=lmin, pmax=lmax, pn=n, res_type="log",
                               domain="wavelength", inplace=True)
            # Even with lambda monotone in T, interpolating V on a log-lambda grid and
            # recomputing f = V/lambda is NOT order-preserving where the curve is steep
            # (riehen overtone group at (18,39): last two f reversed). Validate, else fall back.
            if _valid(cand):
                tgt = cand; done = True
            else:
                warnings.warn("%s: log-wavelength resample gave unordered frequencies -- "
                              "resampling in log-FREQUENCY instead" % wave)
        else:
            warnings.warn("%s: lambda=V*T not monotone in T (%d reversals) -- resampling in "
                          "log-FREQUENCY, not log-wavelength"
                          % (wave, int((np.diff(lam) >= 0).sum())))
        if not done:
            fq = np.asarray(tgt.frequency, float)
            tgt.easy_resample(pmin=float(fq.min()), pmax=float(fq.max()), pn=n, res_type="log",
                              domain="frequency", inplace=True)
        if not _valid(tgt):
            raise ValueError("%s: resampled target is invalid (non-finite / non-positive / "
                             "unordered frequencies) -- refusing to write it" % wave)
    return tgt


def wavelength_range(cell, prefer="fund_phase", fallback="fund"):
    """(lambda_min, lambda_max) in METRES that size the SWinvert layering (lmin/3, lmax/df).

    Uses the fundamental PHASE curve when the cell has one -- the LN/LR rules are defined on
    phase wavelength. Falls back to group U*T with a warning: that is not a true wavelength,
    so dmax comes out ~30% shallower than the phase-based value on the same cell.

    The cell must ALREADY be period-restricted (vs_inversion.restrict_periods); sizing off the
    unvalidated range inflates lmax and pushes dmax past what the data resolve. There is no way
    to detect that here, so callers are responsible -- run_dinver_cell.py records the range it
    used in the npz for exactly this reason.
    """
    for key, note in ((prefer, None), (fallback, "no %s curve; layering sized from GROUP U*T"
                                                  " (not a true wavelength)" % prefer)):
        if key in cell.curves and len(cell.curves[key][0]) > 1:
            T, U, _ = cell.curves[key]
            lam = np.asarray(U, float) * np.asarray(T, float) * KM
            if note:
                warnings.warn(note)
            return float(lam.min()), float(lam.max()), key
    raise ValueError("wavelength_range: cell has neither %r nor %r" % (prefer, fallback))


# --------------------------------------------------------------------- .target writer
# GROUP-SLOWNESS GRID PADDING. Dinver evaluates the forward ONLY at the union of the target
# curves' x values (ModalFactory::setX), and derives group slowness from the phase curve by a
# central finite difference ACROSS THAT GRID (Dispersion::setGroupSlowness), invalidating the
# first and last grid points. Measured on a 3-layer synthetic: gpdc group at 14 log-spaced
# points is off by up to 3.7% vs disba / vs gpdc at 400 points (which agree to 0.05%) -- the
# size of our data COV -- and a group-only target loses its two extreme periods outright,
# which also trips the misfit's x(1+nData-nValues) penalty. A data point flagged
# <valid>false</valid> is skipped by the misfit (StatisticalValue::misfit: neither nData nor
# nValues) but its x STILL enters the forward grid, so padding each Group curve with dense
# invalid points fixes all three effects without touching what is fitted.
GROUP_PAD_PER_DECADE = 40        # ~gpdc -n 60 over 1.4 decades matched disba to ~0.2%
GROUP_PAD_MARGIN = 0.15          # extend the grid this fraction beyond fmin/fmax in log-f


def _pad_frequencies(f, per_decade=GROUP_PAD_PER_DECADE, margin=GROUP_PAD_MARGIN, min_sep=0.02):
    """Dense log-f grid spanning [fmin/(1+m), fmax*(1+m)], minus points within min_sep of data."""
    f = np.asarray(f, float)
    lo, hi = f.min() / (1 + margin), f.max() * (1 + margin)
    n = max(3, int(np.ceil(np.log10(hi / lo) * per_decade)) + 1)
    grid = np.geomspace(lo, hi, n)
    keep = np.array([np.min(np.abs(np.log(g / f))) > min_sep for g in grid])
    return grid[keep]


def _modalcurve_xml(target, slowness, name, pad_per_decade=None):
    """One <ModalCurve> block, swprepost 3.4.2 layout, with our slowness tag (+ grid padding)."""
    if slowness not in ("Group", "Phase"):
        raise ValueError("slowness must be 'Group' or 'Phase', got %r" % (slowness,))
    target._sort_data()
    out = ["      <ModalCurve>",
           f"        <name>{name}</name>",
           "        <log>noisepy.dinver_target (swprepost 3.4.2 layout)</log>",
           "        <enabled>true</enabled>"]
    for pol, mode in target.description:
        out += ["        <Mode>",
                "          <value>Signed</value>",
                f"          <slowness>{slowness}</slowness>",
                f"          <polarization>{pol.capitalize()}</polarization>",
                "          <ringIndex>0</ringIndex>",
                f"          <index>{mode}</index>",
                "        </Mode>"]
    pts = [(x, m, s, True) for x, m, s in zip(target.frequency, target.slowness, target.logstd)]
    if pad_per_decade:
        f = target.frequency; ls = np.log(target.slowness)
        std_pad = float(np.median(target.logstd))
        for x in _pad_frequencies(f, pad_per_decade):
            # any positive mean is fine for an invalid point; use the log-log interpolant so the
            # file still reads sensibly in the Dinver GUI
            m = float(np.exp(np.interp(np.log(x), np.log(f), ls)))
            pts.append((x, m, std_pad, False))
    pts.sort(key=lambda p: p[0])
    for x, mean, std, valid in pts:
        out += ["        <RealStatisticalPoint>",
                f"          <x>{x}</x>",
                f"          <mean>{mean}</mean>",
                f"          <stddev>{std}</stddev>",
                "          <weight>1</weight>",
                f"          <valid>{'true' if valid else 'false'}</valid>",
                "        </RealStatisticalPoint>"]
    out.append("      </ModalCurve>")
    return out


def target_xml(entries, misfit_weight=1, group_pad_per_decade=GROUP_PAD_PER_DECADE):
    """contents.xml text for a dispersion-only .target.

    entries : list of (swprepost.ModalTarget, "Group"|"Phase", name)
    group_pad_per_decade : invalid grid-padding density for Group curves (see above); 0/None
        disables it (only for experiments -- it re-introduces a few-% forward bias).
    The non-dispersion targets (autocorr, ellipticity, refraction, MT) are emitted disabled
    exactly as swprepost does, so Dinver's TargetList reads a complete document.
    """
    if not entries:
        raise ValueError("target_xml: no curves")
    c = ["<Dinver>",
         "  <pluginTag>DispersionCurve</pluginTag>",
         "  <pluginTitle>Surface Wave Inversion</pluginTitle>",
         "  <TargetList>",
         "    <position>0 0 0</position>",
         "    <DispersionTarget type=\"dispersion\">",
         "      <selected>true</selected>",
         f"      <misfitWeight>{misfit_weight}</misfitWeight>",
         "      <minimumMisfit>0</minimumMisfit>",
         "      <misfitType>L2_LogNormalized</misfitType>"]
    for tgt, slowness, name in entries:
        c += _modalcurve_xml(tgt, slowness, name,
                             pad_per_decade=(group_pad_per_decade if slowness == "Group" else None))
    c += ["    </DispersionTarget>",
          "    <AutocorrTarget>",
          "      <selected>false</selected>",
          "      <misfitWeight>1</misfitWeight>",
          "      <minimumMisfit>0</minimumMisfit>",
          "      <misfitType>L2_NormalizedBySigmaOnly</misfitType>",
          "      <AutocorrCurves>",
          "      </AutocorrCurves>",
          "    </AutocorrTarget>",
          "    <ModalCurveTarget type=\"ellipticity\">",
          "      <selected>false</selected>",
          "      <misfitWeight>1</misfitWeight>",
          "      <minimumMisfit>0</minimumMisfit>",
          "      <misfitType>L2_Normalized</misfitType>",
          "    </ModalCurveTarget>",
          "    <EllipticityPeakTarget type=\"ellipticity peak\">",
          "      <minimumAmplitude>0</minimumAmplitude>",
          "      <RealStatisticalValue>",
          "        <mean>0</mean>",
          "        <stddev>0</stddev>",
          "        <weight>1</weight>",
          "        <valid>false</valid>",
          "      </RealStatisticalValue>",
          "    </EllipticityPeakTarget>",
          "    <RefractionTarget type=\"Vp\">",
          "      <selected>false</selected>",
          "      <misfitWeight>1</misfitWeight>",
          "      <minimumMisfit>0</minimumMisfit>",
          "      <misfitType>L2_Normalized</misfitType>",
          "    </RefractionTarget>",
          "    <RefractionTarget type=\"Vs\">",
          "      <selected>false</selected>",
          "      <misfitWeight>1</misfitWeight>",
          "      <minimumMisfit>0</minimumMisfit>",
          "      <misfitType>L2_Normalized</misfitType>",
          "    </RefractionTarget>",
          "    <MagnetoTelluricTarget>",
          "      <selected>false</selected>",
          "      <misfitWeight>1</misfitWeight>",
          "      <minimumMisfit>0</minimumMisfit>",
          "      <misfitType>L2_Normalized</misfitType>",
          "    </MagnetoTelluricTarget>",
          "  </TargetList>",
          "</Dinver>\n"]
    return "\n".join(c)


def write_target(entries, path, misfit_weight=1, group_pad_per_decade=GROUP_PAD_PER_DECADE):
    """Write a Dinver .target: gzip tar holding a UTF-16-LE (BOM) contents.xml.

    Container recipe copied from swprepost.TargetSet._to_target (3.4.2 branch), so the file is
    indistinguishable from a swprepost one apart from the per-curve <slowness> tag and the
    invalid grid-padding points on Group curves."""
    text = "﻿" + target_xml(entries, misfit_weight, group_pad_per_decade)
    data = text.encode("utf_16_le")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.DEFAULT_FORMAT) as tar:
        info = tarfile.TarInfo(name="contents.xml")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    with open(path, "wb") as f:
        f.write(buf.getvalue())
    return path


def read_target(path):
    """Parse a .target back into [(frequency, velocity_ms, velstd_ms, slowness, description)].

    swprepost.TargetSet.from_file drops the slowness tag (it assumes Phase), so this reader
    exists to make the round-trip test able to check the ONE thing we changed."""
    import re
    import swprepost
    from swprepost.regex import modalcurve_exec
    with tarfile.open(path, "r:gz") as tar:
        text = tar.extractfile(tar.getmember("contents.xml")).read().decode("utf_16_le")
    text = text.lstrip("﻿")
    out = []
    for m in modalcurve_exec.finditer(text):
        block = m.group(1)
        f, v, s, desc = swprepost.ModalTarget._parse_modeltarget_from_text(block, "3.4.2")
        # drop the invalid grid-padding points (swprepost's regex reads every triple)
        valid = np.array([t == "true" for t in re.findall(r"<valid>(true|false)</valid>", block)])
        if valid.size == f.size:
            f, v, s = f[valid], v[valid], s[valid]
        slow = re.findall(r"<slowness>(Group|Phase)</slowness>", block)
        out.append((f, v, s, slow[0] if slow else None, desc))
    return out


def cell_target_entries(cell, waves, measure_of, min_cov=0.05, n_resample=30,
                        wavelength_range_m=None):
    """[(ModalTarget, slowness, name)] for the requested curve keys of a CellData.

    measure_of : callable curve_key -> "group"|"phase" (vs_inversion.parse_curve_key()[1]).
    Curve keys follow the repo convention: "fund" is GROUP, "fund_phase" is PHASE, so a joint
    group+phase cell yields two ModalCurves for the same mode with different slowness tags --
    which is the case swprepost cannot express and the reason this module exists."""
    entries = []
    for key in waves:
        if key not in cell.curves or len(cell.curves[key][0]) == 0:
            continue
        base, meas = measure_of(key)
        T, V, S = cell.curves[key]
        tgt = to_modal_target(T, V, S, base, min_cov=min_cov, n_resample=n_resample,
                              wavelength_range=wavelength_range_m)
        entries.append((tgt, "Group" if meas == "group" else "Phase", key))
    return entries


# --------------------------------------------------------------------- self-test
def _selftest(tmpdir=None):
    """Round-trip: pipeline curve -> .target -> parse -> same curve; group tag survives."""
    import os
    import tempfile
    tmpdir = tmpdir or tempfile.mkdtemp(prefix="dinver_target_")
    T = np.array([0.5, 1.0, 2.0, 4.0]); V = np.array([1.2, 1.6, 2.1, 2.6]); S = 0.03 * V
    g = to_modal_target(T, V, S, "fund", min_cov=None, n_resample=None)
    p = to_modal_target(T * 1.1, V * 1.05, S, "fund", min_cov=None, n_resample=None)
    o = to_modal_target(T[:3], V[:3] * 1.4, S[:3], "overtone", min_cov=None, n_resample=None)
    path = os.path.join(tmpdir, "rt.target")
    write_target([(g, "Group", "fund"), (p, "Phase", "fund_phase"), (o, "Group", "overtone")], path)
    back = read_target(path)
    assert len(back) == 3, len(back)
    assert [b[3] for b in back] == ["Group", "Phase", "Group"], [b[3] for b in back]
    assert back[2][4] == [("rayleigh", 1)], back[2][4]
    for (f, v, s, _, _), tgt in zip(back, (g, p, o)):
        assert np.allclose(f, tgt.frequency), (f, tgt.frequency)
        assert np.allclose(v, tgt.velocity, rtol=1e-6), (v, tgt.velocity)
        assert np.allclose(s, tgt.velstd, rtol=1e-6), (s, tgt.velstd)
    Tb, Vb, Sb = curve_from_hz_ms(back[0][0], back[0][1], back[0][2])
    assert np.allclose(Tb, T) and np.allclose(Vb, V) and np.allclose(Sb, S, rtol=1e-6)
    # a swprepost-written phase-only target must parse identically through our reader
    import swprepost
    swprepost.TargetSet([p]).to_file(os.path.join(tmpdir, "sw.target"), version="3.4.2")
    sw = read_target(os.path.join(tmpdir, "sw.target"))
    assert sw[0][3] == "Phase" and np.allclose(sw[0][1], p.velocity)

    # non-monotone lambda (Love group through an Airy minimum): must fall back to log-frequency
    # and still yield a valid, positive, ordered, on-curve target -- the pre-fix code produced
    # negative frequencies here.
    Tl = np.array([0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 3.6, 4.3, 4.8])
    Ul = np.array([1.60, 1.30, 0.95, 0.75, 0.70, 0.90, 1.30, 1.70, 2.10, 2.33])   # Airy min
    assert not np.all(np.diff(Ul * Tl) > 0), "test curve should be non-monotone in lambda"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        lt = to_modal_target(Tl, Ul, 0.05 * Ul, "love", n_resample=8)
    assert any("log-FREQUENCY" in str(x.message) for x in w), "expected the fallback warning"
    fl = np.asarray(lt.frequency); vl = np.asarray(lt.velocity)
    assert np.all(fl > 0) and np.all(np.diff(fl) > 0), fl
    assert np.allclose(vl / KM, np.interp(1 / fl, Tl, Ul), rtol=0.08), (vl / KM, np.interp(1 / fl, Tl, Ul))
    return path


if __name__ == "__main__":
    print("round-trip OK:", _selftest())
