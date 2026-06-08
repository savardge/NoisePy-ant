"""
Synthetic verification / calibration for the FTAN period correction and phase-velocity
extraction added to noisepy/dispersion.py.

Builds a single-mode dispersive causal Green's function with a KNOWN phase-velocity curve
c(T) and the implied group velocity U(T) = dw/dk, runs it through the CWT pipeline, and:

  1. checks that the measured group velocity matches U_true(T);
  2. calibrates the Morlet phase offset (the constant phase_offset that makes
     phase_velocity recover c_true(T) across the band) and checks it is ~constant;
  3. checks the Shapiro centroid correction: with a deliberately steep (falling) amplitude
     spectrum the centroid period is shorter than the nominal period.

Run:  python scripts/picking/synthetic_phase_calibration.py
"""
import numpy as np
from noisepy import dispersion as d


def build_synthetic(dist=30.0, dt=0.02, npts=8192, spectral_slope=0.0,
                    band=(0.25, 3.0), c0=2.6, c1=0.30):
    """
    Causal single-mode dispersive signal at distance `dist`.

    Phase velocity model: c(T) = c0 + c1 * T  (km/s), for periods within `band`.
    Returns (signal_pos, c_func, U_func, dt, dist).
    spectral_slope: amplitude ~ f**spectral_slope inside the band (negative => falls off
    toward high frequency, the Shapiro scenario).
    """
    freqs = np.fft.rfftfreq(npts, dt)            # Hz, >= 0
    w = 2 * np.pi * freqs
    T = np.zeros_like(freqs)
    T[1:] = 1.0 / freqs[1:]

    fmin, fmax = 1.0 / band[1], 1.0 / band[0]
    in_band = (freqs >= fmin) & (freqs <= fmax)

    # Phase velocity and wavenumber on the grid
    c = c0 + c1 * np.where(T > 0, T, 0.0)        # km/s
    k = np.zeros_like(w)
    k[in_band] = w[in_band] / c[in_band]

    # Amplitude: cosine-tapered band, optional spectral slope
    A = np.zeros_like(freqs)
    f_in = freqs[in_band]
    taper = 0.5 * (1 - np.cos(2 * np.pi * (f_in - f_in.min()) / (f_in.max() - f_in.min())))
    slope = np.ones_like(f_in) if spectral_slope == 0 else (f_in / f_in.mean()) ** spectral_slope
    A[in_band] = taper * slope

    # Causal propagating signal: x(t) = Re int A exp(i(w t - kΔ)) dw.
    # numpy's irfft reconstructs with exp(+i w t), so use exp(-i kΔ) to put the group
    # arrival at the causal time t = +Δ/U(w).
    spec = A * np.exp(-1j * k * dist)
    x = np.fft.irfft(spec, n=npts)               # t = 0, dt, 2dt, ...

    # Analytic phase/group velocity models for checking
    def c_func(period):
        return c0 + c1 * np.asarray(period, dtype=float)

    def U_func(period):
        # U = dw/dk computed numerically from the model on a fine grid
        Tg = np.asarray(period, dtype=float)
        wf = 2 * np.pi / Tg
        kf = wf / (c0 + c1 * Tg)
        # central differences in w
        dw = 1e-4 * wf
        Tp = 2 * np.pi / (wf + dw)
        Tm = 2 * np.pi / (wf - dw)
        kp = (wf + dw) / (c0 + c1 * Tp)
        km = (wf - dw) / (c0 + c1 * Tm)
        return (2 * dw) / (kp - km)

    return x, c_func, U_func, dt, dist


def main():
    dist, dt = 60.0, 0.02
    band = (0.3, 3.0)
    tau_max = dist / 12.0  # Bensen Δ/12 phase-measurement cutoff
    x, c_func, U_func, dt, dist = build_synthetic(dist=dist, dt=dt, npts=16384, band=band)

    cwt = d.compute_cwt(x, dist, dt, Tmin=band[0], vmin=1.5, vmax=4.5, vave=2.3)

    print(f"dist={dist} km, tau_max=Δ/12={tau_max:.2f} s (phase reliable below this period)\n")
    test_periods = np.arange(0.5, 2.61, 0.2)
    print(f"{'T':>5} {'U_true':>7} {'U_meas':>7} {'c_true':>7} {'phi(tu)':>8} {'Tcen':>6} {'Tinst':>6}")
    rows = []
    for T in test_periods:
        U_true = float(U_func(T))
        # initial group-velocity guess from the global envelope peak on that scale row
        j = d._scale_index_for_period(cwt, T)
        it = int(np.argmax(np.abs(cwt['cwt'][j, :])))
        U_guess = dist / cwt['tvec'][it]
        m = d.measure_point(cwt, T, U_guess, dist)
        U_meas = m['U']
        c_true = float(c_func(T))
        phi = m['phase']
        Tcen = d.centroid_period(cwt, T)
        Tinst = m['T_inst']
        rows.append((T, U_true, U_meas, c_true, phi))
        print(f"{T:5.2f} {U_true:7.3f} {U_meas:7.3f} {c_true:7.3f} {phi:8.3f} {Tcen:6.3f} {Tinst:6.3f}")

    # --- Calibrate the phase offset by grid search: the constant phase_offset that
    # minimises the in-band phase-velocity recovery error (N is resolved per period by the
    # reference curve, so this is an honest end-to-end calibration). ---
    cref = d.load_reference_curve((test_periods, c_func(test_periods)))
    inband = [r for r in rows if r[0] <= tau_max]

    def inband_err(off):
        errs = []
        for (T, U_true, U_meas, c_true, phi) in inband:
            c_rec, _ = d.phase_velocity(phi, U_meas, dist, T, cref,
                                        phase_shift=0.0, phase_offset=off)
            if np.isfinite(c_rec):
                errs.append(abs(c_rec - c_true) / c_true)
        return np.mean(errs) if errs else np.inf

    grid = np.linspace(-np.pi, np.pi, 361)
    errs = np.array([inband_err(o) for o in grid])
    off_mean = float(grid[np.argmin(errs)])
    print(f"\nBest-fit Morlet phase_offset = {off_mean:+.3f} rad "
          f"(in-band mean err {100*errs.min():.2f}%).")
    print(f"Robustness: in-band mean err is {100*inband_err(0.0):.2f}% at offset 0 and "
          f"{100*inband_err(-np.pi/4):.2f}% at -pi/4 -- at Δ={dist:g} km the result is\n"
          f"weakly sensitive to the absolute offset because the 2*pi*N reference resolution "
          f"dominates (offset matters more at small distance*period).")

    # --- Recover c with the calibrated offset and a reference curve, check error ---
    print(f"\n{'T':>5} {'c_true':>7} {'c_rec':>7} {'N':>3} {'err%':>6}  (* = T>tau_max)")
    max_err_inband = 0.0
    for (T, U_true, U_meas, c_true, phi) in rows:
        c_rec, N = d.phase_velocity(phi, U_meas, dist, T, cref,
                                    phase_shift=0.0, phase_offset=off_mean)
        err = 100 * abs(c_rec - c_true) / c_true
        flag = '' if T <= tau_max else ' *'
        if T <= tau_max:
            max_err_inband = max(max_err_inband, err)
        print(f"{T:5.2f} {c_true:7.3f} {c_rec:7.3f} {N:3d} {err:6.2f}{flag}")
    print(f"\nMax phase-velocity recovery error within tau_max: {max_err_inband:.2f}%")

    # --- Shapiro centroid check with a steeply falling spectrum ---
    xs, *_ = build_synthetic(dist=dist, dt=dt, band=band, spectral_slope=-4.0)
    cwts = d.compute_cwt(xs, dist, dt, Tmin=band[0], vmin=1.0, vmax=4.5, vave=2.4)
    print(f"\nShapiro steep-spectrum check (slope -4): {'T':>5} {'Tcen':>6} {'shift%':>7}")
    for T in test_periods:
        Tcen = d.centroid_period(cwts, T)
        print(f"{'':>5}{T:5.2f} {Tcen:6.3f} {100*(Tcen-T)/T:7.2f}")
    print("(Falling spectrum => centroid period shorter than nominal => negative shift.)")


if __name__ == "__main__":
    main()
