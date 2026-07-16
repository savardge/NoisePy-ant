"""Parallel-tempering ladder defaults -- the single source of truth.

These lived duplicated in `scripts/picking/well_vs_qc.py` and `scripts/picking/run_bayhunter_cell.py`
(both hardcoding `max(1, N // 2)` and `2.0`). well_vs_qc always writes a resolved value into the
cfg, but six other cfg builders (grid_vs_inversion, chaincount_well_study, noisepy.vs_inversion,
synth_radial_gate, ...) never set the keys and fall back to the runner's copy -- so a change in one
file and not the other diverges silently, per-driver. Import from here instead.

WHY THESE VALUES
----------------
The previous defaults (`t1chains = nchains // 2`, `maxtemp = 2.0`) cannot do PT's job, and the
verdict that they were fine was based on the wrong statistic.

`maxtemp = 2.0` puts the hottest chain at beta = 0.5: it merely HALVES every log-likelihood
difference. A 25-logL barrier is still 12.5 logL after tempering (e^-12.5 ~ 4e-6) -- uncrossable.
So the ladder melts nothing while discarding every sample not at T=1 (BayHunter keeps only those,
Plotting.py:271). All cost, no benefit; PT could only ever look neutral-to-worse.

The old verdict -- "31% swap acceptance, squarely in the healthy 20-30% band, so the ladder was
never the problem" -- optimised ladder SPACING and never checked ladder RANGE. Swap acceptance is
necessary, not sufficient: 31% only says the rungs are well spaced *within [1, 2]*; it says nothing
about whether [1, 2] is the right interval. A HIGH acceptance is in fact a symptom of a ladder so
tight that adjacent rungs are nearly the same distribution, so swapping is free and buys no
exploration. Measured in this fork on a synthetic (12 chains, 400k iters):

    t1chains=6,  maxtemp=2   ->  swap acceptance 44.5%,  T=1 sample fraction 0.50
    t1chains=3,  maxtemp=50  ->  swap acceptance 19.0%,  T=1 sample fraction 0.25

Correct knob order: set the RANGE from the physics (barrier height), THEN add chains until the
SPACING gives ~20-30% acceptance. Tuning spacing by shrinking range is how you land on a ladder
that is perfectly spaced and useless.

Reference: Jan Dettmer's Fortran/MPI rj-McMC (~/Codes/receiver_rjmcmc_varpar_sourceinv_joint,
template_parameter.dat: NPTCHAINS1=3, dTlog=1.35). With 16 chains that gives
T = 1,1,1,1.35,1.82,...,36.6,49.5, i.e. hottest beta = 0.02 -- the same 25-logL barrier becomes
0.51 logL and is crossed freely. It pins only THREE chains at T=1, not half of them: chains at
equal T never swap with each other (mcmcOptimizer._swap_temperatures skips equal-T pairs), so
`t1chains = nchains // 2` is 8 redundant cold chains while only 8 carry the whole ladder.

COST, STATED PLAINLY
--------------------
t1chains=3 of 16 leaves ~19% of samples at T=1 versus ~50% before, i.e. roughly 2.7x fewer
posterior samples per unit wall-clock. Raise --n-chains or --iter-main to compensate. This is a
real price for working mixing, not a free improvement.

DIAGNOSTIC
----------
Judge a ladder by ROUND-TRIP RATE (chains travelling T=1 -> T_max -> T=1), not swap acceptance --
swap acceptance alone cannot distinguish a good ladder from a too-cold one. Computable from the
`pt_temp_history` array that run_bayhunter_cell.py saves into the npz.
"""

# Chains pinned at T=1. Matches Dettmer's NPTCHAINS1=3. Chains at equal temperature never swap
# with each other, so extra T=1 chains add no mixing -- they are just independent cold chains.
PT_T1CHAINS = 3

# Top of the temperature ladder. Must be set by the BARRIER HEIGHT, not by chasing a swap rate.
# 50.0 gives beta = 0.02 at the hottest rung, matching Dettmer's 16-chain default (T_max ~ 49).
PT_MAXTEMP = 50.0


def round_trips(temp_history):
    """Round trips per chain: T=1 -> hottest rung -> T=1. THE diagnostic for ladder adequacy.

    `temp_history` is (nchains, nsamples), e.g. the thinned `pt_temp_history` saved in the npz.

    Why not swap acceptance: it measures whether adjacent rungs OVERLAP (spacing). It cannot tell
    you whether the ladder REACHES far enough to melt the barrier -- a ladder crammed into [1, 2]
    can post a textbook 20-30% acceptance while no chain ever samples anything the cold chains
    could not reach alone. A round trip is the thing that actually transports a state from the hot
    end (where barriers are crossable) down to T=1 (where the posterior is collected), so it is
    what PT has to deliver. Rule of thumb: want >= ~1 round trip per chain per run; near zero means
    the ladder is too coarse OR too short, and swap acceptance tells you which (low => too coarse,
    add rungs; healthy but no round trips => range is fine but the ladder is not transporting).

    Returns (round_trips_per_chain, total). Returns zeros if PT was off (all temperatures 1.0).
    """
    import numpy as np
    th = np.asarray(temp_history, float)
    if th.ndim != 2 or th.size == 0:
        return np.array([]), 0
    tmax = np.nanmax(th)
    if not np.isfinite(tmax) or tmax <= 1.0:
        return np.zeros(th.shape[0], int), 0          # PT off: every temperature is 1.0
    out = np.zeros(th.shape[0], int)
    for i, row in enumerate(th):
        # count 1 -> tmax -> 1 cycles: walk the sequence of "at cold" / "at hot" visits and count
        # each cold->hot->cold transition once
        cold = row <= 1.0
        hot = row >= tmax - 1e-6
        state, n = None, 0
        for c, h in zip(cold, hot):
            if c and state == "hot":
                n += 1
                state = "cold"
            elif c:
                state = "cold"
            elif h and state == "cold":
                state = "hot"
        out[i] = n
    return out, int(out.sum())


def resolve(t1chains, maxtemp, nchains):
    """Fill in unset PT knobs. Pass None to take the default; 0 is NOT treated as unset.

    The `x or default` idiom this replaces silently swallowed --t1chains 0 / --maxtemp 0 (both
    falsy). 0 is a real, if bad, value and should reach _create_temperature_ladder's own warning
    rather than be reinterpreted here as "the user said nothing".
    """
    t1 = PT_T1CHAINS if t1chains is None else int(t1chains)
    mt = PT_MAXTEMP if maxtemp is None else float(maxtemp)
    return max(1, min(t1, int(nchains))), mt
