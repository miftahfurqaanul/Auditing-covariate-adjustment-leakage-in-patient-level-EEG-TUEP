# ADDENDUM 3 to PRAREGISTRASI_putaran40.md (2026-10-23; before any result of stages A/C/GB/GC/NM/ROH was seen)

Implementation details of `round40_ringkas.py` not covered by addendum 2:
1. `lk` = log10(clip(kurtosis, min 0.1) + 1), the same as `round39_analisis.py` (the kurtosis of a normal distribution can be negative).
2. TOST is computed against the bounds 0.48 and 0.52 with a one-sided t-test (df = n-1); the TOST p is the maximum of the two p values; equivalence is declared when p < 0.05.
3. The R1 decision for H40-2 is computed for every cell (dist, scen, n, m, p, imb, s, sig, cfrac, cscale) x classifier x transform x adjust != none.
4. H40-1: the two-stage bootstrap is not used; replicates within a cell are bootstrapped and then averaged across cells (same as addendum 2 item 3).
5. Bootstrap seed: `np.random.default_rng(40)`.
