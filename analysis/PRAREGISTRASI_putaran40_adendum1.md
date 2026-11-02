# ADDENDUM 1 to PRAREGISTRASI_putaran40.md (2026-10-23; written BEFORE the first run; no round 40 result had been seen)

Hash of the parent file: see `PRAREGISTRASI_hash_putaran40.txt` (4ea1e89c...). This addendum changes neither the hypotheses nor the decision rules; it aligns the scope with the round 39 data that actually exist.

1. **ExtraTrees (ET) is dropped from stage A.** ET was run only in the round 39 pilot (20 replicates, 7 cells); stages B and C of round 39 used LRpat and RF. Reproduction (H40-0) and comparison are only possible for LRpat and RF. H40-1 is tested on LRpat and RF (not ET).
2. **Stage names and scope (details of section 3 of the parent):** `P` (cost pilot + reproduction: base H1 lognormal and contamination cells, 12 replicates, CPU); `A` (stage B of round 39 repeated, CPU, 50 replicates, heavy cell 30); `C` (stage C of round 39 repeated, CPU, 100 replicates); `GB` (GPU: base H1 lognormal and contamination, each with s=1 and s=0; 20 replicates); `GC` (GPU: normal, t3, chi-square H1 s=1; 20 replicates); `NM` (50 replicates H1 and H2a, N0 and V4); `ROH` (Rohira RF-none/RF-logpat on log features and LR-none/LR-pat on raw features, 30 replicates H1 and H2a, seed 33000+rep).
3. **Rohira LR cells (stage B of the parent):** taken from `round36_diagnosis_rohira.py` (LR-none, LR-pat; raw features). Replicates 0-9 are checked against `hasil36/diagnosis_null_replikasi.csv`; the RF cells are checked against `hasil36/paired_rohira_null.csv` (replicates 0-29).
4. **Pilot P** only measures cost and checks H40-0 on CPU; its result does not change the scope of the full run.
5. **Saving predictions for the independent check:** stage A saves the out-of-fold predictions per patient for replicates 0-2 in the base cells (lognormal and contamination H1) to `hasil40/pred/`.
6. **H40-3 diagnostics** are computed per replicate (per training fold, averaged), stored in the columns `skew`, `Dq05`, `cusp` of every row of stages A, C, GB, GC; definitions as in the parent.
