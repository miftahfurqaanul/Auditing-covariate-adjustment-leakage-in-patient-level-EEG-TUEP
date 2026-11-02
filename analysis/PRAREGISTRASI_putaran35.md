# Analysis plan, round 35: V4t (adjustment fitted only on the patients that train the classifier) for Rohira and Tang (2026-10-22)

## Background (results of rounds 33-34 already seen)
The permuted-feature test showed LEAKAGE in the V4 variants applied at window/second level: Rohira-V4 H1 0.573, H2a 0.562 (10 replicates); Tang-V4 H1 0.605, H2a 0.596 (20 replicates), whereas the variants without adjustment (F) are clean (Rohira 0.503, Tang 0.500/0.509) and the patient-level DNM-V4 is clean (0.500/0.514). The real-data results of Rohira-V4 and Tang-V4 must NOT be used as leak-free comparators.
## Mechanism hypothesis (fixed before the round 35 run)
Rohira and Tang fit V4 on ALL training patients (training + validation) but train the classifier on only 80% of them (the `tr_p` patients; the 20% `va_p` patients are used for the threshold/epoch selection). The regression residuals are orthogonal to the design only on the full set, not on `tr_p` alone: an asymmetry similar to controls vs epilepsy in V0, so covariate estimation error remains in the training features and is carried to the test patients. The V4t fix: the coefficients B are fitted ONLY on the `tr_p` patients (the set that trains the classifier); validation and test patients are treated as out-of-sample data. This is also a candidate explanation of the NM-V4 null slightly above 0.50 (0.515; NM also fits the tokens on all training patients and trains on 80%).
## Design
Rohira (CPU, `round35_rohira_v4t.py`): variants V4t and V4t+AS (age/sex as per-window columns) with the same Rohira-F recipe (round33); permuted-feature test H1 (global) and H2a (within era), 20 replicates with seed 33000 + r (the first 10 identical to round 33), without age/sex; rule R1: mean <= 0.52 and mean + 1.645 SE <= 0.54 = no evidence of leakage; > 0.55 leakage. Real data are reported only as a test if R1 is met on H1 and H2a; otherwise they are flagged as leaking and not used.
Tang (GPU, `round35_tang_v4t.py`): variant V4t (B from the `tr_p` patients); the reduced permuted-feature test as in round 34 (repetition 0, 5 folds, H1/H2a, 20 replicates, seeds 34000 + rep and 34300 + rep) is run BEFORE the real data; the real data (50 folds) are run afterwards and evaluated with `round35_eval_tang.py` (CPU; same contrasts as round 34: Tang-V4t - Tang-F, I2-adj@199 - Tang-V4t, Tang-V4t+AS - Tang-F+AS, Holm over 3 tests).
Mandatory check: B from `tr_p` only (assert on the features: the `tr_p` residuals are orthogonal to the design; changing the data of validation/test patients does not change B).
## Predictions
- Rohira V4t null H1 <= 0.52 (70%); H2a <= 0.52 (55%).
- Tang V4t null H1 and H2a <= 0.52 (65%).
- If they pass: |Rohira-V4t+AS - Rohira+AS| < 0.05 and not significant (65%); |Tang-V4t - Tang-F| < 0.06 and not significant (65%).
- If V4t passes on Rohira and Tang, NM-V4 (and therefore F2-adj) must be rerun as NM-V4t (GPU, about 4 hours) together with the reduced null; the decision to execute rests with the owner.
## Note
Round 35 was designed AFTER seeing the V4 leakage in Rohira and Tang (written before any V4t result). The choice of this fix was guided by the null test, not by the real-data AUC; the leaking V4 AUC had already been seen. No claim of superiority comes from this round.
