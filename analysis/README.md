# analysis/ -- follow-up analyses, rounds 28--61

Scripts, analysis plans, and result tables of the follow-up analyses on the TUEP v2.0.1 first-session cohort (200 patients, 50 canonical patient-level folds); rounds 50 and later also use the public Emotiv cohort (Zenodo 1252141, 299 subjects). The code of the earlier rounds (feature extraction, models, rounds 3--27) is in `../code/`. The manuscript (`../manuscript/`, Supplement S1--S3) describes each round and its status.

All analyses are **follow-up analyses**: every plan was written after the results of earlier rounds were known, so none is confirmatory. The `PRAREGISTRASI_*.md` files are version-controlled analysis plans (the file names keep the original wording), not formal registrations.

## Names that come from Indonesian

File and folder names, CSV columns, and decision labels were kept as they were produced (the plans and scripts refer to them, and the stored results use them). Prose, comments, and messages are in English for rounds up to 50; the plans, hash notes, and script comments of rounds 51--61 are the original Indonesian files (so their SHA-256 values can be checked). Glossary:

| Name | Meaning |
|---|---|
| `hasilNN/` | results of round NN |
| `PRAREGISTRASI_putaranNN.md`, `..._hash_putaranNN.txt` | analysis plan of round NN and the record of SHA-256 hashes taken before the run |
| `putaran` / `lipatan` / `pasien` | round / fold / patient |
| `ringkas*`, `ringkasan` | summary |
| `bruto` | raw (unsummarized) output |
| `cek_independen` | independent check (second implementation) |
| `anggaran_setara` | equal tuning budget |
| `pembanding` | comparator |
| `kebocoran` | leakage |
| `kalibrasi` | calibration (operating characteristics of the decision rules, round 49B) |
| `struktur` | structure (within-patient structure, round 49D) |
| `deskriptif` | descriptive (outside the plan) |
| `sebelum_perbaikan_cetak` | before the print correction (copy kept as evidence, round 50 addendum 1) |
| `emotiv_fitur.npz` | cache of the Emotiv window features written by `round50_emotiv.py` (not included; recomputed on the first run) |
| `nesting_ketat`, `nm_ketat` | strict nesting (round 49A) |
| `ringkas` / `nyata` / `cek_independen` | summary / real data / independent check |
| `referensi_kontrol`, `referensi_acak` | control reference (round 55), random reference of equal size (round 56) |
| `lr_mentah`, `lr_bobot`, `lrw_disamakan` | logistic regression on raw features (53), with window weights (54), window-level LR with a training configuration matched to the adjustment (58) |
| `transformasi`, `keputusan_gabungan`, `validasi_aturan` | transformation (57), combined decision (59), validation of the rules on new simulation runs (60) |
| `perlipatan` | per fold |
| `tindaklanjut` | follow-up |
| `*_SELESAI` | "done" marker files (not included) |
| decision labels `tidak ada bukti` / `tidak pasti` / `kebocoran` | `no evidence` / `uncertain` / `leakage` (rules R1/R2) |
| CSV columns of `hasil50/ringkas_A*.csv`, `ringkas_B_uji.csv`: `aturan`, `n_rep`, `tidak_ada_bukti`, `tidak_pasti`, `kebocoran` (+ `_lo`/`_hi`), `delta_awal`, `sd_awal`, `famili`, `uji` | rule, number of replicates per draw, % no evidence, % uncertain, % leakage (+ Monte Carlo CI), difference and SD in the first 30/20 replicates, family, test |
| values `primer` / `sekunder`; `R2 normal (selisih 0)` | primary / secondary family; R2 under a normal model with a true difference of zero |
| CSV columns of `hasil49/B_*.csv`: `sumber`, `kumpulan`, `aturan`, `tingkat`, `jenis`, `ukuran_kumpulan`, `rata2`, `p_tidak_ada_bukti`, `p_tidak_pasti`, `p_kebocoran`, `ci_*` | source, pool, rule, level, type, pool size, mean, P(no evidence), P(uncertain), P(leakage), CI |

## Rounds

| Round | Content | Main scripts |
|---|---|---|
| 28 | Correlation between views, decision curves, strata sizes, Spisak-style confounder tests, Hamdan-style feature-shuffle leakage test | `round28_*.py` |
| 29 | Sensitivity of the normative layer: V0--V4; V4 (covariate regression on all training patients) removes the leak | `round29_*.py` |
| 30 | P3, NM, and F2 with V4/V5; dynamic leakage audit v2 | `round30_*.py`, `AUDIT_kode_putaran30.md` |
| 31--35 | Frozen BENDR/LaBraM with V4; equal tuning budgets; DNM, Rohira, Tang with V4 and V4t | `round31_*.py` ... `round35_*.py` |
| 36--38 | Mechanism of window-level adjustment leakage; paired null tests (Rohira log, Tang symlog, NM) | `round36*.py` ... `round38*.py` |
| 39 | Simulation: leakage map of adjustment on clustered data with zero disease signal | `round39_*.py` |
| 40 | Decision rules R1/R2 on the mean per-fold AUC; mechanism diagnostics | `round40_*.py`, `hasil40/LAPORAN_putaran40.md` |
| 41 | AUC within year x device strata with an age caliper | `round41_*.py`, `hasil41/LAPORAN_putaran41.md` |
| 42 | Plasmode: real EEG features, covariate-generated labels | `round42_*.py` |
| 43 | Equalized foundation-model pipelines, anatomical compression, selection-optimism correction | `round43_*.py`, `hasil43/LAPORAN_putaran43.md` |
| 44--45 | F2-Clean, IPW comparator, 30-window sensitivity; strata contrasts and patient bootstrap | `round44_*.py`, `round45_*.py` |
| 46 | R1 on the per-fold AUC for C1 and V0--V4 | `round46_*.py` |
| 47 | F2 without NM | `round47_*.py` |
| 48 | Dynamic leakage audit on all 50 folds | `round48_*.py` |
| 49 | Strict nesting (49A), operating characteristics of R1/R2 (49B), EEG over non-EEG predictors (49C), within-patient structure (49D), strata weighting and kurtosis table (49E/F) | `round49_*.py`, `table_builders/hitung_49D.py` |
| 50 | Plasmode replicates for raw-feature trees extended to 120 per cell (50A); plasmode repeated on the independent Emotiv cohort (50B); reporting completeness from stored results (50C); addendum 2: descriptive bootstrap CI of the Kural demographic baseline | `round50_*.py` (including `round50_kural_demografi_ci.py`), `run_chain_50.sh`, `hasil50/deskriptif_120_semua_pipeline.py`, `table_builders/bangun_tabel_putaran50.py` |
| 51 | Mechanism of the covariate-adjustment leak and candidate remedies (not reported in the manuscript) | `round51_*.py`, `run_chain_51*.sh` |
| 52 | Adaptive shrinkage (ash) of the adjustment coefficients extended to the other F2 components and comparators (not reported) | `round52_*.py`, `run_chain_52.sh` |
| 53 | Logistic regression on the same raw window features as the trees (window-level versus patient-level fitting), TUEP and Emotiv | `round53_*.py` |
| 54 | Test of the fitting-mismatch proposition for window-level logistic regression; **cancelled** before any summary (`hasil54/` is the untouched partial output; not reported) | `round54_*.py` |
| 55 | Simulation of the control-referenced adjustment: number of reference controls, sparse versus mirror era design, label--covariate association | `round55_*.py` |
| 56 | Real null test with a random reference of as many training patients as there are training controls | `round56_*.py` |
| 57 | Transformed pipelines without adjustment (adjustment effect within each transformation) | `round57_*.py` |
| 58 | Window-level logistic regression with a training configuration matched to the adjustment (factorial variant) | `round58_*.py` |
| 59 | Decision frequencies of the combined rule | `round59_*.py` |
| 60 | Validation of the frozen rules on new simulation runs with unused seeds (independent audits) | `round60_*.py` |
| 61 | Strictly nested inner selection for the control, random, and all-patient references (editorial review); descriptive re-summaries of stored results (decisions by rule and audit size, planning example, within-era differences, Emotiv subject flow) | `round61_*.py` |
| 61 | Strictly nested inner selection for the control, random, and all-patient references (editorial review); descriptive re-summaries of stored results (decisions by rule and audit size, planning example, within-era differences, Emotiv subject flow) | `round61_*.py` |
| 61 | Strictly nested inner selection for the control, random, and all-patient references (editorial review); descriptive re-summaries of stored results (decisions by rule and audit size, planning example, within-era differences, Emotiv subject flow) | `round61_*.py` |
| 61 | Strictly nested inner selection for the control, random, and all-patient references (editorial review); descriptive re-summaries of stored results (decisions by rule and audit size, planning example, within-era differences, Emotiv subject flow) | `round61_*.py` |
| 61 | Strictly nested inner selection for the control, random, and all-patient references (editorial review); descriptive re-summaries of stored results (decisions by rule and audit size, planning example, within-era differences, Emotiv subject flow) | `round61_*.py` |
| 61 | Strictly nested inner selection for the control, random, and all-patient references (editorial review); descriptive re-summaries of stored results (decisions by rule and audit size, planning example, within-era differences, Emotiv subject flow) | `round61_*.py` |
| 61 | Strictly nested inner selection for the control, random, and all-patient references (editorial review); descriptive re-summaries of stored results (decisions by rule and audit size, planning example, within-era differences, Emotiv subject flow) | `round61_*.py` |
| 61 | Strictly nested inner selection for the control, random, and all-patient references (editorial review); descriptive re-summaries of stored results (decisions by rule and audit size, planning example, within-era differences, Emotiv subject flow) | `round61_*.py` |
| 61 | Strictly nested inner selection for the control, random, and all-patient references (editorial review); descriptive re-summaries of stored results (decisions by rule and audit size, planning example, within-era differences, Emotiv subject flow) | `round61_*.py` |
| 61 | Strictly nested inner selection for the control, random, and all-patient references (editorial review); descriptive re-summaries of stored results (decisions by rule and audit size, planning example, within-era differences, Emotiv subject flow) | `round61_*.py` |
| 61 | Strictly nested inner selection for the control, random, and all-patient references (editorial review); descriptive re-summaries of stored results (decisions by rule and audit size, planning example, within-era differences, Emotiv subject flow) | `round61_*.py` |
| 61 | Strictly nested inner selection for the control, random, and all-patient references (editorial review); descriptive re-summaries of stored results (decisions by rule and audit size, planning example, within-era differences, Emotiv subject flow) | `round61_*.py` |
| 61 | Strictly nested inner selection for the control, random, and all-patient references (editorial review); descriptive re-summaries of stored results (decisions by rule and audit size, planning example, within-era differences, Emotiv subject flow) | `round61_*.py` |
| 61 | Strictly nested inner selection for the control, random, and all-patient references (editorial review); descriptive re-summaries of stored results (decisions by rule and audit size, planning example, within-era differences, Emotiv subject flow) | `round61_*.py` |
| 61 | Strictly nested inner selection for the control, random, and all-patient references (editorial review); descriptive re-summaries of stored results (decisions by rule and audit size, planning example, within-era differences, Emotiv subject flow) | `round61_*.py` |
| 61 | Strictly nested inner selection for the control, random, and all-patient references (editorial review); descriptive re-summaries of stored results (decisions by rule and audit size, planning example, within-era differences, Emotiv subject flow) | `round61_*.py` |

Each plan lists its hypotheses, decision rules, predictions (written before the results), and the checks that were promised. Rounds 28--50 and 53 have an independent check (`*_cek_independen.py`) that recomputes the main numbers with a second implementation. For rounds 51--61, the hash files also record the `--check` reproductions of earlier outputs that were run before each round.

## What is not included

* TUEP data and anything derived from it that carries patient identifiers (for example per-fold prediction files keyed by subject). The result tables and prediction arrays kept in `hasil*/` carry no identifiers.
* Run logs and "done" marker files.
* `hasil52/nm_ash/` (per-fold NormMamba outputs keyed by TUEP subject). Some hash notes name log files (for example `hasil53/ringkas_pertama_sebelum_perbaikan.log`); these logs are not included.
* Scripts that generated the Indonesian version of the manuscript tables. Some presentation scripts named in the plans (for example the table builders listed in addendum 5 of round 49) are therefore not part of this repository. For round 50, `table_builders/bangun_tabel_putaran50.py` is an English-only version of the presentation script; `--check` confirms that it reproduces the Emotiv rows of the plasmode table and Table S2.16 in `../manuscript/naskah/`.
* For rounds up to 50, the plans, hash notes, and script comments were translated into English for publication. The SHA-256 values in `PRAREGISTRASI_hash_*.txt` were computed on the original Indonesian files and do not match the English files here.
