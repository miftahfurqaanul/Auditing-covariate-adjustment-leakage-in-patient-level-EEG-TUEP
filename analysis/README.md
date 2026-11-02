# analysis/ -- follow-up analyses, rounds 28--49

Scripts, analysis plans, and result tables of the follow-up analyses on the TUEP v2.0.1 first-session cohort (200 patients, 50 canonical patient-level folds). The code of the earlier rounds (feature extraction, models, rounds 3--27) is in `../code/`. The manuscript (`../manuscript/`, Supplement S1--S3) describes each round and its status.

All analyses are **follow-up analyses**: every plan was written after the results of earlier rounds were known, so none is confirmatory. The `PRAREGISTRASI_*.md` files are version-controlled analysis plans (the file names keep the original wording), not formal registrations.

## Names that come from Indonesian

File and folder names, CSV columns, and decision labels were kept as they were produced (the plans and scripts refer to them, and the stored results use them). Prose, comments, and messages are in English. Glossary:

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
| `nesting_ketat`, `nm_ketat` | strict nesting (round 49A) |
| `perlipatan` | per fold |
| `tindaklanjut` | follow-up |
| `*_SELESAI` | "done" marker files (not included) |
| decision labels `tidak ada bukti` / `tidak pasti` / `kebocoran` | `no evidence` / `uncertain` / `leakage` (rules R1/R2) |
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
| 49 | Strict nesting (49A), operating characteristics of R1/R2 (49B), EEG over non-EEG predictors (49C), within-patient structure (49D), strata weighting and kurtosis table (49E/F) | `round49_*.py`, `table_builders/` |

Each plan lists its hypotheses, decision rules, predictions (written before the results), and the checks that were promised. Every round has an independent check (`*_cek_independen.py`) that recomputes the main numbers with a second implementation.

## What is not included

* TUEP data and anything derived from it that carries patient identifiers (for example per-fold prediction files keyed by subject). The result tables and prediction arrays kept in `hasil*/` carry no identifiers.
* Run logs and "done" marker files.
* Scripts that generated the Indonesian version of the manuscript tables. Some presentation scripts named in the plans (for example the table builders listed in addendum 5 of round 49) are therefore not part of this repository.
* The plans, hash notes, and script comments were translated into English for publication. The SHA-256 values in `PRAREGISTRASI_hash_*.txt` were computed on the original Indonesian files and do not match the English files here.
