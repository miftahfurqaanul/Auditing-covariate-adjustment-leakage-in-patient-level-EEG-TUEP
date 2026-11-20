# Auditing covariate-adjustment leakage in patient-level EEG classification: simulations and a TUEP case study

Code, analysis plans, result tables and manuscript sources for a methodological study of **covariate-adjustment
leakage** in patient-level epilepsy classification from routine EEG (TUH EEG Epilepsy Corpus, TUEP v2.0.1;
200 patients, first session, 50 patient-level folds).

The study asks how much spurious discrimination appears when covariates (age, sex, recording era) are removed
from EEG features while the EEG itself carries no label information, and how often a feature-shuffle audit with
explicit decision rules (R1, R2) raises false alarms or misses the leak. The pipeline that is audited (F2, a
mean of three view-specific probabilities) is a test case, not a validated clinical model. The label is the TUEP
curation label (derived from reports, clinical history and medication), not an independent clinical diagnosis.

> Status: the current manuscript version (2026-11-20, "Covariate-adjustment leakage in patient-level EEG classification:
> effects of fitting population, repeated measurements, and audit replication") is in `manuscript/jbi/`. The analyses are
> follow-up analyses on one retrospective archive over 60 rounds (rounds 51, 52, and the cancelled round 54 are not reported);
> they are exploratory, not confirmatory. Round 50 and later rounds also use the independent Emotiv cohort.

## Repository layout

| Folder | Content |
|---|---|
| `manuscript/jbi/` | **Current version (2026-11-20).** Single-file LaTeX source of the main text (`main_jbi_en.tex`, compiles with the included `elsarticle.cls`, `.bst`, `.bbl`, and `supl_jbi_en.aux`), Figures 1--4, and the PDFs of the main text and the Supplementary Material S1--S4 |
| `manuscript/` (other folders) | Earlier version (2026-11-03): LaTeX sources of the main text (`main_en.tex`) and the supplementary material S1--S3 (`supl_S1S3_en.tex`), section files in `naskah/`, figures in `figures_en/`, bibliography, build script, and the compiled PDFs in `manuscript/pdf/` |
| `analysis/` | Scripts, analysis plans (`PRAREGISTRASI_*.md`) with their SHA-256 records (`PRAREGISTRASI_hash_*.txt`), and result tables for rounds 28--60 (`hasil*/`); see `analysis/README.md` for the round index and a glossary of the Indonesian-derived file names |
| `code/` | Pipeline code shared by all rounds (feature extraction, normative adjustment, evaluation kit, models, rounds 3--27) |
| `tools/` | `verify_sha256.sh`, `rewrite_paths.sh` |
| `SHA256SUMS.txt` | Checksums of every file in this repository |
| `requirements.txt` | Frozen Python environment (Python 3.12.3) |

## Data (not redistributed)

* **TUEP v2.0.1** (Temple University Hospital EEG Epilepsy Corpus): free registration with the Neural Engineering
  Data Consortium (https://isip.piconepress.com/projects/nedc/html/tuh_eeg/). The data use terms do not allow
  redistribution, so no EEG, no patient index and no per-patient outputs are included here.
* Result files that contain TUEP subject identifiers (per-fold predictions keyed by subject) were removed;
  `analysis/hasil*/` keeps the aggregate result tables and prediction arrays that carry no identifiers (run logs are not included).
* The patient-level folds are deterministic: `code/folds.py` rebuilds the 10 x 5 stratified folds
  (`StratifiedKFold`, seeds 1000 + repetition) from the patient index.
* External sets: Kural et al. (Zenodo 4688758) and the Emotiv Guinea-Bissau/Nigeria set (Zenodo 1252141). Both are used
  for prediction transfer; the Emotiv set is also used for the round-50 plasmode (`analysis/round50_emotiv.py`, which
  recomputes its cached band-power features from the preprocessed Emotiv windows).

## Rebuilding the manuscript

Install [Tectonic](https://tectonic-typesetting.github.io/) (or any TeX engine; edit the script accordingly) and run

```bash
cd manuscript
TECTONIC=tectonic bash build/build_en.sh      # main_en.pdf and supl_S1S3_en.pdf
```

The build order is main text, supplement, main text (the two documents cross-reference each other). The PDFs are
written to `manuscript/build/out/`. Their text matches the PDFs in `manuscript/pdf/`.

## Re-running the analyses

1. `bash tools/verify_sha256.sh` (before any change).
2. Obtain TUEP and build the working directory described at the top of `tools/rewrite_paths.sh`.
3. `bash tools/rewrite_paths.sh /path/to/WORKDIR` points the hard-coded paths (the scripts were written on a
   single workstation with absolute paths) to this clone and your working directory.
4. Create the environment from `requirements.txt` and run the scripts of the round of interest, e.g.
   `analysis/round50_*.py`. Each round's plan (`analysis/PRAREGISTRASI_putaranNN.md`) describes its scripts and outputs.
   Some rounds need a GPU (NormMamba, foundation-model embeddings); LaBraM, CBraMod, BENDR and EEG-Bench checkpoints must be
   obtained from their original repositories.

## Analysis plans and their status

Every round was planned in a version-controlled document written after the results of earlier rounds were
known. Rounds 3--27 were committed to Git before their code ran; rounds 28--60 were hashed with SHA-256 (the
`PRAREGISTRASI_hash_*.txt` files). For rounds 51--60 the plans and hash files are the original Indonesian files, so
`sha256sum analysis/PRAREGISTRASI_putaran5*.md` reproduces the last value recorded for each plan (plans that received
an addendum were hashed again, and the hash file lists every value with its timestamp). A hash identifies a document but does not by itself prove when it was written, and
these plans are **not** registrations in the confirmatory sense. Addenda made after results were seen are
listed in the plans and in the manuscript (Supplement S1).

## Known limitations of this repository

* The scripts contain absolute paths of the original workstation (see above). Their logic is unchanged.
* For rounds up to 50, the analysis plans, the notes in the hash files, and the comments and messages of the scripts were translated from
  Indonesian into English for publication (only text; a check of the code structure confirmed that no executable code
  changed). The SHA-256 values recorded in `analysis/PRAREGISTRASI_hash_*.txt` were computed on the original Indonesian
  files and therefore do not match the English files in this repository. The files of rounds 51--60 were not translated, so
  their hashes can be checked directly. File and folder names, CSV columns, and the
  decision labels stored in result files were kept as produced; `analysis/README.md` explains them.
* The Git history of the original working directory is not included; this repository starts from the final state.
* No licence has been chosen yet; until the authors add one, all rights are reserved.

## Citation

A citation will be added when the manuscript is published.
