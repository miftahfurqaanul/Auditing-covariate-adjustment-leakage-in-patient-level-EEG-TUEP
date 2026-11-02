"""Inventory of every evaluation that touched the 50 EEG-Bench test patients (P4), built from the result files.
Rows from the degenerate BENDR time-mean representation are kept but flagged (reports/reconciliation.md §1)."""
import re, pandas as pd
R = "/home/miftah_f/RTX-5090/tuep_fresh/reports"; rows = []
add = lambda src, method, auc, bac, lo=None, hi=None, note="": rows.append(dict(source=src, method=method, AUC=auc, BAC=bac, BAC_lo=lo, BAC_hi=hi, note=note))

for r in pd.read_csv(f"{R}/strategyB_results.csv").query("protocol == 'P4'").itertuples():
    lo, hi = (map(float, r.BAC_CI.split("-")) if isinstance(r.BAC_CI, str) else (None, None))
    add("strategyB (2026-10-04)", f"{r.features} {r.model}", r.AUC, r.BAC, lo, hi)
for r in pd.read_csv(f"{R}/step1_p4.csv").itertuples():
    add("step1 (median aggregation)", r.name, r.AUC, r.BAC, r.BAC_lo, r.BAC_hi, "pre-declared primary P4 analysis" if r.name == "bandpower EEG + age+sex" else "")
for r in pd.read_csv(f"{R}/step2a_p4.csv").itertuples():
    add("step2a (mean aggregation)", r.name, r.AUC, r.BAC, r.BAC_lo, r.BAC_hi, "EXCLUDED: degenerate BENDR time-mean" if r.name in ("bendr", "fusion") else "")
for r in pd.read_csv(f"{R}/step2b_mil_p4.csv").itertuples():
    add("step2b MIL", r.name, r.AUC, r.BAC, r.BAC_lo, r.BAC_hi)
for r in pd.read_csv(f"{R}/eegbench_variants.csv").itertuples():
    add("EEG-Bench recipe replication", f"variant {r.variant} ({r.desc}) {r.encoder}", r.P4_AUC, r.P4_BAC, r.P4_BAC_lo, r.P4_BAC_hi,
        "recipe with CAR over all channels (shortcut)" if r.variant in ("A", "B") else "")
for r in pd.read_csv(f"{R}/round2_p4.csv").itertuples():
    if r.cand != "NESTED":
        add("round 2", r.cand, r.AUC, r.BAC, r.BAC_lo, r.BAC_hi)
for r in pd.read_csv(f"{R}/cbramod_p4.csv").itertuples():
    if r.config != "C1":
        add("frozen foundation models", r.config, r.AUC, r.BAC, r.BAC_lo, r.BAC_hi)
for r in pd.read_csv(f"{R}/threshold_p4.csv").itertuples():
    add("threshold selection", f"{r.cand} {r.threshold}", r.AUC, r.BAC, r.lo, r.hi, "chosen inside training (reported best, 0.704)" if r.chosen else "")
for r in pd.read_csv(f"{R}/fusion_p4.csv").itertuples():
    if r.config not in ("M1",):
        add("fusion", r.config, r.AUC, r.BAC, r.lo, r.hi)
for line in open(f"{R}/cnn_results.txt"):
    g = re.match(r"P4 (\w+)\s+5-seed ensemble (thr [^:]+?)\s*: AUC ([\d.]+)\s+BAC ([\d.]+) \(95% CI ([\d.]+)-([\d.]+)\)", line)
    if g:
        add("end-to-end CNN", f"{g[1]} {g[2].strip()}", float(g[3]), float(g[4]), float(g[5]), float(g[6]))
for line in open(f"{R}/cbramod_finetune_results.txt"):
    g = re.match(r"P4 FT-CBraMod 3-seed ensemble: AUC ([\d.]+) BAC ([\d.]+) CI ([\d.]+)-([\d.]+)", line)
    if g:
        add("CBraMod fine-tune v1 (15 fixed epochs)", "3-seed ensemble thr 0.5", *map(float, g.groups()))
for line in open(f"{R}/cbramod_ft_results.txt"):
    g = re.match(r"ENSEMBLE (thr [^:]+?)\s*: AUC ([\d.]+)\s+BAC ([\d.]+) \(95% CI ([\d.]+)-([\d.]+)\)", line)
    if g:
        add("CBraMod fine-tune v2 (official recipe)", f"3-seed ensemble {g[1].strip()}", *map(float, g.groups()[1:]))

T = pd.DataFrame(rows); T.to_csv(f"{R}/p4_inventory.csv", index=False)
valid = T[~T.note.str.startswith("EXCLUDED")]
print(f"P4 evaluations on the same 50 test patients: {len(T)} rows ({len(valid)} valid, {len(T) - len(valid)} excluded); "
      f"per-seed runs of CNN/fine-tuning not counted separately")
print(f"valid BAC: median {valid.BAC.median():.3f}, range {valid.BAC.min():.3f}-{valid.BAC.max():.3f}; rows with BAC >= 0.70: {(valid.BAC >= 0.70).sum()}")
print(valid.sort_values("BAC", ascending=False).head(8)[["source", "method", "AUC", "BAC", "note"]].to_string(index=False))
