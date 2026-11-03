#!/usr/bin/env python3
"""Round 50, addendum 2 (DESCRIPTIVE, OUTSIDE the plan; written after the third automated-reviewer output, 2026-11-03):
95% subject-level bootstrap CI for the Kural demographic baseline (age + sex; 10 x 5 cross-validation within Kural), which the transfer table reported only as an AUC of 63.4 without a CI.
Same procedure as round12_kural.py (subject S49 excluded -> 99; StratifiedKFold(5) with random_state 2000 + r, r = 0..9; ek.nested_lr on [age, male]);
AUC = mean over repeats of the pooled AUC. CI: 2,000 subject bootstraps (indices drawn with replacement; draws without both classes dropped), 2.5th and 97.5th percentiles, seed 20261103.
Check: the recomputed AUC must equal round12_kural_results.csv (row 'demographics', difference < 1e-9).
Usage: python round50_kural_demografi_ci.py   (output hasil50/kural_demografi_ci.csv)"""
import os, sys, zipfile
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["PYTHONWARNINGS"] = "ignore"
sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import xml.etree.ElementTree as ET
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import evalkit as ek

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "hasil50")
K = "/home/miftah_f/RTX-5090/tuep_fresh/external/kural2021/raw"; R12 = "/home/miftah_f/RTX-5090/tuep_fresh/reports/round12_kural_results.csv"


def kural_labels():
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}; z = zipfile.ZipFile(f"{K}/Demographics_&_gold_standard.xlsx")
    ss = ["".join(t.itertext()) for t in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)]; rows = []
    for r in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).find("m:sheetData", ns).findall("m:row", ns):
        rows.append([("" if c.find("m:v", ns) is None else (ss[int(c.find("m:v", ns).text)] if c.get("t") == "s" else c.find("m:v", ns).text)) for c in r.findall("m:c", ns)])
    assert rows[0] == ["Samples", "Gender", "Age", "Gold Standard"]
    return pd.DataFrame([dict(subject=f"S{int(r[0][1:]):02d}", male=float(r[1] == "M"), age=float(r[2]), y=int(r[3] == "Epileptic")) for r in rows[1:]]).set_index("subject")


if __name__ == "__main__":
    L = kural_labels(); assert len(L) == 100; D = L.drop("S49"); yk = D.y.values; ASk = D[["age", "male"]].values
    assert len(D) == 99 and yk.sum() == 54 and (1 - yk).sum() == 45
    P = np.zeros((10, len(yk)))
    for r in range(10):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=2000 + r).split(ASk, yk):
            P[r, te] = ek.nested_lr().fit(ASk[tr], yk[tr]).predict_proba(ASk[te])[:, 1]
    auc = float(np.mean([roc_auc_score(yk, p) for p in P])); ref = float(pd.read_csv(R12).set_index("method").loc["demographics", "auc"])
    print(f"Kural demographic baseline AUC recomputed {auc:.10f}; stored {ref:.10f}; difference {abs(auc - ref):.2e}", flush=True); assert abs(auc - ref) < 1e-9
    rng = np.random.default_rng(20261103); b = []
    for _ in range(2000):
        i = rng.integers(0, len(yk), len(yk))
        if len(np.unique(yk[i])) == 2:
            b.append(np.mean([roc_auc_score(yk[i], p[i]) for p in P]))
    lo, hi = np.percentile(b, [2.5, 97.5])
    pd.DataFrame([dict(n=len(yk), auc=auc, lo=lo, hi=hi, n_boot=len(b), seed=20261103)]).to_csv(f"{OUT}/kural_demografi_ci.csv", index=False)
    print(f"AUC {auc * 100:.1f} [95% CI {lo * 100:.1f}--{hi * 100:.1f}] (n_boot {len(b)})", flush=True)
