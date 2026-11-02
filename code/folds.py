"""Canonical patient-level folds shared by every method (10 repeats x 5 stratified folds).

Every model in this project is evaluated on exactly these splits, so differences between
methods are paired comparisons on the same test patients.
"""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
D = "/home/miftah_f/RTX-5090/tuep_fresh/data"
N_REPEATS, N_FOLDS = 10, 5

if __name__ == "__main__":
    idx = pd.read_csv(f"{D}/index.csv")
    pts = idx.groupby("subject").label.first().sort_index()
    rows = []
    for r in range(N_REPEATS):
        cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=1000 + r)
        for k, (_, te) in enumerate(cv.split(pts.index, pts.values)):
            rows += [dict(subject=s, label=int(pts[s]), repeat=r, fold=k) for s in pts.index[te]]
    f = pd.DataFrame(rows)
    f.to_csv(f"{D}/folds_patient.csv", index=False)
    print(f.groupby(["repeat", "fold"]).label.agg(["count", "sum"]).head(5))
    print("patients:", f.subject.nunique(), "| rows:", len(f))
