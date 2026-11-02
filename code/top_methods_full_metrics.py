"""Full metric set (accuracy, sensitivity, specificity, precision, F1, BAC, AUC) for our pre-specified strict-protocol (P3) procedures,
re-run with the original functions and seeds of cbramod_eval.py (C1, BENDR, LaBraM, CBraMod, ENS, floor) and round2_eval.py
(C2 Riemannian, C5 ensemble, NESTED). Check: AUC and BAC must equal the earlier reports (cbramod_p3.csv, round2_p3.csv). CPU only."""
import os, numpy as np, pandas as pd, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import evalkit as ek, cbramod_eval as ce, round2_eval as r2
warnings.filterwarnings("ignore"); R = "/home/miftah_f/RTX-5090/tuep_fresh/reports"


def full(y, p):
    q = p > 0.5; tp = (q & (y == 1)).sum(); fp = (q & (y == 0)).sum(); se = q[y == 1].mean(); sp = 1 - q[y == 0].mean(); pr = tp / max(tp + fp, 1)
    return dict(acc=(q == y).mean(), sens=se, spec=sp, prec=pr, f1=2 * pr * se / max(pr + se, 1e-9), bac=(se + sp) / 2, auc=roc_auc_score(y, p))


subjects, y, P3, *_ = ce.tables(); F = ek.folds(); pos = {s: i for i, s in enumerate(subjects)}
jobs = []
for r in range(10):
    for k in range(5):
        te = np.array([pos[s] for s in F[(F.repeat == r) & (F.fold == k)].subject]); jobs.append((r, k, np.setdiff1d(np.arange(len(y)), te), te))
res_ce = Parallel(n_jobs=20)(delayed(ce.procedure)(P3, y, tr, te, 100 * r + k, None, False) for r, k, tr, te in jobs)

T = np.load(f"{ek.D}/p3_patient_table.npz", allow_pickle=True); bp, agesex = T["bandpower"], T["agesex"]
rz = np.load(f"{ek.D}/riemann_p3.npz", allow_pickle=True); assert (rz["subjects"] == subjects).all()
lay = P3["LaBraM"][:, :, :200]
X = {"C1": (np.hstack([bp, agesex]), bp.shape[1], 0), "C2": (np.hstack([rz["X"], agesex]), rz["X"].shape[1], 128)}
for l in range(12):
    X[f"C3_L{l + 1}"] = (np.hstack([lay[:, l], agesex]), 200, 0)
res_r2 = Parallel(n_jobs=10)(delayed(r2.outer_fold)(X, y, tr, te, 100 * r + k) for r, k, tr, te in jobs)

rows = []
for name, getter in [("C1 band power + age/sex", lambda i: res_ce[i][0]["C1"]), ("ENS (C1 + LaBraM + CBraMod + BENDR)", lambda i: res_ce[i][0]["ENS"]),
                     ("C5 (C1 + Riemannian + LaBraM layer)", lambda i: res_r2[i][1]["C5"]), ("NESTED selection (round 2)", lambda i: res_r2[i][1]["NESTED"]),
                     ("BENDR flatten", lambda i: res_ce[i][0]["BENDR"]), ("LaBraM (inner layer)", lambda i: res_ce[i][0]["LaBraM"]),
                     ("CBraMod (inner layer)", lambda i: res_ce[i][0]["CBraMod"]), ("C2 Riemannian", lambda i: res_r2[i][1]["C2"]),
                     ("acquisition-metadata floor", lambda i: res_ce[i][0]["floor"])]:
    per = []
    for r in range(10):
        p = np.zeros(len(y))
        for i, (rr, k, tr, te) in enumerate(jobs):
            if rr == r:
                p[te] = getter(i)
        per.append(full(y, p))
    D = pd.DataFrame(per); rows.append(dict(method=name, **{k: D[k].mean() for k in D}, **{f"{k}_sd": D[k].std() for k in D}))
out = pd.DataFrame(rows); out.to_csv(f"{R}/top_methods_full_metrics.csv", index=False)
old = {**pd.read_csv(f"{R}/cbramod_p3.csv").set_index("config")[["AUC", "BAC"]].T.to_dict(), **pd.read_csv(f"{R}/round2_p3.csv").set_index("cand")[["AUC", "BAC"]].T.to_dict()}
key = {"C1 band power + age/sex": "C1", "ENS (C1 + LaBraM + CBraMod + BENDR)": "ENS", "C5 (C1 + Riemannian + LaBraM layer)": "C5", "NESTED selection (round 2)": "NESTED",
       "BENDR flatten": "BENDR", "LaBraM (inner layer)": "LaBraM", "CBraMod (inner layer)": "CBraMod", "C2 Riemannian": "C2", "acquisition-metadata floor": "floor"}
for r in out.itertuples():
    o = old[key[r.method]]; same = abs(o["AUC"] - r.auc) < 1e-6 and abs(o["BAC"] - r.bac) < 1e-6
    print(f"{'PASS' if same else 'DIFF'} {r.method:38s} acc {100*r.acc:5.1f} sens {100*r.sens:5.1f} spec {100*r.spec:5.1f} prec {100*r.prec:5.1f} "
          f"f1 {100*r.f1:5.1f} bac {100*r.bac:5.1f} auc {100*r.auc:5.1f}  (reported earlier: AUC {100*o['AUC']:.1f}, BAC {100*o['BAC']:.1f})")
