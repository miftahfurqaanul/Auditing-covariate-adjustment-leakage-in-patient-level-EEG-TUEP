#!/usr/bin/env python3
"""Round 28, POST HOC analysis (addendum 2 in PRAREGISTRASI_analisis_tambahan.md). CPU only."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import round28_analisis as A
v, c17, J, Y, N, BP = A.v, A.c17, A.J, A.Y, A.N, A.BP
OUT = A.OUT

# ---- P1: data asli vs distribusi nol
R = pd.read_csv(f"{OUT}/5_hamdan_replikasi.csv"); ref = pd.read_json(f"{OUT}/5_hamdan_acuan_data_asli.json", typ="series")
rows = []
for metric, real in (("pooled", ref["I2_noAS"] - ref["C1_noAS"]), ("within_era", ref["I2_noAS_wera"] - ref["C1_noAS_wera"]), ("within_year_x_device", ref["I2_noAS_wyd"] - ref["C1_noAS_wyd"])):
    suf = {"pooled": "", "within_era": "_wera", "within_year_x_device": "_wyd"}[metric]
    for sc, g in R.groupby("scenario"):
        d = (g[f"I2_noAS{suf}"] - g[f"C1_noAS{suf}"]).values
        rows.append(dict(metric=metric, scenario=sc, real_I2_minus_C1=real, null_mean=d.mean(), null_sd=d.std(ddof=1), null_p95=np.percentile(d, 95), frac_null_ge_real=(d >= real).mean(), excess_over_null_mean=real - d.mean()))
P1 = pd.DataFrame(rows); P1.to_csv(f"{OUT}/6_posthoc_data_asli_vs_nol.csv", index=False); print(P1.round(4).to_string(index=False), flush=True)

# ---- P2: mechanism (design of the normative layer)
BPm = BP; D0 = A.D_I2
AGE_SEX = D0[:, :3]; ERA_ONLY = np.column_stack([D0[:, 0], D0[:, 3:]]); INTERCEPT = D0[:, :1]
DESIGNS = {"intersep": INTERCEPT, "age+sex": AGE_SEX, "era only": ERA_ONLY, "full (intercept, age, sex, era)": D0}
def fold_pred(X, tr, te, D):
    c = tr[Y[tr] == 0]; B = np.linalg.lstsq(D[c], X[c], rcond=None)[0]; sd = (X[c] - D[c] @ B).std(0, ddof=np.linalg.matrix_rank(D[c]))
    return v.lr_pred((X - D @ B) / sd, tr, te)
def rep_auc(X, D):
    out = Parallel(n_jobs=12)(delayed(fold_pred)(X, tr, te, D) for _, _, tr, te in J)
    return c17.pooled_auc(c17.to_matrix(J, out))
rows = []
for name, D in DESIGNS.items():
    vals = []
    for r in range(100):
        Xs = A.shuffle_cols(BP, [np.arange(N)], np.random.default_rng(28700 + r)); vals.append(rep_auc(Xs, D))
    vals = np.array(vals); rows.append(dict(design=name, n_cols=D.shape[1], mean=vals.mean(), sd=vals.std(ddof=1), p5=np.percentile(vals, 5), p95=np.percentile(vals, 95), frac_gt_052=(vals > 0.52).mean()))
    print(rows[-1], flush=True)
real = {name: rep_auc(BP, D) for name, D in DESIGNS.items()}
M = pd.DataFrame(rows); M["real_data_AUC"] = [real[n] for n in M.design]; M.to_csv(f"{OUT}/6_posthoc_mekanisme_desain.csv", index=False); print(M.round(4).to_string(index=False))
