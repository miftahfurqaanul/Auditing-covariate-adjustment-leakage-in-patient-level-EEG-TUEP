#!/usr/bin/env python3
"""Round 30: evaluation of F2 without the leak (F2-adj = mean of I2-adj, P3-adj, NM-adj). See PRAREGISTRASI_putaran30.md section 3. CPU only.
Usage: CUDA_VISIBLE_DEVICES="" python round30_eval.py [--smoke]   (--smoke uses the old NM instead of NM-adj, only to test the code)"""
import os, sys, json, hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import round28_analisis as A, round29_sensitivitas as S
import round14 as r14, round10 as r10, round9 as r9, round21_age as r21, round4 as r4, round17_common as c17

SMOKE = "--smoke" in sys.argv
OUT = os.path.join(HERE, "hasil30" if not SMOKE else "hasil30/smoke"); os.makedirs(OUT, exist_ok=True)
Y, N, J, SUBJ, ERA = A.Y, A.N, A.J, c17.SUBJ, A.ERA; POS = {s: i for i, s in enumerate(SUBJ)}


def load_nm_v4():
    nm = np.full((10, N), np.nan)
    for r, k, _, te in J:
        d = json.load(open(os.path.join(HERE, "hasil30", "nm_v4", f"r{r}_k{k}.json")))["test"]; nm[r, te] = [d[s] for s in SUBJ[te]]
    return nm


P0 = A.load_models(); Z29 = np.load(os.path.join(HERE, "hasil29", "B_prediksi_data_asli.npz")); P3VAR = os.environ.get("P3VAR", "V5"); Z30 = np.load(os.path.join(HERE, "hasil30", "p3_prediksi_data_asli.npz" if P3VAR == "V4" else "p3_v5_prediksi_data_asli.npz"))
NMV4 = P0["NM"] if SMOKE else load_nm_v4()
rr = np.load(f"{c17.OUT}/p3.npz")["Riemann_raw"]
mamba_raw = np.full((10, N), np.nan)
for r, k, _, te in J:
    d = json.load(open(f"{S.ek.D}/round16/r{r}_k{k}.json"))["raw"]["test"]; mamba_raw[r, te] = [d[s] for s in SUBJ[te]]
AGEPRED = np.load(f"{S.ek.D}/round21/age_preds.npz")
P = {"I2-adj": Z29["V4_AS"], "P3-adj": Z30[f"P3_{P3VAR}_AS"], "NM-adj": NMV4, "I2": P0["I2"], "P3": P0["P3"], "NM": P0["NM"], "F2-lama": P0["F2"], "C1": P0["C1"], "Floor": P0["Floor"],
     "Riemann-mentah": rr, "Mamba-mentah": mamba_raw, "LaBraM-FT": P0["LaBraM-FT"], "CBraMod-FT": P0["CBraMod-FT"], "Demo": AGEPRED["Demo"]}
P["F2-adj"] = (P["I2-adj"] + P["P3-adj"] + P["NM-adj"]) / 3
assert abs(c17.pooled_auc(P["F2-lama"]) - 0.72709) < 1e-5 and abs(c17.pooled_auc(P["I2-adj"]) - 0.71210 + 0.0) < 0.05
for k, m in P.items():
    assert m.shape == (10, N) and not np.isnan(m).any(), k
FA = {k: c17.fold_aucs(J, m) for k, m in P.items()}


def holm(p):
    return A.holm(p)


# ---------- metrik
rows = []
CIs = dict(zip(["I2-adj", "P3-adj", "NM-adj", "F2-adj", "F2-lama", "C1"], Parallel(n_jobs=6)(delayed(r14.boot)(P[m], 31000 + i) for i, m in enumerate(["I2-adj", "P3-adj", "NM-adj", "F2-adj", "F2-lama", "C1"]))))
for m in ["I2-adj", "P3-adj", "NM-adj", "F2-adj", "F2-lama", "C1", "Floor", "Riemann-mentah", "Mamba-mentah", "LaBraM-FT", "CBraMod-FT"]:
    pt = r14.rep_metrics(P[m], np.arange(N)); fa = FA[m]; se = np.sqrt(fa.var(ddof=1) * (1 / 50 + 40 / 160)); tc = stats.t.ppf(0.975, 49)
    row = dict(model=m, auc_fold=100 * fa.mean(), fold_lo=100 * (fa.mean() - tc * se), fold_hi=100 * (fa.mean() + tc * se), sd_fold=100 * fa.std(ddof=1), auc_pooled=100 * pt.auc)
    if m in CIs:
        row.update(pooled_lo=100 * CIs[m].loc[0.025, "auc"], pooled_hi=100 * CIs[m].loc[0.975, "auc"], brier_lo=CIs[m].loc[0.025, "brier"], brier_hi=CIs[m].loc[0.975, "brier"], slope_lo=CIs[m].loc[0.025, "calib_slope"], slope_hi=CIs[m].loc[0.975, "calib_slope"])
    row.update(sens=100 * pt.sens, spec=100 * pt.spec, bac=100 * pt.bac, brier=pt.brier, slope=pt.calib_slope); rows.append(row)
M1 = pd.DataFrame(rows); M1.to_csv(f"{OUT}/eval_metrik.csv", index=False); print(M1.round(3).to_string(index=False), flush=True)

# ---------- per-fold difference
def nbd(a, b):
    d, lo, hi, p = S.nb_diff(FA[a], FA[b]); return dict(test=f"{a} - {b}", diff=d, lo=lo, hi=hi, p=p, folds_better=int((FA[a] > FA[b]).sum()))
pairs = [("F2-adj", "I2-adj"), ("F2-adj", "C1"), ("F2-adj", "F2-lama"), ("I2-adj", "C1"), ("P3-adj", "Riemann-mentah"), ("NM-adj", "Mamba-mentah"), ("P3-adj", "P3"), ("NM-adj", "NM"), ("I2-adj", "I2"),
         ("F2-adj", "Floor"), ("F2-adj", "LaBraM-FT"), ("F2-adj", "CBraMod-FT")]
T = pd.DataFrame([nbd(a, b) for a, b in pairs])
comb = (FA["I2-adj"] - FA["C1"] + FA["P3-adj"] - FA["Riemann-mentah"] + FA["NM-adj"] - FA["Mamba-mentah"]) / 3
d, lo, hi, p = S.nb_diff(comb, np.zeros(50)); T = pd.concat([T, pd.DataFrame([dict(test="efek gabungan (V4 - mentah, rata-rata 3 pandangan)", diff=d, lo=lo, hi=hi, p=p, folds_better=int((comb > 0).sum()))])], ignore_index=True)
T["family"] = "descriptive"; T["p_holm"] = np.nan
fm = T.test.isin(["F2-adj - LaBraM-FT", "F2-adj - CBraMod-FT"]); T.loc[fm, "family"] = "primary"; T.loc[fm, "p_holm"] = holm(T.loc[fm, "p"].values)
T["q_bh_semua_uji_baru"] = stats.false_discovery_control(np.nan_to_num(T.p.values, nan=1.0)); T.to_csv(f"{OUT}/eval_selisih_per_lipatan.csv", index=False); print(T.round(4).to_string(index=False), flush=True)
sens = pd.Series(c17.pooled_auc(P["F2-adj"][:, :]), name="x")

# ---------- within strata
age3 = r21.AGE3; strata = dict(r9.STRATA); strata["year x montage"] = c17.STRATA["year x montage"]; strata["age3"] = age3
Pw = {k: P[k] for k in ["F2-adj", "I2-adj", "P3-adj", "NM-adj", "C1", "Floor", "LaBraM-FT", "CBraMod-FT", "Demo"]}
tests = [("F2-adj", "Floor"), ("F2-adj", "C1"), ("F2-adj", "I2-adj"), ("I2-adj", "Floor"), ("I2-adj", "C1"), ("F2-adj", "LaBraM-FT"), ("F2-adj", "CBraMod-FT"), ("F2-adj", "Demo")]
W = r10.within(Pw, np.arange(N), strata, tests, 31500); W.to_csv(f"{OUT}/eval_dalam_strata.csv", index=False)
print(W[["strata", "test", "diff", "lo", "hi", "p"]].round(4).to_string(index=False), flush=True)

# ---------- sensitivity without era <= 2009
keep = ERA != "<=2009"; rows = []
for m in ["F2-adj", "I2-adj", "P3-adj", "NM-adj", "F2-lama", "C1", "LaBraM-FT", "CBraMod-FT"]:
    rows.append(dict(model=m, auc_200=100 * c17.pooled_auc(P[m]), auc_169=100 * np.mean([roc_auc_score(Y[keep], p[keep]) for p in P[m]])))
pd.DataFrame(rows).to_csv(f"{OUT}/eval_tanpa_era2009.csv", index=False); print(pd.DataFrame(rows).round(2).to_string(index=False), flush=True)

# ---------- confounder test
S29 = {m: P[m].mean(0) for m in ["F2-adj", "I2-adj", "P3-adj", "NM-adj", "F2-lama", "C1", "Floor", "LaBraM-FT", "CBraMod-FT"]}; rp, rf, i = [], [], 0
for m, s in S29.items():
    for cn, (c, kind) in A.CONF_PARTIAL.items():
        i += 1; r2, p = A.partial_test(s, c, kind, 31200 + i); rp.append(dict(model=m, confounder=cn, R2=r2, p=p))
    for cn, c in A.CONF_FULL.items():
        i += 1; r2, p = A.full_test(s, c, 31200 + i); rf.append(dict(model=m, confounder=cn, R2=r2, p=p))
Cp, Cf = pd.DataFrame(rp), pd.DataFrame(rf)
for X in (Cp, Cf):
    X["family"] = "descriptive"; X["p_holm"] = np.nan
msk = Cp.model.isin(["F2-adj", "NM-adj"]) & (Cp.confounder == "era"); Cp.loc[msk, "family"] = "primary"; Cp.loc[msk, "p_holm"] = holm(Cp.loc[msk, "p"].values)
Cp.to_csv(f"{OUT}/eval_confounder_parsial.csv", index=False); Cf.to_csv(f"{OUT}/eval_confounder_penuh.csv", index=False); print(Cp.round(4).to_string(index=False)); print(Cf.round(4).to_string(index=False), flush=True)

# ---------- correlation, decision curve
Pc = {"I2": P["I2-adj"], "P3": P["P3-adj"], "NM": P["NM-adj"], "Floor": P["Floor"]}; K, KD = A.a1_korelasi(Pc); K.to_csv(f"{OUT}/eval_korelasi.csv", index=False); KD.to_csv(f"{OUT}/eval_korelasi_deskriptif.csv", index=False); print(K.round(3).to_string(index=False))
Pn = {"F2": P["F2-adj"], "I2": P["I2-adj"], "Floor": P["Floor"]}; NB = A.nb_boot(Pn); NB.to_csv(f"{OUT}/eval_net_benefit.csv", index=False); NBP = A.nb_prev(Pn); NBP.to_csv(f"{OUT}/eval_net_benefit_prevalensi.csv", index=False)
print(NB[["pt", "NB_F2", "NB_I2", "NB_Floor", "d_F2-Floor_lo", "d_F2-Floor_hi", "d_F2-max(all,none)_lo", "d_F2-max(all,none)_hi"]].round(4).to_string(index=False))
np.savez(f"{OUT}/eval_prediksi_v4.npz", **{k.replace("-", "_"): v for k, v in P.items()})
json.dump(dict(smoke=SMOKE, code_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest()), open(f"{OUT}/meta_eval.json", "w"))
print("done")
