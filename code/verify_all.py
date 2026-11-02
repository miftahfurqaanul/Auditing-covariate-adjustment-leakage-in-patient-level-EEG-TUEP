"""Final re-check of the experiment fixes (requested 2026-10-10). Every check prints PASS or FAIL. CPU only.
1 pre-registration order in git; 2 round-3 split integrity; 3 independent recomputation of headline numbers from saved predictions;
4 determinism (re-runs must match exactly); 5 EDF checksums; 6 numbers quoted in the reports match the result tables;
7-8 rounds 7-10; 9 rounds 16-26 incl. 25 (added 2026-10-12/14): pre-registration order, independent recomputation from saved predictions, re-runs, quoted numbers;
10 external transfer (Kural, Emotiv) in the tables and documents, and the post hoc Emotiv registry sensitivity (added 2026-10-14).
Note: section 9 reads prediction files under data/round17, data/round19, data/round20, data/round21, data/round24, data/round26 (.npz, not tracked by git) and data/round16, data/round19 (.json, tracked)."""
import os, glob, json, hashlib, subprocess, numpy as np, pandas as pd, warnings
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
ROOT = "/home/miftah_f/RTX-5090/tuep_fresh"; R = f"{ROOT}/reports"; D = f"{ROOT}/data"; FAILS = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def first_added(path):
    out = subprocess.run(["git", "-C", ROOT, "log", "--diff-filter=A", "--format=%at", "--", path], capture_output=True, text=True).stdout.split()
    return int(out[-1]) if out else None


# 1 ------------------------------------------------------------ pre-registration order
print("== 1. pre-registration committed before results ==")
for pred, res in [("matched_prediction.md", "matched_results.txt"), ("rohira_repro_prediction.md", "rohira_results.txt"),
                  ("tang_fixed_prediction.md", "tang_fixed_results.txt"), ("rohira_fixed_prediction.md", "rohira_fixed_results.txt"),
                  ("eegbench_fixed_prediction.md", "eegbench_fixed_results.txt"), ("round3_prediction.md", "round3_confound.txt"),
                  ("round3_prediction.md", "round3_results_fc_matched_temporal.csv"), ("round4_prediction.md", "round4_results.txt"),
                  ("round4_prediction.md", "round4_posthoc.txt"), ("round5_prediction.md", "round5_results.txt"),
                  ("round6_atm_prediction.md", "round6_results.txt"),
                  ("round7_prediction.md", "round7_results.txt"), ("round8_zanardini_prediction.md", "round8_results.txt"), ("round9_prediction.md", "round9_results.txt"),
                  ("round10_prediction.md", "round10_results.txt")]:
    a, b = first_added(f"reports/{pred}"), first_added(f"reports/{res}")
    check(a is not None and b is not None and a < b, f"{pred} committed before {res}")
print("note: analyses before 2026-10-09 (no git yet) cannot be verified this way; listed as a limitation")

# 2 ------------------------------------------------------------ round-3 split integrity
print("\n== 2. round-3 splits ==")
import round3 as r3
for prot in ["fc", "matched", "temporal"]:
    S = {n: (tr, va, te) for n, tr, va, te in r3.splits(prot)}; bad = 0; n = 0
    for p in glob.glob(f"{r3.OUT}/{prot}/*/*.json"):
        d = json.load(open(p)); tr, va, te = S[os.path.basename(p)[:-5]]; n += 1
        bad += not (set(d["test"]) == set(te) and set(d["val"]) == set(va) and d["train"] == len(tr))
    check(bad == 0 and n > 0, f"{prot}: {n} result files use exactly the defined train/val/test patients")

# 3 ------------------------------------------------------------ independent recomputation from saved predictions
print("\n== 3. headline numbers recomputed with independent code ==")
T = np.load(f"{D}/p3_patient_table.npz", allow_pickle=True); YL = pd.Series(T["y"], index=T["subjects"])
def pooled_auc(files):
    by = {}
    for p in files:
        d = json.load(open(p)); r = os.path.basename(p).split("_")[0]; by.setdefault(r, {}).update(d["test"])
    return np.mean([roc_auc_score(YL.loc[list(v)].values, list(v.values())) for v in by.values()])
tf_auc = pooled_auc(glob.glob(f"{D}/tang/fixed/r*_f*.json")); rep = pd.read_csv(f"{R}/tang_fixed_results.csv")["Tang-F|AUC"].mean()
check(abs(tf_auc - rep) < 1e-9, f"Tang-F AUC {tf_auc:.4f} == reported {rep:.4f}")
r3c = pd.read_csv(f"{R}/round3_results_fc_matched_temporal.csv")
for prot, m in [("fc", "C1-V"), ("fc", "Tang-F-T"), ("fc", "Rohira-F-T"), ("matched", "M-EEG-V")]:
    a = pooled_auc(glob.glob(f"{r3.OUT}/{prot}/{m}/*.json")); b = r3c[(r3c.protocol == prot) & (r3c.method == m)].auc.iloc[0]
    check(abs(a - b) < 1e-9, f"round 3 {prot} {m} AUC {a:.4f} == reported {b:.4f}")
import tang_eval as tev
M = tev.meta(250); P = np.full(len(M), np.nan)
for f in glob.glob(f"{D}/tang/r2_250/fold*.npz"):
    z = np.load(f); P[z["te"]] = z["p"]
per = M.assign(c=(P > 0.5) == M.label.values).groupby("subj").agg(y=("label", "first"), a=("c", "mean"))
check(abs(100 * per[per.y == 1].a.mean() - 68.79) < 0.005, f"Tang R2-250 epilepsy per-patient accuracy {100 * per[per.y == 1].a.mean():.2f} == 68.79")

# 4 ------------------------------------------------------------ determinism: re-runs must equal saved results
print("\n== 4. re-runs reproduce saved results exactly ==")
import matched_eval as me
a = me.procedure(me.X["M-EEG"], me.y)[0]; check(abs(a - pd.read_csv(f"{R}/matched_results.csv", index_col=0).loc["M-EEG", "AUC"]) < 1e-12, f"matched M-EEG AUC re-run {a:.4f}")
import eegbench_fixed_eval as ebf
z = np.load(f"{D}/eegbench_rep_D.npz"); rec = z["rec"]; X = z["labram"].astype(np.float32); s = ebf.subj_r[rec]
Xp = np.vstack([X[ebf.first[rec] & (s == p)].mean(0) for p in ebf.subjects]); a = ebf.p3(Xp)[0].auc.mean()
b = pd.read_csv(f"{R}/eegbench_fixed_results.csv").query("variant == 'D' and encoder == 'labram'").p3_auc.iloc[0]
check(abs(a - b) < 1e-12, f"EEG-Bench-F D LaBraM P3 AUC re-run {a:.4f} == {b:.4f}")
for name, tr, va, te in [x for x in r3.splits("fc") if x[0] in ("r3_f1", "r7_f4")]:
    new = r3.lr_method("C1-V", tr, va, te)["test"]; old = json.load(open(f"{r3.OUT}/fc/C1-V/{name}.json"))["test"]
    check(max(abs(new[k] - old[k]) for k in old) < 1e-12, f"round 3 C1-V {name} re-run identical")
    new = r3.rohira_method(tr, va, te, 7 * [x[0] for x in r3.splits("fc")].index(name)); old = json.load(open(f"{r3.OUT}/fc/Rohira-F-T/{name}.json"))
    check(new["chosen"]["max_features"] == old["chosen"]["max_features"] and max(abs(new["test"][k] - old["test"][k]) for k in old["test"]) < 1e-12,
          f"round 3 Rohira-F-T {name} re-run identical")

import round4_posthoc as r4p
a = r4p.i2_auc(r4p.r4.YEAR_G); check(abs(a - pd.read_csv(f"{R}/round4_results.csv", index_col=0).loc["I2", "auc"]) < 1e-12, f"round 4 I2 AUC re-run {a:.4f} identical")

# 5 ------------------------------------------------------------ EDF checksums
print("\n== 5. EDF checksums ==")
md = pd.read_csv(f"{D}/edf_md5.csv").sample(30, random_state=0); bad = 0
for r in md.itertuples():
    h = hashlib.md5(open(f"/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1/{r.path}", "rb").read()).hexdigest(); bad += h != r.md5
check(bad == 0, f"30 random EDF files match data/edf_md5.csv ({len(pd.read_csv(f'{D}/edf_md5.csv'))} files listed)")

# 6 ------------------------------------------------------------ numbers in reports == tables
print("\n== 6. numbers quoted in the reports ==")
fmt = lambda v, d=3: f"{v:.{d}f}".replace(".", ",")
txt = lambda f: open(f"{R}/{f}").read()
pairs = [("tang_fixed_prediction.md", fmt(pd.read_csv(f"{R}/tang_fixed_results.csv")["Tang-F|AUC"].mean())),
         ("rohira_fixed_prediction.md", fmt(100 * pd.read_csv(f"{R}/rohira_fixed_setA.csv")["Rohira-F|AUC"].mean(), 1)),
         ("eegbench_fixed_prediction.md", fmt(100 * pd.read_csv(f"{R}/eegbench_fixed_results.csv").query("variant=='D' and encoder=='labram'").p3_auc.iloc[0], 1)),
         ("matched_prediction.md", fmt(pd.read_csv(f"{R}/matched_results.csv", index_col=0).loc["M-EEG", "AUC"])),
         ("rohira_repro_prediction.md", fmt(pd.read_csv(f"{R}/rohira_results.csv").query("set=='A' and proto=='W'").acc.mean(), 2)),
         ("round4_prediction.md", fmt(100 * pd.read_csv(f"{R}/round4_results.csv", index_col=0).loc["I2", "auc"], 2)),
         ("round4_prediction.md", fmt(100 * pd.read_csv(f"{R}/round4_results.csv", index_col=0).loc["C1", "auc"], 2)),
         ("round5_prediction.md", fmt(100 * pd.read_csv(f"{R}/round5_results.csv", index_col=0).loc["H5a", "auc"], 2)),
         ("round6_atm_prediction.md", fmt(100 * pd.read_csv(f"{R}/round6_results.csv", index_col=0).loc["DNM-F", "auc"], 2))]
for f, s in pairs:
    check(s in txt(f), f"{f} quotes {s}")

# 7 ------------------------------------------------------------ rounds 7-9 statistics recomputed from saved artifacts (independent code)
print("\n== 7. rounds 7-9 recomputed ==")
from scipy import stats as _st
def _nb(a, b, n1=160, n2=40):
    d = np.asarray(a) - np.asarray(b); J = d.size; t = d.mean() / np.sqrt((1 / J + n2 / n1) * d.var(ddof=1)); return 2 * _st.t.sf(abs(t), J - 1)
fa = np.load(f"{R}/round7_fold_aucs.npz"); T7 = pd.read_csv(f"{R}/round7_tests.csv")
pairs7 = {"I2 vs Tang-F (199)": ("I2@199", "Tang-F@199"), "I2 vs Rohira-F": ("I2", "Rohira-F"), "I2 vs DNM-F": ("I2", "DNM-F"),
          "I2 vs BENDR": ("I2", "BENDR"), "I2 vs LaBraM": ("I2", "LaBraM"), "I2 vs metadata floor": ("I2", "Floor")}
pr = np.array([_nb(fa[a], fa[b]) for a, b in [pairs7[t] for t in T7.test]]); o = np.argsort(pr); hm = np.empty(6); run = 0
for k, i in enumerate(o):
    run = max(run, min(1, pr[i] * (6 - k))); hm[i] = run
check(np.allclose(pr, T7.p, rtol=1e-9) and np.allclose(hm, T7.p_holm, rtol=1e-9), "round 7: six corrected t-tests and Holm recomputed from saved fold AUCs")
r8 = open(f"{R}/round8_results.txt").read()
for cond in ["rest", "ips"]:
    nz = np.load(f"{R}/round8_{cond}_null.npz"); cfg = pd.read_csv(f"{R}/round8_{cond}_configs.csv"); best = cfg.auc.max()
    pv = (1 + (nz["best"] >= best).sum()) / 51
    check(f"best config {cfg.config[cfg.auc.idxmax()]} AUC {best:.3f}" in r8 and f"observed {best:.3f}, p = {pv:.3f}" in r8, f"round 8 {cond}: best config AUC {best:.3f} and null p {pv:.3f} recomputed")
C = pd.read_csv(f"{R}/round9_C_speccurve.csv"); pv9 = C.pivot_table(index=["feats", "cov", "clf"], columns="harm", values="auc")
npos = int((pv9["age+sex+year"] > pv9["none"]).sum()); check(npos == 12 and abs(_st.binomtest(npos, 12).pvalue - 0.000488) < 1e-5, f"round 9: year harmonisation positive in {npos}/12 matched specifications")
A9 = pd.read_csv(f"{R}/round9_A_tests.csv"); check(abs(A9.query("strata=='year' and test=='I2 - Floor'").p_holm.iloc[0] - 0.015) < 1e-9, "round 9: within-year I2 - floor Holm p 0.015")
for f, q in [("round7_prediction.md", "71,21"), ("round7_prediction.md", "0,015"), ("round8_zanardini_prediction.md", "85,5"), ("round8_zanardini_prediction.md", "48,3"),
             ("round9_prediction.md", "0,688"), ("round9_prediction.md", "12/12")]:
    check(q in txt(f), f"{f} quotes {q}")

# 8 ------------------------------------------------------------ round 10
print("\n== 8. round 10 recomputed ==")
NF = pd.read_csv(f"{R}/round10_nested_folds.csv"); cands = ["C1", "I1", "I2", "I2b", "H5a", "H5b", "DNM-F", "DNM-M"]
re_best = [cands[int(np.argmax(row))] for row in NF[[f"inner_{m}" for m in cands]].values]
check(re_best == list(NF.best) and len(NF) == 50, "round 10: selected candidate == argmax of saved inner AUCs in all 50 outer folds")
check(NF.best.value_counts().to_dict() == {"I2": 23, "H5b": 15, "I2b": 5, "H5a": 3, "DNM-F": 2, "C1": 1, "I1": 1}, "round 10: selection frequencies match the report")
for f_ in ["round10_within_200.csv", "round10_within_169.csv"]:
    W = pd.read_csv(f"{R}/{f_}"); o = np.argsort(W.p.values); hm = np.empty(len(W)); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1, W.p.values[i] * (len(W) - k))); hm[i] = run
    check(np.allclose(hm, W.p_holm) and ((W["diff"] >= W.lo) & (W["diff"] <= W.hi)).all(), f"{f_}: Holm recomputed; every point estimate lies inside its bootstrap CI")
for f, q in [("round10_prediction.md", "69,46"), ("round10_prediction.md", "0,056"), ("round9_prediction.md", "Koreksi"), ("limitations_and_tripod.md", "e078378")]:
    check(q in txt(f), f"{f} quotes {q}")
# 9 ------------------------------------------------------------ rounds 16-21 (added 2026-10-12): independent recomputation from saved predictions
print("\n== 9. rounds 16-21 ==")
for pred, res in [("round16_normmamba_prediction.md", "round16_results.txt"), ("round17_P1-P5_prediction.md", "round17_results.csv"),
                  ("round17_implementation_notes.md", "round17_results.csv"), ("round18_prediction.md", "round18_results.txt"),
                  ("round19_prediction.md", "round19_results.txt"), ("round20_prediction.md", "round20_harmonisation.log"), ("round21_prediction.md", "round21_age.log")]:
    a, b = first_added(f"reports/{pred}"), first_added(f"reports/{res}")
    check(a is not None and b is not None and a < b, f"{pred} committed before {res}")
for pred, code in [("round17_implementation_notes.md", "code/round17_p1.py"), ("round19_prediction.md", "code/round19_nm_inner.py"),
                   ("round20_prediction.md", "code/round20_harmonisation.py"), ("round21_prediction.md", "code/round21_age.py")]:
    a, b = first_added(f"reports/{pred}"), first_added(code)
    check(a is not None and b is not None and a < b, f"{pred} committed before {code}")
_T = np.load(f"{D}/p3_patient_table.npz", allow_pickle=True); _S = list(_T["subjects"]); _y = _T["y"]; _pos = {s_: i for i, s_ in enumerate(_S)}
_F = pd.read_csv(f"{D}/folds_patient.csv"); _JOBS = []
for _r in range(10):
    for _k in range(5):
        _te = np.array([_pos[s_] for s_ in _F[(_F.repeat == _r) & (_F.fold == _k)].subject]); _JOBS.append((_r, _k, np.setdiff1d(np.arange(200), _te), _te))
_pooled = lambda P: float(np.mean([roc_auc_score(_y[~np.isnan(p)], p[~np.isnan(p)]) for p in P]))
_fold = lambda P: np.array([roc_auc_score(_y[te], P[r, te]) for r, _, _, te in _JOBS])
def _holm(p):
    p = np.asarray(p, float); o = np.argsort(p); h = np.empty(len(p)); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1, p[i] * (len(p) - k))); h[i] = run
    return h
def _nm(variant, folder="round16", key=None):
    P = np.full((10, 200), np.nan)
    for r, k, tr, te in _JOBS:
        d = json.load(open(f"{D}/{folder}/r{r}_k{k}.json"))[variant]["test"]; P[r, te] = [d[_S[i]] for i in te]
    return P
# round 16
T16 = pd.read_csv(f"{R}/round16_results.csv", index_col=0)
for v, name in [("F", "NormMamba-F"), ("S", "NormMamba-S"), ("raw", "Mamba-raw")]:
    a = _pooled(_nm(v)); check(abs(a - T16.loc[name, "auc"]) < 1e-9, f"round 16: {name} pooled AUC {a:.5f} recomputed from the 50 fold JSON files")
# round 17
L17 = {i: np.load(f"{D}/round17/p{i}.npz") for i in range(1, 6)}
M17 = {"I2-repro": (1, "I2"), "C1-repro": (1, "C1"), "P1": (1, "P1"), "P2": (2, "P2"), "P2ekor-saja": (2, "P2ekor"), "P2-tanpa-beta-gamma": (2, "P2_noBG"),
       "P3": (3, "P3"), "P3-global": (3, "P3_global"), "Riemann-tanpa-normatif-repro": (3, "Riemann_raw"), "P4": (4, "P4"), "P4-ens": (4, "P4_ens"),
       "P4-tanpa-total": (4, "P4_nototal"), "P5-adapter": (5, "P5"), "P5-global": (5, "P5_global"), "P5-whiten": (5, "P5_whiten"),
       "LaBraM-median": (5, "LaBraM_median"), "LaBraM-mentah-repro": (5, "LaBraM_raw")}
T17 = pd.read_csv(f"{R}/round17_results.csv").set_index("method")
bad = [m for m, (i, k_) in M17.items() if abs(_pooled(L17[i][k_]) - T17.loc[m, "auc"]) > 1e-9]
check(not bad, f"round 17: pooled AUC of {len(M17)} methods recomputed from saved predictions {'(mismatch: ' + str(bad) + ')' if bad else ''}")
F17 = {m: _fold(L17[i][k_]) for m, (i, k_) in M17.items()}
prim17 = [("P1", "I2-repro"), ("P2", "I2-repro"), ("P3", "I2-repro"), ("P4", "I2-repro"), ("P5-adapter", "LaBraM-mentah-repro")]
p17 = [_nb(F17[a_], F17[b_]) for a_, b_ in prim17]
check(np.allclose(p17, [T17.loc[a_, "p"] for a_, _ in prim17], rtol=1e-9) and np.allclose(_holm(p17), [T17.loc[a_, "p_holm"] for a_, _ in prim17], rtol=1e-9),
      "round 17: five primary corrected t-tests and Holm recomputed from saved predictions")
# round 18
I2_, P3_, NM_ = L17[1]["I2"], L17[3]["P3"], _nm("F"); F2_ = (I2_ + P3_ + NM_) / 3
T18 = pd.read_csv(f"{R}/round18_results.csv").set_index("method")
check(abs(_pooled(F2_) - T18.loc["F2", "auc"]) < 1e-9 and abs(_pooled(F2_) - 0.72709) < 1e-5, f"round 18: F2 pooled AUC {_pooled(F2_):.5f} recomputed (I2 + P3 + NormMamba-F)")
F18 = {"F1": _fold((I2_ + P3_) / 2), "F2": _fold(F2_), "F3": _fold((I2_ + L17[5]["P5"]) / 2), "I2": _fold(I2_)}
p18 = [_nb(F18[a_], F18["I2"]) for a_ in ("F1", "F2", "F3")]
check(np.allclose(p18, [T18.loc[a_, "p"] for a_ in ("F1", "F2", "F3")], rtol=1e-9) and np.allclose(_holm(p18), [T18.loc[a_, "p_holm"] for a_ in ("F1", "F2", "F3")], rtol=1e-9),
      "round 18: three primary tests and Holm recomputed")
# round 19 (Platt re-implemented with statsmodels GLM)
import statsmodels.api as _sm
_lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
O19 = np.load(f"{D}/round19/cpu_inner.npz"); CF2 = np.full((10, 200), np.nan)
for f_, (r, k, tr, te) in enumerate(_JOBS):
    nmi = json.load(open(f"{D}/round19/nm_inner_r{r}_k{k}.json"))["oof"]
    inner = {"I2": O19["I2"][f_, tr], "P3": O19["P3"][f_, tr], "NM": np.array([nmi[_S[i]] for i in tr])}; test = {"I2": I2_[r, te], "P3": P3_[r, te], "NM": NM_[r, te]}
    CF2[r, te] = np.mean([_sm.GLM(_y[tr], _sm.add_constant(_lg(inner[m])), family=_sm.families.Binomial()).fit().predict(_sm.add_constant(_lg(test[m]))) for m in inner], 0)
T19 = pd.read_csv(f"{R}/round19_results.csv").set_index("method")
# tolerance 1e-4: sklearn (lbfgs) and statsmodels (IRLS) Platt fits differ by up to 1.2e-4 in calibrated probability, enough to swap a near-tied pair
check(abs(_pooled(CF2) - T19.loc["CF2", "auc"]) < 1e-4, f"round 19: CF2 pooled AUC {_pooled(CF2):.5f} vs saved {T19.loc['CF2', 'auc']:.5f} recomputed with an independent Platt implementation (solver tolerance 1e-4)")
# round 20
P20 = np.load(f"{D}/round20/harmonisation_preds.npz"); T20 = pd.read_csv(f"{R}/round20_harmonisation_results.csv").set_index("method"); TT20 = pd.read_csv(f"{R}/round20_harmonisation_tests.csv")
bad = [m for m in ("I2", "C1", "H-olsall", "H-zall", "H-combat", "P3", "P3-all") if abs(_pooled(P20[m]) - T20.loc[m, "auc"]) > 1e-9]
check(not bad, f"round 20: pooled AUC of 7 methods recomputed from saved predictions {'(mismatch: ' + str(bad) + ')' if bad else ''}")
prim20 = [("I2", "H-olsall"), ("I2", "H-zall"), ("I2", "H-combat"), ("P3", "P3-all")]; p20 = [_nb(_fold(P20[a_]), _fold(P20[b_])) for a_, b_ in prim20]
t20 = TT20.set_index("test"); check(np.allclose(p20, [t20.loc[f"{a_} - {b_}", "p"] for a_, b_ in prim20], rtol=1e-9) and np.allclose(_holm(p20), [t20.loc[f"{a_} - {b_}", "p_holm"] for a_, b_ in prim20], rtol=1e-9),
      "round 20: four primary tests and Holm recomputed")
# round 21 (demographic floor refitted with an independent pipeline; within-age-stratum AUC with an independent pair counter)
from sklearn.pipeline import make_pipeline as _mp
from sklearn.preprocessing import StandardScaler as _SS
from sklearn.linear_model import LogisticRegressionCV as _LRCV
P21 = np.load(f"{D}/round21/age_preds.npz"); AS_ = _T["agesex"]; Dm = np.full((10, 200), np.nan)
for r, k, tr, te in _JOBS:
    Dm[r, te] = _mp(_SS(), _LRCV(Cs=np.logspace(-3, 1, 9), cv=5, scoring="roc_auc", class_weight="balanced", max_iter=5000)).fit(AS_[tr], _y[tr]).predict_proba(AS_[te])[:, 1]
check(np.allclose(Dm, P21["Demo"]), f"round 21: demographic floor refitted independently (pooled AUC {_pooled(Dm):.5f})")
_age = AS_[:, 0]; _g = np.where(_age <= 40, 0, np.where(_age <= 60, 1, 2))
def _strat(p):
    num = den = 0.0
    for s_ in range(3):
        e, c = p[(_g == s_) & (_y == 1)], p[(_g == s_) & (_y == 0)]; d = e[:, None] - c[None, :]; num += (d > 0).sum() + 0.5 * (d == 0).sum(); den += d.size
    return num / den
W21 = pd.read_csv(f"{R}/round21_age_within.csv"); w = W21[(W21.strata == "age3") & (W21.test == "F2-noAS-partial - Demo")].iloc[0]
check(abs(_strat(P21["F2-noAS-partial"].mean(0)) - w["auc_F2-noAS-partial"]) < 1e-12 and abs(_strat(P21["Demo"].mean(0)) - w["auc_Demo"]) < 1e-12,
      "round 21: within-age-stratum AUC of F2-noAS-partial and Demo recomputed with an independent pair counter")
check(np.allclose(_holm(W21.p.values), W21.p_holm.values) and ((W21["diff"] >= W21.lo) & (W21["diff"] <= W21.hi)).all(), "round21_age_within.csv: Holm recomputed; point estimates inside their CIs")
# re-runs: one fold of P1, P4 and H-olsall must equal the saved predictions
import sys as _sys; _sys.path.insert(0, f"{ROOT}/code"); os.environ["CUDA_VISIBLE_DEVICES"] = ""
import round17_p1 as _p1, round17_p4 as _p4, round20_harmonisation as _h20, round4 as _r4
_r, _k, _tr, _te = _JOBS[0]
check(np.allclose(_p1.p1_predict(_tr, _te, _r4.YEAR_G)[0], L17[1]["P1"][_r, _te], atol=1e-12), "re-run: P1 fold r0k0 equals the saved predictions")
check(np.allclose(_p4.p4_predict(_tr, _te, "P4")[0], L17[4]["P4"][_r, _te], atol=1e-12), "re-run: P4 fold r0k0 equals the saved predictions")
check(np.allclose(_h20.spectral_predict(_tr, _te, "H-olsall"), P20["H-olsall"][_r, _te], atol=1e-12), "re-run: H-olsall fold r0k0 equals the saved predictions")
# round 22 (added 2026-10-13)
for pred, res in [("round22_prediction.md", "round22_combined_tests.csv"), ("round22_prediction.md", "round22_AB_tests.csv"), ("round22_prediction.md", "code/round22_nm.py")]:
    a, b = first_added(f"reports/{pred}"), first_added(res if res.startswith("code/") else f"reports/{res}")
    check(a is not None and b is not None and a < b, f"{pred} committed before {res}")
z16f = np.load(f"{R}/round16_fold_aucs.npz"); z17f = np.load(f"{R}/round17_fold_aucs.npz")
dbar = ((z17f["I2"] - z17f["C1"]) + (z16f["NormMamba-F"] - z16f["Mamba-raw"]) + (z17f["P3"] - z17f["Riemann-tanpa-normatif"])) / 3
T22 = pd.read_csv(f"{R}/round22_combined_tests.csv")
check(abs(_nb(dbar, np.zeros(50)) - T22.p.iloc[0]) < 1e-12 and abs(dbar.mean() - T22.dAUC.iloc[0]) < 1e-12, f"round 22 C: combined normative-layer test recomputed (dbar {dbar.mean():+.4f}, p {_nb(dbar, np.zeros(50)):.4f})")
_keep = np.array([g_ != "<=2009" for g_ in _r4.YEAR_G]); _I169 = L17[1]["I2_169"]; _P169 = L17[3]["P3_169"]; _rep = []
for _r in range(10):
    _yy, _pp = [], []
    for _k in range(5):
        _subs = [s_ for s_ in _F[(_F.repeat == _r) & (_F.fold == _k)].subject if _keep[_pos[s_]]]; _d = json.load(open(f"{D}/round22/j169/r{_r}_k{_k}.json"))["F"]["test"]
        _yy += [_y[_pos[s_]] for s_ in _subs]; _pp += [(_I169[_r, _pos[s_]] + _P169[_r, _pos[s_]] + _d[s_]) / 3 for s_ in _subs]
    _rep.append(roc_auc_score(_yy, _pp))
check(abs(np.mean(_rep) - 0.69651) < 1e-5, f"round 22 A: F2 on 169 patients recomputed from the JSON files ({np.mean(_rep):.5f})")
_sh = [_pooled(_nm("F", f"round22/shuf{i}")) for i in range(3)]
check(np.allclose(_sh, [0.6626, 0.6619, 0.6407], atol=5e-5), f"round 22 B: NormMamba year-shuffle pooled AUCs recomputed {np.round(_sh, 4).tolist()}")
check("PASS" in open(f"{R}/round22_parallel_check.log").read(), "round 22: parallel GPU run equals a solo re-run (log)")
# round 23 (added 2026-10-13)
for res in ["round23_calibration.csv", "round23_within_combined.csv", "code/round23.py"]:
    a, b = first_added("reports/round23_prediction.md"), first_added(res if res.startswith("code/") else f"reports/{res}")
    check(a is not None and b is not None and a < b, f"round23_prediction.md committed before {res}")
C23 = pd.read_csv(f"{R}/round23_calibration.csv"); W23 = pd.read_csv(f"{R}/round23_within_combined.csv")
_lgt = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
_sl = lambda P: np.mean([_sm.GLM(_y, _sm.add_constant(_lgt(p)), family=_sm.families.Binomial()).fit().params[1] for p in P])
check(abs(_sl(F2_) - C23.F2.iloc[0]) < 1e-4 and abs(_sl(I2_) - C23.I2.iloc[0]) < 1e-4, f"round 23 A: calibration slopes recomputed with statsmodels (F2 {_sl(F2_):.4f}, I2 {_sl(I2_):.4f})")
check(np.allclose(_holm(C23.p.values), C23.p_holm.values) and ((C23["diff"] >= C23.lo) & (C23["diff"] <= C23.hi)).all(), "round23_calibration.csv: Holm recomputed; point estimates inside CIs")
_g = _r4.YEAR_G
def _sa(p):
    num = den = 0.0
    for s_ in np.unique(_g):
        e, c = p[(_g == s_) & (_y == 1)], p[(_g == s_) & (_y == 0)]; d = e[:, None] - c[None, :]; num += (d > 0).sum() + 0.5 * (d == 0).sum(); den += d.size
    return num / den
_NR = _nm("raw"); _dd = [_sa(I2_.mean(0)) - _sa(L17[1]["C1"].mean(0)), _sa(NM_.mean(0)) - _sa(_NR.mean(0)), _sa(P3_.mean(0)) - _sa(L17[3]["Riemann_raw"].mean(0))]
_cw = W23[(W23.strata == "year") & (W23.effect == "combined (mean of 3)")]["diff"].iloc[0]
check(abs(np.mean(_dd) - _cw) < 1e-12, f"round 23 B: within-year combined normative effect recomputed with an independent pair counter ({np.mean(_dd):+.4f})")
# round 24 (added 2026-10-13): primary family recomputed from saved predictions; gate recomputed from the saved inner out-of-fold scores; re-runs
for res in ["code/round24_wpli.py", "code/round24_riemann.py", "code/round24_agesd.py", "code/round24_eval.py", "round24_primary.csv", "round24_results.csv"]:
    a, b = first_added("reports/round24_prediction.md"), first_added(res if res.startswith("code/") else f"reports/{res}")
    check(a is not None and b is not None and a < b, f"round24_prediction.md committed before {res}")
Q24 = {i: np.load(f"{D}/round24/r{i}.npz") for i in (1, 2, 3)}; PR24 = pd.read_csv(f"{R}/round24_primary.csv"); T24 = pd.read_csv(f"{R}/round24_results.csv").set_index("method")
_I2in = np.load(f"{D}/round19/cpu_inner.npz")["I2"]; from scipy.stats import spearmanr as _sp
_ga = np.mean([roc_auc_score(_y[tr], Q24[1]["inner_oof"][f_, tr]) for f_, (_, _, tr, _) in enumerate(_JOBS)])
_gr = np.mean([_sp(Q24[1]["inner_oof"][f_, tr], _I2in[f_, tr]).correlation for f_, (_, _, tr, _) in enumerate(_JOBS)])
check(abs(_ga - Q24[1]["gate_auc"].mean()) < 1e-12 and abs(_gr - Q24[1]["gate_rho"].mean()) < 1e-12 and bool(Q24[1]["gate"]) == (_ga >= 0.60 and _gr <= 0.70),
      f"round 24 R1: gate recomputed from the saved inner out-of-fold scores (AUC {_ga:.4f}, Spearman {_gr:.4f}, passed {_ga >= 0.60 and _gr <= 0.70})")
_M24 = {"W": Q24[1]["W"], "PLV": Q24[1]["PLV"], "P3-KS": Q24[2]["P3KS"], "P3-K": Q24[2]["P3K"], "R3": Q24[3]["R3"], "F2w": (I2_ + P3_ + NM_ + Q24[1]["W"]) / 4,
        "F2 (P3-KS)": (I2_ + Q24[2]["P3KS"] + NM_) / 3, "F2 (R3 for I2)": (Q24[3]["R3"] + P3_ + NM_) / 3}
bad = [m for m, P in _M24.items() if abs(_pooled(P) - T24.loc[m, "auc"]) > 1e-9]
check(not bad, f"round 24: pooled AUC of {len(_M24)} models recomputed from saved predictions {'(mismatch: ' + str(bad) + ')' if bad else ''}")
_mid = np.where((_age > 40) & (_age <= 60))[0]; _a24, _b24 = Q24[3]["R3"].mean(0), I2_.mean(0); _rng = np.random.default_rng(24400); _bd = []
for _ in range(2000):
    s_ = _rng.choice(_mid, len(_mid))
    if len(set(_y[s_])) == 2:
        _bd.append(roc_auc_score(_y[s_], _a24[s_]) - roc_auc_score(_y[s_], _b24[s_]))
_bd = np.array(_bd); _p3 = max(min(1.0, 2 * min((_bd <= 0).mean(), (_bd >= 0).mean())), 1 / len(_bd))
p24 = [_nb(_fold(_M24["F2w"]), _fold(F2_)), _nb(_fold(_M24["P3-KS"]), _fold(P3_)), _p3]
check(len(_mid) == 89 and np.allclose(p24, PR24.p.values, rtol=1e-9) and np.allclose(_holm(p24), PR24.p_holm.values, rtol=1e-9),
      f"round 24: three primary tests and Holm recomputed (p {np.round(p24, 4).tolist()}, Holm {np.round(_holm(p24), 3).tolist()})")
import round24_wpli as _w24, round24_agesd as _a24m
_r, _k, _tr, _te = _JOBS[0]; _W24 = np.load(f"{D}/round24/wpli_features.npz")["W"]
check(np.allclose(_w24.w_predict(_W24, _tr, _te), Q24[1]["W"][_r, _te], atol=1e-12), "re-run: W fold r0k0 equals the saved predictions")
check(np.allclose(_a24m.r3_predict(_tr, _te), Q24[3]["R3"][_r, _te], atol=1e-12), "re-run: R3 fold r0k0 equals the saved predictions")
# round 26 (added 2026-10-14): within-person era calibration (E1) and state matching (E2)
for res in ["code/round26_features.py", "code/round26_run.py", "code/round26_eval.py", "round26_primary.csv", "round26_results.csv"]:
    a, b = first_added("reports/round26_prediction.md"), first_added(res if res.startswith("code/") else f"reports/{res}")
    check(a is not None and b is not None and a < b, f"round26_prediction.md committed before {res}")
Q26 = np.load(f"{D}/round26/preds.npz"); PR26 = pd.read_csv(f"{R}/round26_primary.csv")
check(np.array_equal(Q26["I2"], I2_), "round 26: I2 predictions identical to round 17")
p26 = [_nb(_fold(Q26["WP"]), _fold(I2_)), _nb(_fold(Q26["SM"]), _fold(I2_))]
check(np.allclose(p26, PR26.p.values, rtol=1e-9) and np.allclose(_holm(p26), PR26.p_holm.values, rtol=1e-9), f"round 26: two primary tests and Holm recomputed (p {np.round(p26, 4).tolist()})")
import round26_run as _r26
_r, _k, _tr, _te = _JOBS[12]
check(np.array_equal(_r26.pred("WP", _tr, _te), Q26["WP"][_r, _te]) and np.array_equal(_r26.pred("SM", _tr, _te), Q26["SM"][_r, _te]), "re-run: I2-WP and I2-SM fold r2k2 equal the saved predictions")
# round 25 (added 2026-10-14): fine-tuned foundation models on the 50 canonical folds
for res in ["code/round25_fm.py", "code/round25_eval.py", "round25_primary.csv", "round25_results.csv"]:
    a, b = first_added("reports/round25_prediction.md"), first_added(res if res.startswith("code/") else f"reports/{res}")
    check(a is not None and b is not None and a < b, f"round25_prediction.md committed before {res}")
_FM = {}
for _m in ("labram", "cbramod"):
    _P = np.full((10, 200), np.nan); _okj = True
    for _r, _k, _tr, _te in _JOBS:
        _d = json.load(open(f"{D}/round25/{_m}/r{_r}_k{_k}.json")); _okj &= set(_d["test"]) == {_S[i] for i in _te}
        for _s, _v in _d["test"].items(): _P[_r, _pos[_s]] = _v
    _FM[_m] = _P; check(_okj and not np.isnan(_P).any(), f"round 25: {_m} 50 JSON files hold exactly the canonical test patients")
PR25 = pd.read_csv(f"{R}/round25_primary.csv"); p25 = [_nb(_fold(F2_), _fold(_FM["labram"])), _nb(_fold(F2_), _fold(_FM["cbramod"]))]
check(np.allclose(p25, PR25.p.values, rtol=1e-9) and np.allclose(_holm(p25), PR25.p_holm.values, rtol=1e-9), f"round 25: two primary tests and Holm recomputed from the JSON files (p {np.round(p25, 4).tolist()})")
# round 27 (added 2026-10-14): validity package
for res in ["code/round27_registry.py", "code/round27_validity.py", "round27_registry.csv", "round27_A4_metrics.csv"]:
    a, b = first_added("reports/round27_prediction.md"), first_added(res if res.startswith("code/") else f"reports/{res}")
    check(a is not None and b is not None and a < b, f"round27_prediction.md committed before {res}")
from statsmodels.stats.multitest import multipletests as _mt
G27 = pd.read_csv(f"{R}/round27_registry.csv")
check(len(G27) == 89 and np.allclose(_mt(G27.p, method="fdr_bh")[1], G27.q_bh_global) and np.allclose(_mt(G27.p, method="holm")[1], G27.p_holm_global), "round 27 A1: 89-test registry; global Holm and BH recomputed")
Q27 = np.load(f"{D}/round27/preds.npz"); _in = np.load(f"{D}/round19/cpu_inner.npz"); _nest = np.full((10, 200), np.nan)
_cands = [["I2"], ["P3"], ["NM"], ["I2", "P3"], ["I2", "NM"], ["P3", "NM"], ["I2", "P3", "NM"]]; _OUT = {"I2": I2_, "P3": P3_, "NM": NM_}
for _f, (_r, _k, _tr, _te) in enumerate(_JOBS):
    _oo = json.load(open(f"{D}/round19/nm_inner_r{_r}_k{_k}.json"))["oof"]; _IN = {"I2": _in["I2"][_f][_tr], "P3": _in["P3"][_f][_tr], "NM": np.array([_oo[_S[i]] for i in _tr])}
    _sc = [roc_auc_score(_y[_tr], np.mean([_IN[m] for m in c], 0)) for c in _cands]; _b = sorted(range(7), key=lambda i: (-round(_sc[i], 12), len(_cands[i]), i))[0]
    _nest[_r, _te] = np.mean([_OUT[m][_r, _te] for m in _cands[_b]], 0)
check(np.allclose(_nest, Q27["F2_nested"]), f"round 27 A2: nested fusion selection recomputed (per-fold mean {_fold(_nest).mean():.4f})")
# numbers quoted in the reports
for f, q in [("round27_prediction.md", "72,4"), ("round27_prediction.md", "0,541"), ("round25_prediction.md", "0,018"), ("round26_prediction.md", "0,950"), ("round26_prediction.md", "0,608"), ("round24_prediction.md", "0,892"), ("round24_prediction.md", "0,647"), ("round23_prediction.md", "0,35"), ("round22_prediction.md", "0,037"), ("round16_normmamba_prediction.md", f"{100 * T16.loc['NormMamba-F', 'auc']:.1f}".replace(".", ",")),
             ("round17_handover.md", f"{100 * T17.loc['P1', 'auc']:.1f}".replace(".", ",")),
             ("round18_prediction.md", f"{100 * T18.loc['F2', 'auc']:.1f}".replace(".", ",")),
             ("round19_prediction.md", f"{100 * T19.loc['CF2', 'auc']:.1f}".replace(".", ",")),
             ("round20_prediction.md", f"+{t20.loc['I2 - H-olsall', 'dAUC']:.3f}".replace(".", ",")),
             ("round21_prediction.md", f"{w['p_holm']:.3f}".replace(".", ","))]:
    check(q in txt(f), f"{f} quotes {q}")

# 10 ----------------------------------------------------------- external transfer documentation (added 2026-10-14)
print("\n== 10. external transfer (Kural, Emotiv) in tables and documents ==")
EXR = pd.read_csv(f"{R}/external_results.csv"); _z4 = EXR.query("analysis == 'Z4'")
def _bh(p):  # independent Benjamini-Hochberg (no statsmodels)
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); q = np.empty(m); run = 1.0
    for j in range(m - 1, -1, -1):
        run = min(run, p[o[j]] * m / (j + 1)); q[o[j]] = run
    return np.minimum(q, 1)
_q89, _q91 = _bh(G27.p), _bh(np.r_[G27.p.values, _z4.p.values])
SEN = pd.read_csv(f"{R}/registry_sensitivity_emotiv.csv")
check(np.allclose(_q89, SEN.q_bh_89.values[:89]) and np.allclose(_q91, SEN.q_bh_91.values) and len(SEN) == 91,
      f"Emotiv sensitivity: BH over 89 and 91 tests recomputed independently (max dq {(_q91[:89] - _q89).max():.4f})")
check(((_q89 < .05) == (_q91[:89] < .05)).all() and ((_q89 < .10) == (_q91[:89] < .10)).all(), "Emotiv sensitivity: no registry test changes side of q 0.05 or 0.10")
_SN = pd.read_csv(f"{R}/semua_nilai_eksperimen.csv"); _H = _SN[_SN.Kelompok.str.startswith("H. ")]
_want = [EXR.query("analysis == 'Z3 TUEP->Emotiv' and model == 'bandpower' and seg == 'closed' and norm == 'per-dataset z'").AUC.iloc[0],
         EXR.query("analysis == 'Z3 TUEP->Emotiv' and model == 'labram' and seg == 'closed' and norm == 'per-dataset z'").AUC.iloc[0],
         EXR.query("analysis == 'Z3 Emotiv->TUEP' and model == 'bandpower' and norm == 'per-dataset z'").AUC.iloc[0],
         EXR.query("analysis == 'Z3 Emotiv->TUEP' and model == 'labram' and norm == 'per-dataset z'").AUC.iloc[0],
         EXR.query("analysis == 'Z2' and model == 'bandpower' and seg == 'closed'").AUC.iloc[0], EXR.query("analysis == 'Z2' and model == 'labram' and seg == 'closed'").AUC.iloc[0],
         EXR.query("analysis == 'Z1'").AUC.iloc[0], _z4.query("model == 'bandpower'").AUC_tuep_only.iloc[0], _z4.query("model == 'bandpower'").AUC_pooled.iloc[0]]
check(len(_H) == 9 and np.allclose(_H.AUC.values, 100 * np.array(_want)), "semua_nilai_eksperimen: 9 Emotiv rows equal external_results.csv")
_zb = EXR.query("analysis == 'Z3 TUEP->Emotiv' and model == 'bandpower' and seg == 'closed' and norm == 'per-dataset z'").iloc[0]
_zl = EXR.query("analysis == 'Z3 TUEP->Emotiv' and model == 'labram' and seg == 'closed' and norm == 'per-dataset z'").iloc[0]
K12 = pd.read_csv(f"{R}/round12_kural_results.csv").set_index("method"); K12b = pd.read_csv(f"{R}/round12b_kural_I2_results.csv").set_index("method")
_c = lambda v, d=1: f"{v:.{d}f}".replace(".", ",")
_L = [f"C1 AUC {_c(100 * K12.loc['C1 transfer', 'auc'])} (CI {_c(100 * K12.loc['C1 transfer', 'lo'])}–{_c(100 * K12.loc['C1 transfer', 'hi'])})",
      f"I2 {_c(100 * K12b.loc['I2 transfer', 'auc'])} (CI {_c(100 * K12b.loc['I2 transfer', 'lo'])}–{_c(100 * K12b.loc['I2 transfer', 'hi'])})",
      f"p Holm {_c(K12.loc['C1 transfer', 'p_holm'], 3)} and {_c(K12b.loc['I2 transfer', 'p_holm'], 3)}", f"lantai demografi Kural ({_c(100 * K12.loc['demographics', 'auc'])})",
      f"band power {_c(_zb.AUC, 3)} (CI {_c(_zb.lo, 3)}–{_c(_zb.hi, 3)})", f"LaBraM beku {_c(_zl.AUC, 3)} (CI {_c(_zl.lo, 3)}–{_c(_zl.hi, 3)})",
      f"−{_c(-_z4.query('model == \"bandpower\"').dAUC.iloc[0], 3)} (p {_c(_z4.query('model == \"bandpower\"').p.iloc[0], 3)}; q global {_c(_q91[89], 3)})",
      f"q naik paling banyak {_c((_q91[:89] - _q89).max(), 4)}"]
for q in _L:
    check(q in txt("limitations_and_tripod.md"), f"limitations_and_tripod.md quotes '{q}'")
_S = [f"{_c(100 * _zb.AUC)} (CI {_c(100 * _zb.lo)}–{_c(100 * _zb.hi)})", f"{_c(100 * _zl.AUC)} (CI {_c(100 * _zl.lo)}–{_c(100 * _zl.hi)})", f"q global {_c(_q91[89], 3)}",
      f"q naik paling banyak {_c((_q91[:89] - _q89).max(), 4)}", "maximum increase in q, 0.003"]
for q in _S:
    check(q in txt("ringkasan_semua_eksperimen.md"), f"ringkasan_semua_eksperimen.md quotes '{q}'")
check(f"{(_q91[:89] - _q89).max():.3f}" == "0.003", "manuscript sentence rounds max dq to 0.003")
check("**Satu dataset, satu pusat**, tanpa validasi eksternal" not in txt("limitations_and_tripod.md"), "L4 no longer claims 'tanpa validasi eksternal'")

print(f"\n{'ALL CHECKS PASSED' if not FAILS else f'{len(FAILS)} CHECK(S) FAILED'}")
