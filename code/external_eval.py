"""External validation Z1-Z4 as pre-specified in reports/external_prediction.md (and its addenda)."""
import numpy as np, pandas as pd, warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import evalkit as ek
warnings.filterwarnings("ignore")
D, Z = ek.D, "/home/miftah_f/RTX-5090/tuep_fresh/external/zenodo_1252141"
rng = np.random.default_rng(0)

def fp(Xtr, ytr, Xte):
    return ek.nested_lr().fit(Xtr, ytr).predict_proba(Xte)[:, 1]

def subject_table(npz, agg):
    z = np.load(npz, allow_pickle=True); X, s = z["X"].astype(np.float32), z["sid"]
    ids = np.array(sorted(set(s)))
    f = np.median if agg == "median" else np.mean
    return ids, np.vstack([f(X[s == i], axis=0) for i in ids])

def zscore(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-6)

def boot_auc(y, p, strata=None, n=2000):
    a = []
    for _ in range(n):
        i = rng.choice(len(y), len(y))
        if len(np.unique(y[i])) == 2:
            a.append(roc_auc_score(y[i], p[i]))
    return np.percentile(a, [2.5, 97.5])

subj = pd.read_csv(f"{D}/ext_zenodo_subjects.csv").set_index("sid")
yz_all = (subj.group == "epilepsy").astype(int)
T = np.load(f"{D}/p3_patient_table.npz", allow_pickle=True); t_ids, yt = T["subjects"], T["y"]
out = []

# ---------------------------------------------------------------- Z1 nuisance floor (Emotiv)
q = subj.copy()
q["ng"] = (q.country == "NG").astype(float); q["closed_first"] = q.eyes.astype(str).str.lower().str.startswith("closed").astype(float)
cols = ["dur_s", "interp_frac", "counter_jumps", "cq_mean", "cq_frac_bad", "gyro_sd", "std_med", "frac_win_ptp_gt_200", "ng", "closed_first"]
Xq, yq = q[cols].values.astype(float), yz_all.loc[q.index].values
strat = (q.country + q.group).values
def cv_rep(X, y, strat, reps=10):
    aucs, P = [], np.zeros(len(y))
    for r in range(reps):
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, strat):
            p[te] = fp(X[tr], y[tr], X[te])
        aucs.append(roc_auc_score(y, p)); P += p / reps
    return np.array(aucs), P
a, _ = cv_rep(Xq, yq, strat)
print(f"Z1 nuisance floor (quality, movement, duration, country, eye order): AUC {a.mean():.3f} ± {a.std():.3f}  (n={len(yq)})")
out.append(dict(analysis="Z1", model="nuisance LR", seg="-", AUC=a.mean(), sd=a.std(), n=len(yq)))
print("   (recording day alone separates the groups perfectly in Nigeria and almost perfectly in Guinea-Bissau)")

# ---------------------------------------------------------------- Z2 within Emotiv
for seg in ["closed", "whole"]:
    for rep, agg in [("bandpower", "median"), ("labram", "mean")]:
        ids, X = subject_table(f"{D}/ext_zenodo_{seg}_{'bp' if rep == 'bandpower' else 'labram'}.npz", agg)
        y = yz_all.loc[ids].values; st = (subj.loc[ids].country + subj.loc[ids].group).values
        a, P = cv_rep(X, y, st)
        per = {c: roc_auc_score(y[subj.loc[ids].country.values == c], P[subj.loc[ids].country.values == c]) for c in ["GB", "NG"]}
        print(f"Z2 within Emotiv [{seg:6s}] {rep:9s}: AUC {a.mean():.3f} ± {a.std():.3f} (n={len(y)}; GB {per['GB']:.3f}, NG {per['NG']:.3f})")
        out.append(dict(analysis="Z2", model=rep, seg=seg, AUC=a.mean(), sd=a.std(), n=len(y), GB=per["GB"], NG=per["NG"]))

# ---------------------------------------------------------------- Z3 transfer
for rep, agg, tf, ef in [("bandpower", "median", "ext_tuep10_first_bp", "bp"), ("labram", "mean", "ext_tuep10_first_labram", "labram")]:
    ti, Xt = subject_table(f"{D}/{tf}.npz", agg); assert (ti == t_ids).all()
    for seg in ["closed", "whole"]:
        zi, Xz = subject_table(f"{D}/ext_zenodo_{seg}_{ef}.npz", agg); yz = yz_all.loc[zi].values; cz = subj.loc[zi].country.values
        for norm in ["per-dataset z", "none"]:
            A, B = (zscore(Xt), zscore(Xz)) if norm == "per-dataset z" else (Xt, Xz)
            p = fp(A, yt, B); auc = roc_auc_score(yz, p); lo, hi = boot_auc(yz, p)
            per = {c: roc_auc_score(yz[cz == c], p[cz == c]) for c in ["GB", "NG"]}
            print(f"Z3 TUEP -> Emotiv [{seg:6s}] {rep:9s} norm={norm:13s}: AUC {auc:.3f} (95% CI {lo:.3f}-{hi:.3f}; GB {per['GB']:.3f}, NG {per['NG']:.3f})")
            out.append(dict(analysis="Z3 TUEP->Emotiv", model=rep, seg=seg, norm=norm, AUC=auc, lo=lo, hi=hi, GB=per["GB"], NG=per["NG"]))
            if seg == "closed":
                p2 = fp(B, yz, A); auc2 = roc_auc_score(yt, p2); lo2, hi2 = boot_auc(yt, p2)
                print(f"Z3 Emotiv -> TUEP [{seg:6s}] {rep:9s} norm={norm:13s}: AUC {auc2:.3f} (95% CI {lo2:.3f}-{hi2:.3f})")
                out.append(dict(analysis="Z3 Emotiv->TUEP", model=rep, seg=seg, norm=norm, AUC=auc2, lo=lo2, hi=hi2))

# ---------------------------------------------------------------- Z4 pooled training, tested on TUEP canonical folds
F = ek.folds(); pos = {s: i for i, s in enumerate(t_ids)}
for rep, agg, tf, ef in [("bandpower", "median", "ext_tuep10_first_bp", "bp"), ("labram", "mean", "ext_tuep10_first_labram", "labram")]:
    _, Xt = subject_table(f"{D}/{tf}.npz", agg); Xt = zscore(Xt)
    zi, Xz = subject_table(f"{D}/ext_zenodo_closed_{ef}.npz", agg); Xz = zscore(Xz); yz = yz_all.loc[zi].values
    fa = {"TUEP only": [], "TUEP + Emotiv": []}; rep_auc = {k: [] for k in fa}
    for r in range(10):
        Fr = F[F.repeat == r]; P = {k: np.zeros(len(yt)) for k in fa}
        for k in range(5):
            te = np.array([pos[s] for s in Fr[Fr.fold == k].subject]); tr = np.setdiff1d(np.arange(len(yt)), te)
            P["TUEP only"][te] = fp(Xt[tr], yt[tr], Xt[te])
            P["TUEP + Emotiv"][te] = fp(np.vstack([Xt[tr], Xz]), np.concatenate([yt[tr], yz]), Xt[te])
            for kk in fa:
                fa[kk].append(roc_auc_score(yt[te], P[kk][te]))
        for kk in fa:
            rep_auc[kk].append(roc_auc_score(yt, P[kk]))
    d, p = ek.corrected_ttest(fa["TUEP + Emotiv"], fa["TUEP only"])
    print(f"Z4 {rep:9s}: TUEP only AUC {np.mean(rep_auc['TUEP only']):.3f} | TUEP + Emotiv AUC {np.mean(rep_auc['TUEP + Emotiv']):.3f} | dAUC {d:+.3f}, p={p:.3f}")
    out.append(dict(analysis="Z4", model=rep, AUC_tuep_only=np.mean(rep_auc["TUEP only"]), AUC_pooled=np.mean(rep_auc["TUEP + Emotiv"]), dAUC=d, p=p))
pd.DataFrame(out).to_csv("/home/miftah_f/RTX-5090/tuep_fresh/reports/external_results.csv", index=False)
