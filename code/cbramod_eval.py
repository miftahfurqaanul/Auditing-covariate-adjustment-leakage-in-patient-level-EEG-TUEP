"""Tasks 1, 2, 5 (reports/cbramod_prediction.md): single backbones, diverse-backbone ensemble, metadata floor,
EEG-Bench split (P4), and a 1000x patient-level permutation test that re-runs the whole procedure."""
import os, sys, numpy as np, pandas as pd, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
import evalkit as ek
warnings.filterwarnings("ignore")
D, R = ek.D, "/home/miftah_f/RTX-5090/tuep_fresh/reports"
N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1000

def lr(n_main=None, pca=0, groups_cv=None):
    steps = []
    if pca:
        steps.append(ColumnTransformer([("pca", make_pipeline(StandardScaler(), PCA(n_components=pca, random_state=0)), list(range(n_main))),
                                        ("cov", "passthrough", [n_main, n_main + 1])]))
    steps += [StandardScaler(), LogisticRegressionCV(Cs=ek.CS, cv=groups_cv or 5, scoring="roc_auc", class_weight="balanced", max_iter=5000)]
    return make_pipeline(*steps)

def fp(X, tr, te, y, pca=0, groups=None):
    cv = list(GroupKFold(5).split(X[tr], y[tr], groups[tr])) if groups is not None else None
    m = lr(X.shape[1] - 2, min(pca, len(tr) - 1) if pca else 0, cv)
    return m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]

def pick_layer(L, tr, y, seed, groups=None):
    """Inner CV on the training units only -> layer with the best AUC."""
    cv = (list(GroupKFold(5).split(tr, y[tr], groups[tr])) if groups is not None
          else list(StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y[tr])))
    best, best_auc = 0, -1
    for l in range(L.shape[1]):
        p = np.zeros(len(tr))
        for itr, ite in cv:
            p[ite] = fp(L[:, l], tr[itr], tr[ite], y, groups=groups)
        a = roc_auc_score(y[tr], p)
        if a > best_auc:
            best, best_auc = l, a
    return best

def procedure(S, y, tr, te, seed, groups=None, all_layers=False):
    """One outer fold of every configuration. S = dict of feature arrays (patient or recording rows)."""
    out = {"floor": fp(S["acq"], tr, te, y, groups=groups) if "acq" in S else None,
           "C1": fp(S["C1"], tr, te, y, groups=groups),
           "BENDR": fp(S["BENDR"], tr, te, y, pca=128, groups=groups)}
    lb, cb = pick_layer(S["LaBraM"], tr, y, seed, groups), pick_layer(S["CBraMod"], tr, y, seed + 1, groups)
    out["LaBraM"] = fp(S["LaBraM"][:, lb], tr, te, y, groups=groups)
    out["CBraMod"] = fp(S["CBraMod"][:, cb], tr, te, y, groups=groups)
    out["ENS"] = np.mean([out[k] for k in ["C1", "LaBraM", "CBraMod", "BENDR"]], axis=0)
    if all_layers:
        for l in range(12):
            out[f"CBraMod_L{l + 1}"] = fp(S["CBraMod"][:, l], tr, te, y, groups=groups)
    return {k: v for k, v in out.items() if v is not None}, lb + 1, cb + 1

def tables():
    T = np.load(f"{D}/p3_patient_table.npz", allow_pickle=True)
    subjects, y, bp, agesex, acq = T["subjects"], T["y"], T["bandpower"], T["agesex"], T["acq"]
    meta = pd.read_csv(f"{D}/features_bandpower_recordings.csv"); meta = meta[meta.n_clean > 0].reset_index(drop=True)
    first = (meta.groupby("subject").session_no.transform("min") == meta.session_no).values
    def layers(name):
        z = np.load(f"{D}/emb_{name}_layers_windows.npz", allow_pickle=True); E, rec = z["E"], z["rec"]
        assert (z["h5_path"] == meta.h5_path.values).all()
        pt = np.stack([E[np.isin(rec, np.where(first & (meta.subject.values == s))[0])].astype(np.float32).mean(0) for s in subjects])
        rc = np.stack([E[rec == i].astype(np.float32).mean(0) for i in range(len(meta))])
        return pt, rc
    lab_p, lab_r = layers("labram"); cbr_p, cbr_r = layers("cbramod")
    zb = np.load(f"{D}/eegbench_rep_C.npz"); brec = zb["rec"]; bfeat = zb["bendr"].astype(np.float32)
    ben_p = np.vstack([bfeat[np.isin(brec, np.where(first & (meta.subject.values == s))[0])].mean(0) for s in subjects])
    add = lambda X: np.hstack([X, agesex])
    addL = lambda L: np.concatenate([L, np.repeat(agesex[:, None, :], L.shape[1], 1)], axis=2)
    P3 = {"acq": np.hstack([acq, agesex[:, :0]]), "C1": add(bp), "BENDR": add(ben_p), "LaBraM": addL(lab_p), "CBraMod": addL(cbr_p)}
    P3["acq"] = np.hstack([acq, np.zeros((len(acq), 2))])      # keeps the "last two columns = covariates" layout
    return subjects, y, P3, meta, lab_r, cbr_r, brec, bfeat

if __name__ == "__main__":
    subjects, y, P3, meta, lab_r, cbr_r, brec, bfeat = tables()
    F = ek.folds(); pos = {s: i for i, s in enumerate(subjects)}
    jobs = []
    for r in range(10):
        Fr = F[F.repeat == r]
        for k in range(5):
            te = np.array([pos[s] for s in Fr[Fr.fold == k].subject]); jobs.append((r, k, np.setdiff1d(np.arange(len(y)), te), te))
    res = Parallel(n_jobs=20)(delayed(procedure)(P3, y, tr, te, 100 * r + k, None, True) for r, k, tr, te in jobs)
    keys = list(res[0][0].keys()); rep = {k: [] for k in keys}; fa = {k: [] for k in keys}
    for r in range(10):
        P = {k: np.zeros(len(y)) for k in keys}
        for (rr, kk, tr, te), (pr, lb, cb) in zip(jobs, res):
            if rr == r:
                for k in keys:
                    P[k][te] = pr[k]; fa[k].append(roc_auc_score(y[te], pr[k]))
        for k in keys:
            rep[k].append(ek.metrics(y, P[k]))
    layer_choice = pd.DataFrame([(lb, cb) for _, lb, cb in res], columns=["LaBraM", "CBraMod"])
    floor_bac = np.mean([m["BAC"] for m in rep["floor"]])
    rows = []
    print("== P3 strict protocol (canonical 10x5 folds) ==")
    for k in keys:
        a = np.array([m["AUC"] for m in rep[k]]); b = np.array([m["BAC"] for m in rep[k]])
        d, p = ek.corrected_ttest(fa[k], fa["C1"]) if k != "C1" else (0.0, 1.0)
        rows.append(dict(config=k, AUC=a.mean(), AUC_sd=a.std(), BAC=b.mean(), BAC_sd=b.std(), dBAC_vs_floor=b.mean() - floor_bac, dAUC_vs_C1=d, p_vs_C1=p))
        print(f"{k:12s} AUC {a.mean():.3f} ± {a.std():.3f}  BAC {b.mean():.3f}  dBAC vs floor {b.mean() - floor_bac:+.3f}  vs C1: dAUC {d:+.3f} p={p:.3f}", flush=True)
    print("inner-selected layers:", {c: layer_choice[c].value_counts().to_dict() for c in layer_choice})
    df = pd.DataFrame(rows)

    # ------------------------------------------------ permutation test (whole procedure, repeat-0 folds)
    rng = np.random.default_rng(2026); perms = [rng.permutation(y) for _ in range(N_PERM)]
    f0 = [(tr, te) for r, k, tr, te in jobs if r == 0]
    def one_perm(yp, i):
        P = {k: np.zeros(len(yp)) for k in ["C1", "CBraMod", "ENS"]}
        for k, (tr, te) in enumerate(f0):
            pr, _, _ = procedure(P3, yp, tr, te, 7 * i + k)
            for c in P:
                P[c][te] = pr[c]
        return {c: roc_auc_score(yp, P[c]) for c in P}
    null = Parallel(n_jobs=20)(delayed(one_perm)(yp, i) for i, yp in enumerate(perms))
    null = pd.DataFrame(null); null.to_csv(f"{R}/cbramod_permutation_null.csv", index=False)
    print(f"\n== permutation test, {N_PERM}x (labels shuffled at patient level, whole procedure refit) ==")
    for c in ["C1", "CBraMod", "ENS"]:
        obs = df.set_index("config").AUC[c]; p = (1 + (null[c] >= obs).sum()) / (1 + len(null))
        df.loc[df.config == c, "p_perm"] = p
        print(f"{c:8s} observed AUC {obs:.3f} | null mean {null[c].mean():.3f}, 95th pct {null[c].quantile(.95):.3f} | p = {p:.4f}")
    df.to_csv(f"{R}/cbramod_p3.csv", index=False)

    # ------------------------------------------------ P4 EEG-Bench split, one shot
    hdr = pd.read_csv(f"{D}/edf_headers.csv")
    hdr["age"] = hdr.patient.str.extract(r"Age:(-?\d+)")[0].astype(float)
    hdr["male"] = (hdr.patient.str.extract(r"^\w+\s+([MF])\s")[0] == "M").astype(float)
    m4 = meta.merge(hdr[["file", "age", "male", "duration_s"]], on="file", how="left")
    ras = m4[["age", "male"]].values.astype(float)
    z = np.load(f"{D}/features_bandpower_windows.npz", allow_pickle=True)
    pr_ = {h: i for i, h in enumerate(meta.h5_path)}; wr = np.array([pr_.get(h, -1) for h in z["h5_path"]]); Xw, wr = z["X"][wr >= 0], wr[wr >= 0]
    rbp = np.vstack([np.median(Xw[wr == i], 0) for i in range(len(meta))])
    keep = np.isin(np.arange(len(meta)), brec)                                 # recordings >= 60 s with BENDR features
    bmap = {r: j for j, r in enumerate(brec)}; ben_r = np.vstack([bfeat[bmap[i]] if i in bmap else np.zeros(bfeat.shape[1], np.float32) for i in range(len(meta))])
    S4 = {"C1": np.hstack([rbp, ras]), "BENDR": np.hstack([ben_r, ras]),
          "LaBraM": np.concatenate([lab_r, np.repeat(ras[:, None, :], 12, 1)], 2), "CBraMod": np.concatenate([cbr_r, np.repeat(ras[:, None, :], 12, 1)], 2)}
    test_subj = ek.eegbench_test_subjects(); subj_r, y_r = m4.subject.values, m4.label.values
    tr4 = np.where(keep & ~np.isin(subj_r, list(test_subj)))[0]; te4 = np.where(keep & np.isin(subj_r, list(test_subj)))[0]
    pr4, lb, cb = procedure(S4, y_r, tr4, te4, 0, groups=subj_r)
    print(f"\n== P4 EEG-Bench split ({len(tr4)} train / {len(te4)} test files; inner-selected layers LaBraM L{lb}, CBraMod L{cb}) ==")
    out4 = []
    for k in ["C1", "BENDR", "LaBraM", "CBraMod", "ENS"]:
        mtr = np.zeros(len(y_r), bool); mtr[tr4] = True; mte = np.zeros(len(y_r), bool); mte[te4] = True
        m, _ = ek.p4_eval(np.zeros((len(y_r), 1)), subj_r, y_r, mtr, mte, lambda *a, _p=pr4[k], **kw: _p)
        out4.append(dict(config=k, **m))
        print(f"{k:8s} AUC {m['AUC']:.3f}  BAC {m['BAC']:.3f} (95% CI {m['BAC_lo']:.3f}-{m['BAC_hi']:.3f})")
    print("reference (EEG-Bench README, BAC): BENDR 0.740, NeuroGPT 0.734, LaBraM 0.565")
    pd.DataFrame(out4).to_csv(f"{R}/cbramod_p4.csv", index=False)
