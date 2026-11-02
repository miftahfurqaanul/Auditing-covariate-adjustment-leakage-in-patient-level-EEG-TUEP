"""Round 2 (pre-specified in reports/round2_prediction.md): candidates C1, C2, C3(layer 1-12), C5 and
nested model selection on the canonical P3 folds, then the same procedure once on the EEG-Bench split (P4)."""
import numpy as np, pandas as pd, warnings
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
import evalkit as ek
warnings.filterwarnings("ignore")
D, R = ek.D, "/home/miftah_f/RTX-5090/tuep_fresh/reports"

def model(n_main, n_cov, pca, inner_cv=5):
    """Standardize -> (PCA on the EEG block only, label-free) -> L2 LR with nested C."""
    steps = []
    if pca:
        steps.append(ColumnTransformer([("pca", make_pipeline(StandardScaler(), PCA(n_components=pca, random_state=0)), list(range(n_main))),
                                        ("cov", "passthrough", list(range(n_main, n_main + n_cov)))]))
    steps += [StandardScaler(), LogisticRegressionCV(Cs=ek.CS, cv=inner_cv, scoring="roc_auc", class_weight="balanced", max_iter=5000)]
    return make_pipeline(*steps)

def fit_predict(cand, X, tr, te, y, groups=None):
    Xc, n_main, pca = X[cand]
    m = model(n_main, Xc.shape[1] - n_main, min(pca, len(tr) - 1) if pca else 0)
    if groups is not None:
        m.set_params(logisticregressioncv__cv=list(GroupKFold(5).split(Xc[tr], y[tr], groups[tr])))
    return m.fit(Xc[tr], y[tr]).predict_proba(Xc[te])[:, 1]

LAYERS = [f"C3_L{l}" for l in range(1, 13)]

def inner_select(X, y, tr, seed, groups=None):
    """Inner CV over the training units only -> AUC per candidate (C5 uses the inner-best layer)."""
    cv = (GroupKFold(5).split(tr, y[tr], groups[tr]) if groups is not None
          else StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y[tr]))
    P = {c: np.zeros(len(tr)) for c in ["C1", "C2"] + LAYERS}
    for itr, ite in cv:
        for c in P:
            P[c][ite] = fit_predict(c, X, tr[itr], tr[ite], y, groups)
    auc = {c: roc_auc_score(y[tr], p) for c, p in P.items()}
    best_layer = max(LAYERS, key=lambda c: auc[c])
    auc["C5"] = roc_auc_score(y[tr], (P["C1"] + P["C2"] + P[best_layer]) / 3)
    return auc, best_layer

def predict_cand(c, best_layer, X, tr, te, y, groups=None):
    if c == "C5":
        return np.mean([fit_predict(k, X, tr, te, y, groups) for k in ["C1", "C2", best_layer]], axis=0)
    return fit_predict(c, X, tr, te, y, groups)

def outer_fold(X, y, tr, te, seed):
    auc, best_layer = inner_select(X, y, tr, seed)
    chosen = max(auc, key=auc.get)
    preds = {c: predict_cand(c, best_layer, X, tr, te, y) for c in ["C1", "C2", "C5"] + LAYERS}
    preds["NESTED"] = preds[chosen]
    return te, preds, chosen, best_layer

if __name__ == "__main__":
    T = np.load(f"{D}/p3_patient_table.npz", allow_pickle=True)
    subjects, y, bp, agesex = T["subjects"], T["y"], T["bandpower"], T["agesex"]
    rz = np.load(f"{D}/riemann_p3.npz", allow_pickle=True); assert (rz["subjects"] == subjects).all()
    meta = pd.read_csv(f"{D}/features_bandpower_recordings.csv"); meta = meta[meta.n_clean > 0].reset_index(drop=True)
    first = (meta.groupby("subject").session_no.transform("min") == meta.session_no).values
    L = np.load(f"{D}/emb_labram_layers_windows.npz", allow_pickle=True); E, rec = L["E"], L["rec"]
    assert (L["h5_path"] == meta.h5_path.values).all()
    lay = np.stack([E[np.isin(rec, np.where(first & (meta.subject.values == s))[0])].astype(np.float32).mean(0) for s in subjects])  # [200, 12, 200]
    X = {"C1": (np.hstack([bp, agesex]), bp.shape[1], 0), "C2": (np.hstack([rz["X"], agesex]), rz["X"].shape[1], 128)}
    for l in range(12):
        X[f"C3_L{l + 1}"] = (np.hstack([lay[:, l], agesex]), 200, 0)
    F = ek.folds(); pos = {s: i for i, s in enumerate(subjects)}
    jobs = []
    for r in range(10):
        Fr = F[F.repeat == r]
        for k in range(5):
            te = np.array([pos[s] for s in Fr[Fr.fold == k].subject]); tr = np.setdiff1d(np.arange(len(y)), te)
            jobs.append((r, k, tr, te))
    res = Parallel(n_jobs=10)(delayed(outer_fold)(X, y, tr, te, 100 * r + k) for r, k, tr, te in jobs)
    cands = ["C1", "C2", "C5"] + LAYERS + ["NESTED"]
    fold_auc = {c: [] for c in cands}; rep = {c: [] for c in cands}; chosen = []
    for r in range(10):
        P = {c: np.zeros(len(y)) for c in cands}
        for (rr, k, tr, te), (te2, preds, ch, bl) in zip(jobs, res):
            if rr != r:
                continue
            chosen.append((ch, bl))
            for c in cands:
                P[c][te2] = preds[c]; fold_auc[c].append(roc_auc_score(y[te2], preds[c]))
        for c in cands:
            rep[c].append(ek.metrics(y, P[c]))
    print("== P3 strict protocol, canonical 10x5 folds (AUC / BAC mean ± sd over repeats) ==")
    rows = []
    for c in cands:
        a = np.array([m["AUC"] for m in rep[c]]); b = np.array([m["BAC"] for m in rep[c]])
        d, p = ek.corrected_ttest(fold_auc[c], fold_auc["C1"]) if c != "C1" else (0.0, 1.0)
        rows.append(dict(cand=c, AUC=a.mean(), AUC_sd=a.std(), BAC=b.mean(), BAC_sd=b.std(), dAUC_vs_C1=d, p_vs_C1=p))
        print(f"{c:8s} AUC {a.mean():.3f} ± {a.std():.3f}  BAC {b.mean():.3f} ± {b.std():.3f}   vs C1: dAUC {d:+.3f} p={p:.3f}")
    ch = pd.Series([c for c, _ in chosen]).value_counts(); bl = pd.Series([b for _, b in chosen]).value_counts()
    print("\nnested selection chose:", ch.to_dict(), "| inner-best LaBraM layer:", bl.to_dict())
    pd.DataFrame(rows).to_csv(f"{R}/round2_p3.csv", index=False)

    # ------------------------------------------------------------- P4, one shot
    hdr = pd.read_csv(f"{D}/edf_headers.csv")
    hdr["age"] = hdr.patient.str.extract(r"Age:(-?\d+)")[0].astype(float)
    hdr["male"] = (hdr.patient.str.extract(r"^\w+\s+([MF])\s")[0] == "M").astype(float)
    m4 = meta.merge(hdr[["file", "age", "male", "duration_s"]], on="file", how="left")
    z = np.load(f"{D}/features_bandpower_windows.npz", allow_pickle=True)
    pos_r = {h: i for i, h in enumerate(meta.h5_path)}; wr = np.array([pos_r.get(h, -1) for h in z["h5_path"]])
    Xw, wr = z["X"][wr >= 0], wr[wr >= 0]
    rbp = np.vstack([np.median(Xw[wr == i], 0) for i in range(len(meta))])
    rr = np.load(f"{D}/riemann_recordings.npz", allow_pickle=True); assert (rr["h5_path"] == meta.h5_path.values).all()
    rlay = np.stack([E[rec == i].astype(np.float32).mean(0) for i in range(len(meta))])
    ras = m4[["age", "male"]].values.astype(float)
    X4 = {"C1": (np.hstack([rbp, ras]), rbp.shape[1], 0), "C2": (np.hstack([rr["X"], ras]), rr["X"].shape[1], 128)}
    for l in range(12):
        X4[f"C3_L{l + 1}"] = (np.hstack([rlay[:, l], ras]), 200, 0)
    test_subj = ek.eegbench_test_subjects(); keep = (m4.duration_s >= 60).values
    subj_r, y_r = m4.subject.values, m4.label.values
    tr4 = np.where(keep & ~np.isin(subj_r, list(test_subj)))[0]; te4 = np.where(keep & np.isin(subj_r, list(test_subj)))[0]
    auc_in, best_layer = inner_select(X4, y_r, tr4, 0, groups=subj_r)
    chosen4 = max(auc_in, key=auc_in.get)
    print(f"\n== P4 EEG-Bench split: inner (training patients only) chose {chosen4} (best layer {best_layer}) ==")
    out4 = []
    for c in ["C1", "C2", "C5", best_layer, "NESTED"]:
        cc = chosen4 if c == "NESTED" else c
        p = predict_cand(cc, best_layer, X4, tr4, te4, y_r, groups=subj_r)
        msk = np.zeros(len(y_r), bool); msk[tr4] = True; msk2 = np.zeros(len(y_r), bool); msk2[te4] = True
        m, _ = ek.p4_eval(np.c_[np.zeros(len(y_r))], subj_r, y_r, msk, msk2, lambda *a, **k: p)
        out4.append(dict(cand=c, chosen=cc, **m))
        print(f"{c:8s} ({cc:7s}) AUC {m['AUC']:.3f}  BAC {m['BAC']:.3f} (95% CI {m['BAC_lo']:.3f}-{m['BAC_hi']:.3f})")
    print("reference (EEG-Bench README, BAC): BENDR 0.740, NeuroGPT 0.734, LaBraM 0.565")
    pd.DataFrame(out4).to_csv(f"{R}/round2_p4.csv", index=False)
