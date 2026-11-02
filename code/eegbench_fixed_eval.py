"""EEG-Bench-F evaluation (reports/eegbench_fixed_prediction.md, commit 213718d). Variants A, B, C (features unchanged) and D (fixed CAR),
encoders BENDR and LaBraM. Official split per file and per patient, strict P3 (canonical folds, first session), full metrics,
C1 and metadata floor on the same patients and folds. CPU only."""
import os, numpy as np, pandas as pd, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import evalkit as ek
warnings.filterwarnings("ignore")
D, R = ek.D, "/home/miftah_f/RTX-5090/tuep_fresh/reports"
meta = pd.read_csv(f"{D}/features_bandpower_recordings.csv"); meta = meta[meta.n_clean > 0].reset_index(drop=True)
subj_r, y_r = meta.subject.values, meta.label.values
first = (meta.groupby("subject").session_no.transform("min") == meta.session_no).values
test_subj = ek.eegbench_test_subjects()
T = np.load(f"{D}/p3_patient_table.npz", allow_pickle=True); subjects, y = T["subjects"], T["y"]; F = ek.folds(); pos = {s: i for i, s in enumerate(subjects)}


def lr_fp(Xtr, ytr, Xte, groups=None):                     # identical to eval_variants.lr_fp
    m = ek.nested_lr()
    if Xtr.shape[1] > 2000:
        m.steps.insert(1, ("pca", PCA(n_components=min(128, len(Xtr) - 1), random_state=0)))
    if groups is not None:
        m.set_params(logisticregressioncv__cv=list(GroupKFold(5).split(Xtr, ytr, groups)))
    return m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def full(yy, p, thr=0.5):
    q = p > thr; tp = (q & (yy == 1)).sum(); fp = (q & (yy == 0)).sum(); se = q[yy == 1].mean(); sp = 1 - q[yy == 0].mean(); pr = tp / max(tp + fp, 1)
    return dict(acc=(q == yy).mean(), sens=se, spec=sp, prec=pr, f1=2 * pr * se / max(pr + se, 1e-9), bac=(se + sp) / 2, auc=roc_auc_score(yy, p))


def boot_bac(yy, p, units, n=2000):
    rng = np.random.default_rng(0); us = np.unique(units); rows = {u: np.where(units == u)[0] for u in us}; b = []
    for _ in range(n):
        i = np.concatenate([rows[u] for u in rng.choice(us, len(us))])
        if len(np.unique(yy[i])) == 2:
            b.append(full(yy[i], p[i])["bac"])
    return np.percentile(b, [2.5, 97.5])


def p3(Xp):
    out, fa = [], []
    for r in range(10):
        p = np.zeros(len(y))
        for k in range(5):
            te = np.array([pos[s] for s in F[(F.repeat == r) & (F.fold == k)].subject]); tr = np.setdiff1d(np.arange(len(y)), te)
            p[te] = lr_fp(Xp[tr], y[tr], Xp[te]); fa.append(roc_auc_score(y[te], p[te]))
        out.append(full(y, p))
    return pd.DataFrame(out), fa


def one(v, enc):
    z = np.load(f"{D}/eegbench_rep_{v}.npz"); rec = z["rec"]; X = z[enc].astype(np.float32); s, yy = subj_r[rec], y_r[rec]
    trm, tem = ~np.isin(s, list(test_subj)), np.isin(s, list(test_subj))
    pf = lr_fp(X[trm], yy[trm], X[tem], groups=s[trm]); yt, st = yy[tem], s[tem]
    pp = pd.DataFrame(dict(s=st, y=yt, p=pf)).groupby("s").agg(y=("y", "first"), p=("p", "mean"))
    Xp = np.vstack([X[first[rec] & (s == p)].mean(0) for p in subjects]); m3, fa = p3(Xp)
    return dict(variant=v, encoder=enc, file=full(yt, pf), file_ci=boot_bac(yt, pf, st), patient=full(pp.y.values, pp.p.values),
                patient_ci=boot_bac(pp.y.values, pp.p.values, pp.index.values), p3=m3, p3_folds=fa)


if __name__ == "__main__":
    jobs = [(v, e) for v in "ABCD" for e in ["bendr", "labram"]]
    res = Parallel(n_jobs=6)(delayed(one)(v, e) for v, e in jobs)
    base = {"C1 band power + age/sex": np.hstack([T["bandpower"], T["agesex"]]), "acquisition-metadata floor": T["acq"]}
    bres = {k: p3(v) for k, v in base.items()}
    desc = {"A": "EEG-Bench recipe as written (CAR all channels, notch 50)", "B": "recipe, notch 60 (CAR all channels)",
            "C": "our clean data (19-ch CAR, 0.5-45 Hz)", "D": "FIXED recipe: CAR over 19 EEG channels, notch 60"}
    rows = []; fmt = lambda d: "  ".join(f"{k} {100 * v:5.1f}" for k, v in d.items())
    print("reference (EEG-Bench README, BAC on the official split, per file): BENDR 74.0, Neuro-GPT 73.4, LaBraM 56.5\n")
    for r in res:
        print(f"[{r['variant']}] {desc[r['variant']]} | {r['encoder']}")
        print(f"   official split, per file   : {fmt(r['file'])}   BAC CI {100 * r['file_ci'][0]:.1f}-{100 * r['file_ci'][1]:.1f}")
        print(f"   official split, per patient: {fmt(r['patient'])}   BAC CI {100 * r['patient_ci'][0]:.1f}-{100 * r['patient_ci'][1]:.1f}")
        print(f"   strict P3 (10x5, per pt)   : {fmt(r['p3'].mean().to_dict())}")
        d, p = ek.corrected_ttest(r["p3_folds"], bres["C1 band power + age/sex"][1]); print(f"   P3 minus C1: dAUC {d:+.3f}, corrected p = {p:.3f}")
        rows.append(dict(variant=r["variant"], encoder=r["encoder"], **{f"file_{k}": v for k, v in r["file"].items()}, file_bac_lo=r["file_ci"][0], file_bac_hi=r["file_ci"][1],
                         **{f"patient_{k}": v for k, v in r["patient"].items()}, **{f"p3_{k}": v for k, v in r["p3"].mean().items()}, p3_dAUC_vs_C1=d, p3_p_vs_C1=p))
    for k, (m3, _) in bres.items():
        print(f"[baseline] {k:30s} strict P3: {fmt(m3.mean().to_dict())}")
    pd.DataFrame(rows).to_csv(f"{R}/eegbench_fixed_results.csv", index=False)
