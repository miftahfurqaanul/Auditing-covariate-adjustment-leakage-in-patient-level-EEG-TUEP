"""Evaluate the Tang et al. 2024 reproduction runs (reports/tang_repro_prediction.md). CPU only."""
import os, glob, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
D = "/home/miftah_f/RTX-5090/tuep_fresh/data/tang"; SEQ = 60
PAPER = {250: dict(acc=95.60, sens=96.07, spec=95.12, prec=95.21, f1=95.64), 256: dict(acc=99.87, sens=99.82, spec=99.91, prec=99.91, f1=99.87)}


def meta(fs):
    idx = pd.read_csv(f"{D}/index_{fs}.csv"); idx["n_seq"] = idx.n_sec // SEQ
    return idx.loc[idx.index.repeat(idx.n_seq), ["file", "subj", "label"]].reset_index(drop=True)


def m(y, p, thr=0.5):
    q = p > thr; tp, tn = (q & (y == 1)).sum(), (~q & (y == 0)).sum(); fp, fn = (q & (y == 0)).sum(), (~q & (y == 1)).sum()
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1)
    return dict(acc=100 * (tp + tn) / len(y), sens=100 * sens, spec=100 * spec, prec=100 * prec, f1=100 * 2 * prec * sens / max(prec + sens, 1e-12),
                bac=100 * (sens + spec) / 2, auc=100 * roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan)


def fmt(d):
    return "  ".join(f"{k} {v:6.2f}" for k, v in d.items())


for fs in [250, 256]:
    M = meta(fs); y = M.label.values
    print(f"\n================ {fs} Hz: {len(M)} sequences of 60 s, {M.subj.nunique()} subjects "
          f"({M[M.label == 1].subj.nunique()} epilepsy / {M[M.label == 0].subj.nunique()} non); majority-class accuracy {100 * max(y.mean(), 1 - y.mean()):.2f}%")
    r1 = sorted(glob.glob(f"{D}/r1_{fs}/fold*.npz"))
    if r1:
        rows = []
        for f in r1:
            z = np.load(f); j = json.load(open(f.replace(".npz", ".json")))
            rows.append(dict(m(y[z["te"]], z["p"]), overlap=j["subject_overlap"], loss=j["final_loss"], secs=j["train_seconds"]))
        R = pd.DataFrame(rows)
        print(f"R1 sequence-level 5-fold (paper protocol), {len(R)} folds done; subjects shared between train and test per fold: {R.overlap.tolist()}")
        print("   ours  mean: " + fmt(R[["acc", "sens", "spec", "prec", "f1", "bac", "auc"]].mean().to_dict()))
        print("   ours  sd  : " + fmt(R[["acc", "sens", "spec", "prec", "f1", "bac", "auc"]].std().to_dict()))
        print("   paper     : " + fmt(PAPER[fs]))
        print(f"   final training loss per fold {np.round(R.loss, 4).tolist()}, seconds per fold {np.round(R.secs).tolist()}")
    r2 = sorted(glob.glob(f"{D}/r2_{fs}/fold*.npz"))
    if r2:
        mode = json.load(open(f"{D}/r2_{fs}/mode.json")); P = np.full(len(M), np.nan)
        for f in r2:
            z = np.load(f); P[z["te"]] = z["p"]
        done = ~np.isnan(P); Md = M[done].assign(p=P[done], correct=((P[done] > 0.5) == y[done]))
        print(f"R2 subject left out ({mode['mode']}, {len(r2)}/{mode['n_folds']} folds, probe {mode['probe_train_seconds']:.0f}s on GPU)")
        print("   sequence-level pooled: " + fmt(m(Md.label.values, Md.p.values)))
        per = Md.groupby("subj").agg(label=("label", "first"), acc=("correct", "mean"), p=("p", "mean"))
        print(f"   mean per-subject accuracy: epilepsy {100 * per[per.label == 1].acc.mean():.2f}% (n={int((per.label == 1).sum())}; paper 86.90% on 86 patients), "
              f"non-epilepsy {100 * per[per.label == 0].acc.mean():.2f}% (n={int((per.label == 0).sum())}; paper: not reported)")
        sl = m(per.label.values, per.p.values); rng = np.random.default_rng(0); b = []
        for _ in range(2000):
            i = rng.integers(0, len(per), len(per)); yy, pp = per.label.values[i], per.p.values[i]
            if len(np.unique(yy)) == 2:
                b.append(m(yy, pp)["bac"])
        lo, hi = np.percentile(b, [2.5, 97.5])
        print(f"   subject-level (mean probability per subject, threshold 0.5): BAC {sl['bac']:.2f} (95% CI {lo:.2f}-{hi:.2f}), sens {sl['sens']:.2f}, spec {sl['spec']:.2f}, AUC {sl['auc']:.2f}")
