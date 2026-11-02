"""Evaluate round 3 parts A (fc), B3 (matched) and C (temporal) exactly as pre-registered in reports/round3_prediction.md (commit 0d442ba).
Thresholds from validation patients only: max-BAC threshold (primary) and 90%-specificity threshold; also 0.5. CPU only."""
import os, sys, glob, json, numpy as np, pandas as pd, warnings
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import evalkit as ek, threshold_select as ts
warnings.filterwarnings("ignore")
R = "/home/miftah_f/RTX-5090/tuep_fresh/reports"; OUT = f"{ek.D}/round3"; NB = 2000
T = np.load(f"{ek.D}/p3_patient_table.npz", allow_pickle=True); YLAB = pd.Series(T["y"], index=T["subjects"])
ORDER = {"fc": ["C1-V", "Floor-V", "Tang-F-T", "Rohira-F-T", "LaBraM-V", "BENDR-V"],
         "matched": ["M-EEG-V", "C1-V", "Floor-V", "Tang-F-T", "Rohira-F-T", "LaBraM-V", "BENDR-V"],
         "temporal": ["C1-V", "Floor-V", "Tang-F-T", "Rohira-F-T", "LaBraM-V", "BENDR-V"]}
REF = {"fc": "C1-V", "matched": "M-EEG-V", "temporal": "C1-V"}


def load(protocol, method):
    out = {}
    for p in sorted(glob.glob(f"{OUT}/{protocol}/{method}/*.json")):
        d = json.load(open(p)); v = pd.Series(d["val"]); t = pd.Series(d["test"]); yv = YLAB.loc[v.index].values
        thr, _ = ts.best_threshold(yv, v.values); thr90 = float(np.quantile(v.values[yv == 0], 0.90))
        out[os.path.basename(p)[:-5]] = (t, thr, thr90)
    return out


def binm(y, q):
    se, sp = q[y == 1].mean(), 1 - q[y == 0].mean(); tp, fp = (q & (y == 1)).sum(), (q & (y == 0)).sum(); pr = tp / max(tp + fp, 1)
    return dict(acc=(q == y).mean(), sens=se, spec=sp, prec=pr, f1=2 * pr * se / max(pr + se, 1e-9), bac=(se + sp) / 2)


def calib(y, p):
    lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    m = LogisticRegression(penalty=None).fit(lg[:, None], y); return m.intercept_[0], m.coef_[0, 0]


def repeat_tables(F):
    """F: split name -> (test probs, thr, thr90). Returns per repeat: subjects, y, p, q_bac, q90, q05."""
    reps = {}
    for name, (t, thr, thr90) in F.items():
        r = name.split("_")[0]; reps.setdefault(r, []).append((t, thr, thr90))
    out = []
    for r, parts in sorted(reps.items()):
        s = np.concatenate([t.index.values for t, _, _ in parts]); p = np.concatenate([t.values for t, _, _ in parts])
        qb = np.concatenate([t.values > thr for t, thr, _ in parts]); q9 = np.concatenate([t.values > thr90 for t, _, thr90 in parts])
        out.append((s, YLAB.loc[s].values, p, qb, q9, p > 0.5))
    return out


def metrics(tabs, idx=None):
    rows = []
    for s, y, p, qb, q9, q5 in tabs:
        if idx is not None:
            k = pd.Series(np.arange(len(s)), index=s).loc[idx].values; y, p, qb, q9, q5 = y[k], p[k], qb[k], q9[k], q5[k]
        m = binm(y, qb); m9 = binm(y, q9)
        rows.append(dict(auc=roc_auc_score(y, p), **m, bac05=binm(y, q5)["bac"], sens_at_spec90=m9["sens"], spec_at_spec90=m9["spec"], brier=np.mean((p - y) ** 2)))
    return pd.DataFrame(rows).mean()


def fold_aucs(F):
    return [roc_auc_score(YLAB.loc[t.index].values, t.values) for _, (t, _, _) in sorted(F.items())]


if __name__ == "__main__":
    protocols = sys.argv[1:] or ["fc", "matched", "temporal"]; rows = []
    for prot in protocols:
        meths = [m for m in ORDER[prot] if len(glob.glob(f"{OUT}/{prot}/{m}/*.json")) == (1 if prot == "temporal" else 50)]
        print(f"\n================ {prot}: complete methods {meths}"); FS = {m: load(prot, m) for m in meths}; TABS = {m: repeat_tables(FS[m]) for m in meths}
        pts = np.array(sorted(TABS[meths[0]][0][0])); rng = np.random.default_rng(0); boots = [rng.choice(pts, len(pts)) for _ in range(NB)]
        ref = REF[prot]; pv = {}
        for m in meths:
            est = metrics(TABS[m]); B = pd.DataFrame([metrics(TABS[m], b) for b in boots]); lo, hi = B.quantile(0.025), B.quantile(0.975)
            ci_int, ci_slope = calib(*[np.concatenate(x) for x in zip(*[(y, p) for _, y, p, *_ in TABS[m]])])
            if m != ref and ref in TABS:
                if prot == "temporal":
                    d = B.auc.values - pd.DataFrame([metrics(TABS[ref], b) for b in boots]).auc.values; dd = est.auc - metrics(TABS[ref]).auc
                    pv[m] = (dd, min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
                else:
                    pv[m] = ek.corrected_ttest(fold_aucs(FS[m]), fold_aucs(FS[ref]), n_train=160 if prot == "fc" else 69, n_test=40 if prot == "fc" else 17)
            print(f"{m:11s} AUC {100*est.auc:5.1f} [{100*lo.auc:4.1f}-{100*hi.auc:4.1f}]  acc {100*est.acc:5.1f}  sens {100*est.sens:5.1f} [{100*lo.sens:4.1f}-{100*hi.sens:4.1f}]  "
                  f"spec {100*est.spec:5.1f} [{100*lo.spec:4.1f}-{100*hi.spec:4.1f}]  prec {100*est.prec:5.1f}  F1 {100*est.f1:5.1f}  BAC {100*est.bac:5.1f} [{100*lo.bac:4.1f}-{100*hi.bac:4.1f}]  "
                  f"BAC@0.5 {100*est.bac05:5.1f}  sens@spec90 {100*est.sens_at_spec90:5.1f} (spec {100*est.spec_at_spec90:5.1f})  Brier {est.brier:.3f}  calib int {ci_int:+.2f} slope {ci_slope:.2f}")
            ppv = {pi: est.sens * pi / (est.sens * pi + (1 - est.spec) * (1 - pi)) for pi in [0.1, 0.3, 0.5]}
            npv = {pi: est.spec * (1 - pi) / (est.spec * (1 - pi) + (1 - est.sens) * pi) for pi in [0.1, 0.3, 0.5]}
            rows.append(dict(protocol=prot, method=m, **{k: est[k] for k in est.index}, **{f"{k}_lo": lo[k] for k in est.index}, **{f"{k}_hi": hi[k] for k in est.index},
                             calib_intercept=ci_int, calib_slope=ci_slope, **{f"PPV@{int(100*k)}%": v for k, v in ppv.items()}, **{f"NPV@{int(100*k)}%": v for k, v in npv.items()}))
        if pv:
            names = list(pv); praw = np.array([pv[n][1] for n in names]); o = np.argsort(praw); adj = np.empty(len(names)); run = 0
            for rank, i in enumerate(o):
                run = max(run, min(1, praw[i] * (len(names) - rank))); adj[i] = run
            print(f"-- AUC difference vs {ref} ({'paired bootstrap' if prot == 'temporal' else 'corrected t-test, 50 folds'}; Holm-adjusted):")
            for n, a in zip(names, adj):
                print(f"   {n:11s} dAUC {pv[n][0]:+.3f}  p {pv[n][1]:.3f}  p_Holm {a:.3f}")
            for r in rows:
                if r["protocol"] == prot and r["method"] in pv:
                    r.update(dAUC_vs_ref=pv[r["method"]][0], p_vs_ref=pv[r["method"]][1], p_holm=adj[names.index(r["method"])])
    pd.DataFrame(rows).to_csv(f"{R}/round3_results_{'_'.join(protocols)}.csv", index=False)
