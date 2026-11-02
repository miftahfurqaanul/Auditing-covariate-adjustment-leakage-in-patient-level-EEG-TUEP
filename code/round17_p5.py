"""Round 17 P5 (reports/round17_P1-P5_prediction.md Bagian 7; choices in round17_implementation_notes.md). PRETRAINED TRACK ONLY.
Control-referenced era adapter on FROZEN LaBraM embeddings: data/eegbench_rep_D.npz (key 'labram', 200-d per recording; the source of
LaBraM-mentah 64.0 in round 7). No extraction, no GPU, no fine-tuning. Per fold: M[g] = mean embedding of the first-session recordings of
the non-epilepsy TRAINING patients of era g (< 3 such recordings -> global training-control mean); E' = E - M[g]; patient = median over
first-session recordings; probe = eegbench_fixed_eval.lr_fp (evalkit.nested_lr), as LaBraM 64.0. Ablations: P5-global (one M; removed by
StandardScaler, so == raw with median aggregation), P5-whiten (secondary: (E - M[g]) / sd[g]). Comparator: LaBraM-mentah reproduced exactly
as round 7 (mean aggregation). Negative controls: year shuffle 20x, without <=2009. CPU only."""
import round17_common as c17
import time, numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4, eegbench_fixed_eval as ebf

Y, N = c17.Y, c17.N
MIN_REC = 3
with np.load(f"{ek.D}/eegbench_rep_D.npz") as _z:                                  # plain arrays (an open NpzFile cannot be sent to workers)
    REC, LAB32 = _z["rec"], _z["labram"].astype(np.float32)
EMB = LAB32.astype(np.float64)
SR = ebf.subj_r[REC]; FIRST = ebf.first[REC]; POS = {s: i for i, s in enumerate(c17.SUBJ)}
E1 = EMB[FIRST]; RP = np.array([POS[s] for s in SR[FIRST]])                       # first-session recordings and their patient index


def raw_mean(tr, te):
    """LaBraM-mentah exactly as round7.predictions (mean over first-session recordings, float32 embeddings)."""
    X = LAB32; s = SR; Xp = np.vstack([X[FIRST & (s == p)].mean(0) for p in c17.SUBJ])
    return ebf.lr_fp(Xp[tr], Y[tr], Xp[te])


def adapted(tr, yg, mode):
    c = tr[Y[tr] == 0]; cr = np.isin(RP, c); era = yg[RP]; Eg = E1.copy()
    mg, sg = E1[cr].mean(0), E1[cr].std(0, ddof=1); fb = []
    for g in np.unique(era):
        sel = era == g; ref = cr & sel
        if mode == "global":
            m, s = mg, sg
        elif ref.sum() < MIN_REC:
            m, s = mg, sg; fb.append(str(g))
        else:
            m, s = E1[ref].mean(0), E1[ref].std(0, ddof=1)
        Eg[sel] = (E1[sel] - m) / s if mode == "whiten" else E1[sel] - m
    if mode == "median_raw":
        Eg = E1
    return np.vstack([np.median(Eg[RP == i], 0) for i in range(N)]), fb


def p5_predict(tr, te, yg, mode):
    Xp, fb = adapted(tr, yg, mode); return ebf.lr_fp(Xp[tr], Y[tr], Xp[te]), fb


def checks(tr):
    print(f"first-session recordings with LaBraM embeddings: {len(E1)}; patients covered {len(np.unique(RP))}/200; per patient median {np.median(np.bincount(RP)):.0f}")
    Xp, fb = adapted(tr, r4.YEAR_G, "era"); c = tr[Y[tr] == 0]; cr = np.isin(RP, c); yc = r4.YEAR_G[RP]
    Eg = E1.copy()
    for g in np.unique(yc):
        Eg[yc == g] -= E1[cr & (yc == g)].mean(0)
    worst = max(np.abs(Eg[cr & (yc == g)].mean(0)).max() for g in np.unique(yc))
    c17.pf(worst < 1e-8, f"centred training-control recordings have mean 0 within every era (max |mean| {worst:.1e}); era fallbacks {fb}")
    print("   training-control first-session recordings per era:", {g: int((cr & (yc == g)).sum()) for g in np.unique(yc)})


if __name__ == "__main__":
    a = c17.args(); c17.header("round17_p5 (pretrained track; frozen embeddings, no GPU)"); t0 = time.time()
    assert len(np.unique(RP)) == N
    if a.dry_run:
        r, k, tr, te = c17.JOBS[0]; checks(tr)
        print(f"dry-run fold r{r}k{k}: P5-adapter {roc_auc_score(Y[te], p5_predict(tr, te, r4.YEAR_G, 'era')[0]):.4f} | LaBraM-mentah {roc_auc_score(Y[te], raw_mean(tr, te)):.4f} | {time.time() - t0:.1f} s")
        raise SystemExit
    J = c17.JOBS; checks(J[0][2]); P = {}
    P["LaBraM_raw"] = c17.to_matrix(J, Parallel(n_jobs=a.jobs)(delayed(raw_mean)(tr, te) for _, _, tr, te in J))
    if not c17.pf(abs(c17.pooled_auc(P["LaBraM_raw"]) - 0.640) < 5e-4, f"LaBraM-mentah == 0.640 (round 7, same tolerance) ({c17.pooled_auc(P['LaBraM_raw']):.5f})"):
        raise SystemExit("STOP: reproduction failed")
    for name, mode in [("P5", "era"), ("P5_global", "global"), ("P5_whiten", "whiten"), ("LaBraM_median", "median_raw")]:
        out = Parallel(n_jobs=a.jobs)(delayed(p5_predict)(tr, te, r4.YEAR_G, mode) for _, _, tr, te in J); P[name] = c17.to_matrix(J, [o[0] for o in out])
        print(f"{name} AUC {c17.pooled_auc(P[name]):.5f} | folds with era fallback {sum(1 for o in out if o[1])}", flush=True)
    c17.pf(np.allclose(P["P5_global"], P["LaBraM_median"], atol=1e-6), "P5-global == LaBraM with median aggregation (a common shift is removed by StandardScaler)")
    shuf = []
    for i in range(c17.N_SHUF):
        Ps = c17.to_matrix(J, [o[0] for o in Parallel(n_jobs=a.jobs)(delayed(p5_predict)(tr, te, c17.shuffled_year(i), "era") for _, _, tr, te in J)])
        shuf.append(c17.pooled_auc(Ps)); print(f"   year-shuffle {i:2d}: P5 AUC {shuf[-1]:.4f}", flush=True)
    shuf = np.array(shuf); a5 = c17.pooled_auc(P["P5"])
    print(f"year-shuffle null: mean {shuf.mean():.4f} sd {shuf.std(ddof=1):.4f} p95 {np.percentile(shuf, 95):.4f}; P5 > p95: {a5 > np.percentile(shuf, 95)}; "
          f"rank p = {(1 + (shuf >= a5).sum()) / (1 + len(shuf)):.3f}")
    J2 = c17.J169; P169 = {"P5": c17.to_matrix(J2, [o[0] for o in Parallel(n_jobs=a.jobs)(delayed(p5_predict)(tr, te, r4.YEAR_G, "era") for _, _, tr, te in J2)]),
                           "LaBraM_raw": c17.to_matrix(J2, Parallel(n_jobs=a.jobs)(delayed(raw_mean)(tr, te) for _, _, tr, te in J2))}
    d, p = ek.corrected_ttest(c17.fold_aucs(J2, P169["P5"]), c17.fold_aucs(J2, P169["LaBraM_raw"]), 135, 34)
    print(f"without <=2009 (169): P5 {c17.pooled_auc(P169['P5']):.4f} | LaBraM-mentah {c17.pooled_auc(P169['LaBraM_raw']):.4f} | dAUC {d:+.4f} p {p:.4f}")
    np.savez(f"{c17.OUT}/p5.npz", **P, shuffle_auc=shuf, P5_169=P169["P5"], LaBraM_raw_169=P169["LaBraM_raw"])
    print(f"done in {(time.time() - t0) / 60:.1f} min")
