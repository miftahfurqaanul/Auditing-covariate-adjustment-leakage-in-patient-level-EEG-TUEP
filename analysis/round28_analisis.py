#!/usr/bin/env python3
"""Round 28: additional analysis (see PRAREGISTRASI_analisis_tambahan.md; SHA-256 recorded in PRAREGISTRASI_hash.txt before the run).
CPU only. Reads tuep_fresh (read-only); writes only to analisis_tambahan/hasil/.
Usage: CUDA_VISIBLE_DEVICES="" .../python round28_analisis.py --dry-run | --run
"""
import os, sys, json, time, hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1")
CODE = "/home/miftah_f/RTX-5090/tuep_fresh/code"
sys.path.insert(0, CODE)
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4, round9 as r9, round17_common as c17, round18_fusion as f18, round25_eval as e25
import round27_validity as v

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "hasil"); os.makedirs(OUT, exist_ok=True)
J, Y, N, SUBJ = c17.JOBS, c17.Y, c17.N, c17.SUBJ
BP, D_I2 = r4.BP, r4.DESIGN["I2"]
ERA, DEV, MONT, AGE = np.asarray(r4.YEAR_G), np.asarray(r4.DEV_G), np.asarray(c17.MONT), np.asarray(r4.AGE, float)
B_PERM = 10000


def codes(a):
    return np.unique(a, return_inverse=True)[1]


def load_models():
    M = f18.load_members(); M["C1"] = np.load(f"{c17.OUT}/p1.npz")["C1"]
    F2 = f18.fuse(M, ["I2", "P3", "NM"])
    assert abs(c17.pooled_auc(M["I2"]) - 0.71210) < 1e-5 and abs(c17.pooled_auc(M["C1"]) - 0.66627) < 1e-5 and abs(c17.pooled_auc(F2) - 0.72709) < 1e-5, "reproduction failed"
    A = np.load(f"{ek.D}/round21/age_preds.npz")
    FT = {n: e25.load(m)[0] for n, m in e25.FM.items()}
    Floor = c17.to_matrix(J, Parallel(n_jobs=12)(delayed(r9.lr_acq)(tr, te) for _, _, tr, te in J))
    assert abs(c17.pooled_auc(Floor) - 0.6479) < 5e-4, "Floor not reproduced"
    P = {"I2": M["I2"], "P3": M["P3"], "NM": M["NM"], "F2": F2, "C1": M["C1"], "Floor": Floor, "I2-noAS": A["I2-noAS"], "F2-noAS-partial": A["F2-noAS-partial"], **FT}
    for k, p in P.items():
        assert p.shape == (10, N) and not np.isnan(p).any(), k
    return P


# ----------------------------------------------------------------------------- 1. korelasi
def spear(a, b):
    return stats.spearmanr(a, b).correlation


def a1_korelasi(P):
    S = {m: P[m].mean(0) for m in ("I2", "P3", "NM", "Floor")}; rows = []
    rng = np.random.default_rng(28900); boot = [rng.integers(0, N, N) for _ in range(2000)]
    pairs = [("I2", "P3"), ("I2", "NM"), ("P3", "NM"), ("I2", "Floor"), ("P3", "Floor"), ("NM", "Floor")]
    for a, b in pairs:
        bs = [spear(S[a][i], S[b][i]) for i in boot]
        rows.append(dict(pair=f"{a}-{b}", rho_all=spear(S[a], S[b]), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                         rho_epilepsy=spear(S[a][Y == 1], S[b][Y == 1]), rho_nonepilepsy=spear(S[a][Y == 0], S[b][Y == 0]),
                         rho_repeat_mean=np.mean([spear(P[a][r], P[b][r]) for r in range(10)])))
    X = pd.DataFrame(rows)
    era_ord = pd.Series(ERA).map({"<=2009": 0, "2010": 1, "2011": 2, "2012": 3, "2013": 4}).values
    D = pd.DataFrame([dict(view=m, rho_era=spear(S[m], era_ord), rho_age=spear(S[m], AGE), rho_label=spear(S[m], Y)) for m in S])
    return X, D


# ----------------------------------------------------------------------------- 2. net benefit
PT = np.round(np.arange(0.10, 0.901, 0.05), 2)


def nb_boot(P, models=("F2", "I2", "Floor")):
    treat = {m: (P[m][:, None, :] >= PT[None, :, None]) for m in models}           # (10, T, N)
    odds = PT / (1 - PT)
    def nb(m, w):
        tp = np.einsum("rtn,n->rt", treat[m] * (Y == 1)[None, None, :], w); fp = np.einsum("rtn,n->rt", treat[m] * (Y == 0)[None, None, :], w)
        return ((tp - fp * odds[None, :]) / w.sum()).mean(0)
    def nb_all(w):
        pi = (w * (Y == 1)).sum() / w.sum(); return pi - (1 - pi) * odds
    w0 = np.ones(N); rng = np.random.default_rng(28901); W = [np.bincount(rng.integers(0, N, N), minlength=N).astype(float) for _ in range(2000)]
    rows = []
    pt_nb = {m: nb(m, w0) for m in models}; all0 = nb_all(w0)
    bt = {m: np.array([nb(m, w) for w in W]) for m in models}; ball = np.array([nb_all(w) for w in W])
    for i, pt in enumerate(PT):
        row = dict(pt=pt, treat_all=all0[i], treat_none=0.0)
        for m in models:
            row[f"NB_{m}"] = pt_nb[m][i]; row[f"NB_{m}_lo"] = np.percentile(bt[m][:, i], 2.5); row[f"NB_{m}_hi"] = np.percentile(bt[m][:, i], 97.5)
        for a, b in (("F2", "Floor"), ("F2", "I2")):
            d = bt[a][:, i] - bt[b][:, i]; row[f"d_{a}-{b}"] = pt_nb[a][i] - pt_nb[b][i]; row[f"d_{a}-{b}_lo"] = np.percentile(d, 2.5); row[f"d_{a}-{b}_hi"] = np.percentile(d, 97.5)
        d = bt["F2"][:, i] - np.maximum(ball[:, i], 0); row["d_F2-max(all,none)"] = pt_nb["F2"][i] - max(all0[i], 0)
        row["d_F2-max(all,none)_lo"] = np.percentile(d, 2.5); row["d_F2-max(all,none)_hi"] = np.percentile(d, 97.5)
        rows.append(row)
    return pd.DataFrame(rows)


def nb_prev(P, models=("F2", "I2", "Floor"), pis=(0.10, 0.20, 0.30)):
    rows = []
    for pi in pis:
        for pt in PT:
            odds_t = (pt / (1 - pt)) * (1 - pi) / pi; thr = odds_t / (1 + odds_t)
            row = dict(prevalence=pi, pt=pt, treat_all=pi - (1 - pi) * pt / (1 - pt))
            for m in models:
                vals = []
                for r in range(10):
                    pred = P[m][r] >= thr; sens = pred[Y == 1].mean(); spec = (~pred)[Y == 0].mean()
                    vals.append(sens * pi - (1 - spec) * (1 - pi) * pt / (1 - pt))
                row[f"NB_{m}"] = float(np.mean(vals))
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 3. strata
def a3_strata():
    age3 = np.where(AGE <= 40, "<=40", np.where(AGE <= 60, "41-60", ">60"))
    S = {"year": ERA, "device (3 grup)": DEV, "year x device": np.char.add(np.char.add(ERA, "|"), DEV),
         "year x montage": np.char.add(np.char.add(ERA, "|"), MONT), "age3": age3}
    rows, summ = [], []
    for name, g in S.items():
        tot_pairs = 0; contrib = 0; both = 0
        for s in np.unique(g):
            ne, nn = int(((g == s) & (Y == 1)).sum()), int(((g == s) & (Y == 0)).sum()); pairs = ne * nn
            rows.append(dict(strata=name, stratum=s, n_epilepsy=ne, n_nonepilepsy=nn, pairs=pairs, pairs_pct=100 * pairs / 10000)); tot_pairs += pairs
            if ne and nn:
                both += 1; contrib += ne + nn
        summ.append(dict(strata=name, n_strata=len(np.unique(g)), n_strata_both_classes=both, patients_in_both_class_strata=contrib, same_stratum_pairs=tot_pairs, same_stratum_pairs_pct=100 * tot_pairs / 10000,
                         smallest_cell=int(min(((g == s).sum() for s in np.unique(g))))))
    return pd.DataFrame(rows), pd.DataFrame(summ)


# ----------------------------------------------------------------------------- 4. confounder test
def r2_codes(yhat, cod, k):
    cnt = np.bincount(cod, minlength=k).astype(float); sm = np.bincount(cod, weights=yhat, minlength=k); ok = cnt > 0
    ss_b = (cnt[ok] * (sm[ok] / cnt[ok] - yhat.mean()) ** 2).sum(); return ss_b / ((yhat - yhat.mean()) ** 2).sum()


def r2_cont(a, b):
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def perm_within(groups, rng):
    """permutation index that only swaps elements within each group"""
    idx = np.arange(N)
    for g in groups:
        idx[g] = g[rng.permutation(len(g))]
    return idx


def partial_test(yhat, c, kind, seed, B=B_PERM):
    """H0: yhat is independent of c given Y; c is permuted within each class of Y."""
    groups = [np.where(Y == k)[0] for k in (0, 1)]; rng = np.random.default_rng(seed)
    if kind == "cat":
        cod = codes(c); k = cod.max() + 1; t0 = r2_codes(yhat, cod, k)
        null = np.array([r2_codes(yhat, cod[perm_within(groups, rng)], k) for _ in range(B)])
    else:
        t0 = r2_cont(yhat, c); null = np.array([r2_cont(yhat, c[perm_within(groups, rng)]) for _ in range(B)])
    return t0, (1 + (null >= t0).sum()) / (B + 1)


def full_test(yhat, c, seed, B=B_PERM):
    """H0: yhat is independent of Y given c; Y is permuted within each cell of c."""
    cod = codes(c); groups = [np.where(cod == k)[0] for k in range(cod.max() + 1)]; rng = np.random.default_rng(seed)
    t0 = r2_cont(yhat, Y.astype(float)); null = np.array([r2_cont(yhat, Y[perm_within(groups, rng)].astype(float)) for _ in range(B)])
    return t0, (1 + (null >= t0).sum()) / (B + 1)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); h = np.empty(len(p)); run = 0
    for rank, i in enumerate(o):
        run = max(run, min(1.0, p[i] * (len(p) - rank))); h[i] = run
    return h


CONF_PARTIAL = {"era": (ERA, "cat"), "device": (DEV, "cat"), "montage": (MONT, "cat"), "age": (AGE, "cont")}
CONF_FULL = {"era": ERA, "device": DEV, "year x device": np.char.add(np.char.add(ERA, "|"), DEV)}
MODELS = ["Floor", "C1", "I2", "P3", "NM", "F2", "I2-noAS", "F2-noAS-partial", "LaBraM-FT", "CBraMod-FT"]
PRIMARY = ("I2", "F2")


def a4_tests(P):
    S = {m: P[m].mean(0) for m in MODELS}; rows_p, rows_f = [], []; i = 0
    for m in MODELS:
        for cn, (c, kind) in CONF_PARTIAL.items():
            i += 1; t0, p = partial_test(S[m], c, kind, 28000 + i); rows_p.append(dict(model=m, confounder=cn, R2=t0, p=p, family="primary" if m in PRIMARY else "descriptive"))
    i = 0
    for m in MODELS:
        for cn, c in CONF_FULL.items():
            i += 1; t0, p = full_test(S[m], c, 28100 + i); rows_f.append(dict(model=m, confounder=cn, R2=t0, p=p, family="primary" if m in PRIMARY else "descriptive"))
    Tp, Tf = pd.DataFrame(rows_p), pd.DataFrame(rows_f)
    for T in (Tp, Tf):
        T["p_holm"] = np.nan; msk = T.family == "primary"; T.loc[msk, "p_holm"] = holm(T.loc[msk, "p"].values)
    # sensitivitas per pengulangan (parsial, era)
    rows = []
    for m in PRIMARY:
        ps = [partial_test(P[m][r], ERA, "cat", 28200 + 10 * (m == "F2") + r, B=2000)[1] for r in range(10)]
        rows.append(dict(model=m, confounder="era", median_p_over_repeats=float(np.median(ps)), min_p=float(np.min(ps)), max_p=float(np.max(ps)), n_p_lt_005=int(np.sum(np.array(ps) < 0.05))))
    return Tp, Tf, pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 5. Hamdan
YD = np.char.add(np.char.add(ERA, "|"), DEV)


def pooled_from_X(X, within=False):
    out = Parallel(n_jobs=12)(delayed(v.sim_fold)(X, tr, te) for _, _, tr, te in J)
    Ms = [c17.to_matrix(J, [o[i] for o in out]) for i in range(4)]; a = [c17.pooled_auc(m) for m in Ms]
    if not within:
        return a
    # additional diagnostic (addendum 1): AUC within era strata and year x device strata for C1 and I2 without age/sex (scores averaged over repetitions)
    w = {}
    for nm, i in (("C1_noAS", 0), ("I2_noAS", 1)):
        s = Ms[i].mean(0); w[f"{nm}_wera"] = r9.strat_auc(Y, s, ERA); w[f"{nm}_wyd"] = r9.strat_auc(Y, s, YD)
    return a, w


def shuffle_cols(X, groups, rng):
    Z = X.copy()
    for j in range(X.shape[1]):
        for g in groups:
            Z[g, j] = X[g[rng.permutation(len(g))], j]
    return Z


def hamdan_groups():
    allp = [np.arange(N)]
    era = [np.where(ERA == e)[0] for e in np.unique(ERA)]
    yd = np.char.add(np.char.add(ERA, "|"), DEV); yd_g = [np.where(yd == e)[0] for e in np.unique(yd)]
    return {"H1_all": (allp, 28100), "H2a_era": (era, 28300), "H2b_era_device": (yd_g, 28500)}


def a5_hamdan(nrep):
    ref, wref = pooled_from_X(BP, within=True); rows = []; scen = hamdan_groups()
    for name, (groups, seed0) in scen.items():
        for r in range(nrep):
            Xs = shuffle_cols(BP, groups, np.random.default_rng(seed0 + r)); a, w = pooled_from_X(Xs, within=True)
            rows.append(dict(scenario=name, rep=r, C1_noAS=a[0], I2_noAS=a[1], C1_AS=a[2], I2_AS=a[3], **w))
        print(f"  {name}: done", flush=True)
    R = pd.DataFrame(rows)
    S = R.groupby("scenario").agg(**{f"{c}_{f}": (c, fn) for c in ("C1_noAS", "I2_noAS", "C1_AS", "I2_AS", "C1_noAS_wera", "I2_noAS_wera", "C1_noAS_wyd", "I2_noAS_wyd") for f, fn in
                                     (("mean", "mean"), ("sd", "std"), ("p5", lambda x: np.percentile(x, 5)), ("p95", lambda x: np.percentile(x, 95)))}).reset_index()
    S["I2_minus_C1_noAS_mean"] = R.assign(d=R.I2_noAS - R.C1_noAS).groupby("scenario").d.mean().values
    return R, S, {**dict(zip(["C1_noAS", "I2_noAS", "C1_AS", "I2_AS"], ref)), **wref}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--dry-run"; t0 = time.time()
    P = load_models(); print(f"reproduction passed ({time.time() - t0:.0f} s)", flush=True)
    if mode == "--dry-run":
        t1 = time.time(); ref = pooled_from_X(BP); print("acuan data asli [C1noAS, I2noAS, C1+AS, I2+AS]:", np.round(ref, 5), f"{time.time() - t1:.0f} s")
        assert abs(ref[3] - 0.71210) < 5e-3, "I2 not reproduced via sim_fold"
        t1 = time.time(); Xs = shuffle_cols(BP, hamdan_groups()["H2a_era"][0], np.random.default_rng(28300)); print("H2a one replicate:", pooled_from_X(Xs, within=True), f"{time.time() - t1:.0f} s")
        sys.exit()
    meta = dict(prereg_sha256=sha(os.path.join(HERE, "PRAREGISTRASI_analisis_tambahan.md")), code_sha256=sha(__file__), versions=dict(numpy=np.__version__, scipy=__import__("scipy").__version__, sklearn=__import__("sklearn").__version__))
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
    K, KD = a1_korelasi(P); K.to_csv(f"{OUT}/1_korelasi_antar_pandangan.csv", index=False); KD.to_csv(f"{OUT}/1_korelasi_deskriptif.csv", index=False); print(K.round(3).to_string(index=False)); print(KD.round(3).to_string(index=False), flush=True)
    NB = nb_boot(P); NB.to_csv(f"{OUT}/2_net_benefit.csv", index=False); NBP = nb_prev(P); NBP.to_csv(f"{OUT}/2_net_benefit_prevalensi.csv", index=False)
    print(NB[["pt", "treat_all", "NB_F2", "NB_I2", "NB_Floor", "d_F2-Floor", "d_F2-Floor_lo", "d_F2-Floor_hi", "d_F2-I2_lo", "d_F2-I2_hi"]].round(4).to_string(index=False), flush=True)
    SC, SS = a3_strata(); SC.to_csv(f"{OUT}/3_strata_sel.csv", index=False); SS.to_csv(f"{OUT}/3_strata_ringkasan.csv", index=False); print(SS.round(1).to_string(index=False), flush=True)
    Tp, Tf, Tr = a4_tests(P); Tp.to_csv(f"{OUT}/4_confounder_parsial.csv", index=False); Tf.to_csv(f"{OUT}/4_confounder_penuh.csv", index=False); Tr.to_csv(f"{OUT}/4_parsial_per_pengulangan.csv", index=False)
    print(Tp.round(4).to_string(index=False)); print(Tf.round(4).to_string(index=False)); print(Tr.to_string(index=False), flush=True)
    nrep = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    R, S, ref = a5_hamdan(nrep); R.to_csv(f"{OUT}/5_hamdan_replikasi.csv", index=False); S.to_csv(f"{OUT}/5_hamdan_ringkasan.csv", index=False)
    json.dump(ref, open(f"{OUT}/5_hamdan_acuan_data_asli.json", "w")); print("acuan data asli:", {k: round(x, 4) for k, x in ref.items()}); print(S.round(4).T.to_string())
    print(f"\ndone in {(time.time() - t0) / 60:.1f}minutes")
