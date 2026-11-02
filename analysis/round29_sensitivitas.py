#!/usr/bin/env python3
"""Round 29: sensitivity tests of the normative layer without the leak (see PRAREGISTRASI_putaran29.md; SHA-256 recorded before the run).
CPU only. Reads tuep_fresh (read-only); writes to analisis_tambahan/hasil29/.
Usage: CUDA_VISIBLE_DEVICES="" .../python round29_sensitivitas.py --dry-run | --run [n_replicates]"""
import os, sys, json, time, hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
import round28_analisis as A
import round10 as r10
ek, c17, r9, J, Y, N, BP, ERA, DEV, MONT, AGE = A.ek, A.c17, A.r9, A.J, A.Y, A.N, A.BP, A.ERA, A.DEV, A.MONT, A.AGE
OUT = os.path.join(HERE, "hasil29"); os.makedirs(OUT, exist_ok=True)
D_FULL = A.D_I2; D_NOERA = D_FULL[:, :3]; AS = c17.AS
VARIANTS = ["C1", "V0", "V1", "V2", "V3", "V4"]


def fit_norm(X, D, ctrl):
    B = np.linalg.lstsq(D[ctrl], X[ctrl], rcond=None)[0]
    sd = (X[ctrl] - D[ctrl] @ B).std(0, ddof=np.linalg.matrix_rank(D[ctrl]))
    return B, sd


def split_reference(tr, f):
    """V1: training controls split in two, stratified by era; R only for fitting B and s."""
    rng = np.random.default_rng(29000 + f); tc = tr[Y[tr] == 0]; R = []
    for e in np.unique(ERA[tc]):
        idx = tc[ERA[tc] == e]; idx = idx[rng.permutation(len(idx))]; R.extend(idx[: int(np.ceil(len(idx) / 2))])
    R = np.array(sorted(R)); return R, np.setdiff1d(tr, R)


def crossfit_groups(tr, f):
    """V2: training patients split into 5 random class-stratified groups."""
    rng = np.random.default_rng(29000 + f); g = np.full(N, -1)
    for k in (0, 1):
        idx = tr[Y[tr] == k]; idx = idx[rng.permutation(len(idx))]; g[idx] = np.arange(len(idx)) % 5
    return g


def features(X, f, tr, te, variant):
    """returns (feature matrix for all patients, indices for model training)"""
    ctrl = tr[Y[tr] == 0]
    if variant == "C1":
        return X, tr
    if variant == "V0":
        B, sd = fit_norm(X, D_FULL, ctrl); return (X - D_FULL @ B) / sd, tr
    if variant == "V3":
        B, sd = fit_norm(X, D_NOERA, ctrl); return (X - D_NOERA @ B) / sd, tr
    if variant == "V4":
        B, sd = fit_norm(X, D_FULL, tr); return (X - D_FULL @ B) / sd, tr
    if variant == "V1":
        R, trL = split_reference(tr, f); B, sd = fit_norm(X, D_FULL, R); return (X - D_FULL @ B) / sd, trL
    if variant == "V2":
        g = crossfit_groups(tr, f); Z = np.zeros_like(X); B, sd = fit_norm(X, D_FULL, ctrl); Z[te] = (X[te] - D_FULL[te] @ B) / sd
        for k in range(5):
            members = tr[g[tr] == k]; ctrl_k = ctrl[g[ctrl] != k]; Bk, sdk = fit_norm(X, D_FULL, ctrl_k)
            Z[members] = (X[members] - D_FULL[members] @ Bk) / sdk
        return Z, tr
    raise ValueError(variant)


def predict(X, f, tr, te, variant, with_as):
    F, trfit = features(X, f, tr, te, variant)
    if with_as:
        F = np.hstack([F, AS])
    return ek.nested_lr().fit(F[trfit], Y[trfit]).predict_proba(F[te])[:, 1]


def fold_job(X, f, tr, te, variants, with_as):
    return {v: predict(X, f, tr, te, v, with_as) for v in variants}


def run_matrix(X, variants, with_as):
    out = Parallel(n_jobs=12)(delayed(fold_job)(X, f, tr, te, variants, with_as) for f, (_, _, tr, te) in enumerate(J))
    return {v: c17.to_matrix(J, [o[v] for o in out]) for v in variants}


ERA_STRATA = ERA; YD = np.char.add(np.char.add(ERA, "|"), DEV)


def null_replicate(X):
    M = run_matrix(X, VARIANTS, False); row = {}
    for v in VARIANTS:
        s = M[v].mean(0); row[v] = c17.pooled_auc(M[v]); row[v + "_wera"] = r9.strat_auc(Y, s, ERA_STRATA); row[v + "_wyd"] = r9.strat_auc(Y, s, YD)
    return row


def nb_diff(a, b, n1=160, n2=40):
    d = a - b; J_ = len(d); se = np.sqrt(d.var(ddof=1) * (1 / J_ + n2 / n1)); t = d.mean() / se; tc = stats.t.ppf(0.975, J_ - 1)
    return d.mean(), d.mean() - tc * se, d.mean() + tc * se, 2 * stats.t.sf(abs(t), J_ - 1)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--dry-run"; t0 = time.time()
    P0 = A.load_models(); print(f"reproduction passed ({time.time() - t0:.0f} s)", flush=True)
    # split properties (check iv)
    for f in (0, 7, 23, 49):
        _, _, tr, te = J[f]; R, trL = split_reference(tr, f); ctrl = tr[Y[tr] == 0]
        assert set(R) <= set(ctrl) and not set(R) & set(trL) and not set(R) & set(te) and set(trL) | set(R) == set(tr) and len(set(trL) & set(ctrl)) + len(R) == len(ctrl)
        g = crossfit_groups(tr, f); assert (g[tr] >= 0).all() and (g[te] == -1).all() and set(np.bincount(g[tr])) <= set(range(30, 50))
    print("split properties of V1 and V2 passed", flush=True)
    if mode == "--dry-run":
        t1 = time.time(); Mr = run_matrix(BP, VARIANTS, False); Ma = run_matrix(BP, VARIANTS, True)
        print("real data without AS:", {v: round(c17.pooled_auc(Mr[v]), 5) for v in VARIANTS}); print("real data with AS:", {v: round(c17.pooled_auc(Ma[v]), 5) for v in VARIANTS}, f"{time.time() - t1:.0f} s")
        assert abs(c17.pooled_auc(Mr["V0"]) - 0.71042) < 1e-5 and abs(c17.pooled_auc(Ma["V0"]) - 0.71210) < 1e-5 and abs(c17.pooled_auc(Ma["C1"]) - 0.66627) < 1e-5, "V0/C1 not reproduced"
        t1 = time.time(); Xs = A.shuffle_cols(BP, [np.arange(N)], np.random.default_rng(29100)); print("H1 one replicate:", {k: round(x, 4) for k, x in null_replicate(Xs).items() if not k.endswith(("wera", "wyd"))}, f"{time.time() - t1:.0f} s")
        sys.exit()
    nrep = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    json.dump(dict(prereg_sha256=sha(os.path.join(HERE, "PRAREGISTRASI_putaran29.md")), code_sha256=sha(__file__)), open(f"{OUT}/meta.json", "w"), indent=1)
    # ---------- A. leakage (noise features)
    rows = []
    for name, groups, seed0 in (("H1_all", [np.arange(N)], 29100), ("H2a_era", [np.where(ERA == e)[0] for e in np.unique(ERA)], 29300)):
        for r in range(nrep):
            Xs = A.shuffle_cols(BP, groups, np.random.default_rng(seed0 + r)); rows.append(dict(scenario=name, rep=r, **null_replicate(Xs)))
        print(f"  {name}done ({(time.time() - t0) / 60:.1f}minutes)", flush=True)
    Rn = pd.DataFrame(rows); Rn.to_csv(f"{OUT}/A_kebocoran_replikasi.csv", index=False)
    S = Rn.groupby("scenario").agg(**{f"{v}{s}_{f}": (v + s, fn) for v in VARIANTS for s in ("", "_wera") for f, fn in (("mean", "mean"), ("sd", "std"), ("p5", lambda x: np.percentile(x, 5)), ("p95", lambda x: np.percentile(x, 95)))}).reset_index()
    S.to_csv(f"{OUT}/A_kebocoran_ringkasan.csv", index=False); print(S[["scenario"] + [f"{v}_mean" for v in VARIANTS]].round(4).to_string(index=False), flush=True)
    # ---------- B. data asli
    Mr = run_matrix(BP, VARIANTS, False); Ma = run_matrix(BP, VARIANTS, True)
    np.savez(f"{OUT}/B_prediksi_data_asli.npz", **{f"{v}_noAS": Mr[v] for v in VARIANTS}, **{f"{v}_AS": Ma[v] for v in VARIANTS})
    rowsB = []; FA = {}
    for tag, M in (("noAS", Mr), ("AS", Ma)):
        for v in VARIANTS:
            fa = c17.fold_aucs(J, M[v]); FA[(tag, v)] = fa
            se = np.sqrt(fa.var(ddof=1) * (1 / len(fa) + 40 / 160)); tc = stats.t.ppf(0.975, len(fa) - 1)
            rowsB.append(dict(input=tag, variant=v, auc_fold=fa.mean() * 100, lo=100 * (fa.mean() - tc * se), hi=100 * (fa.mean() + tc * se), auc_pooled=100 * c17.pooled_auc(M[v]),
                              within_era=100 * r9.strat_auc(Y, M[v].mean(0), ERA_STRATA), within_yd=100 * r9.strat_auc(Y, M[v].mean(0), YD)))
    B1 = pd.DataFrame(rowsB); B1.to_csv(f"{OUT}/B_kinerja_data_asli.csv", index=False); print(B1.round(2).to_string(index=False), flush=True)
    rowsD = []
    for tag in ("noAS", "AS"):
        for v in ("V0", "V1", "V2", "V3", "V4"):
            for ref in ("C1", "V0"):
                if v == ref:
                    continue
                d, lo, hi, p = nb_diff(FA[(tag, v)], FA[(tag, ref)]); rowsD.append(dict(input=tag, test=f"{v} - {ref}", diff=d, lo=lo, hi=hi, p=p))
    D1 = pd.DataFrame(rowsD); D1.to_csv(f"{OUT}/B_selisih_per_lipatan.csv", index=False); print(D1.round(4).to_string(index=False), flush=True)
    # difference real data vs. null distribution (pooled AUC, within era, within year x device); difference = variant - C1, without AS
    rowsN = []
    for v in ("V0", "V1", "V2", "V3", "V4"):
        for sc, g in Rn.groupby("scenario"):
            for metric, suf, real in (("pooled", "", c17.pooled_auc(Mr[v]) - c17.pooled_auc(Mr["C1"])), ("within_era", "_wera", r9.strat_auc(Y, Mr[v].mean(0), ERA_STRATA) - r9.strat_auc(Y, Mr["C1"].mean(0), ERA_STRATA)),
                                      ("within_year_x_device", "_wyd", r9.strat_auc(Y, Mr[v].mean(0), YD) - r9.strat_auc(Y, Mr["C1"].mean(0), YD))):
                d = (g[v + suf] - g["C1" + suf]).values
                rowsN.append(dict(variant=v, scenario=sc, metric=metric, real_diff=real, null_mean=d.mean(), null_sd=d.std(ddof=1), frac_null_ge_real=(d >= real).mean(), excess=real - d.mean()))
    N1 = pd.DataFrame(rowsN); N1.to_csv(f"{OUT}/B_data_asli_vs_nol.csv", index=False); print(N1.round(4).to_string(index=False), flush=True)
    # test within strata (bootstrap) with age/sex
    Pw = {"C1": Ma["C1"], "Floor": P0["Floor"], "V0": Ma["V0"], "V1": Ma["V1"], "V2": Ma["V2"], "V3": Ma["V3"], "V4": Ma["V4"]}
    tests = [(v, "Floor") for v in ("V0", "V1", "V2", "V3", "V4")] + [(v, "C1") for v in ("V0", "V1", "V2", "V3", "V4")]
    W = r10.within(Pw, np.arange(N), r9.STRATA, tests, 29500); W.to_csv(f"{OUT}/B_dalam_strata.csv", index=False)
    print(W[["strata", "test", "diff", "lo", "hi", "p"]].round(4).to_string(index=False), flush=True)
    # ---------- C. confounder test
    SC = {}
    for tag, M in (("noAS", Mr), ("AS", Ma)):
        for v in ("V0", "V1", "V2", "V3", "V4"):
            SC[f"{v}_{tag}"] = M[v].mean(0)
    rp, rf, i = [], [], 0
    for m, s in SC.items():
        for cn, (c, kind) in A.CONF_PARTIAL.items():
            i += 1; t0_, p = A.partial_test(s, c, kind, 29200 + i); rp.append(dict(model=m, confounder=cn, R2=t0_, p=p))
        for cn, c in A.CONF_FULL.items():
            i += 1; t0_, p = A.full_test(s, c, 29200 + i); rf.append(dict(model=m, confounder=cn, R2=t0_, p=p))
    Cp, Cf = pd.DataFrame(rp), pd.DataFrame(rf)
    for T in (Cp, Cf):
        T["family"] = "descriptive"; T["p_holm"] = np.nan
    msk = (Cp.model.isin(["V1_AS", "V2_AS"])) & (Cp.confounder == "era"); Cp.loc[msk, "family"] = "primary"; Cp.loc[msk, "p_holm"] = A.holm(Cp.loc[msk, "p"].values)
    Cp.to_csv(f"{OUT}/C_confounder_parsial.csv", index=False); Cf.to_csv(f"{OUT}/C_confounder_penuh.csv", index=False)
    print(Cp.round(4).to_string(index=False)); print(Cf.round(4).to_string(index=False), flush=True)
    print(f"\ndone in {(time.time() - t0) / 60:.1f}minutes")
