#!/usr/bin/env python3
"""Round 40: summary (CPU) - R1/R2 on pooled AUC vs per-fold mean, H40-1 (pooled bias), H40-3 (LODO diagnostics), NM and ROH stages. Implementation follows addendum2 (+ addendum3).
Raw results are not changed; all output goes to hasil40/ringkas_*.csv. Usage: CUDA_VISIBLE_DEVICES="" python round40_ringkas.py [sim|nm|roh|semua]"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "hasil40")
import numpy as np, pandas as pd
from scipy import stats

CELLK = ["dist", "scen", "n", "m", "p", "imb", "s", "sig", "cfrac", "cscale"]; FC = [f"fold_auc_{i}" for i in range(5)]; TREE = {"RF", "ET", "XGB"}; RNG = np.random.default_rng(40)


def dec_R1(m, se):
    return "tidak ada bukti" if (m <= 0.52 and m + 1.645 * se <= 0.54) else ("kebocoran" if m > 0.55 else "tidak pasti")


def dec_R2(m, se):
    return "tidak ada bukti" if (m <= 0.01 and m + 1.645 * se <= 0.03) else ("kebocoran" if m > 0.03 else "tidak pasti")


def msd(x):
    x = np.asarray(x, float); n = len(x); return x.mean(), (x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan), n


def load():
    parts = []
    for st, ch in (("A", "cpu"), ("C", "cpu"), ("GB", "gpu"), ("GC", "gpu")):
        f = f"{OUT}/{st}_{ch}.csv"
        if os.path.exists(f):
            d = pd.read_csv(f); d["stage"] = st; parts.append(d)
    D = pd.concat(parts, ignore_index=True)
    for c in FC + ["auc", "skew", "Dq05", "cusp", "kurtosis", "s"]:
        D[c] = D[c].astype(float)
    D["fold_mean"] = D[FC].mean(axis=1); return D


def paired(D, metric):
    keys = CELLK + ["rep", "clf"]; b = D[D["adjust"] == "none"][keys + ["transform", metric]].rename(columns={metric: "base"})
    bT = b[(b["transform"] == "raw") & b["clf"].isin(TREE)].drop(columns="transform").drop_duplicates(); bN = b[~b["clf"].isin(TREE)].drop_duplicates()
    X = D[D["adjust"] != "none"]
    A = pd.concat([X[~X["clf"].isin(TREE)].merge(bN, on=keys + ["transform"], how="left"), X[X["clf"].isin(TREE)].merge(bT, on=keys, how="left")], ignore_index=True)
    A["delta"] = A[metric].astype(float) - A["base"].astype(float); return A.dropna(subset=["delta"])


def sim_tables(D):
    G = CELLK + ["clf", "transform", "adjust"]; rows = []
    P = {met: paired(D, met) for met in ("auc", "fold_mean")}
    for met, A in P.items():
        for k, g in A.groupby(G):
            m1, s1, n1 = msd(g[met]); m2, s2, _ = msd(g["delta"]); r = dict(zip(G, k)); r.update(metrica=met, nrep=n1, auc_mean=m1, auc_se=s1, R1=dec_R1(m1, s1), delta_mean=m2, delta_se=s2, R2=dec_R2(m2, s2) if r["scen"] == "H1" else "n/a (H2a)")
            if n1 >= 30 and s1 > 0:
                tl, th = (m1 - 0.48) / s1, (0.52 - m1) / s1; r["TOST_p"] = max(stats.t.sf(tl, n1 - 1), stats.t.sf(th, n1 - 1))
            rows.append(r)
    T = pd.DataFrame(rows); T.to_csv(f"{OUT}/ringkas_sel_R1R2.csv", index=False)
    W = T.pivot_table(index=G, columns="metrica", values=["R1", "R2"], aggfunc="first")
    W.columns = [f"{a}_{b}" for a, b in W.columns]; W = W.reset_index(); W["R1_berganti"] = W["R1_auc"] != W["R1_fold_mean"]; W["R2_berganti"] = (W["R2_auc"] != W["R2_fold_mean"]) & (W["scen"] == "H1"); W.to_csv(f"{OUT}/ringkas_keputusan_berganti.csv", index=False)
    n1 = len(W); n2 = int((W["scen"] == "H1").sum())
    print(f"[R1] sel (adjust != none): {n1}; berganti kategori: {int(W['R1_berganti'].sum())} ({100 * W['R1_berganti'].mean():.1f}%)")
    print(f"[R2] sel H1: {n2}; berganti kategori: {int(W['R2_berganti'].sum())} ({100 * W[W['scen'] == 'H1']['R2_berganti'].mean():.1f}%)")
    for c in ("R1", "R2"):
        print(pd.crosstab(W[f"{c}_auc"], W[f"{c}_fold_mean"]).to_string())
    return T, P


def h401(D):
    N = D[(D["scen"] == "H1") & (D["transform"] == "raw") & (D["adjust"] == "none")].copy(); N["bias"] = N["fold_mean"] - N["auc"]; rows = []
    for clf, g in N.groupby("clf"):
        cells = [x["bias"].values for _, x in g.groupby(CELLK)]; fm = [x["fold_mean"].values for _, x in g.groupby(CELLK)]
        cm = np.mean([c.mean() for c in cells]); fmm = np.mean([c.mean() for c in fm])
        bs = [np.mean([RNG.choice(c, len(c)).mean() for c in cells]) for _ in range(2000)]; lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append(dict(clf=clf, ncells=len(cells), nrep_total=len(g), auc_gab=g["auc"].mean(), fold_mean=fmm, bias=cm, ci_lo=lo, ci_hi=hi, didukung=bool(lo > 0.003 and abs(fmm - 0.5) <= 0.010)))
    H = pd.DataFrame(rows); H.to_csv(f"{OUT}/ringkas_H40_1.csv", index=False); print("[H40-1]\n" + H.round(4).to_string(index=False)); return H


def lodo(U, cols, groups):
    y = U["y"].values; X = np.column_stack([np.ones(len(U))] + [U[c].values for c in cols]); pred = np.full(len(U), np.nan)
    for g in np.unique(groups):
        te = groups == g; tr = ~te
        if tr.sum() < 2:
            continue
        b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; pred[te] = X[te] @ b
    ok = ~np.isnan(pred); return 1 - np.sum((y[ok] - pred[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2)


def h403(D, P):
    A = P["fold_mean"]; A = A[(A["clf"] == "RF") & (A["scen"] == "H1") & (A["adjust"] == "pat")]
    if A.empty:
        print("[H40-3] sem dados RF"); return
    dg = D[(D["clf"] == "RF") & (D["scen"] == "H1")].groupby(CELLK).agg(skew=("skew", "mean"), Dq05=("Dq05", "mean"), cusp=("cusp", "mean"), kurt=("kurtosis", "mean")).reset_index()
    U = A.groupby(CELLK + ["transform"])["delta"].mean().reset_index().rename(columns={"delta": "y"}).merge(dg, on=CELLK); U["lk"] = np.log10(U["kurt"].clip(lower=0.1) + 1)
    U["ln"], U["lm"], U["lp"], U["imbi"] = np.log(U["n"]), np.log(U["m"]), np.log(U["p"]), U["imb"].astype(int)
    for t in ("symlog", "symlogc", "rank", "log"):
        U[f"t_{t}"] = (U["transform"] == t).astype(float)
    U["cell_id"] = U.groupby(CELLK).ngroup(); grp = U["dist"].values; base = ["lk", "ln", "lm", "lp", "imbi", "s"] + [f"t_{t}" for t in ("symlog", "symlogc", "rank", "log")]; full = base + ["skew", "Dq05", "cusp"]
    r0, r1 = lodo(U, base, grp), lodo(U, full, grp); cells = U.drop_duplicates("cell_id")[["cell_id", "dist"]]; bs = []
    for _ in range(2000):
        pick = np.concatenate([RNG.choice(cells[cells["dist"] == d]["cell_id"].values, (cells["dist"] == d).sum()) for d in cells["dist"].unique()])
        B = pd.concat([U[U["cell_id"] == c] for c in pick], ignore_index=True)
        if len(np.unique(B["dist"])) < 2:
            continue
        bs.append(lodo(B, full, B["dist"].values) - lodo(B, base, B["dist"].values))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); R = pd.DataFrame([dict(unidades=len(U), celulas=U["cell_id"].nunique(), R2_M0=r0, R2_M1=r1, dR2=r1 - r0, ci_lo=lo, ci_hi=hi, didukung=bool(r1 - r0 >= 0.10 and lo > 0))]); R.to_csv(f"{OUT}/ringkas_H40_3.csv", index=False); print("[H40-3]\n" + R.round(4).to_string(index=False))
    U.to_csv(f"{OUT}/ringkas_H40_3_unidades.csv", index=False)
    S = U[(U["dist"].isin(["normal", "t3", "chi2"])) & U["transform"].isin(["raw", "symlog", "symlogc", "rank"])][["dist", "transform", "y", "skew", "Dq05", "cusp"]]; print("[H40-3 deskriptif: sel normal/t3/chi2, delta fold_mean RF]\n" + S.round(4).to_string(index=False))


def nm_table():
    f = f"{OUT}/nm_perlipatan_replikasi.csv"
    if not os.path.exists(f):
        print("[NM] not available yet"); return
    d = pd.read_csv(f); FCs = [c for c in d.columns if c.startswith("fold_auc_")]; d["fold_mean"] = d[FCs].mean(axis=1); rows = []
    for scen, g in d.groupby("scenario"):
        W = g.pivot_table(index="rep", columns="variant", values=["auc", "fold_mean"]); W = W.dropna()
        for met in ("auc", "fold_mean"):
            n0, v4 = W[(met, "N0")], W[(met, "V4")]; dd = v4 - n0; mv, sv, n = msd(v4); m0, s0, _ = msd(n0); md, sd, _ = msd(dd)
            rows.append(dict(scenario=scen, metrica=met, nrep=n, N0_mean=m0, N0_se=s0, V4_mean=mv, V4_se=sv, R1_V4=dec_R1(mv, sv), delta=md, delta_se=sd, R2=dec_R2(md, sd) if scen == "H1" else "n/a (H2a)"))
    T = pd.DataFrame(rows); T.to_csv(f"{OUT}/ringkas_NM.csv", index=False); print("[NM]\n" + T.round(4).to_string(index=False))
    H = d[(d["scenario"] == "H1") & (d["variant"] == "N0")]; bias = (H["fold_mean"] - H["auc"]).values; bs = [RNG.choice(bias, len(bias)).mean() for _ in range(2000)]; lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"[H40-1 NM-N0 H1] n={len(H)} auc_gab={H['auc'].mean():.4f} fold_mean={H['fold_mean'].mean():.4f} bias={bias.mean():.4f} [{lo:.4f},{hi:.4f}] didukung={bool(lo > 0.003 and abs(H['fold_mean'].mean() - 0.5) <= 0.01)}")
    old = f"{HERE}/hasil30/nm_paired_replikasi.csv"
    if os.path.exists(old):
        o = pd.read_csv(old); m = d.merge(o, on=["scenario", "rep", "variant"], suffixes=("_40", "_30")); m["dd"] = (m["auc_40"] - m["auc_30"]).abs()
        print(f"[NM soft reproduction] pairs {len(m)}; max difference {m['dd'].max():.4f}; mean difference {m['dd'].mean():.4f}; signed mean {(m['auc_40'] - m['auc_30']).mean():.4f}")


def roh_table():
    f = f"{OUT}/rohira_perlipatan_replikasi.csv"
    if not os.path.exists(f):
        print("[ROH] not available yet"); return
    d = pd.read_csv(f); rows = []
    for (cell, scen), g in d.groupby(["cell", "scenario"]):
        a = msd(g["auc"]); b = msd(g["fold_mean"]); rows.append(dict(cell=cell, scenario=scen, nrep=a[2], auc_gab=a[0], gab_se=a[1], R1_gab=dec_R1(*a[:2]), fold_mean=b[0], fold_se=b[1], R1_fold=dec_R1(*b[:2]), bias=b[0] - a[0]))
    T = pd.DataFrame(rows); T.to_csv(f"{OUT}/ringkas_ROH.csv", index=False); print("[ROH]\n" + T.round(4).to_string(index=False))
    rows = []
    for scen in ("H1", "H2a"):
        for met in ("auc", "fold_mean"):
            W = d[d["scenario"] == scen].pivot_table(index="rep", columns="cell", values=met).dropna(); dd = W["RF-logpat"] - W["RF-none"]; m, s, n = msd(dd); rows.append(dict(scenario=scen, metrica=met, nrep=n, delta=m, se=s, R2=dec_R2(m, s) if scen == "H1" else "n/a (H2a)"))
    T2 = pd.DataFrame(rows); T2.to_csv(f"{OUT}/ringkas_ROH_pareado.csv", index=False); print("[ROH RF-logpat - RF-none]\n" + T2.round(4).to_string(index=False))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "semua"
    if mode in ("sim", "semua"):
        D = load(); print(f"simulation rows: {len(D)}; stage: {D['stage'].value_counts().to_dict()}"); T, P = sim_tables(D); h401(D); h403(D, P)
    if mode in ("nm", "semua"):
        nm_table()
    if mode in ("roh", "semua"):
        roh_table()
