#!/usr/bin/env python3
"""Round 42: plasmode summary according to PRAREGISTRASI_putaran42.md (section 3). CPU only; reads hasil42/{LR,RF,XGB,Tang,NM}.csv. Usage: python round42_ringkas.py"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "hasil42")
import numpy as np, pandas as pd
from scipy import stats

FC = [f"fold_auc_{i}" for i in range(5)]; TREE = {"RF", "XGB"}; AUC_ETA_REAL = 0.7102
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)


def dec_R1(m, se):
    return "tidak ada bukti" if (m <= 0.52 and m + 1.645 * se <= 0.54) else ("kebocoran" if m > 0.55 else "tidak pasti")


def dec_R2(m, se):
    return "tidak ada bukti" if (m <= 0.01 and m + 1.645 * se <= 0.03) else ("kebocoran" if m > 0.03 else "tidak pasti")


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); h = np.empty(len(p)); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, p[i] * (len(p) - r))); h[i] = run
    return h


def load():
    parts = []
    for f in ("LR", "RF", "XGB", "Tang", "NM"):
        p = f"{OUT}/{f}.csv"
        if os.path.exists(p):
            parts.append(pd.read_csv(p))
    D = pd.concat(parts, ignore_index=True)
    for c in FC + ["auc", "gamma", "auc_cov"]:
        D[c] = D[c].astype(float)
    D["fold_mean"] = D[FC].mean(axis=1); D["pipe"] = D["transform"] + "-" + D["adjust"]; return D


def is_adj(fam, adjust):
    return adjust not in ("none", "N0")


def baseline_pipe(fam, transform):
    return "raw-none" if fam in TREE else (("raw-N0") if fam == "NM" else f"{transform}-none")


def main():
    D = load(); print(f"rows: {len(D)}; families: {D["fam"].value_counts().to_dict()}")
    rows = []; drows = []
    for (fam, cond, g), G in D.groupby(["fam", "cond", "gamma"]):
        W = {m: G.pivot_table(index="rep", columns="pipe", values=m) for m in ("fold_mean", "auc")}
        for pipe in W["fold_mean"].columns:
            t, a = pipe.split("-"); r = dict(fam=fam, cond=cond, gamma=g, pipe=pipe, adjusted=is_adj(fam, a), nrep=int(W["fold_mean"][pipe].notna().sum()))
            for m in ("fold_mean", "auc"):
                x = W[m][pipe].dropna().values; mean, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x)); r[f"{m}_mean"], r[f"{m}_se"] = mean, se
                r[f"R1_{m}"] = dec_R1(mean, se) if r["adjusted"] and cond in ("H1", "H2a") else "n/a"
                bp = baseline_pipe(fam, t)
                if r["adjusted"] and bp in W[m].columns:
                    d = (W[m][pipe] - W[m][bp]).dropna().values; dm, ds = d.mean(), d.std(ddof=1) / np.sqrt(len(d)); r[f"delta_{m}"], r[f"delta_se_{m}"] = dm, ds
                    r[f"R2_{m}"] = dec_R2(dm, ds) if cond == "H1" else "n/a"; r[f"delta_p_{m}"] = stats.ttest_1samp(d, 0).pvalue
            r["auc_cov"] = G["auc_cov"].mean(); rows.append(r)
    S = pd.DataFrame(rows); S.to_csv(f"{OUT}/ringkas_pipelines.csv", index=False)
    cols = ["fam", "cond", "gamma", "pipe", "nrep", "fold_mean_mean", "fold_mean_se", "R1_fold_mean", "delta_fold_mean", "R2_fold_mean", "auc_mean", "R1_auc"]
    for fam in S["fam"].unique():
        print(f"\n===== {fam}: H1 =====\n" + S[(S.fam == fam) & (S.cond == "H1")].sort_values(["gamma", "pipe"])[cols].round(4).to_string(index=False))
        if (S.fam == fam).any() and (S[S.fam == fam].cond == "H2a").any():
            print(f"\n----- {fam}: H2a (adjustment only) -----\n" + S[(S.fam == fam) & (S.cond == "H2a") & S.adjusted].sort_values(["gamma", "pipe"])[cols].round(4).to_string(index=False))
        if (S[S.fam == fam].cond == "nyata").any():
            print(f"\n----- {fam}: kondisi nyata (deskriptif) -----\n" + S[(S.fam == fam) & (S.cond == "nyata")].sort_values(["gamma", "pipe"])[["gamma", "pipe", "fold_mean_mean", "fold_mean_se", "delta_fold_mean"]].round(4).to_string(index=False))
    # dose-response (H1): slope of the AUC fold_mean against gamma per pipeline
    sl = []
    for (fam, pipe), G in D[D.cond == "H1"].groupby(["fam", "pipe"]):
        if G["gamma"].nunique() < 2:
            continue
        r = stats.linregress(G["gamma"].values, G["fold_mean"].values); sl.append(dict(fam=fam, pipe=pipe, slope=r.slope, se=r.stderr, p=r.pvalue, n=len(G)))
    SL = pd.DataFrame(sl); SL["p_holm"] = SL.groupby("fam")["p"].transform(lambda p: holm(p.values)); SL.to_csv(f"{OUT}/ringkas_dosis_respons.csv", index=False); print("\n===== Slope of AUC against gamma (H1; Holm within family) =====\n" + SL.round(4).to_string(index=False))
    # keluarga primer
    pr = []
    for fam in ("RF", "XGB", "Tang"):
        G = D[(D.fam == fam) & (D.cond == "H1") & (D.gamma == 1.0)]
        if G.empty:
            continue
        W = G.pivot_table(index="rep", columns="pipe", values="fold_mean"); d = (W["raw-pat"] - W["rank-pat"]).dropna().values; t = stats.ttest_1samp(d, 0); p1 = t.pvalue / 2 if t.statistic > 0 else 1 - t.pvalue / 2
        r1 = S[(S.fam == fam) & (S["cond"] == "H1") & (S["gamma"] == 1.0) & (S["pipe"] == "rank-pat")]["R1_fold_mean"].iloc[0]; pr.append(dict(fam=fam, nrep=len(d), D=d.mean(), se=d.std(ddof=1) / np.sqrt(len(d)), p_satu_sisi=p1, R1_rank_pat=r1))
    if pr:
        P = pd.DataFrame(pr); P["p_holm"] = holm(P.p_satu_sisi.values); P["klaim"] = (P.p_holm < 0.05) & (P.R1_rank_pat != "kebocoran"); P.to_csv(f"{OUT}/ringkas_primer.csv", index=False); print("\n===== Keluarga primer: AUC(raw-pat) - AUC(rank-pat), H1, gamma=1 =====\n" + P.round(4).to_string(index=False))
    # controls
    print("\n===== Controls =====")
    for fam in S["fam"].unique():
        A = S[(S.fam == fam) & S.adjusted & (S.gamma == 0.0) & S.cond.isin(["H1", "H2a"])]; print(f"K1 {fam}: gamma=0, cells with adjustment 'tidak ada bukti' = {int((A.R1_fold_mean == 'tidak ada bukti').sum())}/{len(A)} ({100 * (A.R1_fold_mean == 'tidak ada bukti').mean():.1f}%; limit 95%)")
        U = S[(S.fam == fam) & (~S.adjusted) & (S.cond == "H1")]; print(f"K3 {fam}: without adjustment H1, mean fold_mean per gamma = {U.groupby('gamma')['fold_mean_mean'].mean().round(4).to_dict()}; maximum across cells {U['fold_mean_mean'].max():.4f}")
    cov = D.groupby("gamma")["auc_cov"].mean().round(4).to_dict(); print(f"K2 covariate AUC (eta -> y_sim) mean per gamma: {cov}; AUC eta -> real label {AUC_ETA_REAL}")
    k4 = S[(S.fam == "RF") & (S.cond == "H1") & (S["pipe"] == "raw-win")]
    if len(k4):
        print("K4 RF raw-win H1:\n" + k4[["gamma", "fold_mean_mean", "fold_mean_se", "R1_fold_mean", "delta_fold_mean", "R2_fold_mean"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
