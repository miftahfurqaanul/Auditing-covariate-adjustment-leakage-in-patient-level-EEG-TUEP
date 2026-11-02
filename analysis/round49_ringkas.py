#!/usr/bin/env python3
"""Round 49: summary (CPU) of the raw output in hasil49/. Parts: A (strict nesting), C (EEG over non-EEG), B (operating characteristics of R1/R2),
D (repeated structure), E/F (weighted strata, kurtosis). Raw results are not changed; output to hasil49/ringkas_*.csv.
Usage: CUDA_VISIBLE_DEVICES="" python round49_ringkas.py [A|C|B|D|E|semua]"""
import os, sys, json, glob
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
OUT = os.path.join(HERE, "hasil49")


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1.0, run)
    return adj


def nb(a, b=None, n1=160, n2=40):
    a = np.asarray(a, float); d = a - (0.0 if b is None else np.asarray(b, float)); J = len(d); se = np.sqrt(d.var(ddof=1) * (1 / J + n2 / n1)); tc = stats.t.ppf(0.975, J - 1)
    return d.mean(), d.mean() - tc * se, d.mean() + tc * se, 2 * stats.t.sf(abs(d.mean() / se), J - 1) if b is not None else np.nan


def matrices():
    import round28_analisis as A
    J, Y, N = A.J, A.Y, A.N; Z = np.load(os.path.join(HERE, "hasil30", "eval_prediksi_v4.npz")); K = np.load(f"{OUT}/A_prediksi_ketat.npz")
    orig = {"I2": Z["I2_adj"], "P3": Z["P3_adj"], "NM": Z["NM_adj"], "C1": Z["C1"], "Floor": Z["Floor"], "F2": Z["F2_adj"]}; orig["F2 tanpa NM"] = (orig["I2"] + orig["P3"]) / 2
    strict = {k: K[k] for k in ("I2", "P3", "C1", "Floor", "N0", "N1", "N2")}
    fs = sorted(glob.glob(f"{OUT}/nm_ketat/r*_k*.json"))
    if len(fs) == 50:
        import round4 as r4
        pos = {s: i for i, s in enumerate(r4.SUBJ)}; M = np.full((10, N), np.nan)
        for f in fs:
            d = json.load(open(f))
            for s, p in d["test"].items():
                M[d["r"], pos[s]] = p
        strict["NM"] = M; strict["F2"] = (strict["I2"] + strict["P3"] + M) / 3
    strict["F2 tanpa NM"] = (strict["I2"] + strict["P3"]) / 2
    fold = lambda P: np.array([roc_auc_score(Y[te], P[r, te]) for r, _, _, te in J]); pooled = lambda P: float(np.mean([roc_auc_score(Y, p) for p in P]))
    return orig, strict, fold, pooled, Y


def part_A():
    orig, strict, fold, pooled, Y = matrices(); rows = []
    for m in ("I2", "P3", "NM", "C1", "Floor", "F2", "F2 tanpa NM"):
        if m not in strict:
            continue
        fo, fs = fold(orig[m]), fold(strict[m]); d = nb(fs, fo); ci = nb(fs)
        rows.append(dict(model=m, asli_per_lipatan=fo.mean(), ketat_per_lipatan=fs.mean(), ketat_ci_lo=ci[1], ketat_ci_hi=ci[2], asli_gabungan=pooled(orig[m]), ketat_gabungan=pooled(strict[m]),
                         selisih=d[0], selisih_ci_lo=d[1], selisih_ci_hi=d[2], selisih_p=d[3], max_abs_beda_prediksi=float(np.nanmax(np.abs(strict[m] - orig[m])))))
    T = pd.DataFrame(rows); T.to_csv(f"{OUT}/ringkas_A_ketat_vs_asli.csv", index=False); print("== 49A: ketat vs asli ==\n" + T.round(4).to_string(index=False))
    rows = []
    for a, b in (("F2", "Floor"), ("F2", "C1"), ("F2", "I2"), ("I2", "C1"), ("F2 tanpa NM", "Floor"), ("F2 tanpa NM", "C1")):
        if a in strict and b in strict:
            d = nb(fold(strict[a]), fold(strict[b])); rows.append(dict(kontras=f"{a} - {b} (keduanya ketat)", selisih=d[0], ci_lo=d[1], ci_hi=d[2], p=d[3]))
    T2 = pd.DataFrame(rows); T2.to_csv(f"{OUT}/ringkas_A_kontras_ketat.csv", index=False); print("\n== 49A: kontras di antara model ketat ==\n" + T2.round(4).to_string(index=False))


def part_C():
    orig, strict, fold, pooled, Y = matrices(); rows = []; F = {m: fold(strict[m]) for m in ("N0", "N1", "N2", "Floor")}
    for m in ("N0", "N1", "N2", "Floor"):
        ci = nb(F[m]); rows.append(dict(model=m, per_lipatan=F[m].mean(), ci_lo=ci[1], ci_hi=ci[2], gabungan=pooled(strict[m])))
    T = pd.DataFrame(rows); print("== 49C: AUC ==\n" + T.round(4).to_string(index=False))
    rng = np.random.default_rng(49300); pos, neg = np.where(Y == 1)[0], np.where(Y == 0)[0]; rows2 = []; ps = []
    for a in ("N1", "N2"):
        d = nb(F[a], F["N0"]); ps.append(d[3]); bs = []
        for _ in range(2000):
            idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]; bs.append(np.mean([roc_auc_score(Y[idx], strict[a][r, idx]) - roc_auc_score(Y[idx], strict["N0"][r, idx]) for r in range(10)]))
        lo, hi = np.percentile(bs, [2.5, 97.5]); rows2.append(dict(kontras=f"{a} - N0", selisih=d[0], ci_lo=d[1], ci_hi=d[2], p=d[3], boot_lo=lo, boot_hi=hi))
    T2 = pd.DataFrame(rows2); T2["holm"] = holm(ps); d = nb(F["N0"], F["Floor"]); T2 = pd.concat([T2, pd.DataFrame([dict(kontras="N0 - Floor (deskriptif)", selisih=d[0], ci_lo=d[1], ci_hi=d[2], p=d[3])])])
    T.to_csv(f"{OUT}/ringkas_C_auc.csv", index=False); T2.to_csv(f"{OUT}/ringkas_C_kontras.csv", index=False); print("\n== 49C: contrasts (Holm over two primary tests) ==\n" + T2.round(4).to_string(index=False))


def part_B():
    F = pd.read_csv(f"{OUT}/B_frekuensi.csv"); E = F[F.sumber == "EEG"]
    print("== 49B: kumpulan EEG ==\n" + E[["kumpulan", "aturan", "jenis", "ukuran_kumpulan", "rata2", "sd", "n", "p_tidak_ada_bukti", "p_tidak_pasti", "p_kebocoran", "ci_kebocoran"]].round(3).to_string(index=False))
    for src in ("plasmode", "simulasi"):
        G = F[F.sumber == src].groupby(["aturan", "jenis", "n"])[["p_tidak_ada_bukti", "p_tidak_pasti", "p_kebocoran"]].agg(["mean", "max", "count"])
        print(f"\n== 49B: {src} (mean and maximum across cells) ==\n" + G.round(3).to_string())
    S = F[(F.sumber == "simulasi") & (F.n == 20)].copy()
    for rule, bins in (("R1", [0, 0.51, 0.52, 0.53, 0.54, 0.55, 0.57, 0.60, 1]), ("R2", [-1, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 1])):
        q = S[S.aturan == rule].copy(); q["bin"] = pd.cut(q.rata2, bins); g = q.groupby("bin", observed=True)[["p_tidak_ada_bukti", "p_tidak_pasti", "p_kebocoran"]].agg(["mean", "count"])
        g.to_csv(f"{OUT}/ringkas_B_oc_empiris_{rule}.csv"); print(f"\n== 49B: empirical OC {rule} (per-fold simulation cells, n = 20) ==\n" + g.round(3).to_string())
    Sx = pd.read_csv(f"{OUT}/B_sensitivitas_ambang.csv"); g = Sx.groupby(["sumber", "aturan"])[["berubah_ketat", "berubah_longgar"]].agg(["sum", "count"]); g.to_csv(f"{OUT}/ringkas_B_ambang.csv")
    print("\n== 49B: published decisions that change ==\n" + g.to_string()); print(Sx[(Sx.sumber == "EEG") & (Sx.berubah_ketat | Sx.berubah_longgar)].to_string(index=False))
    O = pd.read_csv(f"{OUT}/B_oc_parametrik.csv"); q = O[(O.sd_label == "median") & (O.mu.isin([0.5, 0.52, 0.53, 0.55, 0.56, 0.58, 0.6]) | (O.aturan == "R2") & O.mu.isin([0, 0.01, 0.02, 0.03, 0.04, 0.05]))]
    print("\n== 49B: OC parametrik (sd median) ==\n" + q.round(3).to_string(index=False))
    if os.path.exists(f"{OUT}/B_pasien_prediksi.npz"):
        import round28_analisis as A
        Y = A.Y; z = np.load(f"{OUT}/B_pasien_prediksi.npz"); rng = np.random.default_rng(49400); pos, neg = np.where(Y == 1)[0], np.where(Y == 0)[0]; rows = []
        a29 = pd.read_csv(os.path.join(HERE, "hasil29", "A_kebocoran_replikasi.csv")); a29 = a29[a29.scenario == "H1_all"].sort_values("rep")
        for v in ("C1", "V4"):
            Ms = [z[f"{v}_{r}"] for r in range(100)]; rep = np.array([np.mean([roc_auc_score(Y, p) for p in M]) for M in Ms]); err = np.abs(rep - a29[v].values).max()
            Pall = np.vstack(Ms); bs = []                                                                      # (100 replicates x 10 repetitions, 200 patients)
            for _ in range(2000):
                idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]; yy = Y[idx]; n1 = yy.sum(); n0 = len(yy) - n1; R = stats.rankdata(Pall[:, idx], axis=1)
                bs.append(((R[:, yy == 1].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)).mean())
            rows.append(dict(varian=v, rata2_100=rep.mean(), sd_replikasi=rep.std(ddof=1), se_mc_n100=rep.std(ddof=1) / 10, se_mc_n20=rep.std(ddof=1) / np.sqrt(20), sd_pasien_100rep=np.std(bs, ddof=1), cek_vs_r29=err))
        T = pd.DataFrame(rows); T.to_csv(f"{OUT}/ringkas_B_pasien.csv", index=False); print("\n== 49B: Monte Carlo error vs patient variation ==\n" + T.round(4).to_string(index=False))


def part_D():
    D = pd.read_csv(f"{OUT}/D_replikasi.csv"); arms = {"A (per jendela, RF-jendela)": ("RFwin_A", "RFwin_none"), "B (per pasien, RF-jendela)": ("RFwin_B", "RFwin_none"), "C (agregasi dulu, RF-pasien)": ("RFpat_C", "RFpat_none"),
                                                        "LR per jendela": ("LRpat_win", "LRpat_none"), "LR per pasien": ("LRpat_pat", "LRpat_none")}
    for k, (a, b) in arms.items():
        D[k] = D[a] - D[b]
    G = D.groupby(["dist", "m", "icc"])[list(arms)].agg(["mean", "sem"]); G.to_csv(f"{OUT}/ringkas_D_delta.csv"); print("== 49D: delta (mean, SE) ==\n" + G.round(3).to_string())
    L = D[(D.dist == "lognormal") & (D.m == 50)]; a1 = L[L.icc == 0.95]["A (per jendela, RF-jendela)"].values; a0 = L[L.icc == 0.0]["A (per jendela, RF-jendela)"].values
    t1 = stats.ttest_ind(a1, a0, equal_var=False, alternative="greater"); q = L[L.icc == 0.0]; t2 = stats.ttest_rel(q["A (per jendela, RF-jendela)"], q["C (agregasi dulu, RF-pasien)"], alternative="greater")
    T = pd.DataFrame([dict(uji="D1: delta A ICC 0.95 - ICC 0 (lognormal, m=50)", selisih=a1.mean() - a0.mean(), p=t1.pvalue), dict(uji="D2: delta A - delta C pada ICC 0 (lognormal, m=50)", selisih=(q["A (per jendela, RF-jendela)"] - q["C (agregasi dulu, RF-pasien)"]).mean(), p=t2.pvalue)])
    T["holm"] = holm(T.p.values); T.to_csv(f"{OUT}/ringkas_D_primer.csv", index=False); print("\n== 49D: keluarga primer ==\n" + T.round(4).to_string(index=False))
    R = D.groupby(["dist", "m", "icc"])[[c for c in D.columns if c.startswith(("RF", "LR"))]].mean(); R.to_csv(f"{OUT}/ringkas_D_auc.csv")


def part_E():
    E = pd.read_csv(f"{OUT}/E_strata_bobot.csv"); print("== 49E ==\n" + E.round(4).to_string(index=False)); K = pd.read_csv(f"{OUT}/F_kurtosis.csv"); print("\n== 49F ==\n" + K.to_string(index=False))


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "semua"
    for p, f in (("A", part_A), ("C", part_C), ("B", part_B), ("D", part_D), ("E", part_E)):
        if part in (p, "semua"):
            f(); print()
