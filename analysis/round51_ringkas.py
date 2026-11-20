#!/usr/bin/env python3
"""Putaran 51: ringkasan sesuai PRAREGISTRASI_putaran51.md (ditulis dan di-hash SEBELUM keluaran utama dilihat). CPU saja; membaca hasil51/.
Bagian: E2 (plasmode: tabel pipeline, keluarga primer, sekunder, kontrol), E1 (hukum keterpulihan), T (Bi-LSTM, sekunder), E3 (label nyata), E3b (uji confounder parsial).
Pemakaian: CUDA_VISIBLE_DEVICES="" python round51_ringkas.py [E2|E1|T|E3|semua]"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); OUT = os.environ.get("R51_OUT", os.path.join(HERE, "hasil51"))          # R51_OUT: hanya untuk uji dengan data palsu
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from round42_ringkas import dec_R1, dec_R2, holm
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
FC = [f"fold_auc_{i}" for i in range(5)]; MARGIN = 0.01


def load(fname):
    p = os.path.join(OUT, fname)
    if not os.path.exists(p):
        return None
    D = pd.read_csv(p); D["fold_mean"] = D[FC].astype(float).mean(axis=1); D["pipe"] = D["transform"] + "-" + D["adjust"]; return D


def table(D):
    rows = []
    for (coh, fam, cond, g), G in D.groupby(["cohort", "fam", "cond", "gamma"]):
        W = G.pivot_table(index="rep", columns="pipe", values="fold_mean")
        for pipe in W.columns:
            x = W[pipe].dropna().values; m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x)); adj = not pipe.endswith("-none")
            r = dict(cohort=coh, fam=fam, cond=cond, gamma=g, pipe=pipe, nrep=len(x), auc=m, se=se, R1=dec_R1(m, se) if adj else "--")
            if adj and "raw-none" in W.columns:
                d = (W[pipe] - W["raw-none"]).dropna().values; dm, ds = d.mean(), d.std(ddof=1) / np.sqrt(len(d)); r.update(delta=dm, delta_se=ds, R2=dec_R2(dm, ds) if cond == "H1" else "n/a")
            rows.append(r)
    return pd.DataFrame(rows)


def paired(D, coh, fam, cond, a, b, g=1.0):
    """d = AUC(a) - AUC(b) per replikasi (rata-rata 5 lipatan)."""
    W = D[(D.cohort == coh) & (D.fam == fam) & (D.cond == cond) & (D.gamma == g)].pivot_table(index="rep", columns="pipe", values="fold_mean"); return (W[a] - W[b]).dropna().values


def one_sided(d, null=0.0, greater=True):
    se = d.std(ddof=1) / np.sqrt(len(d)); t = (d.mean() - null) / se; p = stats.t.sf(t, len(d) - 1) if greater else stats.t.cdf(t, len(d) - 1); return d.mean(), se, p


def part_E2():
    parts = [x for x in (load("tuep_RF.csv"), load("tuep_XGB.csv"), load("emotiv_RF.csv"), load("emotiv_XGB.csv")) if x is not None]
    if not parts:
        print("E2: belum ada keluaran"); return
    D = pd.concat(parts, ignore_index=True); S = table(D); S.to_csv(f"{OUT}/ringkas_E2_pipeline.csv", index=False)
    for (coh, fam), G in S.groupby(["cohort", "fam"]):
        print(f"\n===== E2 {coh} {fam} =====\n" + G.sort_values(["cond", "gamma", "pipe"]).drop(columns=["cohort", "fam"]).round(4).to_string(index=False))
    # keluarga primer per kohort (Holm, 4 uji): (i) H1: rank-pat - rank-ash > 0; (ii) E: rank-ash - rank-pat < MARGIN (non-inferioritas)
    prim = []
    for coh in ("tuep", "emotiv"):
        rows = []
        for fam in ("RF", "XGB"):
            if not ((D.cohort == coh) & (D.fam == fam)).any():
                continue
            m, se, p = one_sided(paired(D, coh, fam, "H1", "rank-pat", "rank-ash")); rows.append(dict(cohort=coh, fam=fam, uji="H1: rank-pat - rank-ash > 0", d=m, se=se, p=p))
            m, se, p = one_sided(paired(D, coh, fam, "E", "rank-ash", "rank-pat"), MARGIN, greater=False); rows.append(dict(cohort=coh, fam=fam, uji=f"E: rank-ash - rank-pat < {MARGIN}", d=m, se=se, p=p))
        if rows:
            R = pd.DataFrame(rows); R["p_holm"] = holm(R.p.values); prim.append(R)
    if prim:
        P = pd.concat(prim, ignore_index=True); P.to_csv(f"{OUT}/ringkas_E2_primer.csv", index=False); print("\n===== E2 keluarga primer (Holm per kohort, 4 uji) =====\n" + P.round(5).to_string(index=False))
    # sekunder (Holm per kohort x famili, 4 uji), deskriptif bila famili tidak lengkap
    sec = []
    for (coh, fam), _ in D.groupby(["cohort", "fam"]):
        rows = []
        for cond, a, b, lab in (("H1", "raw-pat", "raw-ash", "H1: raw-pat - raw-ash > 0"), ("H1", "log-pat", "log-ash", "H1: log-pat - log-ash > 0"),
                                ("E", "rank-eb", "rank-ash", "E: rank-eb - rank-ash > 0"), ("H1", "rank-eb", "rank-ash", "H1: rank-eb - rank-ash > 0")):
            m, se, p = one_sided(paired(D, coh, fam, cond, a, b)); rows.append(dict(cohort=coh, fam=fam, uji=lab, d=m, se=se, p=p))
        R = pd.DataFrame(rows); R["p_holm"] = holm(R.p.values); sec.append(R)
    T = pd.concat(sec, ignore_index=True); T.to_csv(f"{OUT}/ringkas_E2_sekunder.csv", index=False); print("\n===== E2 sekunder (Holm per kohort x famili) =====\n" + T.round(5).to_string(index=False))
    # kontrol
    print("\n===== E2 kontrol =====")
    for (coh, fam), G in S.groupby(["cohort", "fam"]):
        A = G[(G.gamma == 0.0) & (G.pipe != "raw-none")]; k = int((A.R1 != "kebocoran").sum()); print(f"K1 {coh} {fam}: gamma 0, sel tersesuaikan bukan 'kebocoran' = {k}/{len(A)} ({100 * k / max(len(A), 1):.1f}%; batas 95%)")
        e = G[(G.cond == "E") & (G.gamma == 1.0) & (G.pipe == "raw-none")]
        if len(e):
            print(f"K2 {coh} {fam}: kontrol positif E, raw-none gamma 1 AUC {e.auc.iloc[0]:.4f} (SE {e.se.iloc[0]:.4f}); syarat > 0.55")
    print(f"K3 AUC kovariat (eta -> y_sim) per kohort x gamma: {D.groupby(['cohort', 'gamma'])['auc_cov'].mean().round(4).to_dict()}")


def part_E1():
    for coh in ("tuep", "emotiv"):
        R = load_rho(coh); E2 = [load(f"{coh}_{f}.csv") for f in ("RF", "XGB")]
        if R is None or all(x is None for x in E2):
            print(f"E1 {coh}: belum ada keluaran"); continue
        rho = R.groupby(["cond", "pipe"])["rho"].agg(["mean", lambda v: v.std(ddof=1) / np.sqrt(len(v))]).rename(columns={"mean": "rho", "<lambda_0>": "rho_se"}).reset_index()
        rows = []
        for x in E2:
            if x is None:
                continue
            fam = x.fam.iloc[0]; A = x[x.gamma == 1.0].groupby(["cond", "pipe"])["fold_mean"].mean().rename("auc").reset_index(); M = rho.merge(A, on=["cond", "pipe"])
            s = stats.spearmanr(M.rho, M.auc); low = M[M.rho <= 0.10]; rows.append(dict(cohort=coh, fam=fam, n_sel=len(M), spearman=s.correlation, p=s.pvalue, n_rho_le_010=len(low), maks_auc_rho_le_010=low.auc.max() if len(low) else np.nan))
            M.assign(fam=fam).to_csv(f"{OUT}/ringkas_E1_sel_{coh}_{fam}.csv", index=False); print(f"\n----- E1 {coh} {fam}: sel (gamma 1) -----\n" + M.sort_values("rho").round(4).to_string(index=False))
        L = pd.DataFrame(rows); L.to_csv(f"{OUT}/ringkas_E1_{coh}.csv", index=False); print(f"\n===== E1 {coh}: hukum keterpulihan =====\n" + L.round(4).to_string(index=False))


def load_rho(coh):
    p = os.path.join(OUT, f"rho_{coh}.csv")
    if not os.path.exists(p):
        return None
    R = pd.read_csv(p); R["pipe"] = R["transform"] + "-" + R["adjust"]; return R


def part_T():
    D = load("tuep_Tang.csv")
    if D is None:
        print("T: belum ada keluaran"); return
    W = []
    for g, G in D.groupby("gamma"):
        P = G.pivot_table(index="rep", columns="pipe", values="fold_mean")
        for pipe in P.columns:
            x = P[pipe].dropna().values; m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x)); r = dict(gamma=g, pipe=pipe, nrep=len(x), auc=m, se=se, R1=dec_R1(m, se) if not pipe.endswith("-none") else "--")
            if not pipe.endswith("-none"):
                d = (P[pipe] - P["raw-none"]).dropna().values; r.update(delta=d.mean(), delta_se=d.std(ddof=1) / np.sqrt(len(d)), R2=dec_R2(d.mean(), d.std(ddof=1) / np.sqrt(len(d))))
            W.append(r)
    T = pd.DataFrame(W); T.to_csv(f"{OUT}/ringkas_T_tang.csv", index=False); print("\n===== Bi-LSTM (Tang), H1 (sekunder) =====\n" + T.round(4).to_string(index=False))
    rows = []
    for a, b in (("raw-pat", "raw-ash"), ("rank-pat", "rank-ash")):
        d = paired(D.assign(cohort="tuep"), "tuep", "Tang", "H1", a, b); m, se, p = one_sided(d); rows.append(dict(uji=f"H1 gamma 1: {a} - {b} > 0", d=m, se=se, p=p))
    R = pd.DataFrame(rows); R["p_holm"] = holm(R.p.values); print(R.round(5).to_string(index=False))


def part_E3():
    p = os.path.join(OUT, "nyata_prediksi.npz")
    if not os.path.exists(p):
        print("E3: belum ada keluaran"); return
    import round49_ringkas as R49s
    orig, strict, fold, pooled, Y = R49s.matrices(); nb = R49s.nb; Z = np.load(p); assert "NM" in strict, "NM ketat tidak lengkap"
    M = dict(strict); M.update({k: Z[k] for k in ("I2eb", "N1eb", "P3eb")}); M["F2eb"] = (M["I2eb"] + M["P3eb"] + M["NM"]) / 3
    rows = []
    for m in ("I2", "I2eb", "P3", "P3eb", "NM", "F2", "F2eb", "C1", "Floor", "N0", "N1", "N1eb"):
        f = fold(M[m]); ci = nb(f); rows.append(dict(model=m, per_lipatan=f.mean(), ci_lo=ci[1], ci_hi=ci[2], gabungan=pooled(M[m])))
    A = pd.DataFrame(rows); A.to_csv(f"{OUT}/ringkas_E3_auc.csv", index=False); print("\n===== E3 AUC (label nyata, nesting ketat, 50 lipatan) =====\n" + A.round(4).to_string(index=False))
    pr = []
    for a, b in (("F2eb", "F2"), ("N1eb", "N0")):
        d = nb(fold(M[a]), fold(M[b])); pr.append(dict(kontras=f"{a} - {b}", selisih=d[0], ci_lo=d[1], ci_hi=d[2], p=d[3]))
    P = pd.DataFrame(pr); P["p_holm"] = holm(P.p.values); P.to_csv(f"{OUT}/ringkas_E3_primer.csv", index=False); print("\n===== E3 keluarga primer (Holm, 2 uji) =====\n" + P.round(4).to_string(index=False))
    de = []
    for a, b in (("I2eb", "I2"), ("P3eb", "P3"), ("N1eb", "N1"), ("I2eb", "C1"), ("F2eb", "Floor"), ("F2eb", "C1")):
        d = nb(fold(M[a]), fold(M[b])); de.append(dict(kontras=f"{a} - {b}", selisih=d[0], ci_lo=d[1], ci_hi=d[2], p=d[3]))
    Q = pd.DataFrame(de); Q.to_csv(f"{OUT}/ringkas_E3_deskriptif.csv", index=False); print("\n----- E3 kontras deskriptif -----\n" + Q.round(4).to_string(index=False))
    L = pd.read_csv(f"{OUT}/nyata_lambda.csv"); print("\n----- E3 faktor penyusutan ash (rata-rata atas 50 lipatan) -----\n" + L.groupby("fitur").mean(numeric_only=True).drop(columns="fold").round(3).to_string())
    part_E3b(M, Y)


def part_E3b(M, Y, B=10000, seed=51003):
    """Uji confounder parsial (H0: prediksi independen dari kovariat bersyarat label). y biner, sehingga permutasi baris kovariat DI DALAM kelas label eksak
    (Berrett dkk. 2020, bagian 2.1). Kovariat: empat dummy era (usia dan jenis kelamin adalah fitur masukan AS semua model ini, sehingga ujinya trivial; lihat pra-registrasi).
    Model tanpa ACQ saja. Statistik: R2([1, y, era]) - R2([1, y]) untuk skor pasien = rata-rata prediksi atas 10 ulangan. Deskriptif (lihat Catatan Nygaard di pra-registrasi)."""
    import round28_analisis as A
    Dc = A.D_I2[:, 3:]; assert Dc.shape[1] == 4 and set(np.unique(Dc)) == {0.0, 1.0}; rng = np.random.default_rng(seed); cls = [np.where(Y == c)[0] for c in (0, 1)]; X0 = np.c_[np.ones(len(Y)), Y]
    def r2(s, X):
        b = np.linalg.lstsq(X, s, rcond=None)[0]; return 1 - ((s - X @ b) ** 2).sum() / ((s - s.mean()) ** 2).sum()
    perms = []
    for _ in range(B):
        pi = np.arange(len(Y))
        for ix in cls:
            pi[ix] = rng.permutation(ix)
        perms.append(pi)
    rows = []
    for m in ("C1", "I2", "I2eb", "P3", "P3eb", "NM", "F2", "F2eb"):
        s = np.nanmean(M[m], axis=0); base = r2(s, X0); obs = r2(s, np.c_[X0, Dc]) - base; null = np.array([r2(s, np.c_[X0, Dc[pi]]) - base for pi in perms])
        rows.append(dict(model=m, dR2=obs, null_median=float(np.median(null)), p=(1 + (null >= obs).sum()) / (1 + B)))
    T = pd.DataFrame(rows); T["p_holm"] = holm(T.p.values); T.to_csv(f"{OUT}/ringkas_E3b_confounder_parsial.csv", index=False); print("\n===== E3b uji confounder parsial (deskriptif) =====\n" + T.round(4).to_string(index=False))


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "semua"
    for k, f in (("E2", part_E2), ("E1", part_E1), ("T", part_T), ("E3", part_E3)):
        if part in (k, "semua"):
            f()
