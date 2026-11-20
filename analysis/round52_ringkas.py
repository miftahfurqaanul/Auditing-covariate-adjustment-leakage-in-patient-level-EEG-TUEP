#!/usr/bin/env python3
"""Putaran 52: ringkasan sesuai PRAREGISTRASI_putaran52.md (ditulis dan di-hash SEBELUM keluaran utama dilihat). CPU saja; membaca hasil52/ dan hasil putaran lama.
Bagian: R (label nyata: NM-ash, F2 penuh ash, pembanding), N (plasmode NM), Q (kondisi nol pembanding), B (uji confounder parsial deskriptif).
Pemakaian: CUDA_VISIBLE_DEVICES="" python round52_ringkas.py [R|N|Q|B|semua]"""
import os, sys, json, glob
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
OUT = os.environ.get("R52_OUT", os.path.join(HERE, "hasil52"))               # R52_OUT: hanya untuk uji dengan data palsu
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from round42_ringkas import dec_R1, dec_R2, holm
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
FC = [f"fold_auc_{i}" for i in range(5)]


def real_matrices():
    import round49_ringkas as R49s, round28_analisis as A
    orig, strict, fold, pooled, Y = R49s.matrices(); assert "NM" in strict and "F2" in strict, "NM ketat tidak lengkap"; J, N = A.J, A.N
    M = dict(strict); K51 = np.load(os.path.join(HERE, "hasil51", "nyata_prediksi.npz")); M.update({k: K51[k] for k in ("I2eb", "N1eb", "P3eb")})
    fs = sorted(glob.glob(f"{OUT}/nm_ash/r*_k*.json")); assert len(fs) == 50, f"NM-ash: {len(fs)} dari 50 lipatan"
    import round4 as r4
    pos = {s: i for i, s in enumerate(r4.SUBJ)}; Mx = np.full((10, N), np.nan)
    for f in fs:
        d = json.load(open(f))
        for s, p in d["test"].items():
            Mx[d["r"], pos[s]] = p
    M["NMash"] = Mx; M["F2ash"] = (M["I2eb"] + M["P3eb"] + Mx) / 3
    Z33 = np.load(os.path.join(HERE, "hasil33", "pembanding_prediksi.npz")); M["DNM-V4"] = Z33["DNM_V4"]; M["Rohira-V4-AS"] = Z33["Rohira_V4_AS_AS"]
    E43 = np.load(os.path.join(HERE, "hasil43", "A_prediksi.npz"))
    for nm in ("LaBraM-eq", "CBraMod-eq", "BENDR-eq", "BP-eq"):
        M[f"{nm}-V4"] = E43[f"{nm}__AS"]
    for nm, f in (("DNM-ash", "DNM-ash"), ("Rohira-ash-AS", "Rohira-ash-AS")) + tuple((f"{n}-ash", f"{n}__AS_ash") for n in ("LaBraM-eq", "CBraMod-eq", "BENDR-eq", "BP-eq")):
        M[nm] = np.load(f"{OUT}/cpu/{f}.npy")
    return M, fold, pooled, Y, R49s.nb


def part_R():
    M, fold, pooled, Y, nb = real_matrices(); rows = []
    for m in ("I2", "I2eb", "P3", "P3eb", "NM", "NMash", "F2", "F2ash", "C1", "Floor", "DNM-V4", "DNM-ash", "Rohira-V4-AS", "Rohira-ash-AS", "LaBraM-eq-V4", "LaBraM-eq-ash", "CBraMod-eq-V4", "CBraMod-eq-ash", "BENDR-eq-V4", "BENDR-eq-ash", "BP-eq-V4", "BP-eq-ash"):
        f = fold(M[m]); ci = nb(f); rows.append(dict(model=m, per_lipatan=f.mean(), ci_lo=ci[1], ci_hi=ci[2], gabungan=pooled(M[m])))
    A_ = pd.DataFrame(rows); A_.to_csv(f"{OUT}/ringkas_R_auc.csv", index=False); print("===== R: AUC (label nyata, 50 lipatan) =====\n" + A_.round(4).to_string(index=False))
    pr = []
    for a, b in (("F2ash", "F2"), ("NMash", "NM")):
        d = nb(fold(M[a]), fold(M[b])); pr.append(dict(kontras=f"{a} - {b}", selisih=d[0], ci_lo=d[1], ci_hi=d[2], p=d[3]))
    P = pd.DataFrame(pr); P["p_holm"] = holm(P.p.values); P.to_csv(f"{OUT}/ringkas_R_primer.csv", index=False); print("\n===== R keluarga primer (Holm, 2 uji; t terkoreksi Nadeau-Bengio, dua sisi) =====\n" + P.round(4).to_string(index=False))
    sec = []
    for a, b in (("DNM-ash", "DNM-V4"), ("Rohira-ash-AS", "Rohira-V4-AS"), ("LaBraM-eq-ash", "LaBraM-eq-V4"), ("CBraMod-eq-ash", "CBraMod-eq-V4"), ("BENDR-eq-ash", "BENDR-eq-V4"), ("BP-eq-ash", "BP-eq-V4")):
        d = nb(fold(M[a]), fold(M[b])); sec.append(dict(kontras=f"{a} - {b}", selisih=d[0], ci_lo=d[1], ci_hi=d[2], p=d[3]))
    S = pd.DataFrame(sec); S["p_holm"] = holm(S.p.values); S.to_csv(f"{OUT}/ringkas_R_sekunder.csv", index=False); print("\n===== R sekunder (Holm, 6 uji) =====\n" + S.round(4).to_string(index=False))
    de = []
    for a, b in (("F2ash", "Floor"), ("F2ash", "C1"), ("F2ash", "F2eb_tanpaNM"), ("F2ash", "N0")):
        if b == "F2eb_tanpaNM":
            M[b] = (M["I2eb"] + M["P3eb"]) / 2
        d = nb(fold(M[a]), fold(M[b])); de.append(dict(kontras=f"{a} - {b}", selisih=d[0], ci_lo=d[1], ci_hi=d[2], p=d[3]))
    Q = pd.DataFrame(de); Q.to_csv(f"{OUT}/ringkas_R_deskriptif.csv", index=False); print("\n----- R kontras deskriptif -----\n" + Q.round(4).to_string(index=False))
    return M, Y


def part_N():
    fs = sorted(glob.glob(f"{OUT}/nm_plasmode_s*.csv"))
    if not fs:
        print("N: belum ada keluaran"); return
    D = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True); old = pd.read_csv(os.path.join(HERE, "hasil42", "NM.csv")); D = pd.concat([D, old[old.adjust.isin(["N0", "V4"])]], ignore_index=True)
    D["fold_mean"] = D[FC].astype(float).mean(axis=1); D["pipe"] = D.transform + "-" + D.adjust; rows = []
    for g, G in D.groupby("gamma"):
        W = G.pivot_table(index="rep", columns="pipe", values="fold_mean").dropna(subset=["raw-ash", "rank-ash"])
        for pipe in ("raw-N0", "raw-V4", "rank-V4", "raw-ash", "rank-ash"):
            x = W[pipe].values; m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x)); r = dict(gamma=g, pipe=pipe, nrep=len(x), auc=m, se=se, R1=dec_R1(m, se) if pipe != "raw-N0" else "--")
            if pipe != "raw-N0":
                d = (W[pipe] - W["raw-N0"]).values; dm, ds = d.mean(), d.std(ddof=1) / np.sqrt(len(d)); r.update(delta=dm, delta_se=ds, R2=dec_R2(dm, ds))
            rows.append(r)
    T = pd.DataFrame(rows); T.to_csv(f"{OUT}/ringkas_N_nm_plasmode.csv", index=False); print("===== N: NM plasmode H1 =====\n" + T.round(4).to_string(index=False))
    W = D[D.gamma == 1.0].pivot_table(index="rep", columns="pipe", values="fold_mean").dropna(subset=["raw-ash", "rank-ash"]); pr = []
    for a, b in (("rank-V4", "rank-ash"), ("raw-V4", "raw-ash")):
        d = (W[a] - W[b]).values; se = d.std(ddof=1) / np.sqrt(len(d)); t = d.mean() / se; pr.append(dict(uji=f"gamma 1: {a} - {b} > 0", d=d.mean(), se=se, p=stats.t.sf(t, len(d) - 1)))
    P = pd.DataFrame(pr); P["p_holm"] = holm(P.p.values); P.to_csv(f"{OUT}/ringkas_N_primer.csv", index=False); print("\n----- N primer (Holm, 2 uji, satu sisi) -----\n" + P.round(5).to_string(index=False))
    K1 = T[(T.gamma == 0.0) & T.pipe.isin(["raw-ash", "rank-ash"])]; print(f"K1 NM-ash gamma 0: bukan 'kebocoran' = {int((K1.R1 != 'kebocoran').sum())}/{len(K1)}")


def part_Q():
    p = f"{OUT}/cpu_null.csv"
    if not os.path.exists(p):
        print("Q: belum ada keluaran"); return
    Dn = pd.read_csv(p); old = pd.read_csv(os.path.join(HERE, "hasil43", "C_null_replikasi.csv")); rows = []
    for (mdl, scen), G in Dn[Dn.model.str.endswith("-ash") & Dn.model.isin(["BP-eq-ash", "LaBraM-eq-ash", "CBraMod-eq-ash", "BENDR-eq-ash"])].groupby(["model", "scen"]):
        b = old[(old.model == mdl.replace("-ash", "")) & (old.scen == scen)].set_index("rep").fold_mean; a = G.set_index("rep").fold_mean; j = a.index.intersection(b.index); d = (a.loc[j] - b.loc[j]).values
        rows.append(dict(model=mdl, scen=scen, nrep=len(j), auc_ash=a.loc[j].mean(), R1_ash=dec_R1(a.loc[j].mean(), a.loc[j].std(ddof=1) / np.sqrt(len(j))), auc_v4=b.loc[j].mean(), delta=d.mean(), delta_se=d.std(ddof=1) / np.sqrt(len(d))))
    T = pd.DataFrame(rows); T.to_csv(f"{OUT}/ringkas_Q_embedding.csv", index=False); print("===== Q: embedding beku, kondisi nol (fold_mean; delta = ash - V4) =====\n" + T.round(4).to_string(index=False))
    pe = pd.read_csv(os.path.join(HERE, "hasil33", "pembanding_null.csv")); S = Dn[Dn.model.isin(["DNM-ash", "Rohira-ash"])].groupby(["model", "scen"]).auc.agg(["count", "mean", "std", lambda v: np.percentile(v, 95)]).reset_index()
    S.columns = ["model", "scen", "n", "auc_ash_gabungan", "sd", "p95"]; print("\n----- Q: DNM dan Rohira, kondisi nol (AUC gabungan); baris V4/mentah putaran 33 sebagai rujukan -----\n" + S.round(4).to_string(index=False) + "\n" + pe.round(4).to_string(index=False))
    S.to_csv(f"{OUT}/ringkas_Q_dnm_rohira.csv", index=False)


def part_B(M=None, Y=None, B=10000, seed=51003):
    """Uji confounder parsial deskriptif, identik dengan round51_ringkas.part_E3b (empat dummy era; permutasi dalam kelas label; benih yang sama), dengan model putaran 52 ditambahkan."""
    if M is None:
        M, fold, pooled, Y, nb = real_matrices()
    import round28_analisis as A
    M = dict(M); M["F2eb"] = (M["I2eb"] + M["P3eb"] + M["NM"]) / 3; Dc = A.D_I2[:, 3:]; assert Dc.shape[1] == 4 and set(np.unique(Dc)) == {0.0, 1.0}
    rng = np.random.default_rng(seed); cls = [np.where(Y == c)[0] for c in (0, 1)]; X0 = np.c_[np.ones(len(Y)), Y]
    def r2(s, X):
        b = np.linalg.lstsq(X, s, rcond=None)[0]; return 1 - ((s - X @ b) ** 2).sum() / ((s - s.mean()) ** 2).sum()
    perms = []
    for _ in range(B):
        pi = np.arange(len(Y))
        for ix in cls:
            pi[ix] = rng.permutation(ix)
        perms.append(pi)
    rows = []
    for m in ("C1", "I2", "I2eb", "P3", "P3eb", "NM", "NMash", "F2", "F2eb", "F2ash"):
        s = np.nanmean(M[m], axis=0); base = r2(s, X0); obs = r2(s, np.c_[X0, Dc]) - base; null = np.array([r2(s, np.c_[X0, Dc[pi]]) - base for pi in perms])
        rows.append(dict(model=m, dR2=obs, null_median=float(np.median(null)), p=(1 + (null >= obs).sum()) / (1 + B)))
    T = pd.DataFrame(rows); T["p_holm"] = holm(T.p.values); T.to_csv(f"{OUT}/ringkas_B_confounder_parsial.csv", index=False); print("===== B uji confounder parsial (deskriptif; empat dummy era) =====\n" + T.round(4).to_string(index=False))


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "semua"; M = Y = None
    if part in ("R", "semua"):
        M, Y = part_R()
    if part in ("N", "semua"):
        part_N()
    if part in ("Q", "semua"):
        part_Q()
    if part in ("B", "semua"):
        part_B(M, Y)
