#!/usr/bin/env python3
"""Putaran 61, bagian deskriptif (2026-11-20, ulasan editorial JBI butir 2, 3, 5): ringkasan ulang hasil TERSIMPAN, tanpa melatih model dan tanpa
replikasi baru. Bukan bagian pra-registrasi putaran 61 (yang hanya mencakup nesting ketat); semua keluaran deskriptif.
Keluaran (hasil61/):
  h2a_berpasangan.csv        butir 5: AUC dan selisih berpasangan terhadap C1 (95% CI t atas 100 replikasi) untuk H1 dan H2a (hasil/5_hamdan_replikasi.csv, hasil29)
  audit_baru_per_aturan.csv  butir 2: putaran 60, hitungan keputusan R1/R2/gabungan per kategori dan ukuran audit (pembilang/penyebut), 95% CI Clopper-Pearson
  audit_baru_per_sel.csv     butir 2: idem per sel
  kumpulan_per_aturan.csv    butir 2: putaran 59, frekuensi keputusan R1/R2/gabungan dari pengambilan ulang kumpulan tersimpan (rata-rata dan maksimum atas kumpulan)
  contoh_perencanaan.csv     butir 3: contoh perencanaan n dengan pilot 20 replikasi kontrol negatif (plasmode gamma=0) dan kepekaannya terhadap ketidakpastian rerata pilot
Pemakaian: python round61_tabel_ulasan.py"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from scipy import stats
O = os.path.join(HERE, "hasil61"); os.makedirs(O, exist_ok=True)


def tci(d):
    d = 100 * np.asarray(d, float); n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n); lo, hi = stats.t.interval(0.95, n - 1, m, se); return m, d.std(ddof=1), lo, hi


# ---- butir 5: H2a sebagai uji sensitivitas kondisional (AUC dua pipeline dan selisih berpasangan)
rows = []
H = pd.read_csv(os.path.join(HERE, "hasil", "5_hamdan_replikasi.csv"))
for sc in ("H1_all", "H2a_era"):
    x = H[H.scenario == sc]; assert len(x) == 100
    m, sd, lo, hi = tci(x.I2_noAS - x.C1_noAS)
    rows.append(dict(seri="audit terencana", skenario=sc, varian="V0", auc_C1=100 * x.C1_noAS.mean(), auc_var=100 * x.I2_noAS.mean(), selisih=m, sd=sd, lo=lo, hi=hi, n=len(x)))
S = pd.read_csv(os.path.join(HERE, "hasil29", "A_kebocoran_replikasi.csv"))
for sc in ("H1_all", "H2a_era"):
    x = S[S.scenario == sc]; assert len(x) == 100
    for v in ("V0", "V1", "V2", "V3", "V4"):
        m, sd, lo, hi = tci(x[v] - x.C1)
        rows.append(dict(seri="seri varian", skenario=sc, varian=v, auc_C1=100 * x.C1.mean(), auc_var=100 * x[v].mean(), selisih=m, sd=sd, lo=lo, hi=hi, n=len(x)))
pd.DataFrame(rows).to_csv(os.path.join(O, "h2a_berpasangan.csv"), index=False)

# ---- butir 2a: audit independen putaran 60 (replikasi baru), per aturan, dengan pembilang/penyebut
A = pd.read_csv(os.path.join(HERE, "hasil60", "audit_independen.csv"))
for r in ("R1", "R2", "gab"):
    A[f"{r}_tp"] = A[f"{r}_tp"]; assert ((A[f"{r}_tb"] + A[f"{r}_tp"] + A[f"{r}_bocor"]) == A.audits).all()


def cp(k, n):
    lo = 0.0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1); hi = 1.0 if k == n else stats.beta.ppf(0.975, k + 1, n - k); return 100 * lo, 100 * hi


cols = [f"{r}_{d}" for r in ("R1", "R2", "gab") for d in ("tb", "tp", "bocor")]
A[["code", "truth", "n", "audits", "mean_d_fresh"] + cols].to_csv(os.path.join(O, "audit_baru_per_sel.csv"), index=False)
G = A.groupby(["truth", "n"])[["audits"] + cols].sum().reset_index(); G["sel"] = A.groupby(["truth", "n"]).size().values
for r in ("R1", "R2", "gab"):
    G[f"{r}_bocor_lo"], G[f"{r}_bocor_hi"] = zip(*[cp(k, n) for k, n in zip(G[f"{r}_bocor"], G.audits)])
tot = A.groupby("truth")[["audits", "gab_bocor", "R1_bocor", "R2_bocor"]].sum()
for t in tot.index:
    k, n = int(tot.loc[t, "gab_bocor"]), int(tot.loc[t, "audits"]); lo, hi = cp(k, n)
    G = pd.concat([G, pd.DataFrame([dict(truth=t, n=0, audits=n, gab_bocor=k, gab_bocor_lo=lo, gab_bocor_hi=hi, R1_bocor=int(tot.loc[t, "R1_bocor"]), R2_bocor=int(tot.loc[t, "R2_bocor"]))])], ignore_index=True)
G.to_csv(os.path.join(O, "audit_baru_per_aturan.csv"), index=False)

# ---- butir 2b: pengambilan ulang kumpulan tersimpan (putaran 59), per aturan
F = pd.read_csv(os.path.join(HERE, "hasil59", "frekuensi_gabungan.csv"))
for r in ("R1", "R2"): F[f"{r}_tp"] = 1 - F[f"{r}_tb"] - F[f"{r}_bocor"]
K = F[F.jenis.isin(["nol", "positif"])].copy(); K["kat"] = K.sumber + " " + K.jenis
key = [f"{r}_{d}" for r in ("R1", "R2", "gab") for d in ("tb", "tp", "bocor")]
M = K.groupby(["kat", "n"])[key].agg(["mean", "max", "min"]); M.columns = [f"{a}_{b}" for a, b in M.columns]
M = (100 * M).reset_index(); M["kumpulan"] = K.groupby(["kat", "n"]).size().values
M["undian"] = M.kat.map(lambda s: {"EEG": 10000, "plasmode": 4000, "simulasi": 2000}[s.split()[0]])
M.to_csv(os.path.join(O, "kumpulan_per_aturan.csv"), index=False)

# ---- butir 3: contoh perencanaan n dari pilot kontrol negatif
# kontrol negatif: pipeline random forest pada fitur mentah dengan penyesuaian rata-rata pasien, label sintetis tanpa hubungan label-kovariat (plasmode gamma=0),
# selisih berpasangan terhadap pipeline yang sama tanpa penyesuaian; pilot = 20 replikasi pertama yang direncanakan (rep 0..19); pembanding = 120 replikasi.
FC = [f"fold_auc_{i}" for i in range(5)]


def prep(D):
    D = D.copy(); D["fold_mean"] = D[FC].astype(float).mean(1); D["pipe"] = D["transform"] + "-" + D["adjust"]; return D


old = prep(pd.read_csv(os.path.join(HERE, "hasil42", "RF.csv"))); old = old[(old.cond == "H1") & (old.gamma == 0.0) & (old.rep < 30)]
new = prep(pd.read_csv(os.path.join(HERE, "hasil50", "A_RF.csv"))); new = new[(new.gamma == 0.0)]
W = pd.concat([old, new]).pivot_table(index="rep", columns="pipe", values="fold_mean").sort_index(); assert len(W) == 120
d = 100 * (W["raw-pat"] - W["raw-none"]).values; c, z = 3.0, 2.33


def n_req(b, s):
    return math.inf if b >= c else math.ceil((z * s / (c - b)) ** 2)


rows = []
for lab, x in (("pilot 20", d[:20]), ("kumpulan 120", d)):
    b, s, m = x.mean(), x.std(ddof=1), len(x); se = s / np.sqrt(m)
    for q, nm in ((None, "rerata"), (0.80, "batas atas satu sisi 80%"), (0.95, "batas atas satu sisi 95%")):
        bq = b if q is None else b + stats.t.ppf(q, m - 1) * se
        rows.append(dict(sumber=lab, m=m, b=b, s=s, se_b=se, skenario_b=nm, b_pakai=bq, n_perlu=n_req(bq, s)))
    for b0 in (0.0, 1.0, 2.0):
        rows.append(dict(sumber=lab, m=m, b=b, s=s, se_b=se, skenario_b=f"b tetap {b0:g}", b_pakai=b0, n_perlu=n_req(b0, s)))
P = pd.DataFrame(rows); P.to_csv(os.path.join(O, "contoh_perencanaan.csv"), index=False)
pd.set_option("display.width", 250)
for f in ("h2a_berpasangan", "audit_baru_per_aturan", "contoh_perencanaan"):
    print(f"\n== {f}"); print(pd.read_csv(os.path.join(O, f + ".csv")).round(2).to_string())
print("\n== kumpulan_per_aturan"); print(M.round(1)[["kat", "n", "kumpulan", "undian"] + [f"{k}_mean" for k in key] + ["gab_bocor_max", "R2_bocor_max"]].to_string())

# ---- butir minor: alur subjek Emotiv (313 -> 299), dari cache jendela tersimpan (tuep_fresh/code/external_prep.py: jendela 10 s, ditolak bila
# amplitudo puncak-ke-puncak atau giroskop melebihi ambang per negara median + 3 MAD tanpa label; round50_emotiv.py: MINWIN = 5)
T = "/home/miftah_f/RTX-5090/tuep_fresh/data"
s = pd.read_csv(f"{T}/ext_zenodo_subjects.csv"); zw = np.load(f"{T}/ext_zenodo_whole_windows.npz"); u, c = np.unique(zw["sid"], return_counts=True)
s["nwin"] = s.sid.map(dict(zip(u, c))).fillna(0).astype(int)
alur = [dict(tahap="subjek dalam rilis", n=len(s), epilepsi=int((s.group == "epilepsy").sum()), kontrol=int((s.group == "control").sum())),
        dict(tahap="tanpa jendela bersih", n=int((s.nwin == 0).sum()), epilepsi=int(((s.nwin == 0) & (s.group == "epilepsy")).sum()), kontrol=int(((s.nwin == 0) & (s.group == "control")).sum())),
        dict(tahap="1-4 jendela bersih", n=int(((s.nwin > 0) & (s.nwin < 5)).sum()), epilepsi=int(((s.nwin > 0) & (s.nwin < 5) & (s.group == "epilepsy")).sum()), kontrol=int(((s.nwin > 0) & (s.nwin < 5) & (s.group == "control")).sum())),
        dict(tahap="dianalisis (>= 5 jendela)", n=int((s.nwin >= 5).sum()), epilepsi=int(((s.nwin >= 5) & (s.group == "epilepsy")).sum()), kontrol=int(((s.nwin >= 5) & (s.group == "control")).sum()),
             jendela=int(s.loc[s.nwin >= 5, "nwin"].sum()))]
assert alur[-1]["n"] == 299 and alur[0]["n"] == 313
pd.DataFrame(alur).to_csv(os.path.join(O, "emotiv_alur.csv"), index=False); print("\n== emotiv_alur"); print(pd.DataFrame(alur).to_string())

# ---- butir 1 (lanjutan, deskriptif di luar pra-registrasi): frekuensi keputusan R1/R2/gabungan pada kumpulan KETAT putaran 61,
# dengan bootstrap indeks replikasi berpasangan yang sama seperti putaran 59 (B = 10,000, benih 61), dan pembanding mode longgar
from round49_kalibrasi import decide, THR
RNG = np.random.default_rng(61)


def freqs(x, d, n, B=10000):
    idx = RNG.integers(0, len(x), (B, n)); X, Dd = x[idx], d[idx]
    r1 = decide(X.mean(1), X.std(1, ddof=1) / np.sqrt(n), THR["R1"]["asli"]); r2 = decide(Dd.mean(1), Dd.std(1, ddof=1) / np.sqrt(n), THR["R2"]["asli"])
    j = np.where((r1 == 2) | (r2 == 2), 2, np.where((r1 == 0) & (r2 == 0), 0, 1)); f = lambda a: np.bincount(a, minlength=3) / B
    return f(r1), f(r2), f(j)


rows = []
for mode in ("longgar", "ketat"):
    Q = pd.read_csv(os.path.join(O, f"replikasi_{mode}.csv"))
    for v in ("V0", "VR", "V4"):
        for n in (20, 50, 100):
            f1, f2, fj = freqs(Q[v].values, (Q[v] - Q.C1).values, n)
            rows.append(dict(mode=mode, varian=v, n=n, mean_x=100 * Q[v].mean(), mean_d=100 * (Q[v] - Q.C1).mean(),
                             **{f"{r}_{k}": 100 * f[i] for r, f in (("R1", f1), ("R2", f2), ("gab", fj)) for i, k in enumerate(("tb", "tp", "bocor"))}))
FK = pd.DataFrame(rows); FK.to_csv(os.path.join(O, "frekuensi_ketat.csv"), index=False)
print("\n== frekuensi_ketat"); print(FK.round(1).to_string())
