#!/usr/bin/env python3
"""Putaran 54: ringkasan (P1, P2, sekunder, kontrol, prediksi) dan pemeriksaan reproduksi A0 terhadap putaran 53 untuk semua unit.
Lihat PRAREGISTRASI_putaran54.md. Pemakaian: python round54_ringkas.py   (CPU; keluaran hasil54/ringkas_P.csv, ringkas_P2.csv)"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import round49_kalibrasi as K
from round53_ringkas import tstat, holm

OUT = os.path.join(HERE, "hasil54"); FC = [f"fold_auc_{i}" for i in range(5)]
ARMS = ["A0", "W", "S", "WS"]; EXP = {"tuep": 120, "emotiv": 30}


def load(c):
    D = pd.read_csv(os.path.join(OUT, f"LRw4_{c}.csv")); D["fold_mean"] = D[FC].mean(1)
    D = D.drop_duplicates(subset=["rep", "gamma", "arm", "transform", "adjust"], keep="last")
    assert D.groupby("gamma").rep.nunique().to_dict() == {0.0: EXP[c], 1.0: EXP[c]}, c
    assert D.groupby(["rep", "gamma"]).size().eq(8).all(), c
    return D


def leak(D, arm, g):
    W = D[(D.arm == arm) & (D.gamma == g)].pivot_table(index="rep", columns="adjust", values="fold_mean"); assert W.notna().all().all()
    return (W["pat"] - W["none"]) * 100, W


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    L = {c: load(c) for c in ("tuep", "emotiv")}
    # reproduksi A0 = LRw putaran 53 (raw-none, raw-pat) untuk semua unit yang dipasangkan
    for c, D in L.items():
        old = pd.read_csv(os.path.join(HERE, "hasil53", f"LRw_{c}.csv")); old = old[(old["transform"] == "raw") & old.adjust.isin(["none", "pat"]) & old.gamma.isin([0.0, 1.0])]
        old = old.drop_duplicates(subset=["rep", "gamma", "adjust"], keep="last")
        m = D[D.arm == "A0"].merge(old, on=["rep", "gamma", "adjust"], suffixes=("", "_o")); assert len(m) == 4 * EXP[c], (c, len(m))
        d = max(np.abs(m[f] - m[f + "_o"]).max() for f in ["auc"] + FC); assert d < 0.01, (c, d)                                  # adendum 1
        nd = int(sum((np.abs(m[f] - m[f + "_o"]) > 1e-9).sum() for f in FC))
        mo = m.assign(fold_mean_o=m[[f + "_o" for f in FC]].mean(1)); g1n = mo[mo.gamma == 1.0].pivot_table(index="rep", columns="adjust", values="fold_mean"); g1o = mo[mo.gamma == 1.0].pivot_table(index="rep", columns="adjust", values="fold_mean_o")
        print(f"{c}: A0 terhadap CSV putaran 53 pada {len(m)} baris: selisih maks {d:.4f}; AUC lipatan berbeda {nd} dari {len(m) * 5}; "
              f"kebocoran raw-pat gamma 1 hitung ulang {(g1n['pat'] - g1n['none']).mean() * 100:.3f} poin, tersimpan {(g1o['pat'] - g1o['none']).mean() * 100:.3f} poin")
    rows = []
    for c, D in L.items():
        for g in (0.0, 1.0):
            for arm in ARMS:
                d, W = leak(D, arm, g); s = tstat(d.values)
                rows.append(dict(cohort=c, gamma=g, arm=arm, n=s["n"], auc_none=W["none"].mean() * 100, auc_pat=W["pat"].mean() * 100, R1_pat=K.LAB[K.dec_pool(W["pat"].values, "R1")],
                                 leak=s["mean"], sd=s["sd"], lo=s["lo"], hi=s["hi"], p2=s["p2"], R2=K.LAB[K.dec_pool(d.values / 100, "R2")]))
    Pt = pd.DataFrame(rows); Pt.to_csv(f"{OUT}/ringkas_P.csv", index=False)
    print("== P1 dan sekunder (kebocoran = raw-pat minus raw-none, AUC per lipatan, poin)"); print(Pt.round(3).to_string(index=False))
    rows = []
    for c, D in L.items():
        a0, _ = leak(D, "A0", 1.0); ws, _ = leak(D, "WS", 1.0); s = tstat((a0 - ws).values)
        rows.append(dict(cohort=c, n=s["n"], leak_A0=a0.mean(), leak_WS=ws.mean(), diff=s["mean"], sd=s["sd"], lo=s["lo"], hi=s["hi"], p1=s["p1"]))
    P2 = pd.DataFrame(rows); P2["holm"] = holm(P2.p1.values); P2.to_csv(f"{OUT}/ringkas_P2.csv", index=False)
    print("== P2 (A0 minus WS, gamma 1, satu sisi, Holm dua uji)"); print(P2.round(4).to_string(index=False))
    k = Pt.set_index(["cohort", "gamma", "arm"])
    ctrl = all(k.loc[(c, 0.0, a), "R2"] != "kebocoran" for c in L for a in ARMS)
    print("== prediksi")
    print(f"P54-1 WS TUEP gamma 1 R2 tidak ada bukti: {k.loc[('tuep', 1.0, 'WS'), 'R2'] == 'tidak ada bukti'} ({k.loc[('tuep', 1.0, 'WS'), 'leak']:.2f} poin; {k.loc[('tuep', 1.0, 'WS'), 'R2']})")
    print(f"P54-2 P2(a) Holm < 0.05: {P2.set_index('cohort').loc['tuep', 'holm'] < 0.05} (Holm {P2.set_index('cohort').loc['tuep', 'holm']:.2g})")
    print(f"P54-3 W < S (TUEP): {k.loc[('tuep', 1.0, 'W'), 'leak'] < k.loc[('tuep', 1.0, 'S'), 'leak']} (W {k.loc[('tuep', 1.0, 'W'), 'leak']:.2f}, S {k.loc[('tuep', 1.0, 'S'), 'leak']:.2f})")
    print(f"P54-4 kontrol gamma 0 lulus semua lengan: {ctrl}")
    print(f"P54-5 |WS Emotiv| <= 1.0: {abs(k.loc[('emotiv', 1.0, 'WS'), 'leak']) <= 1.0} ({k.loc[('emotiv', 1.0, 'WS'), 'leak']:.2f})")
