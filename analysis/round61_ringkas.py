#!/usr/bin/env python3
"""Putaran 61: ringkasan sesuai PRAREGISTRASI_putaran61.md (di-hash sebelum hasil dilihat). Tidak melatih model."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from scipy import stats
from round49_kalibrasi import decide, THR
O = os.path.join(HERE, "hasil61")
S = pd.read_csv(os.path.join(O, "replikasi_ketat.csv")).set_index("rep").sort_index()
A = pd.read_csv(os.path.join(O, "replikasi_longgar.csv")).set_index("rep").sort_index()
P56 = pd.read_csv(os.path.join(HERE, "hasil56", "replikasi.csv")).set_index("rep").sort_index()
assert len(S) == 100 and (S.index == A.index).all() and (S.index == P56.index).all()
LAB = {0: "no evidence", 1: "uncertain", 2: "leakage"}
dec = lambda x, rule: LAB[int(decide(np.array([np.mean(x)]), np.array([np.std(x, ddof=1) / np.sqrt(len(x))]), THR[rule]["asli"])[0])]


def ci(d):
    d = 100 * np.asarray(d); t, p = stats.ttest_1samp(d, 0); lo, hi = stats.t.interval(0.95, len(d) - 1, d.mean(), d.std(ddof=1) / np.sqrt(len(d)))
    return d.mean(), d.std(ddof=1), lo, hi, t, p


rows = []
print("AUC (persen) putaran 56 | longgar -> ketat; keputusan R1 longgar -> ketat")
for v in ("C1", "V0", "VR", "V4"):
    print(f"  {v}: [p56 {100 * P56[v].mean():.1f}, R1 {dec(P56[v], 'R1')}] gabungan {100 * A[v].mean():.1f} -> {100 * S[v].mean():.1f}; per lipatan {100 * A[v + '_fold'].mean():.1f} -> {100 * S[v + '_fold'].mean():.1f}; R1 {dec(A[v], 'R1')} -> {dec(S[v], 'R1')}")
    rows.append(dict(kind="AUC", name=v, p56=100 * P56[v].mean(), dec_p56=dec(P56[v], "R1"), asli=100 * A[v].mean(), ketat=100 * S[v].mean(), asli_fold=100 * A[v + "_fold"].mean(), ketat_fold=100 * S[v + "_fold"].mean(),
                     dec_asli=dec(A[v], "R1"), dec_ketat=dec(S[v], "R1")))
print("\nKontras berpasangan (pp; 95% CI atas 100 replikasi); keputusan R2 bila terhadap C1")
for a, b in (("V0", "C1"), ("V0", "VR"), ("VR", "C1"), ("V4", "C1")):
    m0, s0, l0, h0, _, _ = ci(A[a] - A[b]); m1, s1, l1, h1, _, _ = ci(S[a] - S[b]); m5, _, l5, h5, _, _ = ci(P56[a] - P56[b]); d5 = dec(P56[a] - P56[b], "R2") if b == "C1" else "-"
    d0 = dec(A[a] - A[b], "R2") if b == "C1" else "-"; d1 = dec(S[a] - S[b], "R2") if b == "C1" else "-"
    print(f"  {a} - {b}: [p56 {m5:+.2f} ({l5:+.2f} to {h5:+.2f}) R2 {d5}] longgar {m0:+.2f} (SD {s0:.2f}; {l0:+.2f} to {h0:+.2f}) R2 {d0} -> ketat {m1:+.2f} (SD {s1:.2f}; {l1:+.2f} to {h1:+.2f}) R2 {d1}")
    rows.append(dict(kind="contrast", name=f"{a}-{b}", p56=m5, p56_lo=l5, p56_hi=h5, dec_p56=d5, asli=m0, asli_lo=l0, asli_hi=h0, ketat=m1, ketat_sd=s1, ketat_lo=l1, ketat_hi=h1, dec_asli=d0, dec_ketat=d1))
print("\nKONTRAS UTAMA (perubahan ketat - longgar; Holm atas 2)")
pu = []
for a, b in (("V0", "C1"), ("V0", "VR")):
    m, s, lo, hi, t, p = ci((S[a] - S[b]) - (A[a] - A[b])); pu.append(p)
    rows.append(dict(kind="primary", name=f"delta({a}-{b})", ketat=m, ketat_sd=s, ketat_lo=lo, ketat_hi=hi, t=t, p=p))
ph = np.minimum(1, np.maximum.accumulate(np.sort(pu) * np.array([2, 1])))[np.argsort(np.argsort(pu))]
for i, (a, b) in enumerate((("V0", "C1"), ("V0", "VR"))):
    r = [x for x in rows if x["name"] == f"delta({a}-{b})"][0]; r["p_holm"] = ph[i]
    print(f"  delta({a} - {b}): {r['ketat']:+.2f} pp (SD {r['ketat_sd']:.2f}; 95% CI {r['ketat_lo']:+.2f} to {r['ketat_hi']:+.2f}); t = {r['t']:.2f}, p = {r['p']:.3g}, Holm p = {ph[i]:.3g}")
C = np.load(os.path.join(O, "C_terpilih_ketat.npz"))
print("\nC terpilih (ketat), median [min, max] atas replikasi x lipatan:")
for v in ("C1", "V0", "VR", "V4"): print(f"  {v}: {np.median(C[v]):.3g} [{C[v].min():.3g}, {C[v].max():.3g}]")
pd.DataFrame(rows).to_csv(os.path.join(O, "ringkas.csv"), index=False)
