#!/usr/bin/env python3
"""Gambar 4 versi JBI: kebocoran penyesuaian berbasis kontrol terhadap jumlah kontrol referensi (putaran 55; hasil55/ringkas_sel.csv).
Panel (a) desain jarang (sel era kontrol jarang, seperti TUEP), (b) desain cermin. Garis per kekuatan hubungan s; garis putus-putus abu:
penyesuaian semua pasien (s = 1). Titik = rata-rata selisih AUC gabungan (pp) atas 100 replikasi; batang = 95% CI Monte Carlo.
Assert: nilai yang digambar sama dengan ringkas_sel.csv. Pakai: python make_fig_referensi_jbi.py"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
S = pd.read_csv(os.path.join(ROOT, "analisis_tambahan", "hasil55", "ringkas_sel.csv")); OUT = os.path.join(HERE, "hasil_referensi_kontrol_jbi.pdf")
plt.rcParams.update({"font.family": "DejaVu Sans"})
WARNA = {0.0: "#9b968a", 0.5: "#d4ab6a", 1.0: "#5f8f6e"}; NR = sorted(S.nref.unique())   # palet aturan penulisan §11 (2026-11-20)
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True, gridspec_kw=dict(wspace=0.08)); drawn = []
for ax, (d, judul) in zip(axs, (("jarang", "(a) Sparse design: about 6% of controls\nin the earliest era (as in TUEP)"), ("cermin", "(b) Mirror design: same association,\nsparse era cell among the cases"))):
    ax.axhspan(-1, 1, color="#ecebe6", zorder=0); ax.axhline(3, color="#888", lw=0.8, ls=":", zorder=1); ax.axhline(0, color="#bbb", lw=0.6, zorder=1)
    for s in (0.0, 0.5, 1.0):
        g = S[(S.design == d) & (S.s == s)].sort_values("nref")
        ax.errorbar(g.nref, g.leak_CR, yerr=g.leak_CR_ci, color=WARNA[s], alpha=0.75, mec="white", mew=0.5, marker="o", ms=4.5, lw=1.6, capsize=2, label=f"control-referenced, s = {s:g}", zorder=3)
        drawn += list(zip(g.nref, g.leak_CR))
    g = S[(S.design == d) & (S.s == 1.0)].sort_values("nref")
    ax.errorbar(g.nref, g.leak_AP, yerr=g.leak_AP_ci, color="#444", marker="s", ms=3.5, lw=1.0, ls="--", capsize=2, label="all-patient, s = 1", zorder=3)
    ax.set_xscale("log"); ax.set_xticks(NR); ax.set_xticklabels([str(n) for n in NR], fontsize=9); ax.minorticks_off()
    ax.set_xlabel("Training controls in the reference (per fold)", fontsize=9); ax.set_title(judul, fontsize=9, loc="left"); ax.tick_params(labelsize=9)
axs[0].set_ylabel("Leak: AUC with − without adjustment (pp)", fontsize=9)
axs[1].legend(fontsize=8, frameon=False, loc="upper right")
fig.savefig(OUT, bbox_inches="tight"); plt.close(fig)
assert len(drawn) == 30
print("ok", OUT)
