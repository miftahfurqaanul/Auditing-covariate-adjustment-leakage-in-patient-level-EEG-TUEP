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
WARNA = {0.0: "#4f8584", 0.5: "#b5892f", 1.0: "#7f5f8c"}; NR = sorted(S.nref.unique())   # palet aturan §11, rona dipekatkan dan dibedakan (2026-11-21)
GESER = {0.0: 0.93, 0.5: 1.0, 1.0: 1.0}   # pergeseran horizontal kecil (skala log) agar batang galat tidak bertumpuk; nilai x tetap jumlah kontrol
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True, gridspec_kw=dict(wspace=0.08)); drawn = []
for ax, (d, judul) in zip(axs, (("jarang", "(a) Sparse design: about 6% of controls\nin the earliest era (as in TUEP)"), ("cermin", "(b) Mirror design: same association,\nsparse era cell among the cases"))):
    ax.axhspan(-1, 1, color="#d3e5d9", zorder=0); ax.axhline(3, color="#555555", lw=1.0, ls=":", zorder=1); ax.axhline(0, color="#999999", lw=0.6, zorder=1)
    for s in (0.0, 0.5, 1.0):
        g = S[(S.design == d) & (S.s == s)].sort_values("nref")
        ax.errorbar(g.nref * GESER[s], g.leak_CR, yerr=g.leak_CR_ci, color=WARNA[s], alpha=1.0, mec="#1a1a1a", mew=0.5, marker="o", ms=5.4, lw=1.8, capsize=2.2, label=f"control-referenced, s = {s:g}", zorder=3)
        drawn += list(zip(g.nref, g.leak_CR))
        if s == 1.0:   # 2026-11-21: nilai seri s = 1 (dikutip di teks) dicetak di atas batas CI atas
            for xx, yy, ci in zip(g.nref, g.leak_CR, g.leak_CR_ci):
                ax.text(xx, yy + ci + 0.25, f"+{yy:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color=WARNA[s])
    g = S[(S.design == d) & (S.s == 1.0)].sort_values("nref")
    ax.errorbar(g.nref * 1.08, g.leak_AP, yerr=g.leak_AP_ci, color="#1a1a1a", marker="s", ms=4.2, lw=1.2, ls="--", capsize=2, label="all-patient, s = 1", zorder=4)
    ax.set_xscale("log"); ax.set_xticks(NR); ax.set_xticklabels([str(n) for n in NR], fontsize=9); ax.minorticks_off()
    ax.set_xlabel("Training controls in the reference (per fold)", fontsize=9); ax.set_title(judul, fontsize=9, loc="left"); ax.tick_params(labelsize=9)
axs[0].set_ylabel("Leak: AUC with − without adjustment (pp)", fontsize=9)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=7.4, frameon=False, handlelength=1.8, columnspacing=1.1)   # 2026-11-21: di bawah panel, tidak menimpa garis
for ax in axs: ax.set_ylim(-2.9, 10.4); ax.set_xlim(14, 1500)
fig.savefig(OUT, bbox_inches="tight"); plt.close(fig)
assert len(drawn) == 30
print("ok", OUT)
