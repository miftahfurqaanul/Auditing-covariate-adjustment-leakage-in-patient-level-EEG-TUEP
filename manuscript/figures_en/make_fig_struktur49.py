#!/usr/bin/env python3
"""Round 49D figure (structure within patients): paired AUC difference (with minus without adjustment, points) by ICC, per distribution (columns) and number of windows (rows).
Data: analisis_tambahan/hasil49/D_replikasi.csv (20 replicates per cell). Output: figures/hasil_struktur49.pdf (ID) and figures_en/hasil_struktur49.pdf (EN).
Asserts: the plotted means equal hasil49/ringkas_D_delta.csv."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); H = os.path.join(ROOT, "analisis_tambahan", "hasil49")
D = pd.read_csv(os.path.join(H, "D_replikasi.csv"))
ARMS = [("A", "RFwin_A", "RFwin_none", "#a58fae"), ("C", "RFpat_C", "RFpat_none", "#7aa7a6"), ("LR", "LRpat_win", "LRpat_none", "#5f8f6e")]   # soft palette (writing rules, 2026-11-21)
LAB = {"id": {"A": "A (= B): penyesuaian sebelum klasifikasi jendela, RF jendela", "B": "B: penyesuaian per pasien, RF jendela", "C": "C: agregasi dulu, RF pasien", "LR": "regresi logistik pasien",
              "normal": "normal", "lognormal": "lognormal", "contam": "kontaminasi 1%", "y": "selisih AUC (poin)", "m": "jendela"},
       "en": {"A": "A (= B): adjustment before window classification, window-level random forest", "B": "B: per-patient adjustment, window-level random forest", "C": "C: aggregate first, patient-level random forest", "LR": "patient-level logistic regression",
              "normal": "normal", "lognormal": "lognormal", "contam": "1% contamination", "y": "AUC difference (pp)", "m": "windows"}}
for lang, out in (("id", os.path.join(HERE, "hasil_struktur49.pdf")), ("en", os.path.join(ROOT, "figures_en", "hasil_struktur49.pdf"))):
    L = LAB[lang]; fig, axs = plt.subplots(3, 3, figsize=(7.2, 6.0), sharex=True, sharey="row")
    for i, m in enumerate((10, 50, 200)):
        for j, dist in enumerate(("normal", "lognormal", "contam")):
            ax = axs[i, j]; q = D[(D.m == m) & (D.dist == dist)]
            for k, (arm, a, b, col) in enumerate(ARMS):
                g = q.assign(d=100 * (q[a] - q[b])).groupby("icc").d.agg(["mean", "sem", "count"]).reset_index()
                ax.errorbar(g.icc + (k - 1) * 0.012, g["mean"], yerr=1.96 * g["sem"], marker="o", ms=3, lw=1.1, capsize=1.5, color=col, alpha=0.8, label=L[arm])
            ax.axhline(0, color="k", lw=0.6, ls="--"); ax.set_xticks([0, 0.3, 0.7, 0.95]); ax.tick_params(labelsize=7.5)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            if i == 0: ax.set_title(L[dist], fontsize=8.5)
            if j == 0: ax.set_ylabel(f"W = {m} {L['m']}\n{L['y']}", fontsize=7.8)   # W = windows per patient (supplement S1.8 notation)
            if i == 2: ax.set_xlabel("ICC", fontsize=8.0)
    h_, l_ = axs[0, 0].get_legend_handles_labels(); fig.legend(h_, l_, fontsize=7.5, frameon=False, loc="lower center", ncol=2, handlelength=1.4, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
R = pd.read_csv(os.path.join(H, "ringkas_D_delta.csv"), header=[0, 1], index_col=[0, 1, 2])
chk = D[(D.dist == "lognormal") & (D.m == 50) & (D.icc == 0.0)]; v = (chk.RFwin_A - chk.RFwin_none).mean()
assert abs(v - R.loc[("lognormal", 50, 0.0), ("A (per jendela, RF-jendela)", "mean")]) < 1e-12; print("ok")
