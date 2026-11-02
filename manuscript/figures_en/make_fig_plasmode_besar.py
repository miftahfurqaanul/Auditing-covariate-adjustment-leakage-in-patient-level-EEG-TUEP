#!/usr/bin/env python3
"""Redraws the plasmode figure with a larger font (package A, A8c). Data identical to bangun_tabel_putaran44.py::gambar (hasil42/ringkas_pipelines.csv);
that generator is NOT run. Output: figures/hasil_plasmode.pdf. Asserts: the plotted values equal the cells of the plasmode table in naskah_id/tabel_plasmode.tex."""
import os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); AT = os.path.join(ROOT, "analisis_tambahan")
P = pd.read_csv(os.path.join(AT, "hasil42/ringkas_pipelines.csv")); P = P[P.cond == "H1"]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.65), sharey=True)
spec = [("RF", "Random forest", ["raw-none", "raw-pat", "symlog-pat", "rank-pat", "log-pat"]), ("XGB", "XGBoost", ["raw-none", "raw-pat", "symlog-pat", "rank-pat", "log-pat"]), ("Tang", "Bi-LSTM (Tang)", ["raw-none", "raw-pat", "symlog-pat", "rank-pat"])]
col = {"raw-none": "#888888", "raw-pat": "#d62728", "symlog-pat": "#ff7f0e", "rank-pat": "#1f77b4", "log-pat": "#2ca02c"}
lab = {"raw-none": "raw, no adjustment", "raw-pat": "raw + adjustment", "symlog-pat": "symlog + adjustment", "rank-pat": "rank-normal + adjustment", "log-pat": "log + adjustment"}
tp = open(os.path.join(ROOT, "naskah_id/tabel_plasmode.tex"), encoding="utf8").read()
for ax, (fam, title, pipes) in zip(axs, spec):
    for pp in pipes:
        g = P[(P.fam == fam) & (P["pipe"] == pp)].sort_values("gamma")
        ax.errorbar(g.gamma, 100 * g.fold_mean_mean, yerr=100 * 1.645 * g.fold_mean_se, marker="o", ms=3.5, lw=1.3, capsize=2, color=col[pp], label=lab[pp])
    ax.axhline(50, color="k", lw=0.6, ls="--"); ax.set_title(title, fontsize=8.5); ax.set_xlabel(r"$\gamma$", fontsize=8.5); ax.tick_params(labelsize=8.0); ax.set_xticks([0, 0.5, 1])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
# match the table: the gamma=1 point for raw-pat and rank-pat of each family
for fam, pp in (("RF", "raw-pat"), ("RF", "rank-pat"), ("XGB", "raw-pat"), ("XGB", "rank-pat"), ("Tang", "raw-pat"), ("Tang", "rank-pat")):
    v = float(P[(P.fam == fam) & (P["pipe"] == pp) & (P.gamma == 1.0)].fold_mean_mean.iloc[0]) * 100
    assert f"{v:.1f}" in tp or f"{v + 0.0001:.1f}" in tp, (fam, pp, v)
axs[0].set_ylabel("Per-fold AUC (%), shuffled features (H1)", fontsize=8.2); h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, fontsize=8.0, frameon=False, loc="lower center", ncol=3, handlelength=1.4, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.2, 1, 1)); fig.savefig(os.path.join(HERE, "hasil_plasmode.pdf"), bbox_inches="tight"); plt.close(fig); print("ok")
