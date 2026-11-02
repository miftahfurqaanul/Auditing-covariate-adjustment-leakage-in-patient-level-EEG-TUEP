#!/usr/bin/env python3
"""Redraws the leakage-map figure (hasil_peta_kebocoran.pdf) with a larger font and two stacked panels (adds 1 page, T1).
Data identical to analisis_tambahan/round39_analisis.py (hasil39/peta_sel.csv); that generator is NOT run. Asserts: the plotted cell values equal the annotation numbers
in the old PDF (riwayat/hasil_peta_kebocoran.before_tambah1hal.pdf). CPU only."""
import os, subprocess, numpy as np, pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); AT = os.path.join(ROOT, "analisis_tambahan")
G = pd.read_csv(os.path.join(AT, "hasil39/peta_sel.csv")); G["imb"] = G["imb"].astype(str).str.lower().eq("true")
LN = [("lognormal", sg, 0.01, 100.0, f"LN σ={sg}") for sg in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)]
CT = [("contam", 1.0, f_, c_, f"CT {f_ * 100:g}% ×{c_:g}") for f_, c_ in ((0.01, 10.0), (0.003, 100.0), (0.01, 100.0), (0.03, 100.0), (0.01, 1000.0))]
COLS = LN + CT; CL = ["LRpat", "RF", "XGB", "MLP", "LSTM"]; NAMA = ["LR", "RF", "XGB", "MLP", "LSTM"]
def matriks(transform):
    Q = G[(G.scen == "H1") & (G.adjust == "pat") & (G["transform"] == transform) & (G.n == 200) & (G.m == 50) & (G.p == 50) & (G.s == 1.0) & (~G.imb)]
    M = np.full((len(CL), len(COLS)), np.nan); labs = []
    for j, (d_, sg, f_, c_, lab) in enumerate(COLS):
        q = Q[(Q.dist == d_) & (np.isclose(Q.sig, sg)) & (np.isclose(Q.cfrac, f_)) & (np.isclose(Q.cscale, c_))]; labs.append(f"{lab} (k={q['kurtosis'].mean():.0f})" if len(q) else lab)
        for i, c in enumerate(CL):
            v = q[q.clf == c].delta
            if len(v): M[i, j] = v.iloc[0]
    return M, labs
def fmt(x): return f"{x:+.2f}".replace("+0.00", "0.00").replace("-0.00", "0.00")
fig, axs = plt.subplots(2, 1, figsize=(6.3, 4.9), sharex=True); cell = {}
for ax, (tr, title) in zip(axs, (("raw", "(a) raw features"), ("rank", "(b) rank-normal (fitted on training data)"))):
    M, labs = matriks(tr); cell[tr] = M
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.15, vmax=0.15, aspect="auto"); ax.set_yticks(range(len(CL))); ax.set_yticklabels(NAMA, fontsize=7.4); ax.set_title(title, fontsize=8, loc="left")
    for i in range(len(CL)):
        for j in range(len(COLS)):
            if not np.isnan(M[i, j]): ax.text(j, i, fmt(M[i, j]), ha="center", va="center", fontsize=7.0, color="k")
axs[1].set_xticks(range(len(COLS))); axs[1].set_xticklabels(labs, rotation=40, ha="right", fontsize=7.0)
cb = fig.colorbar(im, ax=axs, shrink=0.9, pad=0.02, aspect=30); cb.set_label("paired difference Δ (AUC with − without adjustment)", fontsize=7.4); cb.ax.tick_params(labelsize=7.0)
fig.savefig(os.path.join(HERE, "hasil_peta_kebocoran.pdf"), bbox_inches="tight"); plt.close(fig)
# assert: every annotated value in the old PDF appears with the same multiplicity
old = subprocess.run(["pdftotext", os.path.join(ROOT, "naskah_id/riwayat/hasil_peta_kebocoran.before_tambah1hal.pdf"), "-"], capture_output=True, text=True).stdout.split()
from collections import Counter
co = Counter(t for t in old if t[:1] in "+-0" and "." in t and len(t) <= 5); cn = Counter(fmt(x) for M in cell.values() for x in M.ravel() if not np.isnan(x))
print("sel lama:", sum(co.values()), "sel baru:", sum(cn.values())); extra = co - cn; assert not (cn - co) and set(extra) <= {"0.", "0.00", "0.15", "0.10", "0.05"} and sum(extra.values()) == 8, (extra, cn - co)   # rest = old colorbar scale marks + text fragments
print("ok")
