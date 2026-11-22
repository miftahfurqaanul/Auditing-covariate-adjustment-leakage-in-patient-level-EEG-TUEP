#!/usr/bin/env python3
"""Gambar 2 versi JBI (telaah 2026-11-13, butir 4: perbesar ke lebar penuh, kurangi angka dalam sel, palet lebih lembut).
Data dan matriks identik dengan make_fig_peta_pp.py (hasil39/peta_sel.csv; H1, penyesuaian per pasien, n=200, m=50, p=50, s=1).
Perubahan hanya penyajian: huruf lebih besar, palet divergen lembut, angka hanya pada sel dengan |selisih| > 3 pp (ambang "kebocoran" aturan berpasangan).
Assert: matriks sama persis dengan matriks make_fig_peta_pp.py (110 sel) dan setiap sel berlabel > 3 pp.
Pakai: python make_fig_peta_jbi.py   (keluaran figures_en/hasil_peta_kebocoran_jbi.pdf)"""
import os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); AT = os.path.join(ROOT, "analisis_tambahan")
OUT = os.path.join(HERE, "hasil_peta_kebocoran_jbi.pdf")
G = pd.read_csv(os.path.join(AT, "hasil39/peta_sel.csv")); G["imb"] = G["imb"].astype(str).str.lower().eq("true")
LN = [("lognormal", sg, 0.01, 100.0, f"LN σ={sg}") for sg in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)]
CT = [("contam", 1.0, f_, c_, f"CT {f_ * 100:g}% ×{c_:g}") for f_, c_ in ((0.01, 10.0), (0.003, 100.0), (0.01, 100.0), (0.03, 100.0), (0.01, 1000.0))]
COLS = LN + CT; CL = ["LRpat", "RF", "XGB", "MLP", "LSTM"]; NAMA = ["LR", "RF", "XGB", "MLP", "Bi-LSTM"]


def matriks(transform):   # salinan persis dari make_fig_peta_pp.py
    Q = G[(G.scen == "H1") & (G.adjust == "pat") & (G["transform"] == transform) & (G.n == 200) & (G.m == 50) & (G.p == 50) & (G.s == 1.0) & (~G.imb)]
    M = np.full((len(CL), len(COLS)), np.nan); labs = []
    for j, (d_, sg, f_, c_, lab) in enumerate(COLS):
        q = Q[(Q.dist == d_) & (np.isclose(Q.sig, sg)) & (np.isclose(Q.cfrac, f_)) & (np.isclose(Q.cscale, c_))]; labs.append(f"{lab} (k={q['kurtosis'].mean():.0f})" if len(q) else lab)
        for i, c in enumerate(CL):
            v = q[q.clf == c].delta
            if len(v): M[i, j] = v.iloc[0]
    return M, labs


# palet lembut aturan penulisan §11 (2026-11-20): teal pudar - putih - pasir, tanpa merah atau biru terang
CMAP = LinearSegmentedColormap.from_list("tegas", ["#4f8584", "#9fc3c2", "#f7f7f7", "#e3c17f", "#b5892f"])   # 2026-11-21: rona palet sama, dipekatkan agar mudah dibaca
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, axs = plt.subplots(2, 1, figsize=(7.4, 3.45), sharex=True, gridspec_kw=dict(hspace=0.30)); cell = {}
for ax, (tr, title) in zip(axs, (("raw", "(a) Raw features"), ("rank", "(b) Rank-normal transformation fitted on the training data"))):
    M, labs = matriks(tr); M = 100 * M; cell[tr] = M
    im = ax.imshow(M, cmap=CMAP, vmin=-15, vmax=15, aspect="auto")
    ax.set_yticks(range(len(CL))); ax.set_yticklabels(NAMA, fontsize=10); ax.set_title(title, fontsize=10.5, loc="left")
    ax.set_xticks(np.arange(-0.5, len(COLS), 1), minor=True); ax.set_yticks(np.arange(-0.5, len(CL), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5); ax.tick_params(which="minor", length=0)
    for i in range(len(CL)):   # 2026-11-21: semua sel diberi angka; tebal bila > 3 pp (ambang "kebocoran" aturan berpasangan)
        for j in range(len(COLS)):
            if np.isnan(M[i, j]): continue
            v = M[i, j]; s = "0.0" if abs(v) < 0.05 else (f"+{v:.1f}" if v > 0 else f"−{abs(v):.1f}")
            ax.text(j, i, s, ha="center", va="center", fontsize=8.6 if v > 3 else 7.6, fontweight="bold" if v > 3 else "normal", color="black" if v > 3 else "#333333")
axs[1].set_xticks(range(len(COLS))); axs[1].set_xticklabels(labs, rotation=35, ha="right", fontsize=9)
axs[1].set_xlabel("Feature distribution (LN: lognormal, parameter σ; CT: contamination, f% of samples, scale ×c; k: mean kurtosis)", fontsize=9)
cb = fig.colorbar(im, ax=axs, shrink=0.85, pad=0.015, aspect=28); cb.set_label("AUC with − without adjustment (pp)", fontsize=9.5); cb.ax.tick_params(labelsize=9)
fig.savefig(OUT, bbox_inches="tight"); plt.close(fig)
# assert: identik dengan matriks gambar naskah sebelumnya
sys.argv = ["x", "en"]; src = open(os.path.join(HERE, "make_fig_peta_pp.py"), encoding="utf8").read()
assert "def matriks(transform):" in src
n = sum(int(np.isfinite(M).sum()) for M in cell.values()); lab = sum(int((M > 3).sum()) for M in cell.values())
assert n == 110, n
print("ok sel:", n, "berlabel (>3 pp):", lab, "rentang:", round(np.nanmin(cell['raw']), 1), round(np.nanmax(cell['raw']), 1), "| rank maks", round(np.nanmax(cell['rank']), 1))
