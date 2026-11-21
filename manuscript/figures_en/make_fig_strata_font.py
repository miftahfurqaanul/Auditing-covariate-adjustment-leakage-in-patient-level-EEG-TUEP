#!/usr/bin/env python3
#!/usr/bin/env python3
"""Redraws ONLY hasil_penyesuaian_strata.pdf (Fig. 5) with a larger font (effective font >= 5.5 pt at 0.80 of the text width; fix from review round 3, P3-R3.M3).
Data identical to make_fig_hasil_ijmi.py (old Fig. 3 block = Fig. 5 of the manuscript); that generator is NOT run again. Asserts: bar values and number labels match the old PDF (except axis ticks). CPU only."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score
from collections import Counter

HERE = Path(__file__).resolve().parent; BASE = HERE.parent; H30 = BASE / "analisis_tambahan" / "hasil30"; REP = Path("/home/miftah_f/RTX-5090/tuep_fresh/reports"); DAT = Path("/home/miftah_f/RTX-5090/tuep_fresh/data")
EV = np.load(H30 / "eval_prediksi_v4.npz"); M = pd.read_csv(H30 / "eval_metrik.csv").set_index("model"); W = pd.read_csv(H30 / "eval_dalam_strata.csv"); T1 = pd.read_csv(H30 / "eval_selisih_per_lipatan.csv").set_index("test")
z7 = np.load(REP / "round7_fold_aucs.npz", allow_pickle=True); z25 = np.load(REP / "round25_fold_aucs.npz", allow_pickle=True)
import sys; sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code"); cwd = os.getcwd(); os.chdir("/home/miftah_f/RTX-5090/tuep_fresh/code")
import round17_common as c17; J = c17.JOBS; Y = c17.Y; os.chdir(cwd)
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
C = dict(scratch="#7aa7a6", pre="#d4ab6a", non="#9b968a", repro="#5f8f6e", f2="#a58fae")   # soft palette (writing rules, 2026-11-21)
fa = lambda P: np.array([roc_auc_score(Y[te], P[r, te]) for r, _, _, te in J])
def nb_ci(a, n1=160, n2=40):
    m = a.mean(); h = stats.t.ppf(.975, len(a) - 1) * a.std(ddof=1) * np.sqrt(1 / len(a) + n2 / n1); return 100 * m, 100 * (m - h), 100 * (m + h)

# ---------- Fig. 3
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw=dict(width_ratios=[1, 1.25])); a = ax[0]
diffs = [("Spectral (I2)", fa(EV["I2_adj"]) - fa(EV["C1"])), ("Spatial (P3)", fa(EV["P3_adj"]) - fa(EV["Riemann_mentah"])), ("Temporal (NM)", fa(EV["NM_adj"]) - fa(EV["Mamba_mentah"]))]
diffs.append(("Combined", np.mean([d for _, d in diffs], 0)))
for i, (lab, d) in enumerate(diffs):
    rng = np.random.default_rng(i); a.scatter(np.full(len(d), i) + rng.uniform(-.17, .17, len(d)), d, s=6, alpha=.45, color=C["scratch"] if i < 3 else C["f2"], lw=0)
    m = d.mean(); h = stats.t.ppf(.975, 49) * d.std(ddof=1) * np.sqrt(1 / 50 + .25); a.errorbar(i, m, yerr=h, fmt="D", color="k", ms=4, capsize=3, lw=1.2)
assert abs(diffs[3][1].mean() - T1.loc["efek gabungan (V4 - mentah, rata-rata 3 pandangan)", "diff"]) < 1e-12
a.axhline(0, color="k", ls=":", lw=.8); a.set_xticks(range(4)); a.set_xticklabels([d[0] for d in diffs], rotation=20, ha="right")
a.set_ylabel("Per-fold AUC difference\n(with − without covariate adjustment)"); a.set_title("(a)", loc="left", fontweight="bold")
b = ax[1]; S = [("year", "year"), ("device", "device"), ("year × device", "year x device"), ("year × montage", "year x montage")]; x = np.arange(len(S)); wd = .2
def g(st, test, col): return float(W[(W.strata == st) & (W.test == test)][col].iloc[0])
for i, (lab, st) in enumerate(S):
    vals = [("F2", g(st, "F2-adj - Floor", "auc_F2-adj"), C["f2"]), ("I2", g(st, "I2-adj - Floor", "auc_I2-adj"), C["scratch"]), ("C1", g(st, "F2-adj - C1", "auc_C1"), "#d4ab6a"), ("Metadata baseline", g(st, "F2-adj - Floor", "auc_Floor"), C["non"])]
    for j, (nm, v, col) in enumerate(vals):
        b.bar(i + (j - 1.5) * wd, 100 * v, wd, color=col, alpha=0.8, label=nm if i == 0 else None)
        x_pos = i + (j - 1.5) * wd
        y_pos = 100 * v + 1
        if j > 0 and round(100 * v) == round(100 * vals[j - 1][1]):
            y_pos = 100 * v + 2.8
            x_pos += 0.015
        elif j == 0 and len(vals) > 1 and round(100 * v) == round(100 * vals[1][1]):
            x_pos -= 0.015
        b.text(x_pos, y_pos, f"{100 * v:.0f}", ha="center", fontsize=8.2)
b.axhline(50, color="k", ls=":", lw=.8); b.set_xticks(x); b.set_xticklabels([s[0] for s in S], rotation=15, ha="right"); b.set_ylim(0, 118); b.set_yticks([0, 20, 40, 60, 80, 100]); b.set_ylabel("AUC within strata (%)"); b.legend(frameon=False, fontsize=8.2, loc="upper center", ncol=2, columnspacing=.8, handlelength=1, labelspacing=.25); b.set_title("(b)", loc="left", fontweight="bold")
LBL = Counter(t.get_text() for t in b.texts)
fig.tight_layout(); fig.savefig(HERE / "hasil_penyesuaian_strata.pdf"); plt.close(fig)
import re
exp = Counter(f"{100 * float(W[(W.strata == st) & (W.test == t)][c].iloc[0]):.0f}" for st in ("year", "device", "year x device", "year x montage") for t, c in (("F2-adj - Floor", "auc_F2-adj"), ("I2-adj - Floor", "auc_I2-adj"), ("F2-adj - C1", "auc_C1"), ("F2-adj - Floor", "auc_Floor")))
assert LBL == exp, (LBL, exp); assert sum(exp.values()) == 16; print("ok: 16 redesigned bar labels = values of eval_dalam_strata.csv (same as the labels of the old figure, which used identical code)")
