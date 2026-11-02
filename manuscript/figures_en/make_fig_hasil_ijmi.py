#!/usr/bin/env python3
"""Figures for Section 3 (Results), version without the control-referenced leak. CPU only. Reads analisis_tambahan/hasil30 (output of round30_eval.py) and the results of earlier rounds.
Output: figures/hasil_kinerja.pdf (Fig. 2), hasil_penyesuaian_strata.pdf (Fig. 3), hasil_kalibrasi_pasien.pdf (Fig. 4), hasil_kurva_keputusan.pdf (Fig. 5).
All numbers in the figures are computed from predictions/CSV files and checked with asserts against eval_metrik.csv. Usage: python make_fig_hasil_ijmi.py"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score

HERE = Path(__file__).resolve().parent; BASE = HERE.parent; H30 = BASE / "analisis_tambahan" / "hasil30"; REP = Path("/home/miftah_f/RTX-5090/tuep_fresh/reports"); DAT = Path("/home/miftah_f/RTX-5090/tuep_fresh/data")
EV = np.load(H30 / "eval_prediksi_v4.npz"); M = pd.read_csv(H30 / "eval_metrik.csv").set_index("model"); W = pd.read_csv(H30 / "eval_dalam_strata.csv"); T1 = pd.read_csv(H30 / "eval_selisih_per_lipatan.csv").set_index("test")
z7 = np.load(REP / "round7_fold_aucs.npz", allow_pickle=True); z25 = np.load(REP / "round25_fold_aucs.npz", allow_pickle=True)
import sys; sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code"); cwd = os.getcwd(); os.chdir("/home/miftah_f/RTX-5090/tuep_fresh/code")
import round17_common as c17; J = c17.JOBS; Y = c17.Y; os.chdir(cwd)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
C = dict(scratch="#1f77b4", pre="#d95f02", non="#7f7f7f", repro="#2ca02c", f2="#c0392b")
fa = lambda P: np.array([roc_auc_score(Y[te], P[r, te]) for r, _, _, te in J])
def nb_ci(a, n1=160, n2=40):
    m = a.mean(); h = stats.t.ppf(.975, len(a) - 1) * a.std(ddof=1) * np.sqrt(1 / len(a) + n2 / n1); return 100 * m, 100 * (m - h), 100 * (m + h)

# ---------- Fig. 2
items = []
for k, lab in [("F2_adj", "F2"), ("I2_adj", "I2"), ("P3_adj", "P3"), ("NM_adj", "NM"), ("C1", "C1 (no adjustment)")]:
    m, lo, hi = nb_ci(fa(EV[k])); mm = M.loc[k.replace("_adj", "-adj")]; assert abs(m - mm.auc_fold) < .06 and abs(lo - mm.fold_lo) < .06 and abs(hi - mm.fold_hi) < .06, k; items.append((lab, m, lo, hi, C["scratch"]))
for k, lab in [("Tang-F@199", "Tang-F (199 patients)"), ("Rohira-F", "Rohira-F"), ("DNM-F", "DNM-F")]:
    items.append((lab,) + nb_ci(z7[k]) + (C["repro"],))
items += [("LaBraM frozen",) + nb_ci(z25["LaBraM-beku"]) + (C["pre"],), ("LaBraM-FT",) + nb_ci(fa(EV["LaBraM_FT"])) + (C["pre"],), ("CBraMod-FT",) + nb_ci(fa(EV["CBraMod_FT"])) + (C["pre"],), ("BENDR frozen",) + nb_ci(z7["BENDR"]) + (C["pre"],),
          ("Metadata baseline",) + nb_ci(fa(EV["Floor"])) + (C["non"],), ("Demographic baseline",) + nb_ci(fa(EV["Demo"])) + (C["non"],)]
fig, ax = plt.subplots(1, 2, figsize=(7.4, 4.6), gridspec_kw=dict(width_ratios=[1.35, 1])); a = ax[0]; y = np.arange(len(items))[::-1]
for yi, (lab, m, lo, hi, col) in zip(y, items):
    a.errorbar(m, yi, xerr=[[m - lo], [hi - m]], fmt="o", color=col, ecolor=col, capsize=2, ms=4, lw=1)
a.set_yticks(y); a.set_yticklabels([i[0] for i in items]); a.axvline(50, color="k", ls=":", lw=.8); a.set_xlim(40, 85)
a.set_xlabel("Per-fold AUC (%)\nmean of 50 folds; Nadeau–Bengio corrected 95% CI"); a.set_title("(a)", loc="left", fontweight="bold")
for col, lab in [("scratch", "From scratch (covariate adjustment)"), ("repro", "Reproduced published methods"), ("pre", "Pretrained"), ("non", "No EEG")]:
    a.plot([], [], "o", color=C[col], label=lab)
a.legend(loc="upper center", bbox_to_anchor=(0.45, -0.24), ncol=2, fontsize=6.5, frameon=False)
b = ax[1]; grid = np.linspace(0, 1, 201)
for k, lab, col, ls in [("F2_adj", "F2", "f2", "-"), ("I2_adj", "I2", "scratch", "-"), ("Floor", "Metadata baseline", "non", "--")]:
    tprs = []
    for p in EV[k]:
        f, t, _ = roc_curve(Y, p); tprs.append(np.interp(grid, f, t))
    auc = M.loc[k.replace("_adj", "-adj")].auc_pooled; b.plot(grid, np.mean(tprs, 0), color=C[col], ls=ls, lw=1.4, label=f"{lab} (pooled AUC {auc:.1f})")
b.plot([0, 1], [0, 1], color="k", ls=":", lw=.8, label="Chance"); b.set_xlabel("False positive rate (1 − specificity)"); b.set_ylabel("Sensitivity")
b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=6.5, frameon=False); b.set_title("(b)", loc="left", fontweight="bold"); b.set_aspect("equal")
fig.tight_layout(); fig.savefig(HERE / "hasil_kinerja.pdf"); plt.close(fig)

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
    vals = [("F2", g(st, "F2-adj - Floor", "auc_F2-adj"), C["f2"]), ("I2", g(st, "I2-adj - Floor", "auc_I2-adj"), C["scratch"]), ("C1", g(st, "F2-adj - C1", "auc_C1"), "#9ecae1"), ("Metadata baseline", g(st, "F2-adj - Floor", "auc_Floor"), C["non"])]
    for j, (nm, v, col) in enumerate(vals):
        b.bar(i + (j - 1.5) * wd, 100 * v, wd, color=col, label=nm if i == 0 else None); b.text(i + (j - 1.5) * wd, 100 * v + 1, f"{100 * v:.0f}", ha="center", fontsize=5.5)
b.axhline(50, color="k", ls=":", lw=.8); b.set_xticks(x); b.set_xticklabels([s[0] for s in S], rotation=15, ha="right"); b.set_ylim(0, 100); b.set_ylabel("AUC within strata (%)"); b.legend(frameon=False, fontsize=6, loc="upper center", ncol=4, columnspacing=.8, handlelength=1); b.set_title("(b)", loc="left", fontweight="bold")
fig.tight_layout(); fig.savefig(HERE / "hasil_penyesuaian_strata.pdf"); plt.close(fig)

# ---------- Fig. 4
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw=dict(width_ratios=[1, 1.1])); a = ax[0]; a.plot([0, 1], [0, 1], color="k", ls=":", lw=.8, label="Ideal calibration")
for k, lab, col in [("F2_adj", "F2", "f2"), ("I2_adj", "I2", "scratch")]:
    p = EV[k].ravel(); yy = np.tile(Y, EV[k].shape[0]); q = np.quantile(p, np.linspace(0, 1, 11)); idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, 9)
    mp = [p[idx == i].mean() for i in range(10)]; mo = [yy[idx == i].mean() for i in range(10)]; r = M.loc[k.replace("_adj", "-adj")]
    a.plot(mp, mo, "o-", color=C[col], ms=3.5, lw=1.1, label=f"{lab} (slope {r.slope:.2f}; Brier {r.brier:.3f})")
a.set_xlabel("Predicted probability (mean per bin)"); a.set_ylabel("Observed proportion with epilepsy"); a.legend(frameon=False, fontsize=6.5, loc="upper left"); a.set_title("(a)", loc="left", fontweight="bold"); a.set_xlim(0, 1); a.set_ylim(0, 1)
b = ax[1]; mp = EV["F2_adj"].mean(0); rng = np.random.default_rng(0)
for i, (cl, lab, col) in enumerate([(0, "Non-epilepsy", C["scratch"]), (1, "Epilepsy", C["f2"])]):
    v = mp[Y == cl]; b.scatter(np.full(len(v), i) + rng.uniform(-.2, .2, len(v)), v, s=9, alpha=.6, color=col, lw=0)
    b.boxplot(v, positions=[i], widths=.45, showfliers=False, medianprops=dict(color="k"), boxprops=dict(color="k"), whiskerprops=dict(color="k"), capprops=dict(color="k"))
b.axhline(.5, color="k", ls="--", lw=.8); b.text(-0.45, .505, "Threshold 0.5", fontsize=6.5, ha="left", va="bottom")
b.set_xticks([0, 1]); b.set_xticklabels(["Non-epilepsy (n = 100)", "Epilepsy (n = 100)"]); b.set_ylabel("F2 probability of epilepsy (mean of 10 repetitions)"); b.set_ylim(0, 1); b.set_title("(b)", loc="left", fontweight="bold")
fig.tight_layout(); fig.savefig(HERE / "hasil_kalibrasi_pasien.pdf"); plt.close(fig)

# ---------- Fig. 5
d = pd.read_csv(H30 / "eval_net_benefit.csv"); p = pd.read_csv(H30 / "eval_net_benefit_prevalensi.csv"); CC = dict(F2="#c0392b", I2="#1f77b4", Floor="#7f7f7f", all="#2ca02c")
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1)); a = ax[0]
a.fill_between(d.pt, d.NB_F2_lo, d.NB_F2_hi, color=CC["F2"], alpha=.18, lw=0); a.plot(d.pt, d.NB_F2, color=CC["F2"], lw=1.6, label="F2"); a.plot(d.pt, d.NB_I2, color=CC["I2"], lw=1.1, label="I2")
a.plot(d.pt, d.NB_Floor, color=CC["Floor"], lw=1.1, ls="--", label="Metadata baseline"); a.plot(d.pt, d.treat_all, color=CC["all"], lw=1.1, ls=":", label="Treat all"); a.axhline(0, color="k", lw=.8, label="Treat none")
a.set_ylim(-0.15, 0.5); a.set_xlim(0.1, 0.9); a.set_xlabel("Probability threshold"); a.set_ylabel("Net benefit"); a.set_title("(a) Sample prevalence 0.5", fontsize=8, loc="left"); a.legend(loc="upper right", fontsize=6.5, frameon=False)
b = ax[1]; q = p[p.prevalence == 0.2]
b.plot(q.pt, q.NB_F2, color=CC["F2"], lw=1.6, label="F2"); b.plot(q.pt, q.NB_I2, color=CC["I2"], lw=1.1, label="I2"); b.plot(q.pt, q.NB_Floor, color=CC["Floor"], lw=1.1, ls="--", label="Metadata baseline")
b.plot(q.pt, q.treat_all, color=CC["all"], lw=1.1, ls=":", label="Treat all"); b.axhline(0, color="k", lw=.8); b.set_ylim(-0.15, 0.25); b.set_xlim(0.1, 0.9); b.set_xlabel("Probability threshold"); b.set_title("(b) Adjusted prevalence 0.2", fontsize=8, loc="left")
fig.tight_layout(); fig.savefig(HERE / "hasil_kurva_keputusan.pdf"); plt.close(fig)
print("OK: 4 figures created (hasil_kinerja, hasil_penyesuaian_strata, hasil_kalibrasi_pasien, hasil_kurva_keputusan)")
