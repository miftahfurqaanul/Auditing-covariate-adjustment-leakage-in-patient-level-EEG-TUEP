#!/usr/bin/env python3
"""Redraws ONLY hasil_kinerja.pdf (Fig. 4) with a legend in 8.2 pt (effective font >= 5.5 pt at 0.80 of the text width; fix from review round 2, P2-R3.M4).
Data identical to make_fig_hasil_ijmi.py (same rows; that generator is NOT run again so the other three figures do not change). Asserts: values and labels match eval_metrik.csv (in the script) and the numeric text of the old figure (below). CPU only."""
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
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
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
a.legend(loc="upper center", bbox_to_anchor=(0.45, -0.24), ncol=2, fontsize=8.2, frameon=False)
b = ax[1]; grid = np.linspace(0, 1, 201)
for k, lab, col, ls in [("F2_adj", "F2", "f2", "-"), ("I2_adj", "I2", "scratch", "-"), ("Floor", "Metadata baseline", "non", "--")]:
    tprs = []
    for p in EV[k]:
        f, t, _ = roc_curve(Y, p); tprs.append(np.interp(grid, f, t))
    auc = M.loc[k.replace("_adj", "-adj")].auc_pooled; b.plot(grid, np.mean(tprs, 0), color=C[col], ls=ls, lw=1.4, label=f"{lab} (pooled AUC {auc:.1f})")
b.plot([0, 1], [0, 1], color="k", ls=":", lw=.8, label="Chance"); b.set_xlabel("False positive rate (1 − specificity)"); b.set_ylabel("Sensitivity")
b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=8.2, frameon=False); b.set_title("(b)", loc="left", fontweight="bold"); b.set_aspect("equal")
fig.tight_layout(); fig.savefig(HERE / "hasil_kinerja.pdf"); plt.close(fig)
import subprocess, re
from collections import Counter
# English version: numeric tokens are compared with the Indonesian figure used by naskah_id (figures/hasil_kinerja.pdf); text labels differ by language
numtk = lambda p: Counter(x for x in subprocess.run(["pdftotext", str(p), "-"], capture_output=True, text=True).stdout.split() if re.fullmatch(r"[-\u2212+]?\d+(\.\d+)?%?", x) and not re.fullmatch(r"[01]\.\d{1,2}", x))
a_, b_ = numtk(BASE / "figures/hasil_kinerja.pdf"), numtk(HERE / "hasil_kinerja.pdf"); assert a_ == b_, (a_ - b_, b_ - a_); print("ok: token numerik gambar Inggris = gambar Indonesia (kecuali tick ROC)")
