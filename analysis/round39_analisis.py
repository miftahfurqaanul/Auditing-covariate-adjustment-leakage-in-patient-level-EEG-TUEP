#!/usr/bin/env python3
"""Round 39, stages C and D (CPU): delta regression model, leakage map (figure), empirical EEG anchor, and operating characteristics of rule R2. Reads hasil39/<stage>_*.csv.
Usage: python round39_analisis.py <stage> [additional_stages ...]   (e.g. B pilot)"""
import os, sys, glob, numpy as np, pandas as pd
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
H = os.path.dirname(os.path.abspath(__file__)); stages = sys.argv[1:] or ["pilot"]; OUT = f"{H}/hasil39"
D = pd.concat([pd.read_csv(f) for st in stages for f in sorted(glob.glob(f"{OUT}/{st}_*.csv")) if "ringkasan" not in f], ignore_index=True)
for c, v in (("sig", 1.0), ("cfrac", 0.01), ("cscale", 100.0)):
    D[c] = (D[c] if c in D.columns else v); D[c] = D[c].fillna(v).astype(float)
for c in ("auc", "kurtosis", "meta_auc", "s", "n", "m", "p"): D[c] = D[c].astype(float)
D["imb"] = D["imb"].astype(str).str.lower().eq("true"); D = D.drop_duplicates(["dist", "scen", "n", "m", "p", "imb", "s", "sig", "cfrac", "cscale", "rep", "clf", "transform", "adjust"])
keys = ["dist", "scen", "n", "m", "p", "imb", "s", "sig", "cfrac", "cscale", "clf"]; base = D[D.adjust == "none"].rename(columns={"auc": "auc_none"})[keys + ["rep", "transform", "auc_none"]]
TREEC = ["RF", "ET", "XGB"]; Dn = D[D.adjust != "none"]; baseT = D[(D.adjust == "none") & (D["transform"] == "raw") & D.clf.isin(TREEC)].rename(columns={"auc": "auc_none"})[keys + ["rep", "auc_none"]]                  # pohon: baseline = raw none (invarian monoton)
A = pd.concat([Dn[~Dn.clf.isin(TREEC)].merge(base, on=keys + ["rep", "transform"], how="left"), Dn[Dn.clf.isin(TREEC)].merge(baseT, on=keys + ["rep"], how="left")], ignore_index=True); A["delta"] = A.auc - A.auc_none; A = A.dropna(subset=["delta"]);  A["logk"] = np.log10(np.clip(A["kurtosis"], 0.1, None) + 1)
NONLIN = {"LRpat": 0, "RF": 1, "ET": 1, "XGB": 1, "MLP": 1, "LSTM": 1}; A["nonlin"] = A.clf.map(NONLIN)
# ---------------- ringkasan per sel x pengklasifikasi x konfigurasi
G = A.groupby(keys + ["transform", "adjust"]).agg(n_rep=("delta", "size"), auc_none=("auc_none", "mean"), auc_adj=("auc", "mean"), delta=("delta", "mean"), sd=("delta", "std"), kurtosis=("kurtosis", "mean")).reset_index(); G["se"] = G.sd / np.sqrt(G.n_rep)
G["R2"] = np.where((G.delta <= 0.01) & (G.delta + 1.645 * G.se <= 0.03), "tidak ada bukti", np.where(G.delta > 0.03, "kebocoran", "kecil/tidak pasti")); G.to_csv(f"{OUT}/peta_sel.csv", index=False)
# ---------------- model regresi (H1; penyesuaian pat; transformasi raw vs rank): delta ~ logk * nonlin + log n + log m + log p + imb + s
R = G[(G.scen == "H1") & (G.adjust == "pat") & (G.s > 0)].copy(); res = None
try:
    import statsmodels.formula.api as smf
    R["ln"] = np.log(R.n); R["lm"] = np.log(R.m); R["lp"] = np.log(R.p); R["logk"] = np.log10(R["kurtosis"].clip(lower=0.1) + 1); R["imbi"] = R.imb.astype(int)
    for tr in R["transform"].unique():
        Rt = R[R["transform"] == tr]
        if len(Rt) > 12:
            fit = smf.ols("delta ~ logk * nonlin + ln + lm + lp + imbi + s", data=Rt.assign(nonlin=Rt.clf.map(NONLIN))).fit(cov_type="HC3"); print(f"\n== regresi delta (transformasi {tr}; n sel x klasifikasi = {len(Rt)}) =="); print(fit.summary().tables[1])
            pd.DataFrame({"koef": fit.params, "se": fit.bse, "p": fit.pvalues}).to_csv(f"{OUT}/peta_regresi_{tr}.csv")
except Exception as e:
    print("regresi dilewati:", e)
# ---------------- peta panas: delta (pat) menurut pengklasifikasi x tingkat keparahan sebaran (H1, n = 200, m = 50, p = 50, s = 1)
LN = [("lognormal", sg, 0.01, 100.0, f"LN sig={sg}") for sg in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)]
CT = [("contam", 1.0, f_, c_, f"CT {f_ * 100:g}% x{c_:g}") for f_, c_ in ((0.01, 10.0), (0.003, 100.0), (0.01, 100.0), (0.03, 100.0), (0.01, 1000.0))]
COLS = LN + CT; CL = ["LRpat", "RF", "XGB", "MLP", "LSTM"]
def panel(ax, transform, title):
    Q = G[(G.scen == "H1") & (G.adjust == "pat") & (G["transform"] == transform) & (G.n == 200) & (G.m == 50) & (G.p == 50) & (G.s == 1.0) & (~G.imb)]
    M = np.full((len(CL), len(COLS)), np.nan); labs = []
    for j, (d_, sg, f_, c_, lab) in enumerate(COLS):
        q = Q[(Q.dist == d_) & (np.isclose(Q.sig, sg)) & (np.isclose(Q.cfrac, f_)) & (np.isclose(Q.cscale, c_))]; labs.append(f"{lab}\n(k={q['kurtosis'].mean():.0f})" if len(q) else lab)
        for i, c in enumerate(CL):
            v = q[q.clf == c].delta
            if len(v): M[i, j] = v.iloc[0]
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.15, vmax=0.15, aspect="auto"); ax.set_xticks(range(len(COLS))); ax.set_xticklabels(labs, rotation=55, ha="right", fontsize=5.5); ax.set_yticks(range(len(CL))); ax.set_yticklabels(CL, fontsize=7); ax.set_title(title, fontsize=8)
    for i in range(len(CL)):
        for j in range(len(COLS)):
            if not np.isnan(M[i, j]): ax.text(j, i, f"{M[i, j]:+.2f}".replace("+0.00", "0.00").replace("-0.00", "0.00"), ha="center", va="center", fontsize=5.5, color="k")
    return im
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=True)
try:
    im = panel(axs[0], "raw", "(a) fitur mentah"); panel(axs[1], "rank", "(b) rank-normal (dipasang pada data latih)"); fig.colorbar(im, ax=axs, shrink=0.8, label="selisih berpasangan delta (AUC dengan - tanpa penyesuaian)"); fig.savefig(f"{OUT}/peta_kebocoran.pdf", bbox_inches="tight")
except Exception as e:
    print("peta panas dilewati:", e)
plt.close(fig)
# ---------------- EEG anchor: delta vs log10(kurtosis) (all basic H1 cells; markers by distribution) and the EEG points
EEG = [("Rohira RF mentah", 8481, 0.573 - 0.503), ("Rohira RF log", 2.96, 0.507 - 0.501), ("Tang LSTM mentah", 17301, 0.605 - 0.500), ("Tang LSTM symlog", -1.57, 0.515 - 0.477), ("NM (token)", 0.8, 0.515 - 0.490)]
fig, ax = plt.subplots(figsize=(5.6, 3.8)); S1 = G[(G.scen == "H1") & (G.adjust == "pat") & (G["transform"] == "raw") & (G.s == 1.0)]
for c, col in (("RF", "C3"), ("XGB", "C1"), ("MLP", "C0"), ("LSTM", "C2"), ("LRpat", "k")):
    for d_, mk in (("lognormal", "o"), ("contam", "s")):
        q = S1[(S1.clf == c) & (S1.dist == d_)]
        if len(q): ax.scatter(np.log10(q["kurtosis"].clip(lower=0.1) + 1), q.delta, marker=mk, s=12, color=col, alpha=0.75, label=f"{c} ({'lognormal' if d_ == 'lognormal' else 'kontaminasi'})")
for nm, k, d in EEG:
    ax.scatter(np.log10(max(k, 0) + 1), d, marker="*", s=80, color="m", zorder=5); ax.annotate(nm, (np.log10(max(k, 0) + 1), d), fontsize=5.5, xytext=(3, 3), textcoords="offset points")
ax.axhline(0, color="gray", lw=0.6); ax.set_xlabel("log10(kurtosis + 1) fitur jendela/detik"); ax.set_ylabel("selisih berpasangan delta"); ax.legend(fontsize=4.5, ncol=2, loc="upper left"); fig.savefig(f"{OUT}/peta_jangkar_eeg.pdf", bbox_inches="tight"); plt.close(fig)
# ---------------- operating characteristics of rule R2 (bootstrap of the replicates): specificity in null cells and power in leaking cells
def decide(d):
    se = d.std(ddof=1) / np.sqrt(len(d)); m = d.mean(); return "ok" if (m <= 0.01 and m + 1.645 * se <= 0.03) else ("leak" if m > 0.03 else "unsure")
rng = np.random.default_rng(39); rows = []
cells = A[(A.scen == "H1") & (A.adjust == "pat") & (A["transform"] == "raw") & (A.clf.isin(["RF", "XGB", "MLP"]))].groupby(keys)
for k, g in cells:
    d = g.delta.values
    if len(d) < 30: continue
    mt = d.mean(); kind = "nol (s = 0)" if k[keys.index("s")] == 0 else ("bocor (delta > 0.03)" if mt > 0.03 else "tengah")
    for R_ in (10, 20, 30, 50):
        if R_ > len(d): continue
        dec = [decide(rng.choice(d, R_, replace=True)) for _ in range(400)]; rows.append(dict(sel="|".join(map(str, k)), jenis=kind, delta_benar=mt, R=R_, p_ok=np.mean([x == "ok" for x in dec]), p_leak=np.mean([x == "leak" for x in dec]), p_unsure=np.mean([x == "unsure" for x in dec])))
OC = pd.DataFrame(rows)
if len(OC):
    OC.to_csv(f"{OUT}/oc_R2_sel.csv", index=False); print("\n== operating characteristics of R2 (mean per cell type and number of replicates) =="); print(OC.groupby(["jenis", "R"])[["p_ok", "p_leak", "p_unsure"]].mean().round(3).to_string())
print(f"\nbaris: A={len(A)}, sel={len(G)}")
