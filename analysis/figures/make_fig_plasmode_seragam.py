#!/usr/bin/env python3
"""Gambar 3 versi JBI dengan baseline SERAGAM (2026-11-15; ulasan 2 butir M2 dan paket 5): setiap titik = AUC(T + penyesuaian) − AUC(T tanpa penyesuaian),
AUC per lipatan, rata-rata dengan 95% CI atas replikasi, gamma = 1. Tata letak dan gaya sama dengan make_fig_plasmode_delta.py (Gambar 3 lama, tidak ditimpa).
Sumber:
- titik raw semua famili, Bi-LSTM dan LR pasien (sudah murni): revisi_pakdwi/reviewer_20261106/plasmode_dispersi.json dan hasil53/bahan_naskah.json (sama dengan Gambar 3 lama);
- titik transformasi pohon dan LR jendela: analisis_tambahan/hasil57/empat_sel.csv (adj_in_T, CI t) = Tabel S2.15;
- NM setelah rank-normal tidak digambar (sel transformasi tanpa penyesuaian tidak dijalankan).
Assert: setiap titik raw sama dengan sel Tabel S2.12 (naskah/tabel_plasmode.tex) dan setiap titik transformasi sama dengan sel Tabel S2.15 (naskah_jbi/S2_putaran57_60.tex),
keduanya dibulatkan setengah ke atas; jumlah titik 33. Pakai: python make_fig_plasmode_seragam.py   (keluaran figures_en/hasil_plasmode_seragam.pdf)"""
import os, json, re
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); OUT = os.path.join(HERE, "hasil_plasmode_seragam.pdf")
R = json.load(open(os.path.join(ROOT, "revisi_pakdwi/reviewer_20261106/plasmode_dispersi.json")))["rows"]
D = {(r["blok"], r["baris"]): r for r in R if "delta" in r}
for k, v in json.load(open(os.path.join(ROOT, "analisis_tambahan/hasil53/bahan_naskah.json"), encoding="utf8"))["fig"].items():
    c, f, pipe = k.split("|"); D[({"tuep": "TUEP", "emotiv": "Emotiv"}[c] + " " + f, pipe)] = v
E = pd.read_csv(os.path.join(ROOT, "analisis_tambahan/hasil57/empat_sel.csv")); E = E[E.gamma == 1.0].set_index(["fam", "cohort", "T"])
def r1(x):
    v = f"{Decimal(repr(round(abs(float(x)), 10))).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP):.1f}"
    return "0.0" if v == "0.0" else (("−" if x < 0 else "+") + v)
def murni(fam, cohort, t):          # estimand seragam dari putaran 57
    r = E.loc[(fam, cohort, t)]; return dict(delta=float(r.adj_in_T), lo=float(r.adj_lo), hi=float(r.adj_hi), src="S2.15")
T = dict(raw="raw", symlog="symlog", log="log", rank="rank-normal", V4="all-patient adj.", RF="Random forest", XGB="XGBoost", Tang="Bi-LSTM (Tang)",
         LR="LR (patients, log)", LRw="LR (windows, raw)", LRp="LR (patients, raw)", NM="NormMamba (NM)", a="(a) TUEP", b="(b) Emotiv (independent cohort)",
         x="AUC with − without adjustment (pp), γ = 1", z=("≤ 1 pp: “no evidence” zone", "1–3 pp: “uncertain”", "> 3 pp: “leakage”"))
COL = {"raw": "#7f5f8c", "symlog": "#b5892f", "log": "#3f6e4f", "rank": "#4f8584", "V4": "#6e695e"}   # palet aturan §11, rona sama dipekatkan (2026-11-21)
KEY = {"raw": "raw-pat", "symlog": "symlog-pat", "log": "log-pat", "rank": "rank-pat", "V4": "raw-V4"}
FAM57 = {"RF": "RF", "XGB": "XGB", "LRw": "LRw"}
PAN = [(T["a"], "tuep", [("TUEP RF", "RF", ["raw", "symlog", "log", "rank"]), ("TUEP XGB", "XGB", ["raw", "symlog", "log", "rank"]), ("TUEP Tang", "Tang", ["raw", "rank"]),
                         ("TUEP LR (I2)", "LR", ["raw", "rank"]), ("TUEP LRw", "LRw", ["raw", "symlog", "log", "rank"]), ("TUEP LRp", "LRp", ["raw"]), ("TUEP NM", "NM", ["V4"])]),
       (T["b"], "emotiv", [("Emotiv RF", "RF", ["raw", "symlog", "log", "rank"]), ("Emotiv XGB", "XGB", ["raw", "symlog", "log", "rank"]), ("Emotiv LR", "LR", ["raw", "rank"]),
                           ("Emotiv LRw", "LRw", ["raw", "symlog", "log", "rank"]), ("Emotiv LRp", "LRp", ["raw"])])]
plotted = []; ymins = []
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw=dict(wspace=0.95))
for ax, (title, cohort, groups) in zip(axs, PAN):
    y = 0; ticks, labs, heads = [], [], []
    for blk, fam, trs in groups:
        heads.append((y - 0.15, T[fam])); y -= 0.85
        for tr in trs:
            r = murni(FAM57[fam], cohort, tr) if (fam in FAM57 and tr in ("symlog", "log", "rank")) else dict(D[(blk, KEY[tr])], src="S2.12")
            plotted.append((blk, tr, r["src"], r1(r["delta"])))
            ax.errorbar(r["delta"], y, xerr=[[r["delta"] - r["lo"]], [r["hi"] - r["delta"]]], fmt="o", ms=5.6, color=COL[tr], alpha=1.0, mec="#1a1a1a", mew=0.5, elinewidth=1.7, capsize=2.2, zorder=3)
            ax.text(1.02, y, r1(r["delta"]), transform=ax.get_yaxis_transform(), va="center", fontsize=7.0, fontweight="bold" if r["delta"] > 3 else "normal", color="#111111")
            ticks.append(y); labs.append(T[tr]); y -= 0.75
        y -= 0.25
    ax.axvspan(-4, 1, color="#d3e5d9", zorder=0); ax.axvspan(1, 3, color="#f0dcb4", zorder=0); ax.axvspan(3, 17, color="#e2cfe8", zorder=0)
    for xb in (1, 3): ax.axvline(xb, color="#777777", lw=0.6, ls=":", zorder=1)
    ax.axvline(0, color="k", lw=0.6, ls="--")
    for yy, h in heads: ax.text(-0.02, yy, h, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.0, fontweight="bold")
    ax.set_yticks(ticks); ax.set_yticklabels(labs, fontsize=6.8); ymins.append(y + 0.3); ax.set_xlim(-4, 17); ax.set_xlabel(T["x"], fontsize=7.2)
    ax.set_title(title, fontsize=8, loc="left"); ax.tick_params(axis="x", labelsize=7)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
for ax in axs: ax.set_ylim(min(ymins), 0.35)
leg = fig.legend(handles=[Patch(facecolor=c, edgecolor="#777777", label=l) for c, l in zip(("#d3e5d9", "#f0dcb4", "#e2cfe8"), T["z"])], loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=7.2, frameon=False)   # 2026-11-21: di bawah panel, tidak menutupi data
fig.savefig(OUT, bbox_inches="tight"); plt.close(fig)
# ---- pemeriksaan: titik raw = sel Tabel S2.12; titik transformasi = sel Tabel S2.15; jumlah 33
tab12 = open(os.path.join(ROOT, "naskah/tabel_plasmode.tex"), encoding="utf8").read().replace("$-$", "−").replace("$+$", "+")
cells12 = re.findall(r"([+−]\d+\.\d); (?:leakage|uncertain|no evidence)", tab12)
s15 = open(os.path.join(ROOT, "naskah_jbi/S2_putaran57_60.tex"), encoding="utf8").read().replace("$-$", "−")
rows15 = {}
for ln in s15.split("\n"):
    m = re.match(r"(Random forest|XGBoost|Window-level LR) & (TUEP|Emotiv) & (symlog|log|rank-normal) & [^&]+& [^&]+& [^&]+& [^&]+& ([+−]?\d+\.\d) \[", ln)
    if m: rows15[(m.group(1), m.group(2), m.group(3))] = m.group(4) if m.group(4)[0] in "+−" else "+" + m.group(4)
nm = {"RF": "Random forest", "XGB": "XGBoost", "LRw": "Window-level LR"}; tn = {"symlog": "symlog", "log": "log", "rank": "rank-normal"}
from collections import Counter
raw_vals = [v for blk, tr, src, v in plotted if src == "S2.12"]
assert not (Counter(raw_vals) - Counter(cells12)), Counter(raw_vals) - Counter(cells12)
n15 = 0
for blk, tr, src, v in plotted:
    if src != "S2.15": continue
    coh, fam = blk.split(" "); want = rows15[(nm[fam], coh, tn[tr])]; want = "+0.0" if want in ("0.0", "+0.0") else want
    got = "+0.0" if v in ("0.0", "+0.0", "−0.0") else v
    assert got == want, (blk, tr, got, want); n15 += 1
assert len(plotted) == 33 and n15 == 18, (len(plotted), n15)
print(f"ok: {len(plotted)} titik ({len(raw_vals)} dari S2.12, {n15} dari S2.15) cocok dengan tabel")
for blk, tr, src, v in plotted:
    if src == "S2.15": print("  ", blk, tr, v)
