#!/usr/bin/env python3
"""Putaran 53 (PENYAJIAN, sesudah keluaran terlihat): baris Tabel S2.17 (plasmode) dan data Gambar 4 untuk LR pada fitur jendela mentah,
dihitung dari CSV mentah hasil53 dan CSV pohon tersimpan; tidak ada model yang dilatih. Angka dibulatkan setengah ke atas.
Keluaran: hasil53/bahan_naskah.json (baris tabel EN/ID, titik gambar, angka teks) dan hasil53/tabel_s2_putaran53_{en,id}.tex.
Pemakaian: python round53_bahan_naskah.py"""
import os, sys, json
from decimal import Decimal, ROUND_HALF_UP
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import round53_ringkas as RK
import round49_kalibrasi as K

OUT = RK.OUT
LAB = {"en": {"tidak ada bukti": "no evidence", "tidak pasti": "uncertain", "kebocoran": "leakage"}, "id": {k: k for k in ("tidak ada bukti", "tidak pasti", "kebocoran")}}


def f1(x):
    return f"{Decimal(repr(round(float(x), 10))).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP):.1f}"


def sgn(x):
    v = f1(abs(x)); return ("$-$" if x < 0 and v != "0.0" else "+") + v


def pval(p):
    return "$<$0.001" if p < 0.001 else f"{Decimal(repr(p)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)}"


if __name__ == "__main__":
    L = {(c, f): RK.prep(pd.read_csv(os.path.join(OUT, f"{f}_{c}.csv"))) for c in ("tuep", "emotiv") for f in ("LRw", "LRp")}
    W = {(c, f, g): RK.wide(L[(c, f)], g) for (c, f) in L for g in (0.0, 0.5, 1.0)}
    NAME = {"en": {"raw-none": "raw, no adjustment", "raw-pat": "raw + per-patient adjustment", "symlog-pat": "symlog + per patient", "log-pat": "log + per patient", "rank-pat": "rank-normal + per patient"},
            "id": {"raw-none": "mentah, tanpa penyesuaian", "raw-pat": "mentah + penyesuaian per pasien", "symlog-pat": "symlog + per pasien", "log-pat": "log + per pasien", "rank-pat": "rank-normal + per pasien"}}
    HEAD = {"en": {("tuep", "LRw"): "Logistic regression, window level (raw Rohira window; round 53)", ("tuep", "LRp"): "Logistic regression, patient level (median of raw Rohira windows; round 53)",
                   ("emotiv", "LRw"): "Logistic regression, window level (raw band power, 60 features; round 53)", ("emotiv", "LRp"): "Logistic regression, patient level (median raw band power; round 53)"},
            "id": {("tuep", "LRw"): "Regresi logistik, tingkat jendela (jendela Rohira mentah; putaran 53)", ("tuep", "LRp"): "Regresi logistik, tingkat pasien (median jendela Rohira mentah; putaran 53)",
                   ("emotiv", "LRw"): "Regresi logistik, tingkat jendela (daya pita mentah, 60 fitur; putaran 53)", ("emotiv", "LRp"): "Regresi logistik, tingkat pasien (median daya pita mentah; putaran 53)"}}
    PIPES = {"LRw": ["raw-none", "raw-pat", "symlog-pat", "log-pat", "rank-pat"], "LRp": ["raw-none", "raw-pat"]}
    rows = {"en": {}, "id": {}}; fig = {}; cells = {}
    for c in ("tuep", "emotiv"):
        for f in ("LRw", "LRp"):
            for lg in ("en", "id"):
                ind = "\\quad" if c == "tuep" else "\\qquad"; hind = "" if c == "tuep" else "\\quad"
                out = [f"\\multicolumn{{6}}{{l}}{{{hind}\\textit{{{HEAD[lg][(c, f)]}}}}} \\\\"]
                for p in PIPES[f]:
                    a = [f1(W[(c, f, g)][p].mean() * 100) for g in (0.0, 0.5, 1.0)]
                    if p == "raw-none":
                        out.append(f"{ind} {NAME[lg][p]} & {a[0]} & {a[1]} & {a[2]} & -- & -- \\\\"); continue
                    w1 = W[(c, f, 1.0)]; d = (w1[p] - w1["raw-none"]).values; r1 = LAB[lg][K.LAB[K.dec_pool(w1[p].values, "R1")]]; r2 = LAB[lg][K.LAB[K.dec_pool(d, "R2")]]
                    s = RK.tstat(d * 100); out.append(f"{ind} {NAME[lg][p]} & {a[0]} & {a[1]} & {a[2]} & {r1} & {sgn(s['mean'])}; {r2} \\\\")
                    if lg == "en":
                        fig[f"{c}|{f}|{p}"] = dict(delta=s["mean"], lo=s["lo"], hi=s["hi"], n=s["n"]); cells[f"{c}|{f}|{p}"] = dict(delta=sgn(s["mean"]), lo=sgn(s["lo"]), hi=sgn(s["hi"]), R1=r1, R2=r2, auc1=a[2])
                rows[lg][f"{c}|{f}"] = "\n".join(out)
    # interaksi (Tabel S2.19) dari ringkas_E2.csv
    E2 = pd.read_csv(os.path.join(OUT, "ringkas_E2.csv")); E2 = E2[E2.gamma == 1.0]
    tab = {}
    for lg in ("en", "id"):
        coh = {"en": {"tuep": "TUEP", "emotiv": "Emotiv"}, "id": {"tuep": "TUEP", "emotiv": "Emotiv"}}[lg]; fam = {"en": {"primer": "primary", "sekunder": "secondary"}, "id": {"primer": "primer", "sekunder": "sekunder"}}[lg]
        cap = {"en": "Round 53: leak of the tree classifier minus leak of window-level logistic regression on the same raw window features (raw per-patient adjustment minus raw without adjustment, $\\gamma=1$, H1; per-fold AUC, pp), paired by replicate. One-sided $t$-test ($H_1$: difference $>0$); Holm within each family (two tests).",
               "id": "Putaran 53: kebocoran pengklasifikasi pohon dikurangi kebocoran regresi logistik tingkat jendela pada fitur jendela mentah yang sama (penyesuaian mentah per pasien dikurangi mentah tanpa penyesuaian, $\\gamma=1$, H1; AUC per lipatan, poin), berpasangan per replikasi. Uji $t$ satu sisi ($H_1$: selisih $>0$); Holm di dalam tiap famili (dua uji)."}[lg]
        hd = {"en": "Family & Tree & Cohort & $n$ & Tree leak & LR leak & Difference [95\\% CI] & Holm", "id": "Famili & Pohon & Kohort & $n$ & Kebocoran pohon & Kebocoran LR & Selisih [CI 95\\%] & Holm"}[lg]
        body = [f"{fam[r.famili]} & {r.tree} & {coh[r.cohort]} & {int(r.n)} & {sgn(r.delta_tree)} & {sgn(r.delta_LRw)} & {sgn(r.I)} [{sgn(r.lo)} to {sgn(r.hi)}] & {pval(r.holm)} \\\\".replace(" to ", " to " if lg == "en" else " sampai ")
                for r in E2.itertuples()]
        tab[lg] = "\n".join(["\\begin{table}[!htbp]", f"\\caption{{{cap}}}\\label{{tab:s2_p53}}", "\\centering\\scriptsize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{llcrrrcc}", "\\toprule", hd + " \\\\", "\\midrule"] + body + ["\\bottomrule", "\\end{tabular}", "\\end{table}"])
        open(os.path.join(OUT, f"tabel_s2_putaran53_{lg}.tex"), "w", encoding="utf8").write(tab[lg] + "\n")
    I = {f"{r.tree}|{r.cohort}": dict(I=sgn(r.I), lo=sgn(r.lo), hi=sgn(r.hi), n=int(r.n), holm=float(r.holm), p1=float(r.p1)) for r in E2.itertuples()}
    json.dump(dict(rows=rows, fig=fig, cells=cells, interaksi=I), open(os.path.join(OUT, "bahan_naskah.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(cells, indent=0)); print(json.dumps(I, indent=0)); print(rows["en"]["tuep|LRw"]); print(tab["en"])
