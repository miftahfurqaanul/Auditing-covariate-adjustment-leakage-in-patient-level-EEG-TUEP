#!/usr/bin/env python3
"""Round 50: builds the Emotiv rows of the plasmode table and the full Table S2.16 from ../hasil50/ringkas_B.csv (half-up rounding of the full values).
English-only version of the presentation script used for the manuscript (the original also wrote the Indonesian tables).
Usage: python bangun_tabel_putaran50.py          (prints the rows and Table S2.16)
       python bangun_tabel_putaran50.py --check  (compares them with ../../manuscript/naskah/tabel_plasmode.tex and tabel_s2_emotiv.tex)"""
import os, sys, io
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); H50 = os.path.join(os.path.dirname(HERE), "hasil50"); NASKAH = os.path.join(os.path.dirname(os.path.dirname(HERE)), "manuscript", "naskah")
B = pd.read_csv(os.path.join(H50, "ringkas_B.csv"))
DEC = {"kebocoran": "leakage", "tidak ada bukti": "no evidence", "tidak pasti": "uncertain"}     # decision labels as stored in the CSV


def r1(x):
    return Decimal(repr(float(x))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def num(x):
    return f"{r1(x)}"


def sgn(x):
    v = r1(x); return f"$-${abs(v)}" if v < 0 else f"+{v}"


def cells(fam, pipe):
    return {gm: B[(B.fam == fam) & (B.gamma == gm) & (B["pipe"] == pipe)].iloc[0] for gm in (0.0, 0.5, 1.0)}


# ------------------------------------------------------------------ Emotiv block of the plasmode table (main text)
LAB = dict(head="\\multicolumn{6}{l}{\\textit{Independent cohort: Emotiv, 299 subjects (round 50)}} \\\\", RF="\\quad\\textit{Random forest (raw band power, 60 features)}", XGB="\\quad\\textit{XGBoost (raw band power, 60 features)}", LR="\\quad\\textit{Logistic regression (subject-level log band power)}",
           rn="\\qquad raw, no adjustment", rp="\\qquad raw + per-patient adjustment", kp="\\qquad rank-normal + per patient")


def rows():
    out = [LAB["head"]]
    for fam in ("RF", "XGB", "LR"):
        out.append(f"\\multicolumn{{6}}{{l}}{{{LAB[fam]}}} \\\\")
        for pipe, key in (("raw-none", "rn"), ("raw-pat", "rp"), ("rank-pat", "kp")):
            g = cells(fam, pipe); c = g[1.0]
            tail = "-- & --" if pipe == "raw-none" else f"{DEC[c.R1]} & {sgn(c.delta)}; {DEC[c.R2]}"
            out.append(f"{LAB[key]} & {num(g[0.0].auc)} & {num(g[0.5].auc)} & {num(g[1.0].auc)} & {tail} \\\\")
    return out


# ------------------------------------------------------------------ full Table S2.16 (all Emotiv pipelines)
NAMA = {"raw-none": "raw, no adjustment", "raw-win": "raw + window-weighted adjustment", "raw-pat": "raw + per-patient adjustment", "symlog-pat": "symlog + per patient", "log-pat": "log + per patient", "rank-pat": "rank-normal + per patient", "rank-none": "rank-normal, no adjustment"}
CAP = "Plasmode on the independent Emotiv cohort (round 50; 299 subjects; H1): per-fold AUC (percent; replicate mean) by $\\gamma$, the paired difference at $\\gamma=1$ against the same transformation without adjustment (for the tree classifiers, the raw pipeline; pp, 95\\% CI over replicates), and the R1 and R2 decisions at $\\gamma=1$. Replicates: 30 (\\textit{random forest}, logistic regression) and 20 (XGBoost)."
HEAD = "Pipeline & $\\gamma=0$ & $\\gamma=0.5$ & $\\gamma=1$ & $\\Delta$ [95\\% CI] & R1; R2"
FAMN = {"RF": "\\textit{Random forest} (raw band power, 60 features)", "XGB": "XGBoost (raw band power, 60 features)", "LR": "Logistic regression (subject-level log band power, 60 features)"}


def tabel_s2():
    L = [f"\\begin{{table}}[h]\n\\caption{{{CAP}}}\\label{{tab:s2_emotiv}}", "\\centering\\scriptsize\n\\setlength{\\tabcolsep}{2.5pt}", "\\begin{tabular}{p{0.34\\textwidth}ccccc}", "\\toprule", HEAD + " \\\\", "\\midrule"]
    order = {"RF": ["raw-none", "raw-win", "raw-pat", "symlog-pat", "log-pat", "rank-pat"], "XGB": ["raw-none", "raw-win", "raw-pat", "symlog-pat", "log-pat", "rank-pat"], "LR": ["raw-none", "raw-pat", "rank-none", "rank-pat"]}
    for fam in ("RF", "XGB", "LR"):
        L.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{FAMN[fam]}}}}} \\\\" if not FAMN[fam].startswith("\\textit") else f"\\multicolumn{{6}}{{l}}{{{FAMN[fam]}}} \\\\")
        for pipe in order[fam]:
            g = cells(fam, pipe); c = g[1.0]
            tail = "-- & --" if pipe.endswith("none") else f"{sgn(c.delta)} [{sgn(c.delta_lo)}, {sgn(c.delta_hi)}] & {DEC[c.R1]}; {DEC[c.R2]}"
            L.append(f"\\quad {NAMA[pipe]} & {num(g[0.0].auc)} & {num(g[0.5].auc)} & {num(g[1.0].auc)} & {tail} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    R, S2 = rows(), tabel_s2()
    if "--check" in sys.argv:
        plas = io.open(os.path.join(NASKAH, "tabel_plasmode.tex"), encoding="utf8").read(); s2 = io.open(os.path.join(NASKAH, "tabel_s2_emotiv.tex"), encoding="utf8").read()
        miss = [r for r in R if r not in plas]
        print(f"plasmode table: {len(R) - len(miss)} of {len(R)} rows found; Table S2.16 identical: {s2 == S2}")
        sys.exit(0 if not miss and s2 == S2 else 1)
    print("\n".join(R)); print(); print(S2)
