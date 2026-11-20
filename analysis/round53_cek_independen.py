#!/usr/bin/env python3
"""Putaran 53: pemeriksaan independen (kode terpisah dari round53_lr_mentah.py / round53_ringkas.py). Lihat PRAREGISTRASI_putaran53.md.
(a) kelengkapan berkas keluaran; (b) satu lipatan LRw dan LRp dihitung ulang dengan implementasi manual (standardisasi numpy, CV dalam eksplisit atas C,
    LogisticRegression lbfgs); (c) selisih utama dan interaksi dihitung ulang dari CSV mentah dengan numpy murni dan dibandingkan dengan ringkasan.
Pemakaian: python round53_cek_independen.py   (CPU)"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import warnings; warnings.filterwarnings("ignore")
import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

OUT = os.path.join(HERE, "hasil53"); N_OK = 0
CS = np.logspace(-3, 1, 9)


def ok(cond, msg):
    global N_OK
    assert cond, msg; N_OK += 1


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def table(path, cond_filter=None):
    """{(rep, gamma, pipe): rata-rata 5 AUC lipatan}; baris ganda: yang terakhir."""
    T = {}
    for r in read(path):
        if cond_filter and r["cond"] != cond_filter:
            continue
        T[(int(r["rep"]), float(r["gamma"]), r["transform"] + "-" + r["adjust"])] = np.mean([float(r[f"fold_auc_{i}"]) for i in range(5)])
    return T


def manual_lr(Xtr, ytr, gtr, Xte, inner_seed, grouped):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0; A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=inner_seed).split(A, ytr, gtr) if grouped else StratifiedKFold(5).split(A, ytr)
    cv = list(cv); sc = np.zeros(len(CS))
    for a, b in cv:
        for k, c in enumerate(CS):
            m = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(A[a], ytr[a]); sc[k] += roc_auc_score(ytr[b], m.decision_function(A[b]))
    c = CS[int(np.argmax(sc))]; return LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(A, ytr).predict_proba(B)[:, 1], c


if __name__ == "__main__":
    # (a) kelengkapan
    for c, want in (("tuep", {0.0: 120, 1.0: 120, 0.5: 30}), ("emotiv", {0.0: 30, 0.5: 30, 1.0: 30})):
        for f in ("LRw", "LRp"):
            R = read(os.path.join(OUT, f"{f}_{c}.csv")); keys = {(int(r["rep"]), float(r["gamma"]), r["transform"], r["adjust"]) for r in R}
            per = {}
            for (rep, g, t, a) in keys:
                per.setdefault(g, set()).add(rep)
            ok({g: len(v) for g, v in per.items()} == want, (c, f, {g: len(v) for g, v in per.items()})); ok(len(keys) == 6 * sum(want.values()), (c, f, len(keys)))
            ok(all(np.isfinite(float(r[f"fold_auc_{i}"])) for r in R for i in range(5)), (c, f, "NaN"))
    # (b) satu lipatan dihitung ulang secara manual (TUEP rep 3, gamma 1, lipatan 1, raw-pat; Emotiv rep 2, gamma 1, lipatan 0, raw-pat)
    _argv = sys.argv; sys.argv = [_argv[0]]
    import round42_common as C, round42_cpu as P42, round50_emotiv as E
    sys.argv = _argv
    for cohort, rep, g, fi in (("tuep", 3, 1.0, 1), ("emotiv", 2, 1.0, 0)):
        M = C if cohort == "tuep" else E; N = M.N
        y = M.gen_labels(rep, g); perm = M.gen_perm(rep, "H1") if cohort == "tuep" else M.gen_perm(rep); folds = M.folds_for(rep, g, y); tr, te = folds[fi]
        X, owner = (P42 if cohort == "tuep" else E).windows_for(perm); Dm = M.D
        # penyesuaian per pasien ditulis ulang: OLS rata-rata pasien latih -> kurangi dari jendela
        Mp = np.vstack([X[owner == p].mean(0) for p in range(N)]); Bc = np.linalg.solve(Dm[tr].T @ Dm[tr], Dm[tr].T @ Mp[tr]); Z = X - Dm[owner] @ Bc
        seed = M.kseed("plasmode_clf" if cohort == "tuep" else "emotiv_clf", rep, g, fi); tr_p, _ = M.split_val(tr, y, seed); fit = np.isin(owner, tr_p); sel = np.isin(owner, te)
        pw, cw = manual_lr(Z[fit], y[owner[fit]], owner[fit], Z[sel], M.kseed("lrw_inner", rep, g, fi), True)
        sw = np.array([pw[owner[sel] == p].mean() for p in te]); auc_w = roc_auc_score(y[te], sw)
        R = [r for r in read(os.path.join(OUT, f"LRw_{cohort}.csv")) if int(r["rep"]) == rep and float(r["gamma"]) == g and r["transform"] == "raw" and r["adjust"] == "pat"][-1]
        ok(abs(auc_w - float(R[f"fold_auc_{fi}"])) < 5e-3, (cohort, "LRw", auc_w, R[f"fold_auc_{fi}"]))
        PM = np.vstack([np.median(X[owner == p], 0) for p in range(N)]); Bp = np.linalg.solve(Dm[tr].T @ Dm[tr], Dm[tr].T @ PM[tr]); Zp = PM - Dm @ Bp
        pp, cp = manual_lr(Zp[tr], y[tr], None, Zp[te], None, False); auc_p = roc_auc_score(y[te], pp)
        R = [r for r in read(os.path.join(OUT, f"LRp_{cohort}.csv")) if int(r["rep"]) == rep and float(r["gamma"]) == g and r["transform"] == "raw" and r["adjust"] == "pat"][-1]
        ok(abs(auc_p - float(R[f"fold_auc_{fi}"])) < 5e-3, (cohort, "LRp", auc_p, R[f"fold_auc_{fi}"]))
        print(f"{cohort}: lipatan manual LRw AUC {auc_w:.4f} (C {cw:g}) vs berkas {float([r for r in read(os.path.join(OUT, f'LRw_{cohort}.csv')) if int(r['rep']) == rep and float(r['gamma']) == g and r['transform'] == 'raw' and r['adjust'] == 'pat'][-1][f'fold_auc_{fi}']):.4f}; "
              f"LRp AUC {auc_p:.4f} (C {cp:g})", flush=True)
    # (c) selisih dan interaksi dari CSV mentah dibandingkan dengan ringkasan
    S1 = {(r["cohort"], r["fam"], float(r["gamma"]), r["pipe"], r["ref"]): r for r in read(os.path.join(OUT, "ringkas_E1.csv"))}
    S2 = {(r["tree"], r["cohort"], float(r["gamma"])): r for r in read(os.path.join(OUT, "ringkas_E2.csv"))}
    for c in ("tuep", "emotiv"):
        Tw = table(os.path.join(OUT, f"LRw_{c}.csv"))
        for g in (0.0, 1.0):
            reps = sorted({k[0] for k in Tw if k[1] == g}); d = np.array([Tw[(r, g, "raw-pat")] - Tw[(r, g, "raw-none")] for r in reps]) * 100
            ok(abs(d.mean() - float(S1[(c, "LRw", g, "raw-pat", "raw-none")]["delta"])) < 1e-9, (c, g, "E1"))
            for fam in ("RF", "XGB"):
                if c == "tuep":
                    start = 30 if fam == "RF" else 20; Tt = {k: v for k, v in table(os.path.join(HERE, "hasil42", f"{fam}.csv"), "H1").items() if k[0] < start}; Tt.update(table(os.path.join(HERE, "hasil50", f"A_{fam}.csv"), "H1"))
                else:
                    Tt = table(os.path.join(HERE, "hasil50", f"B_{fam}.csv"))
                rr = sorted({k[0] for k in Tt if k[1] == g} & set(reps)); I = np.array([(Tt[(r, g, "raw-pat")] - Tt[(r, g, "raw-none")]) - (Tw[(r, g, "raw-pat")] - Tw[(r, g, "raw-none")]) for r in rr]) * 100
                s = S2[(fam, c, g)]; ok(int(s["n"]) == len(rr) and abs(I.mean() - float(s["I"])) < 1e-9, (fam, c, g, "E2"))
    print(f"PEMERIKSAAN INDEPENDEN: {N_OK} lulus", flush=True)
