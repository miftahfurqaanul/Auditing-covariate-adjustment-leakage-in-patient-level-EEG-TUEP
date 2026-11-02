#!/usr/bin/env python3
"""Round 50: independent checks (second implementation) of 50A, 50B, and 50C. Does not use the summary functions of round50_ringkas.py.
Usage: python round50_cek_independen.py   (output hasil50/cek_independen.log via tee)"""
import os, sys, io
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(HERE, "hasil50"); T = "/home/miftah_f/RTX-5090/tuep_fresh/data"; Z = "/home/miftah_f/RTX-5090/tuep_fresh/external/zenodo_1252141"
FC = [f"fold_auc_{i}" for i in range(5)]; G = []


def grup(nama, ok, info=""):
    G.append((nama, bool(ok))); print(f"[{'PASS' if ok else 'FAIL'}] {nama} {info}", flush=True)


def sel(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k].values == v)
    return df[m]


def per_rep(df):
    """{rep: mean per-fold AUC} with numpy (not pivot)."""
    return {int(r): float(np.mean(row)) for r, row in zip(df["rep"].values, df[FC].values.astype(float))}


def ttest(d):
    d = np.asarray(d, float); n = len(d); se = d.std(ddof=1) / np.sqrt(n); return d.mean(), se, 2 * stats.t.sf(abs(d.mean() / se), n - 1)


# 1. reproduction of round-42 replicate 0 (--check log)
txt = "".join(io.open(os.path.join(O, f), encoding="utf8").read() for f in ("hasil50_check_A_RF.log",)) if os.path.exists(os.path.join(O, "hasil50_check_A_RF.log")) else ""
grup("1 reproduction of replicate 0 (RF; XGB recorded in the hash log)", "maximum difference from hasil42 = 5.55e-17" in txt)

# 2. 50B summaries recomputed with numpy
B = pd.read_csv(os.path.join(O, "ringkas_B.csv")); raw = pd.concat([pd.read_csv(os.path.join(O, f"B_{f}.csv")) for f in ("LR", "RF", "XGB")], ignore_index=True); bad = 0; n = 0
for fam in ("LR", "RF", "XGB"):
    for g in (0.0, 0.5, 1.0):
        for t, a in sorted(set(zip(sel(raw, fam=fam, gamma=g)["transform"], sel(raw, fam=fam, gamma=g)["adjust"]))):
            x = per_rep(sel(raw, fam=fam, gamma=g, transform=t, adjust=a)); r = B[(B.fam == fam) & (B.gamma == g) & (B["pipe"] == f"{t}-{a}")].iloc[0]; n += 1
            bad += abs(np.mean(list(x.values())) * 100 - r.auc) > 1e-9
            if a != "none":
                b = per_rep(sel(raw, fam=fam, gamma=g, transform=("raw" if fam != "LR" else t), adjust="none")); d = np.array([x[k] - b[k] for k in sorted(x)]) * 100; m, se, p = ttest(d)
                bad += (abs(m - r.delta) > 1e-9) + (abs(se - r.delta_se) > 1e-9) + (abs(p - r.delta_p) > 1e-9)
grup("2 50B summaries (mean, difference, SE, p) recomputed", bad == 0, f"({n} cells, {bad} differ)")

# 3. Holm, primary and secondary families
U = pd.read_csv(os.path.join(O, "ringkas_B_uji.csv")); ok = True
for fm in ("primer", "sekunder"):
    p = U[U.famili == fm].p.values; o = np.argsort(p); h = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o])); hh = np.empty_like(h); hh[o] = h; ok &= np.allclose(hh, U[U.famili == fm].holm.values)
grup("3 Holm 50B", ok, f"(Holm primary {U[U.famili == 'primer'].holm.max():.2e})")

# 4. Emotiv features: raw band power with a manual Welch (FFT) on 25 random windows; subject-level median log band power
zw = np.load(f"{T}/ext_zenodo_whole_windows.npz"); sid = zw["sid"]; u, c = np.unique(sid, return_counts=True); keep = u[c >= 5]; w = np.where(np.isin(sid, keep))[0]
XR = np.load(os.path.join(O, "emotiv_fitur.npz"))["XR"]; rng = np.random.default_rng(7); pick = rng.choice(len(w), 25, replace=False)
win = np.hanning(401)[:-1] if False else 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400); scale = 1.0 / (200 * (win ** 2).sum()); f = np.fft.rfftfreq(400, 1 / 200); bands = [(1, 4), (4, 7.5), (7.5, 13), (13, 16), (16, 30), (30, 40)]; mx = 0
for k in pick:
    x = zw["W"][w[k]].astype(np.float64); segs = [x[:, s:s + 400] for s in range(0, 2000 - 400 + 1, 200)]
    P = np.mean([np.abs(np.fft.rfft((sg - sg.mean(1, keepdims=True)) * win, axis=1)) ** 2 for sg in segs], axis=0) * scale; P[:, 1:-1] *= 2
    v = np.stack([P[:, (f >= lo) & (f < hi)].sum(1) for lo, hi in bands], axis=1).reshape(-1); mx = max(mx, np.max(np.abs(v - XR[k]) / XR[k]))
grup("4 Emotiv features (manual Welch, 25 windows)", mx < 1e-6, f"(maximum relative difference {mx:.1e}; {len(keep)} subjects, {len(w)} windows)")

# 5. Emotiv covariates from the original metadata files (a path independent of ext_zenodo_subjects.csv)
ng = pd.read_csv(f"{Z}/metadata_nigeria.csv"); gb = pd.read_csv(f"{Z}/metadata_guineabissau.csv"); ix = pd.read_csv(f"{Z}/zenodo_index.csv"); ix["sid"] = ix.country + "-" + ix["subject.id"].astype(str)
ix = ix[~ix.file.str.endswith("signal-124-2.csv.gz")].drop_duplicates("sid").set_index("sid").loc[keep]; rows = []
for s_, r in ix.iterrows():
    if r.country == "NG":
        m = ng[ng["csv.file"] == os.path.basename(r.file)].iloc[0]; rows.append(("NG", m.first_condition.strip().lower() == "closed", m.startTime, m.Group.strip().lower()))
    else:
        m = gb[gb["subject.id"] == r["subject.id"]].iloc[0]; rows.append(("GB", m["Eyes.condition"].lower().startswith("closed"), m.startTime, m.Group.strip().lower()))
M = pd.DataFrame(rows, columns=["country", "closed", "start", "group"]); tt = pd.to_datetime(M.start, dayfirst=True)
day = np.array([(t.normalize() - tt[M.country == ct].min().normalize()).days for t, ct in zip(tt, M.country)], float); hr = (tt.dt.hour + tt.dt.minute / 60).values
zs = lambda v: (v - v.mean()) / v.std()
D = np.column_stack([(M.country == "NG").astype(float), M.closed.astype(float), zs(day), zs(hr)]); Y = (M.group == "epilepsy").astype(int).values
eta = LogisticRegression(penalty=None, max_iter=5000).fit(D, Y).decision_function(D); a = roc_auc_score(Y, eta)
grup("5 Emotiv covariates from the original metadata", abs(a - 0.6854) < 5e-5 and Y.sum() == 163 and int(D[:, 0].sum()) == 206, f"(AUC eta {a:.4f}; epilepsy {Y.sum()}; Nigeria {int(D[:, 0].sum())})")

# 6. round 39: unequal-window cells recomputed from the raw replicates
rr = pd.concat([pd.read_csv(os.path.join(HERE, "hasil39", f)) for f in ("B_cpu.csv", "B_gpu.csv")], ignore_index=True); rr = rr[(rr.imb == True) & (rr["transform"] == "raw")]; diffs = []
for (dist, clf), g in rr.groupby(["dist", "clf"]):
    piv = {a_: dict(zip(sel(g, adjust=a_)["rep"], sel(g, adjust=a_)["auc"])) for a_ in ("none", "win", "pat")}; ks = sorted(set(piv["win"]) & set(piv["pat"]) & set(piv["none"]))
    dw = np.mean([piv["win"][k] - piv["none"][k] for k in ks]); dp = np.mean([piv["pat"][k] - piv["none"][k] for k in ks]); diffs.append((dw - dp) * 100)
grup("6 round 39 unequal windows", len(diffs) == 10 and max(map(abs, diffs)) < 0.85 and round(max(map(abs, diffs)), 1) == 0.8, f"(10 cells; difference win - pat {min(diffs):+.3f} to {max(diffs):+.3f} pp)")

# 7. 50A: 120 replicates per cell, differences and pool R2 decisions recomputed
if os.path.exists(os.path.join(O, "ringkas_A.csv")):
    A = pd.read_csv(os.path.join(O, "ringkas_A.csv")); bad = 0; n = 0
    for fam, start in (("RF", 30), ("XGB", 20)):
        old = pd.read_csv(os.path.join(HERE, "hasil42", f"{fam}.csv")); old = old[(old.cond == "H1") & (old.rep < start)]; new = pd.read_csv(os.path.join(O, f"A_{fam}.csv")); allr = pd.concat([old, new])
        for g in (0.0, 1.0):
            base = per_rep(sel(allr, gamma=g, transform="raw", adjust="none")); assert len(base) == 120
            for adj in ("win", "pat"):
                x = per_rep(sel(allr, gamma=g, transform="raw", adjust=adj)); d = np.array([x[k] - base[k] for k in range(120)]) * 100; m, se, p = ttest(d); n += 1
                r = A[(A.fam == fam) & (A.gamma == g) & (A["pipe"] == f"raw-{adj}")].iloc[0]
                dec = "kebocoran" if m > 3 else ("tidak ada bukti" if (m <= 1 and m + 1.645 * se <= 3) else "tidak pasti")
                bad += (abs(m - r.delta) > 1e-9) + (abs(p - r.p) > 1e-9) + (dec != r.R2_120)
    F = pd.read_csv(os.path.join(O, "ringkas_A_frek.csv")); rng = np.random.default_rng(99); fb = 0
    for fam, start in (("RF", 30), ("XGB", 20)):
        old = pd.read_csv(os.path.join(HERE, "hasil42", f"{fam}.csv")); old = old[(old.cond == "H1") & (old.rep < start)]; new = pd.read_csv(os.path.join(O, f"A_{fam}.csv")); allr = pd.concat([old, new])
        base = per_rep(sel(allr, gamma=0.0, transform="raw", adjust="none"))
        for adj in ("win", "pat"):
            x = per_rep(sel(allr, gamma=0.0, transform="raw", adjust=adj)); d = np.array([x[k] - base[k] for k in range(120)])
            for nn in (20, 50, 100):
                S = d[rng.integers(0, 120, (20000, nn))]; mm = S.mean(1); ss = S.std(1, ddof=1) / np.sqrt(nn); leak = np.mean(mm > 0.03) * 100
                ref = F[(F.fam == fam) & (F.gamma == 0.0) & (F["pipe"] == f"raw-{adj}") & (F.aturan == "R2") & (F.n_rep == nn)].kebocoran.iloc[0]; fb += abs(leak - ref) > 1.5
    grup("7 50A: differences, p, pool decisions, R2 frequencies (other RNG)", bad == 0 and fb == 0, f"({n} cells; {bad} differ; frequencies differing by > 1.5 pp: {fb})")

print(f"\n{sum(ok for _, ok in G)} of {len(G)} groups passed" + ("; ALL CHECK GROUPS PASSED" if all(ok for _, ok in G) else ""))
