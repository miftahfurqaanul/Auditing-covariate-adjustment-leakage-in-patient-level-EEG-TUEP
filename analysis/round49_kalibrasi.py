#!/usr/bin/env python3
"""Round 49B (CPU): operating characteristics of rules R1/R2 (decision frequencies on null and positive controls, parametric OC curves,
threshold sensitivity on the published decisions, Monte Carlo error vs patient variation). See PRAREGISTRASI_putaran49.md section 49B.
Usage: CUDA_VISIBLE_DEVICES="" python round49_kalibrasi.py --check | --pasien | --run   (output to hasil49/B_*.csv)"""
import os, sys, time, json, hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
OUT = os.path.join(HERE, "hasil49"); os.makedirs(OUT, exist_ok=True); H = lambda *p: os.path.join(HERE, *p)
THR = {"R1": {"ketat": (0.51, 0.53, 0.54), "asli": (0.52, 0.54, 0.55), "longgar": (0.53, 0.55, 0.56)},
       "R2": {"ketat": (0.005, 0.020, 0.020), "asli": (0.010, 0.030, 0.030), "longgar": (0.015, 0.040, 0.040)}}
NS = (10, 20, 30, 50, 100); LAB = {0: "tidak ada bukti", 1: "tidak pasti", 2: "kebocoran"}


def decide(m, se, thr):
    """m, se: arrays; returns code 0 = no evidence, 1 = uncertain, 2 = leakage (same rule for R1 and R2, different thresholds)."""
    lo, hi, leak = thr; return np.where((m <= lo) & (m + 1.645 * se <= hi), 0, np.where(m > leak, 2, 1))


def dec_pool(x, rule, setname="asli"):
    x = np.asarray(x, float); return int(decide(np.array([x.mean()]), np.array([x.std(ddof=1) / np.sqrt(len(x))]), THR[rule][setname])[0])


def freq(x, rule, n, rng, B=10000, setname="asli"):
    x = np.asarray(x, float); S = x[rng.integers(0, len(x), (B, n))]; d = decide(S.mean(1), S.std(1, ddof=1) / np.sqrt(n), THR[rule][setname]); return np.bincount(d, minlength=3) / B


def freq_ci(x, rule, n, rng, outer=500, B=2000):
    x = np.asarray(x, float); F = np.array([freq(x[rng.integers(0, len(x), len(x))], rule, n, rng, B) for _ in range(outer)]); return np.percentile(F, [2.5, 97.5], axis=0)


# ------------------------------------------------------------------ stored replicate pools
def eeg_pools():
    """dict name -> (value, rule, level, type). type: 'nol' (null: no adjustment / no label-covariate association), 'positif' (leak by design), 'diuji' (under test)."""
    P = {}; a29 = pd.read_csv(H("hasil29", "A_kebocoran_replikasi.csv")); a46 = pd.read_csv(H("hasil46", "H1_per_replikasi.csv"))
    for sc, tag in (("H1_all", "H1"), ("H2a_era", "H2a")):
        g = a29[a29.scenario == sc]
        for v in ("C1", "V0", "V1", "V2", "V3", "V4"):
            P[f"r29 {v} {tag} (gabungan)"] = (g[v].values, "R1", "gabungan", "nol" if v == "C1" else ("positif" if v in ("V0", "V1", "V2") else "diuji"))
        if tag == "H1":
            for v in ("V0", "V4"):
                P[f"r29 {v}-C1 H1 (gabungan)"] = ((g[v] - g["C1"]).values, "R2", "gabungan", "positif" if v == "V0" else "diuji")
    for v in ("C1", "V0", "V4"):
        P[f"r46 {v} H1 (per lipatan)"] = (a46[a46.variant == v].sort_values("rep").fold_mean.values, "R1", "per lipatan", "nol" if v == "C1" else ("positif" if v == "V0" else "diuji"))
    r = pd.read_csv(H("hasil36", "paired_rohira_null.csv")).pivot_table(index=["scenario", "rep"], columns="variant", values="auc").reset_index(); r = r[r.scenario == "H1"]
    P["r36c Rohira RF-none H1"] = (r["RF-none"].values, "R1", "gabungan", "nol"); P["r36c Rohira RF-logpat - RF-none H1"] = ((r["RF-logpat"] - r["RF-none"]).values, "R2", "gabungan", "diuji")
    t = pd.concat([pd.read_csv(H("hasil37", f)) for f in ("tang_L_null_replikasi.csv", "tang_Lp_null_replikasi.csv")]).pivot_table(index=["scenario", "rep"], columns="variant", values="auc").reset_index(); t = t[t.scenario == "H1"]
    P["r37 Tang L H1"] = (t["L"].values, "R1", "gabungan", "nol"); P["r37 Tang Lp - L H1"] = ((t["Lp"] - t["L"]).values, "R2", "gabungan", "diuji")
    n = pd.read_csv(H("hasil40", "nm_perlipatan_replikasi.csv")); n["fold_mean"] = n[[f"fold_auc_{i}" for i in range(5)]].mean(1); n = n[n.scenario == "H1"].pivot_table(index="rep", columns="variant", values="fold_mean")
    P["r40 NM N0 H1 (per lipatan)"] = (n["N0"].values, "R1", "per lipatan", "nol"); P["r40 NM V4 - N0 H1 (per lipatan)"] = ((n["V4"] - n["N0"]).values, "R2", "per lipatan", "diuji")
    return P


def plasmode_cells():
    """Round 42 plasmode cells (per fold): (fam, cond, gamma, pipe) -> R1 value and R2 delta."""
    import round42_ringkas as R42
    D = R42.load(); out = []
    for (fam, cond, g), G in D[D.cond == "H1"].groupby(["fam", "cond", "gamma"]):
        W = G.pivot_table(index="rep", columns="pipe", values="fold_mean")
        for pipe in W.columns:
            t, a = pipe.split("-"); adj = R42.is_adj(fam, a); bp = R42.baseline_pipe(fam, t); x = W[pipe].dropna().values
            d = (W[pipe] - W[bp]).dropna().values if adj and bp in W.columns else None
            out.append(dict(fam=fam, gamma=g, pipe=pipe, adjusted=adj, x=x, d=d))
    return out


def sim_cells():
    """Round 40 simulation cells (per fold): R1 value (arm with adjustment) and R2 delta; type from the design."""
    import round40_ringkas as R40
    D = R40.load(); A = R40.paired(D, "fold_mean"); G = R40.CELLK + ["clf", "transform", "adjust"]; out = []
    for k, g in A.groupby(G):
        r = dict(zip(G, k)); r["x"] = g["fold_mean"].values; r["d"] = g["delta"].values if r["scen"] == "H1" else None; out.append(r)
    base = D[D["adjust"] == "none"]
    for k, g in base.groupby(R40.CELLK + ["clf", "transform"]):
        r = dict(zip(R40.CELLK + ["clf", "transform"], k)); r["adjust"] = "none"; r["x"] = g["fold_mean"].values; r["d"] = None; out.append(r)
    return out


# ------------------------------------------------------------------ analisis
def run(rng):
    P = eeg_pools(); rows = []
    for name, (x, rule, lvl, kind) in P.items():
        for n in NS:
            if n > len(x):
                continue
            f = freq(x, rule, n, rng); lo, hi = freq_ci(x, rule, n, rng)
            rows.append(dict(sumber="EEG", kumpulan=name, aturan=rule, tingkat=lvl, jenis=kind, ukuran_kumpulan=len(x), rata2=float(np.mean(x)), sd=float(np.std(x, ddof=1)), n=n,
                             p_tidak_ada_bukti=f[0], p_tidak_pasti=f[1], p_kebocoran=f[2], ci_tidak_ada_bukti=f"{lo[0]:.3f}-{hi[0]:.3f}", ci_kebocoran=f"{lo[2]:.3f}-{hi[2]:.3f}"))
    for c in plasmode_cells():
        kind_x = "nol" if (not c["adjusted"] or c["gamma"] == 0) else ("positif" if c["pipe"] in ("raw-win", "raw-pat") and c["fam"] in ("RF", "XGB", "Tang") and c["gamma"] == 1 else "diuji")
        for rule, x, kind in (("R1", c["x"], kind_x), ("R2", c["d"], "nol" if c["gamma"] == 0 else kind_x)):
            if x is None or len(x) < 10:
                continue
            for n in (10, 20, 30):
                if n > len(x):
                    continue
                f = freq(x, rule, n, rng, B=4000); rows.append(dict(sumber="plasmode", kumpulan=f"{c['fam']} {c['pipe']} gamma={c['gamma']:g}", aturan=rule, tingkat="per lipatan", jenis=kind, ukuran_kumpulan=len(x),
                                                                    rata2=float(np.mean(x)), sd=float(np.std(x, ddof=1)), n=n, p_tidak_ada_bukti=f[0], p_tidak_pasti=f[1], p_kebocoran=f[2]))
    for c in sim_cells():
        null_x = (c["adjust"] == "none" and c["scen"] == "H1") or c["s"] == 0          # addendum 2: H2a cells without adjustment with s > 0 are not null (the features carry legitimate covariate information)
        for rule, x in (("R1", c["x"]), ("R2", c["d"])):
            if x is None or len(x) < 20:
                continue
            ref = float(np.mean(x)); kind = "nol" if (null_x if rule == "R1" else c["s"] == 0) else ("positif" if ref > (0.55 if rule == "R1" else 0.03) else "diuji")
            for n in (10, 20, 30):
                f = freq(x, rule, n, rng, B=2000); rows.append(dict(sumber="simulasi", kumpulan="|".join(str(c[k]) for k in ("dist", "scen", "m", "imb", "s", "clf", "transform", "adjust")), aturan=rule, tingkat="per lipatan",
                                                                    jenis=kind, ukuran_kumpulan=len(x), rata2=ref, sd=float(np.std(x, ddof=1)), n=n, p_tidak_ada_bukti=f[0], p_tidak_pasti=f[1], p_kebocoran=f[2]))
    F = pd.DataFrame(rows); F.to_csv(f"{OUT}/B_frekuensi.csv", index=False); print(f"frequencies: {len(F)} rows", flush=True)
    # kurva OC parametrik
    sd_null = F[(F.sumber != "simulasi") & (F.jenis == "nol") & (F.aturan == "R1")].drop_duplicates("kumpulan").sd.values
    sd_d = F[(F.aturan == "R2") & (F.sumber == "EEG")].drop_duplicates("kumpulan").sd.values; oc = []
    sd_pl = F[(F.aturan == "R2") & (F.sumber == "plasmode")].drop_duplicates("kumpulan").sd.max()     # addendum 2: largest delta SD (trees on raw features) as the high-variance scenario
    for rule, sds, grid in (("R1", sd_null, np.round(np.arange(0.48, 0.6201, 0.005), 3)), ("R2", sd_d, np.round(np.arange(0.0, 0.0601, 0.0025), 4))):
        for lab, s in (("min", sds.min()), ("median", np.median(sds)), ("maks", sds.max())) + ((("plasmode maks", sd_pl),) if rule == "R2" else ()):
            for n in NS:
                for mu in grid:
                    S = mu + s * rng.standard_normal((20000, n)); d = decide(S.mean(1), S.std(1, ddof=1) / np.sqrt(n), THR[rule]["asli"]); f = np.bincount(d, minlength=3) / 20000
                    oc.append(dict(aturan=rule, sd_label=lab, sd=s, n=n, mu=mu, p_tidak_ada_bukti=f[0], p_tidak_pasti=f[1], p_kebocoran=f[2]))
    pd.DataFrame(oc).to_csv(f"{OUT}/B_oc_parametrik.csv", index=False); print("parametric OC done", flush=True)
    # threshold sensitivity on the published decisions (EEG pool, plasmode cells, round 40 simulation cells)
    srows = []
    for name, (x, rule, lvl, kind) in P.items():
        srows.append(dict(sumber="EEG", kumpulan=name, aturan=rule, **{s: LAB[dec_pool(x, rule, s)] for s in ("ketat", "asli", "longgar")}))
    for c in plasmode_cells():
        if c["adjusted"]:
            srows.append(dict(sumber="plasmode", kumpulan=f"{c['fam']} {c['pipe']} gamma={c['gamma']:g}", aturan="R1", **{s: LAB[dec_pool(c["x"], "R1", s)] for s in ("ketat", "asli", "longgar")}))
            if c["d"] is not None:
                srows.append(dict(sumber="plasmode", kumpulan=f"{c['fam']} {c['pipe']} gamma={c['gamma']:g}", aturan="R2", **{s: LAB[dec_pool(c["d"], "R2", s)] for s in ("ketat", "asli", "longgar")}))
    for c in sim_cells():
        if c["adjust"] != "none":
            k = "|".join(str(c[q]) for q in ("dist", "scen", "n", "m", "p", "imb", "s", "sig", "cfrac", "cscale", "clf", "transform", "adjust"))
            srows.append(dict(sumber="simulasi", kumpulan=k, aturan="R1", **{s: LAB[dec_pool(c["x"], "R1", s)] for s in ("ketat", "asli", "longgar")}))
            if c["d"] is not None:
                srows.append(dict(sumber="simulasi", kumpulan=k, aturan="R2", **{s: LAB[dec_pool(c["d"], "R2", s)] for s in ("ketat", "asli", "longgar")}))
    S = pd.DataFrame(srows); S["berubah_ketat"] = S.ketat != S.asli; S["berubah_longgar"] = S.longgar != S.asli; S.to_csv(f"{OUT}/B_sensitivitas_ambang.csv", index=False)
    print(S.groupby(["sumber", "aturan"])[["berubah_ketat", "berubah_longgar"]].agg(["sum", "count"]).to_string(), flush=True)


def pasien():
    """Monte Carlo error vs patient variation: C1 and V4 (without age/sex) H1, 100 replicates with the round 29 seed, predictions saved."""
    import round29_sensitivitas as S29
    A = S29.A; Y, N = A.Y, A.N; path = f"{OUT}/B_pasien_prediksi.npz"; done = dict(np.load(path)) if os.path.exists(path) else {}; t0 = time.time()
    for r in range(100):
        if f"C1_{r}" in done:
            continue
        Xs = A.shuffle_cols(A.BP, [np.arange(N)], np.random.default_rng(29100 + r)); M = S29.run_matrix(Xs, ["C1", "V4"], False)
        done[f"C1_{r}"], done[f"V4_{r}"] = M["C1"], M["V4"]; np.savez(path, **done)
        if r % 10 == 9:
            print(f"replicate {r + 1}/100 [{time.time() - t0:.0f}s]", flush=True)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"; rng = np.random.default_rng(49200)
    json.dump(dict(prereg_sha256=sha(H("PRAREGISTRASI_putaran49.md")), code_sha256=sha(__file__)), open(f"{OUT}/meta_B.json", "w"), indent=1)
    if mode == "--check":                                   # the "original" decision of the pool must equal the published decision in the round 40 and 42 summaries
        import round40_ringkas as R40
        T = pd.read_csv(H("hasil40", "ringkas_sel_R1R2.csv")); T = T[T.metrica == "fold_mean"]; mis = 0; nchk = 0
        for c in sim_cells():
            if c["adjust"] == "none":
                continue
            q = T
            for k in ("dist", "scen", "n", "m", "p", "imb", "s", "sig", "cfrac", "cscale", "clf", "transform", "adjust"):
                q = q[q[k].astype(str) == str(c[k])]
            if len(q) != 1:
                continue
            nchk += 1; mis += LAB[dec_pool(c["x"], "R1")] != q.R1.iloc[0]
            if c["d"] is not None:
                nchk += 1; mis += LAB[dec_pool(c["d"], "R2")] != q.R2.iloc[0]
        print(f"round 40 simulation: {nchk} decisions compared, {mis} differ", flush=True); assert nchk > 700 and mis == 0
        S42 = pd.read_csv(H("hasil42", "ringkas_pipelines.csv")); mis = 0; nchk = 0
        for c in plasmode_cells():
            if not c["adjusted"]:
                continue
            q = S42[(S42.fam == c["fam"]) & (S42.cond == "H1") & (S42.gamma == c["gamma"]) & (S42["pipe"] == c["pipe"])]; nchk += 1; mis += LAB[dec_pool(c["x"], "R1")] != q.R1_fold_mean.iloc[0]
            if c["d"] is not None:
                nchk += 1; mis += LAB[dec_pool(c["d"], "R2")] != q.R2_fold_mean.iloc[0]
        print(f"round 42 plasmode: {nchk} decisions compared, {mis} differ", flush=True); assert mis == 0
        P = eeg_pools(); print(f"{len(P)} kumpulan EEG; contoh C1 H1 n={len(P['r29 C1 H1 (gabungan)'][0])}", flush=True)
        assert LAB[dec_pool(P["r29 V0 H1 (gabungan)"][0], "R1")] == "kebocoran" and LAB[dec_pool(P["r29 V4 H1 (gabungan)"][0], "R1")] == "tidak ada bukti"
        print("checks passed"); sys.exit()
    if mode == "--pasien":
        pasien(); open(f"{OUT}/B_pasien_SELESAI", "w").write(time.ctime()); sys.exit()
    run(rng); open(f"{OUT}/B_SELESAI", "w").write(time.ctime())
