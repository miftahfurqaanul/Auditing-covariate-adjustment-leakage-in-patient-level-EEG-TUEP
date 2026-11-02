"""Round 8 (reports/round8_zanardini_prediction.md, commit 397e981): reproduction and audit of Zanardini et al. 2026 (arXiv 2605.22858)
on the Z31 subset (TUEP recordings with an active 'PHOTIC PH' channel). CPU only.
Usage: python zanardini.py subset | features | eval [nonull]"""
import os, sys, glob, itertools, numpy as np, pandas as pd, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["CUDA_VISIBLE_DEVICES"] = ""      # CPU only: XGBoost 3 otherwise opens a CUDA context per worker
from multiprocessing import Pool
from scipy import signal, stats
warnings.filterwarnings("ignore")
D = "/home/miftah_f/RTX-5090/tuep_fresh/data"; R = "/home/miftah_f/RTX-5090/tuep_fresh/reports"; RAW = "/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1"; FS = 250
CH = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ", "C4", "T4", "T5", "P3", "PZ", "P4", "T6", "O1", "O2"]
NB = {"FP1": "FP2 F7 F3", "FP2": "FP1 F4 F8", "F7": "FP1 F3 T3", "F3": "FP1 F7 FZ C3", "FZ": "F3 F4 CZ", "F4": "FP2 FZ F8 C4", "F8": "FP2 F4 T4",
      "T3": "F7 C3 T5", "C3": "F3 T3 CZ P3", "CZ": "FZ C3 C4 PZ", "C4": "F4 CZ T4 P4", "T4": "F8 C4 T6", "T5": "T3 P3 O1", "P3": "C3 T5 PZ O1",
      "PZ": "CZ P3 P4", "P4": "C4 PZ T6 O2", "T6": "T4 P4 O2", "O1": "T5 P3 O2", "O2": "T6 P4 O1"}
DB = ["FP1-F7", "F7-T3", "T3-T5", "T5-O1", "FP2-F8", "F8-T4", "T4-T6", "T6-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
      "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FZ-CZ", "CZ-PZ"]
MONTAGES, LENS, COMBS, TYPES, CONDS = ["CAR", "Cz", "Laplacian", "BipolarDB"], [1, 2, 5, 10, 20, 60], ["mean", "median", "sd", "skew", "kurt"], ["spectral", "utm", "conn"], ["rest", "ips"]
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
CONFIGS = list(itertools.product(TYPES, MONTAGES, LENS, COMBS))
PATHS = None


def edf_path(f):
    global PATHS
    if PATHS is None:
        PATHS = {os.path.basename(p): p for p in glob.glob(f"{RAW}/**/*.edf", recursive=True)}
    return PATHS[f]


def photic_edges(x):
    """Number of large transitions on the photic channel and the sample indices (same rule as the pre-registered subset definition)."""
    d = np.abs(np.diff(x)); thr = 0.5 * (np.percentile(x, 99.5) - np.percentile(x, 0.5))
    return np.where(d > thr)[0] if thr > 0 else np.array([], int)


def subset_job(f):
    import mne
    x = mne.io.read_raw_edf(edf_path(f), include=["PHOTIC PH"], preload=True, verbose="ERROR").get_data()[0]; return f, len(photic_edges(x))


def load(f):
    import mne
    raw = mne.io.read_raw_edf(edf_path(f), preload=True, verbose="ERROR"); assert raw.info["sfreq"] == FS
    x = raw.get_data(picks=[f"EEG {c}-LE" for c in CH]) * 1e6; ph = raw.get_data(picks=["PHOTIC PH"])[0]
    b, a = signal.iirnotch(60, 30, FS); x = signal.filtfilt(b, a, x, axis=1)
    x = signal.sosfiltfilt(signal.butter(4, 1, "highpass", fs=FS, output="sos"), x, axis=1)   # 4th-order zero-phase 1 Hz high-pass
    return x, ph


def condition_masks(ph, n):
    sec = np.unique(photic_edges(ph) // FS); act = np.zeros(n // FS + 1, bool)
    if len(sec):
        runs = np.split(sec, np.where(np.diff(sec) > 16)[0] + 1)              # merge gaps <= 15 s (addendum 1: one block per IPS procedure)
        for r in runs:
            act[r[0]:r[-1] + 1] = True
    near = np.convolve(act.astype(float), np.ones(11), "same") > 0             # within 5 s of IPS
    ips = np.repeat(act, FS)[:n]; rest = ~np.repeat(near, FS)[:n]; return {"ips": ips, "rest": rest}


def montage(x, m):
    i = {c: k for k, c in enumerate(CH)}
    if m == "CAR":
        return x - x.mean(0)
    if m == "Cz":
        return np.delete(x - x[i["CZ"]], i["CZ"], axis=0)
    if m == "Laplacian":
        return np.array([x[i[c]] - x[[i[n] for n in NB[c].split()]].mean(0) for c in CH])
    return np.array([x[i[p.split("-")[0]]] - x[i[p.split("-")[1]]] for p in DB])


def segments(mask, L):
    """Start indices of non-overlapping segments of L seconds lying fully inside the mask."""
    n = L * FS; starts = []; edges = np.diff(np.concatenate([[0], mask.astype(int), [0]])); on, off = np.where(edges == 1)[0], np.where(edges == -1)[0]
    for a, b in zip(on, off):
        starts += list(range(a, b - n + 1, n))
    return np.array(starts, int)


def perm_entropy(s):
    """Normalised permutation entropy, order 3, delay 1. s: (..., T)."""
    a, b, c = s[..., :-2], s[..., 1:-1], s[..., 2:]; code = (a > b) * 4 + (b > c) * 2 + (a > c)
    p = np.stack([(code == k).mean(-1) for k in range(8)], -1); p = np.where(p > 0, p, 1)
    return -(p * np.log(p)).sum(-1) / np.log(6)


def seg_features(W, Wa):
    """W: (S, C, T) segments; Wa: same segments of the 8-13 Hz analytic signal. Returns dict type -> (S, F)."""
    S, C, T = W.shape; f, P = signal.welch(W, FS, nperseg=min(T, 2 * FS), axis=-1)
    bp = np.stack([P[..., (f >= lo) & (f < hi)].sum(-1) * (f[1] - f[0]) for lo, hi in BANDS], -1)        # (S, C, 5)
    spec = np.concatenate([np.log10(bp + 1e-12), bp / bp.sum(-1, keepdims=True)], -1).reshape(S, -1)
    d1, d2 = np.diff(W, axis=-1), np.diff(W, 2, axis=-1); v0, v1, v2 = W.var(-1), d1.var(-1), d2.var(-1)
    mob = np.sqrt(v1 / v0); comp = np.sqrt(v2 / v1) / mob
    peaks = ((W[..., 1:-1] > W[..., :-2]) & (W[..., 1:-1] > W[..., 2:])).sum(-1) / (T / FS)
    utm = np.stack([peaks, np.abs(d1).mean(-1), v0, mob, comp, perm_entropy(W)], -1).reshape(S, -1)
    Z = (W - W.mean(-1, keepdims=True)) / (W.std(-1, keepdims=True) + 1e-12); cc = np.abs(np.einsum("sct,sdt->scd", Z, Z) / T)
    ph = np.exp(1j * np.angle(Wa)); plv = np.abs(np.einsum("sct,sdt->scd", ph, ph.conj()) / T)
    off = ~np.eye(C, dtype=bool); conn = np.stack([cc[:, off].reshape(S, C, C - 1).mean(-1), plv[:, off].reshape(S, C, C - 1).mean(-1)], -1).reshape(S, -1)
    return {"spectral": spec, "utm": utm, "conn": conn}


def combine(F, how):
    if len(F) == 0:
        return np.full(F.shape[1], np.nan)
    if how in ("mean", "median", "sd"):
        return {"mean": F.mean(0), "median": np.median(F, 0), "sd": F.std(0)}[how]
    if len(F) < 3:
        return np.zeros(F.shape[1])
    return np.nan_to_num(stats.skew(F, 0) if how == "skew" else stats.kurtosis(F, 0))


def rec_features(f):
    x, ph = load(f); masks = condition_masks(ph, x.shape[1]); out, nseg = {}, {}
    ba = signal.butter(4, [8, 13], "bandpass", fs=FS, output="sos")
    for m in MONTAGES:
        xm = montage(x, m); xa = signal.hilbert(signal.sosfiltfilt(ba, xm, axis=1), axis=1)
        for cond in CONDS:
            for L in LENS:
                st = segments(masks[cond], L); n = L * FS
                W = np.stack([xm[:, s:s + n] for s in st]) if len(st) else np.zeros((0, len(xm), n))
                if len(W):
                    rms = np.sqrt((W ** 2).mean(-1)); keep = (rms <= 5 * np.median(rms, 0)).all(1); W, st = W[keep], st[keep]
                nseg[(cond, m, L)] = len(W)
                if len(W):
                    Wa = np.stack([xa[:, s:s + n] for s in st]); feats = seg_features(W, Wa)
                for t in TYPES:
                    d = {"spectral": 10, "utm": 6, "conn": 2}[t] * len(xm)
                    for c in COMBS:
                        out[(cond, t, m, L, c)] = combine(feats[t], c) if len(W) else np.full(d, np.nan)
    return f, out, nseg, int(masks["ips"].sum() // FS), int(masks["rest"].sum() // FS)


# ------------------------------------------------------------------ evaluation
def meta():
    z = pd.read_csv(f"{D}/zanardini_z31.csv").sort_values("file").reset_index(drop=True)
    h = pd.read_csv(f"{D}/edf_headers.csv").set_index("file").loc[z.file]
    z["age"] = h.patient.str.extract(r"Age:(\d+)")[0].astype(float).values; z["male"] = (h.patient.str.split().str[1] == "M").astype(float).values
    z["y"] = (z.cls == "00_epilepsy").astype(int); return z


def xgb(seed):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=100, max_depth=6, subsample=0.9, random_state=seed, n_jobs=1, verbosity=0)


def loso_scores(X, y, subj, seeds, subjects=None):
    """Subject-level LOSO: train on recordings of the other subjects, average the held-out subject's recording probabilities.
    Returns subject scores averaged over seeds (aligned with `subjects`)."""
    from sklearn.metrics import roc_auc_score
    subjects = np.unique(subj) if subjects is None else subjects; sc = np.zeros(len(subjects))
    for seed in seeds:
        for i, s in enumerate(subjects):
            tr, te = subj != s, subj == s
            if len(np.unique(y[tr])) < 2:
                sc[i] += 0.5; continue
            sc[i] += xgb(seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1].mean()
    return sc / len(seeds)


def subj_auc(scores, ys):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(ys, scores)


def config_job(args):
    X, y, subj, seeds = args; return loso_scores(X, y, subj, seeds)


def nested_job(args):
    """Outer subject o: choose the config by inner LOSO on the remaining subjects (seed 0), then score o with it."""
    o, XS, y, subj = args; inner = subj != o; ys_in = pd.Series(y[inner]).groupby(subj[inner]).first()
    best, bauc = None, -1
    for k, X in enumerate(XS):
        sc = loso_scores(X[inner], y[inner], subj[inner], [0], ys_in.index.values)
        a = subj_auc(sc, ys_in.values)
        if a > bauc:
            best, bauc = k, a
    X = XS[best]; p = xgb(0).fit(X[inner], y[inner]).predict_proba(X[subj == o])[:, 1].mean(); return o, best, bauc, p


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "subset":
        h = pd.read_csv(f"{D}/edf_headers.csv"); files = list(h[h["labels"].str.contains("PHOTIC PH", regex=False)].file)
        with Pool(16) as pool:
            n = dict(pool.map(subset_job, files))
        sel = sorted(f for f in files if n[f] > 1000); saved = sorted(pd.read_csv(f"{D}/zanardini_z31.csv").file)
        gap = sorted(n.values()); print(f"PHOTIC PH files {len(files)}; transitions: largest inactive {max(v for v in gap if v <= 1000)}, smallest active {min(v for v in gap if v > 1000)}")
        print(("PASS" if sel == saved else "FAIL") + f" subset rule re-derived from the EDFs gives the saved Z31 list ({len(sel)} recordings)")
    elif step == "features":
        z = meta()
        with Pool(20) as pool:
            res = pool.map(rec_features, list(z.file))
        res = {r[0]: r for r in res}; arr = {}
        for cond, t, m, L, c in [(cd,) + cf for cd in CONDS for cf in CONFIGS]:
            arr[f"{cond}|{t}|{m}|{L}|{c}"] = np.vstack([res[f][1][(cond, t, m, L, c)] for f in z.file])
        nseg = pd.DataFrame([{**{"file": f}, **{f"{k[0]}|{k[1]}|{k[2]}": v for k, v in res[f][2].items()}, "ips_s": res[f][3], "rest_s": res[f][4]} for f in z.file])
        np.savez_compressed(f"{D}/zanardini_features.npz", files=z.file.values, **arr); nseg.to_csv(f"{R}/round8_segments.csv", index=False)
        print(f"recordings {len(z)}, configs {len(arr)}; IPS seconds median {nseg.ips_s.median():.0f} (min {nseg.ips_s.min()}), rest seconds median {nseg.rest_s.median():.0f} (min {nseg.rest_s.min()})")
        allnan = [k for k, v in arr.items() if np.isnan(v).all(1).any()]; print(f"configs with at least one recording without segments: {len(allnan)}")
        print(f"finite values (excluding no-segment rows): {all(np.isfinite(v[~np.isnan(v).all(1)]).all() for v in arr.values())}")
    elif step == "eval":
        from sklearn.metrics import roc_auc_score
        import evalkit as ek
        z = meta(); y, subj = z.y.values, z.subj.values; S = np.unique(subj); ys = pd.Series(y).groupby(subj).first().loc[S].values
        Z = np.load(f"{D}/zanardini_features.npz", allow_pickle=True); assert list(Z["files"]) == list(z.file)
        sm = z.sort_values("year").groupby("subj").first().loc[S]                 # subject metadata from the earliest Z31 recording
        print(f"Z31: {len(z)} recordings, {len(S)} subjects ({ys.sum()} epilepsy / {len(S) - ys.sum()} non-epilepsy)")
        # ---- C1 metadata floor under the paper's protocol
        a_year = roc_auc_score(ys, -sm.year.values); a_year_rec = roc_auc_score(y, -z.year.values)
        M = sm[["year", "age", "male"]].values.astype(float); pm = np.zeros(len(S))
        for i in range(len(S)):
            tr = np.arange(len(S)) != i; pm[i] = ek.nested_lr(n_inner=5).fit(M[tr], ys[tr]).predict_proba(M[i:i + 1])[:, 1][0]
        a_meta = roc_auc_score(ys, pm)
        print(f"\n[C1] recording year alone (no training): subject AUC {a_year:.3f}, recording AUC {a_year_rec:.3f}; metadata LR (year, age, sex) LOSO: AUC {a_meta:.3f}")
        out = {"year_auc": a_year, "year_auc_rec": a_year_rec, "meta_lr_auc": a_meta}; NULL = 0 if "nonull" in sys.argv else 50
        for cond in CONDS:
            keys = [f"{cond}|{t}|{m}|{L}|{c}" for t, m, L, c in CONFIGS]; XS = [Z[k] for k in keys]
            # ---- C3.1 as in the paper: rank configs by their own LOSO AUC (5 seeds)
            with Pool(12) as pool:
                sc = pool.map(config_job, [(X, y, subj, range(5)) for X in XS])
            auc = np.array([roc_auc_score(ys, s) for s in sc]); o = np.argsort(-auc)
            ens = {k: roc_auc_score(ys, np.mean([sc[i] for i in o[:k]], 0)) for k in (3, 7)}
            pt = pd.Series(auc, index=keys); best_type = pt.groupby([k.split("|")[1] for k in keys]).max()
            print(f"\n[{cond}] as in the paper: best config {keys[o[0]]} AUC {auc[o[0]]:.3f}; top-3 ensemble {ens[3]:.3f}; top-7 ensemble {ens[7]:.3f}")
            print(f"   best per feature type: {best_type.round(3).to_dict()}; median config AUC {np.median(auc):.3f}; configs >= 0.80: {(auc >= 0.8).sum()} / {len(auc)}")
            # ---- C3.2 nested selection (honest)
            with Pool(12) as pool:
                nest = pool.map(nested_job, [(s, XS, y, subj) for s in S])
            pn = np.array([n[3] for n in nest]); a_nest = roc_auc_score(ys, pn); chosen = pd.Series([keys[n[1]] for n in nest]).value_counts()
            rng = np.random.default_rng(8000); perm = np.array([roc_auc_score(rng.permutation(ys), pn) for _ in range(200)])
            p_nest = (1 + (perm >= a_nest).sum()) / 201
            boot = []
            for _ in range(2000):
                b = rng.integers(0, len(S), len(S))
                if len(np.unique(ys[b])) == 2:
                    boot.append(roc_auc_score(ys[b], pn[b]) - roc_auc_score(ys[b], -sm.year.values[b]))
            ci = np.percentile(boot, [2.5, 97.5])
            print(f"   nested selection: AUC {a_nest:.3f} (permutation p vs 0.5 = {p_nest:.3f}); nested - year alone {a_nest - a_year:+.3f}, 95% CI {ci[0]:+.3f} to {ci[1]:+.3f}")
            print(f"   inner-CV AUC of the chosen config: median {np.median([n[2] for n in nest]):.3f}; distinct configs chosen {len(chosen)}; most frequent {chosen.index[0]} ({chosen.iloc[0]}x)")
            res = dict(best=auc[o[0]], best_cfg=keys[o[0]], top3=ens[3], top7=ens[7], nested=a_nest, p_nested=p_nest, nest_minus_year_lo=ci[0], nest_minus_year_hi=ci[1])
            pd.DataFrame(dict(config=keys, auc=auc)).to_csv(f"{R}/round8_{cond}_configs.csv", index=False)
            # ---- C3.3 null: the whole as-in-the-paper procedure with permuted subject labels
            if NULL:
                rng = np.random.default_rng(8100 + CONDS.index(cond)); nb, n3, n7 = [], [], []
                for i in range(NULL):
                    yp_s = rng.permutation(ys); yp = pd.Series(yp_s, index=S).loc[subj].values
                    with Pool(12) as pool:
                        scn = pool.map(config_job, [(X, yp, subj, range(5)) for X in XS])
                    an = np.array([roc_auc_score(yp_s, s) for s in scn]); on = np.argsort(-an)
                    nb.append(an[on[0]]); n3.append(roc_auc_score(yp_s, np.mean([scn[j] for j in on[:3]], 0))); n7.append(roc_auc_score(yp_s, np.mean([scn[j] for j in on[:7]], 0)))
                    print(f"      null {i + 1}/{NULL}: best {nb[-1]:.3f} top3 {n3[-1]:.3f} top7 {n7[-1]:.3f}", flush=True)
                nb, n3, n7 = map(np.array, (nb, n3, n7)); np.savez(f"{R}/round8_{cond}_null.npz", best=nb, top3=n3, top7=n7)
                for name, obs, nul in [("best", auc[o[0]], nb), ("top-3", ens[3], n3), ("top-7", ens[7], n7)]:
                    print(f"   null {name}: mean {nul.mean():.3f}, 95th pct {np.percentile(nul, 95):.3f}, max {nul.max():.3f}; observed {obs:.3f}, p = {(1 + (nul >= obs).sum()) / (1 + NULL):.3f}")
                    res[f"null_{name}_mean"], res[f"null_{name}_p95"], res[f"p_{name}"] = nul.mean(), np.percentile(nul, 95), (1 + (nul >= obs).sum()) / (1 + NULL)
            out[cond] = res
        pd.DataFrame({k: v for k, v in out.items() if isinstance(v, dict)}).to_csv(f"{R}/round8_results.csv")
        print("\nsummary:", {k: (round(v, 3) if not isinstance(v, dict) else None) for k, v in out.items()})
