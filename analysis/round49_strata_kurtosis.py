#!/usr/bin/env python3
"""Round 49E + 49F (CPU): (E) weighting sensitivity of the caliper strata (equal weight per cell; without the largest cell 2012|X) from the stored predictions;
(F) the S3 kurtosis/skew table rebuilt from the data (float64, all observation units). See PRAREGISTRASI_putaran49.md sections 49E and 49F.
Usage: CUDA_VISIBLE_DEVICES="" python round49_strata_kurtosis.py --check | --run"""
import os, sys, time, json, hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["PYTHONWARNINGS"] = "ignore"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import round41_strata_kaliper as R41
OUT = os.path.join(HERE, "hasil49"); os.makedirs(OUT, exist_ok=True)
Y, AGE, G = R41.Y, R41.AGE, R41.STR["year x device"]; MODELS = ["F2_adj", "I2_adj", "C1", "Floor"]; IDX = [R41.MODELS.index(m) for m in MODELS]; KS = (5.0, 10.0, 15.0)


def weighted_aucs(S, ei, ni, k, drop=None):
    """S: (N, nmodel). Returns (pair-weighted AUC, equal-weight-per-cell AUC, AUC without cell `drop`)."""
    m = np.abs(AGE[ei] - AGE[ni]) <= k; e, n = ei[m], ni[m]; C = (S[e] > S[n]).astype(float) + 0.5 * (S[e] == S[n]); cell = G[e]
    pair = C.mean(0); per = np.array([C[cell == c].mean(0) for c in np.unique(cell)]); keep = cell != drop
    return pair, per.mean(0), C[keep].mean(0)


def run():
    S = R41.MEAN[:, IDX]; ei, ni = R41.make_pairs(Y, G, np.arange(R41.N)); rng = np.random.default_rng(49500); groups = [np.where(G == s)[0] for s in np.unique(G)]; rows = []
    for k in KS:
        pt = weighted_aucs(S, ei, ni, k, "2012|X"); B = np.full((2000, 3, len(MODELS)), np.nan)
        for b in range(2000):
            idx = np.concatenate([rng.choice(v, len(v)) for v in groups]); e, n = R41.make_pairs(Y, G, idx)
            if len(e):
                B[b] = np.array(weighted_aucs(S, e, n, k, "2012|X"))
        for j, w in enumerate(("bobot pasangan (asli)", "bobot sama per sel", "tanpa sel 2012|X")):
            for i, mdl in enumerate(MODELS):
                lo, hi = np.nanpercentile(B[:, j, i], [2.5, 97.5]); rows.append(dict(k=k, pembobotan=w, model=mdl, auc=pt[j][i], ci_lo=lo, ci_hi=hi))
    T = pd.DataFrame(rows); T.to_csv(f"{OUT}/E_strata_bobot.csv", index=False); print(T.round(4).to_string(index=False), flush=True)


def kurtosis_table():
    import round7 as r7, round34_tang_v4 as T, normmamba as nm
    rows = []
    def add(feat, scale, unit, M):
        M = np.asarray(M, np.float64); k = stats.kurtosis(M, axis=0); s = stats.skew(M, axis=0)
        rows.append(dict(fitur=feat, skala=scale, unit=unit, n_baris=M.shape[0], n_fitur=M.shape[1], kurtosis_median=np.median(k), kurtosis_maks=k.max(), skew_median=np.median(s), nilai_mutlak_maks=np.abs(M).max()))
    XR, WS, _, _ = r7.rohira_inputs(); pidx = pd.factorize(WS)[0]; XL = np.log1p(XR.astype(np.float64))
    pm = lambda X: pd.DataFrame(X).groupby(pidx).mean().values
    add("Rohira", "mentah", "jendela 10 s", XR); add("Rohira", "mentah", "rata-rata pasien", pm(XR.astype(np.float64))); add("Rohira", "log1p", "jendela 10 s", XL); add("Rohira", "log1p", "rata-rata pasien", pm(XL))
    X, meta = T.tf.load(); R = X.reshape(-1, X.shape[-1]).astype(np.float64); add("Tang", "mentah", "detik", R); add("Tang", "symlog", "detik", np.sign(R) * np.log1p(np.abs(R)))
    add("NM", "mentah (sebelum V4)", "segmen 4 s", nm.XSEG)
    K = pd.DataFrame(rows); K.to_csv(f"{OUT}/F_kurtosis.csv", index=False); print(K.to_string(index=False), flush=True)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    json.dump(dict(prereg_sha256=sha(os.path.join(HERE, "PRAREGISTRASI_putaran49.md")), code_sha256=sha(__file__)), open(f"{OUT}/meta_EF.json", "w"), indent=1)
    if mode == "--check":                                   # pair weights must equal the round 41 point (bruto_year_x_device.npz)
        z = np.load(os.path.join(HERE, "hasil41", "bruto_year_x_device.npz")); ei, ni = R41.make_pairs(Y, G, np.arange(R41.N))
        for i, k in enumerate(R41.KS[:3]):
            pair = weighted_aucs(R41.MEAN[:, IDX], ei, ni, k)[0]; assert np.allclose(pair, z["point"][i, IDX]), (k, pair, z["point"][i, IDX])
        print("checks passed (pair weights = round 41 point for k = 5, 10, 15)"); sys.exit()
    run(); kurtosis_table(); open(f"{OUT}/EF_SELESAI", "w").write(time.ctime())
