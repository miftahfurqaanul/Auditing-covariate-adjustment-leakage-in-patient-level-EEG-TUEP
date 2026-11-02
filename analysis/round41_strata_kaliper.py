#!/usr/bin/env python3
"""Round 41 (CPU): AUC within year x device strata with an age caliper (k = 5, 10, 15, infinity). See PRAREGISTRASI_putaran41.md. No training: uses the stored out-of-fold predictions.
Usage: CUDA_VISIBLE_DEVICES="" python round41_strata_kaliper.py --run   (raw results to hasil41/ before the summary) | --ringkas"""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np, pandas as pd
import round9 as r9, round4 as r4
OUT = os.path.join(HERE, "hasil41"); os.makedirs(OUT, exist_ok=True)
MODELS = ["F2_adj", "I2_adj", "P3_adj", "NM_adj", "C1", "Floor", "LaBraM_FT", "CBraMod_FT", "Demo"]; KS = [5.0, 10.0, 15.0, np.inf]; NB = 2000; NPERM = 200
Y = r9.Y; N = len(Y); AGE = r4.AGE.astype(float); STR = {"year x device": r9.STRATA["year x device"], "year": r9.STRATA["year"]}
Z = np.load(os.path.join(HERE, "hasil30", "eval_prediksi_v4.npz")); PR = {m: np.asarray(Z[m], float) for m in MODELS}; MEAN = np.column_stack([np.nanmean(PR[m], 0) for m in MODELS])


def make_pairs(y, g, idx):
    """All (epilepsy, non-epilepsy) pairs within the same cell; idx = patient indices (duplicates allowed, for the bootstrap)."""
    gi, yi = g[idx], y[idx]; ei, ni = [], []
    for s in np.unique(gi):
        e = idx[(gi == s) & (yi == 1)]; n = idx[(gi == s) & (yi == 0)]
        if len(e) and len(n):
            ei.append(np.repeat(e, len(n))); ni.append(np.tile(n, len(e)))
    return (np.concatenate(ei), np.concatenate(ni)) if ei else (np.array([], int), np.array([], int))


def auc_by_k(S, ei, ni):
    """S: (N, nmodels) scores. Returns (nK, nmodels) AUC and (nK,) number of pairs."""
    C = (S[ei] > S[ni]).astype(float) + 0.5 * (S[ei] == S[ni]); d = np.abs(AGE[ei] - AGE[ni]); A = np.full((len(KS), S.shape[1]), np.nan); npairs = np.zeros(len(KS), int)
    for i, k in enumerate(KS):
        m = d <= k; npairs[i] = int(m.sum())
        if npairs[i]:
            A[i] = C[m].sum(0) / npairs[i]
    return A, npairs


def run():
    t0 = time.time(); res = {}
    for sname, g in STR.items():
        ei, ni = make_pairs(Y, g, np.arange(N)); A0, np0 = auc_by_k(MEAN, ei, ni)
        rep = np.array([auc_by_k(np.column_stack([PR[m][r] for m in MODELS]), ei, ni)[0] for r in range(10)]).mean(0)                       # media dos AUC por repetição
        rng = np.random.default_rng(41000 + (0 if sname == "year x device" else 1)); groups = [np.where(g == s)[0] for s in np.unique(g)]; B = np.full((NB, len(KS), len(MODELS)), np.nan); NP = np.zeros((NB, len(KS)), int)
        for b in range(NB):
            idx = np.concatenate([rng.choice(v, len(v)) for v in groups]); e, n = make_pairs(Y, g, idx); B[b], NP[b] = auc_by_k(MEAN, e, n)
        np.savez(os.path.join(OUT, f"bruto_{sname.replace(' ', '_')}.npz"), point=A0, npairs=np0, rep_mean=rep, boot=B, boot_npairs=NP, ks=np.array(KS), models=np.array(MODELS))
        print(f"{sname}: pasangan per k {np0.tolist()}; F2 AUC_k {np.round(A0[:, 0], 4).tolist()} [{time.time() - t0:.0f}s]", flush=True)
    # permutation control: labels shuffled within each year x device cell
    g = STR["year x device"]; rng = np.random.default_rng(41001); cells = [np.where(g == s)[0] for s in np.unique(g)]; PERM = np.full((NPERM, len(KS), len(MODELS)), np.nan)
    for i in range(NPERM):
        yp = Y.copy()
        for v in cells:
            yp[v] = rng.permutation(Y[v])
        e, n = make_pairs(yp, g, np.arange(N)); PERM[i] = auc_by_k(MEAN, e, n)[0]
    np.savez(os.path.join(OUT, "bruto_permutasi.npz"), perm=PERM, ks=np.array(KS), models=np.array(MODELS)); print(f"permutation done [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    if "--run" in sys.argv:
        run()
    else:
        print("use --run (ringkasan: round41_ringkas.py)")
