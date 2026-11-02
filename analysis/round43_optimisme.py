#!/usr/bin/env python3
"""Round 43 part D (CPU): selection-optimism correction and max-T test (Reality Check/Westfall-Young) on the stored out-of-fold prediction matrix. See PRAREGISTRASI_putaran43.md section 4.
Usage: CUDA_VISIBLE_DEVICES="" python round43_optimisme.py --run | --perm   (raw results to hasil43/D_*.npz before the summary: round43_ringkas.py)"""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "hasil43")
import numpy as np
from scipy.stats import rankdata

NB = 2000; Yp = None


def load_candidates():
    """Returns (S_utama, S_semua, ref, Y): dict name -> matrix (10, 200)."""
    sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
    import round28_analisis as A
    Y = A.Y; L = lambda f: np.load(os.path.join(HERE, f), allow_pickle=True); h30, h29, h31, h32, h33 = L("hasil30/eval_prediksi_v4.npz"), L("hasil29/B_prediksi_data_asli.npz"), L("hasil31/fm_beku_prediksi.npz"), L("hasil32/anggaran_prediksi.npz"), L("hasil33/pembanding_prediksi.npz")
    d43 = L("hasil43/A_prediksi.npz") if os.path.exists(f"{OUT}/A_prediksi.npz") else None; M = lambda z, k: np.asarray(z[k], float)
    main = {"I2-adj": M(h30, "I2_adj"), "P3-adj": M(h30, "P3_adj"), "NM-adj": M(h30, "NM_adj"), "F2-adj": M(h30, "F2_adj"), "C1": M(h30, "C1"), "LaBraM-FT": M(h30, "LaBraM_FT"), "CBraMod-FT": M(h30, "CBraMod_FT"),
            "BENDR-V4+AS": M(h31, "BENDR_V4+AS"), "LaBraM-V4+AS": M(h31, "LaBraM_V4+AS"), "DNM-V4": M(h33, "DNM_V4"), "BP-V4-tuned": M(h32, "BP_V4_tuned")}
    if d43 is not None:
        for k in ("LaBraM-eq", "CBraMod-eq", "BENDR-eq", "Comp50"):
            main[k] = M(d43, f"{k}__AS")
    allc = dict(main)
    for k in ("C1_AS", "C1_noAS", "V4_AS", "V4_noAS"):
        allc[f"h29:{k}"] = M(h29, k)
    for k in ("Riemann_mentah", "Mamba_mentah"):
        allc[f"h30:{k}"] = M(h30, k)
    for z, tag in ((h31, "h31"), (h32, "h32")):
        for k in z.files:
            allc[f"{tag}:{k}"] = M(z, k)
    for k in ("DNM_F", "DNM_V4", "Rohira_F", "Rohira_AS_AS"):
        allc[f"h33:{k}"] = M(h33, k)
    if d43 is not None:
        for k in d43.files:
            if not k.endswith("__C"):
                allc[f"h43:{k}"] = M(d43, k)
    uniq = {}
    for k, v in allc.items():                                                                              # drop identical duplicates (order: S_utama first)
        if not any(np.allclose(v, u, atol=1e-12) for u in uniq.values()):
            uniq[k] = v
    ref = {"Floor": M(h30, "Floor"), "C1": M(h30, "C1")}
    for z in (main, uniq):
        for k, v in z.items():
            assert v.shape == (10, 200) and not np.isnan(v).any(), k
    return main, uniq, ref, Y


def auc_stack(P, y, idx):
    """P: (K, 10, 200); idx: patient indices (duplicates allowed). Mean AUC over repetitions, (K,)."""
    yy = y[idx]; n1 = yy.sum(); n0 = len(yy) - n1; R = rankdata(P[:, :, idx], axis=-1); a = (R[:, :, yy == 1].sum(-1) - n1 * (n1 + 1) / 2) / (n1 * n0); return a.mean(1)


def boot_idx(rng, pos, neg):
    return np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def run_set(P, names, y, refs, seed_opt, seed_mt, nb=NB):
    """Optimism + max-T for one candidate set. Returns the raw dict."""
    K = len(names); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; full = auc_stack(P, y, np.arange(len(y))); w0 = int(np.argmax(full))
    rng = np.random.default_rng(seed_opt); opt = np.zeros(nb); win = np.zeros(nb, int); ain = np.zeros((nb, K)); aoob = np.zeros((nb, K))
    for b in range(nb):
        while True:
            idx = boot_idx(rng, pos, neg); oob = np.setdiff1d(np.arange(len(y)), idx)
            if y[oob].sum() >= 5 and (len(oob) - y[oob].sum()) >= 5:
                break
        ain[b] = auc_stack(P, y, idx); aoob[b] = auc_stack(P, y, oob); win[b] = int(np.argmax(ain[b])); opt[b] = ain[b, win[b]] - aoob[b, win[b]]
    res = dict(names=np.array(names), full=full, w0=w0, opt=opt, win=win, ain=ain, aoob=aoob)
    rng = np.random.default_rng(seed_mt); boots = np.zeros((nb, K)); refb = {r: np.zeros(nb) for r in refs}
    for b in range(nb):
        idx = boot_idx(rng, pos, neg); boots[b] = auc_stack(P, y, idx)
        for r, Rm in refs.items():
            refb[r][b] = auc_stack(Rm[None], y, idx)[0]
    res["boots"] = boots
    for r, Rm in refs.items():
        res[f"refboot_{r}"] = refb[r]; res[f"reffull_{r}"] = auc_stack(Rm[None], y, np.arange(len(y)))[0]
    return res


def maxT(res, r, exclude=()):
    keep = np.array([i for i, n in enumerate(res["names"]) if n not in exclude]); full = res["full"][keep]; d = full - res[f"reffull_{r}"]; db = res["boots"][:, keep] - res[f"refboot_{r}"][:, None]
    se = db.std(0, ddof=1); se = np.where(se < 1e-9, 1e-9, se); t_obs = (d / se).max(); tb = ((db - d) / se).max(1); p = (1 + (tb >= t_obs).sum()) / (1 + len(tb)); return dict(ref=r, k=len(keep), best=str(res["names"][keep][int(np.argmax(d / se))]), d_best=float(d.max()), t_obs=float(t_obs), p=float(p))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--run"; t0 = time.time(); main, allc, ref, Y = load_candidates()
    if mode == "--run":
        for tag, S in (("utama", main), ("semua", allc)):
            names = list(S); P = np.stack([S[k] for k in names]); res = run_set(P, names, Y, ref, 43100, 43101); np.savez(f"{OUT}/D_bruto_{tag}.npz", **res); print(f"{tag}: K={len(names)} winner {names[res['w0']]} AUC {res['full'][res['w0']]:.4f}; mean optimism {res['opt'].mean():.4f} [{time.time() - t0:.0f}s]", flush=True)
    elif mode == "--perm":                                                                                  # kalibrasi prosedur: label dipermutasi (nol sejati); 40 permutasi x 400 bootstrap
        names = list(main); P = np.stack([main[k] for k in names]); rows = []
        for i in range(40):
            yp = np.random.default_rng(43200 + i).permutation(Y); res = run_set(P, names, yp, {"Floor": ref["Floor"]}, 43300 + i, 43400 + i, nb=400); m = maxT(res, "Floor")
            rows.append((res["full"][res["w0"]], res["opt"].mean(), res["full"][res["w0"]] - res["opt"].mean(), m["p"]))
        np.save(f"{OUT}/D_perm.npy", np.array(rows)); r = np.array(rows); print(f"permutasi label (40): AUC pemenang {r[:, 0].mean():.4f}; optimisme {r[:, 1].mean():.4f}; AUC terkoreksi {r[:, 2].mean():.4f}; p max-T < 0.05: {(r[:, 3] < 0.05).mean():.3f} [{time.time() - t0:.0f}s]", flush=True)
