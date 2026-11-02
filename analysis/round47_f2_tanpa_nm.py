#!/usr/bin/env python3
"""Round 47: AUC per fold for F2 without NM (mean of the I2_adj and P3_adj probabilities) from the stored out-of-fold predictions (PRAREGISTRASI_putaran47.md; SHA-256 recorded before the run).
CPU only (CUDA_VISIBLE_DEVICES=""). No new training. Usage: python round47_f2_tanpa_nm.py --cek-kunci | --run
--cek-kunci only checks the shape of the files and that F2_adj = mean of the three views; it does not compute or print any AUC."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/home/miftah_f/RTX-5090/tuep_fresh/code")
import numpy as np
import round28_analisis as A
c17, J = A.c17, A.J
OUT = os.path.join(HERE, "hasil47"); os.makedirs(OUT, exist_ok=True)
Z = np.load(os.path.join(HERE, "hasil30/eval_prediksi_v4.npz")); K = ["I2_adj", "P3_adj", "NM_adj", "F2_adj", "C1", "Floor"]
assert all(Z[k].shape == (10, 200) for k in K)
assert np.abs(Z["F2_adj"] - (Z["I2_adj"] + Z["P3_adj"] + Z["NM_adj"]) / 3).max() < 1e-9, "F2_adj is not the mean of the three views"
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cek-kunci":
        print("shape (10, 200) for all keys; F2_adj = mean of I2_adj, P3_adj, NM_adj (max difference < 1e-9); AUC not computed"); sys.exit()
    S = {"F2_tanpa_NM": (Z["I2_adj"] + Z["P3_adj"]) / 2, "F2_adj": Z["F2_adj"], "C1": Z["C1"], "Floor": Z["Floor"], "I2_adj": Z["I2_adj"], "P3_adj": Z["P3_adj"]}
    FA = {k: c17.fold_aucs(J, v) for k, v in S.items()}; PO = {k: c17.pooled_auc(v) for k, v in S.items()}
    for k in FA: assert len(FA[k]) == 50
    np.savez(f"{OUT}/auc_per_lipatan.npz", **{f"fold_{k}": v for k, v in FA.items()}, **{f"pooled_{k}": v for k, v in PO.items()})  # raw output is saved before summarizing
    print("done; raw output saved in hasil47/auc_per_lipatan.npz")
