"""Round 50 (descriptive, OUTSIDE the plan; written after ringkas_A was seen): all RF/XGB pipelines over 120 replicates (H1, gamma 0 and 1), pool decisions R1/R2."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, pandas as pd, round49_kalibrasi as K
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); FC = [f"fold_auc_{i}" for i in range(5)]; rows = []
for fam, st in (("RF", 30), ("XGB", 20)):
    o = pd.read_csv(f"{H}/hasil42/{fam}.csv"); o = o[(o.cond == "H1") & o.gamma.isin([0.0, 1.0]) & (o.rep < st)]; D = pd.concat([o, pd.read_csv(f"{H}/hasil50/A_{fam}.csv")]); D["fm"] = D[FC].mean(1); D["pp"] = D["transform"] + "-" + D["adjust"]
    for g, G in D.groupby("gamma"):
        W = G.pivot_table(index="rep", columns="pp", values="fm"); assert len(W) == 120
        for p in W.columns:
            if p == "raw-none": continue
            d = (W[p] - W["raw-none"]).values * 100; x = W[p].values
            rows.append(dict(fam=fam, gamma=g, pipe=p, auc=x.mean() * 100, delta=d.mean(), se=d.std(ddof=1) / np.sqrt(120), R1=K.LAB[K.dec_pool(x, "R1")], R2=K.LAB[K.dec_pool(d / 100, "R2")]))
R = pd.DataFrame(rows); R.to_csv(f"{H}/hasil50/deskriptif_120_semua_pipeline.csv", index=False); print(R.round(3).to_string())
