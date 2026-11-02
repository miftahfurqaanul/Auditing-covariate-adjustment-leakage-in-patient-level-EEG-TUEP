"""Round 3 Tang-F-T on the GPU (reports/round3_prediction.md): Tang-F with lr chosen on validation from {5e-3, 1e-3, 5e-4}
(epoch chosen on validation within each lr). Same splits as round3.py. Usage: python round3_tang.py <protocol> [check]"""
import os, sys, json, torch, numpy as np
import tang_fixed as tf, round3 as r3
LRS = [5e-3, 1e-3, 5e-4]
if __name__ == "__main__":
    protocol, check = sys.argv[1], len(sys.argv) > 2
    dev = torch.device("cuda"); X, meta = tf.load(); y = meta.label.values; subj = meta.subj.values; ylab = meta.groupby("subj").label.first()
    Xg = torch.from_numpy(X).to(dev); out = f"{r3.OUT}/{protocol}/Tang-F-T"; os.makedirs(out, exist_ok=True)
    for i, (name, tr, va, te) in enumerate(r3.splits(protocol)):
        assert set(tr + va + te) <= set(ylab.index), "patient without Tang features"
        if os.path.exists(f"{out}/{name}.json") and not check:
            continue
        runs = [tf.train_fold(Xg, y, subj, tr, va, te, ylab, 3000 + i, dev, epochs=2 if check else 100, lr=lr) for lr in LRS]
        b = int(np.argmax([r["val_auc"] for r in runs])); best = runs[b]
        res = dict(val=best["val"], test=best["test"], chosen=dict(lr=LRS[b], best_epoch=best["best_epoch"], val_auc=best["val_auc"],
                                                                     val_auc_per_lr=[r["val_auc"] for r in runs]), train=len(tr))
        assert set(res["test"]) == set(te) and set(res["val"]) == set(va)
        if check:
            print(f"check {protocol} {name}: per-lr val AUC {np.round(res['chosen']['val_auc_per_lr'], 3)}, chosen lr {res['chosen']['lr']}, test {len(te)}: PASS"); break
        json.dump(res, open(f"{out}/{name}.json", "w"))
        print(f"{protocol} {name}: lr {LRS[b]}, epoch {best['best_epoch']}, val AUC {best['val_auc']:.3f}", flush=True)
