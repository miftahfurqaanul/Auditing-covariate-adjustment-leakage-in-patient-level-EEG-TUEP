"""Double check of the GPU-resident R1b path in tang_train.py before running R1b (run alone on the GPU)."""
import time, numpy as np, torch
import tang_train as tt
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m, flush=True)
dev = torch.device("cuda"); torch.manual_seed(0)
X, meta = tt.load_overlap(250, 10); y = meta.label.values; G = tt.GpuSeqView(X, dev)
idx = np.sort(np.random.default_rng(0).choice(len(X), 1000, replace=False))
ok(torch.equal(G.batch(idx).cpu(), torch.from_numpy(X[idx])), "GPU gather == CPU gather (1000 random sequences, exact)")
tr = np.sort(np.random.default_rng(1).choice(len(X), 5000, replace=False))
mu, sd = tt.standardise(X, tr); mu_g, sd_g = tt.standardise_gpu(G.batch, tr, dev)
r = max(np.abs(mu - mu_g.cpu().numpy()).max() / np.abs(mu).max(), np.abs(sd - sd_g.cpu().numpy()).max() / np.abs(sd).max())
ok(r < 1e-6, f"standardise on GPU == CPU (max rel diff {r:.1e})")
# short training: old path (materialised numpy array) vs new path (SeqView on GPU), same seed and indices
sub = np.sort(np.random.default_rng(2).choice(len(X), 2500, replace=False)); Xs = X[sub]; ys = y[sub]
Vs = tt.SeqView(X.F, X.starts[sub]); trl, tel = np.arange(2000), np.arange(2000, 2500); tt.EPOCHS = 3
p_old1, _, _ = tt.train_eval(Xs, ys, trl, tel, 7, dev); p_old2, _, _ = tt.train_eval(Xs, ys, trl, tel, 7, dev)
p_new, _, _ = tt.train_eval(Vs, ys, trl, tel, 7, dev)
base, diff = np.abs(p_old1 - p_old2).max(), np.abs(p_old1 - p_new).max()
print(f"     3-epoch training, max |dp|: old vs old (GPU nondeterminism) {base:.2e}; old vs new {diff:.2e}; "
      f"test acc old {((p_old1 > .5) == ys[tel]).mean():.3f} new {((p_new > .5) == ys[tel]).mean():.3f}")
ok(diff <= max(10 * base, 1e-3), "new path reproduces old path within GPU run-to-run variation")
# speed: one epoch at full R1b fold size
tt.EPOCHS = 1; trf = np.arange(int(0.8 * len(X))); tef = np.arange(int(0.8 * len(X)), len(X))
t0 = time.time(); tt.train_eval(X, y, trf, tef, 0, dev); t_new = time.time() - t0
print(f"     one epoch on {len(trf)} sequences + test prediction (new path): {t_new:.1f}s -> 100 epochs ~{100 * t_new / 60:.0f} min per fold (+ per-epoch test curve)")
