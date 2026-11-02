"""End-to-end CNNs (braindecode ShallowFBCSPNet / Deep4Net) as pre-specified in reports/cnn_prediction.md.
Usage: python cnn_train.py p3 shallow|deep4   |   python cnn_train.py p4 shallow|deep4   |   python cnn_train.py smoke
One GPU process at a time; per-fold results are saved so a run can resume."""
import os, sys, time, json, numpy as np, pandas as pd, torch, torch.nn as nn, warnings
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")
from braindecode.models import ShallowFBCSPNet, Deep4Net
import evalkit as ek
D, R = ek.D, "/home/miftah_f/RTX-5090/tuep_fresh/reports"
dev = torch.device("cuda")
CFG = {"shallow": dict(cls=ShallowFBCSPNet, lr=6.25e-4, wd=0.0), "deep4": dict(cls=Deep4Net, lr=1e-3, wd=5e-4)}
EPOCHS, BATCH = 35, 64

def load():
    W = np.load(f"{D}/cnn_windows100.npy"); wrec = np.load(f"{D}/cnn_windows100_rec.npy")
    meta = pd.read_csv(f"{D}/features_bandpower_recordings.csv"); meta = meta[meta.n_clean > 0].reset_index(drop=True)
    hdr = pd.read_csv(f"{D}/edf_headers.csv"); meta = meta.merge(hdr[["file", "duration_s"]], on="file", how="left")
    meta["first"] = meta.groupby("subject").session_no.transform("min") == meta.session_no
    return W, wrec, meta

def train(arch, W, idx, y_win, seed, epochs=EPOCHS):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    c = CFG[arch]; m = c["cls"](n_chans=19, n_outputs=2, n_times=1000, final_conv_length="auto").to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=c["lr"], weight_decay=c["wd"])
    steps = epochs * int(np.ceil(len(idx) / BATCH)); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    n1 = y_win.sum(); n0 = len(y_win) - n1
    lossf = nn.CrossEntropyLoss(weight=torch.tensor([len(y_win) / (2 * n0), len(y_win) / (2 * n1)], dtype=torch.float32, device=dev))
    yt = torch.from_numpy(y_win.astype(np.int64))
    m.train()
    for ep in range(epochs):
        order = rng.permutation(len(idx))
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            x = torch.from_numpy(W[idx[b]].astype(np.float32)).to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = m(x)
            loss = lossf(out.float(), yt[b].to(dev))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    return m.eval()

@torch.no_grad()
def predict(m, W, idx):
    out = []
    for s in range(0, len(idx), 512):
        x = torch.from_numpy(W[idx[s:s + 512]].astype(np.float32)).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.softmax(m(x).float(), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)

def run_p3(arch):
    W, wrec, meta = load(); F = ek.folds()
    subjects = np.array(sorted(F.subject.unique())); y = meta.groupby("subject").label.first().loc[subjects].values
    win_subj = meta.subject.values[wrec]; win_first = meta["first"].values[wrec]; win_lab = meta.label.values[wrec]
    path = f"{D}/cnn_p3_{arch}.json"; done = json.load(open(path)) if os.path.exists(path) else {}
    for r in range(10):
        Fr = F[F.repeat == r]
        for k in range(5):
            key = f"{r}-{k}"
            if key in done:
                continue
            t0 = time.time(); te_p = set(Fr[Fr.fold == k].subject)
            tr_idx = np.where(win_first & ~np.isin(win_subj, list(te_p)))[0]; te_idx = np.where(win_first & np.isin(win_subj, list(te_p)))[0]
            m = train(arch, W, tr_idx, win_lab[tr_idx], seed=1000 + 10 * r + k)
            p = pd.Series(predict(m, W, te_idx)).groupby(win_subj[te_idx]).mean()
            done[key] = p.to_dict(); json.dump(done, open(path, "w"))
            print(f"[{arch}] repeat {r} fold {k}: {len(tr_idx)} train windows, fold AUC {roc_auc_score([y[list(subjects).index(s)] for s in p.index], p.values):.3f}, {time.time() - t0:.0f}s", flush=True)
            del m; torch.cuda.empty_cache()
    print(f"[{arch}] P3 done")

def run_p4(arch, n_seeds=5):
    W, wrec, meta = load()
    test_subj = ek.eegbench_test_subjects(); keep = (meta.duration_s >= 60).values
    subj_r, y_r = meta.subject.values, meta.label.values
    tr_rec = np.where(keep & ~np.isin(subj_r, list(test_subj)))[0]; te_rec = np.where(keep & np.isin(subj_r, list(test_subj)))[0]
    win_lab = y_r[wrec]
    path = f"{D}/cnn_p4_{arch}.json"; done = json.load(open(path)) if os.path.exists(path) else {}
    def rec_scores(m, recs):
        idx = np.where(np.isin(wrec, recs))[0]; p = predict(m, W, idx)
        return pd.Series(p).groupby(wrec[idx]).mean()
    # inner 5-fold grouped CV on training patients -> out-of-fold recording scores for the threshold
    for f, (itr, ite) in enumerate(GroupKFold(5).split(tr_rec, y_r[tr_rec], subj_r[tr_rec])):
        key = f"inner{f}"
        if key in done:
            continue
        t0 = time.time(); idx = np.where(np.isin(wrec, tr_rec[itr]))[0]
        m = train(arch, W, idx, win_lab[idx], seed=500 + f)
        done[key] = {int(k): float(v) for k, v in rec_scores(m, tr_rec[ite]).items()}; json.dump(done, open(path, "w"))
        print(f"[{arch}] P4 inner fold {f}: {len(idx)} windows, {time.time() - t0:.0f}s", flush=True); del m; torch.cuda.empty_cache()
    for s in range(n_seeds):
        key = f"seed{s}"
        if key in done:
            continue
        t0 = time.time(); idx = np.where(np.isin(wrec, tr_rec))[0]
        m = train(arch, W, idx, win_lab[idx], seed=s)
        done[key] = {int(k): float(v) for k, v in rec_scores(m, te_rec).items()}; json.dump(done, open(path, "w"))
        print(f"[{arch}] P4 seed {s}: {len(idx)} windows, {time.time() - t0:.0f}s", flush=True); del m; torch.cuda.empty_cache()
    print(f"[{arch}] P4 done")

if __name__ == "__main__":
    what = sys.argv[1]
    if what == "smoke":
        W, wrec, meta = load(); idx = np.where(meta["first"].values[wrec])[0][:4000]
        t0 = time.time(); m = train("shallow", W, idx, meta.label.values[wrec][idx], 0, epochs=1)
        print(f"shallow 1 epoch on {len(idx)} windows: {time.time() - t0:.1f}s"); p = predict(m, W, idx[:1000]); print("pred range", p.min(), p.max())
        t0 = time.time(); m = train("deep4", W, idx, meta.label.values[wrec][idx], 0, epochs=1)
        print(f"deep4 1 epoch on {len(idx)} windows: {time.time() - t0:.1f}s")
    elif what == "p3":
        run_p3(sys.argv[2])
    elif what == "p4":
        run_p4(sys.argv[2])
