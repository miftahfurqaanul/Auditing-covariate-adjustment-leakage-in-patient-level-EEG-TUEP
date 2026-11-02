"""Fine-tune CBraMod with its official binary recipe (reports/cbramod_ft_prediction.md).
Usage: python cbramod_finetune.py smoke | p4 <seed> | p3 <repeat> <fold>
Per-run results are written to data/cbramod_ft/<run>.json so the queue can resume."""
import os, sys, json, time, numpy as np, pandas as pd, h5py, torch, torch.nn as nn, warnings
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/miftah_f/RTX-5090/CBraMod")
from models.cbramod import CBraMod
import evalkit as ek, threshold_select as ts
D = ek.D; OUT = f"{D}/cbramod_ft"; os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda")
CH = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ", "C4", "T4", "T5", "P3", "PZ", "P4", "T6", "O1", "O2"]
ORDER = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]
PERM = [CH.index(c) for c in ORDER]
WIN, EDGE, PTP = 2000, 2000, 800.0
EPOCHS, BATCH, LR_BB, LR_HEAD, WD = 50, 64, 1e-4, 1e-3 * (64 / 256) ** 0.5, 5e-2

def rec_windows(h5):
    with h5py.File(f"{D}/{h5}") as f:
        x = f["eeg"][()].astype(np.float32)
    x = x[:, EDGE:x.shape[1] - EDGE]; n = x.shape[1] // WIN
    if n == 0:
        return np.zeros((0, 19, 10, 200), np.float16)
    w = x[:, :n * WIN].reshape(19, n, WIN).transpose(1, 0, 2)
    w = w[(w.max(2) - w.min(2)).max(1) <= PTP][:, PERM] / 100.0
    return w.reshape(len(w), 19, 10, 200).astype(np.float16)

def meta_table():
    meta = pd.read_csv(f"{D}/features_bandpower_recordings.csv"); meta = meta[meta.n_clean > 0].reset_index(drop=True)
    hdr = pd.read_csv(f"{D}/edf_headers.csv"); meta = meta.merge(hdr[["file", "duration_s"]], on="file", how="left")
    meta["first"] = meta.groupby("subject").session_no.transform("min") == meta.session_no
    return meta

def load_windows(meta, recs):
    with Pool(12) as pool:
        ws = pool.map(rec_windows, meta.h5_path.values[recs], chunksize=4)
    wrec = np.concatenate([np.full(len(w), r, np.int32) for r, w in zip(recs, ws)])
    return np.concatenate(ws), wrec

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
        self.backbone.load_state_dict(torch.load("/home/miftah_f/RTX-5090/CBraMod/pretrained_weights/pretrained_weights.pth", map_location="cpu"), strict=True)
        self.backbone.proj_out = nn.Identity()
        self.classifier = nn.Linear(200, 1)                               # avgpooling_patch_reps head
    def forward(self, x):
        return self.classifier(self.backbone(x).mean((1, 2))).squeeze(-1)

@torch.no_grad()
def predict(m, W, idx):
    m.eval(); out = []
    for s in range(0, len(idx), 256):
        x = torch.from_numpy(W[idx[s:s + 256]].astype(np.float32)).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(m(x).float()).cpu().numpy())
    return np.concatenate(out)

def unit_scores(p, units):
    return pd.Series(p).groupby(units).mean()

def finetune(W, tr_idx, y_win, va_idx, va_units, va_y_unit, seed, log):
    """Official recipe; returns the model state at the epoch with the best validation AUROC (unit level)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = Model().to(dev)
    bb = [p for n, p in m.named_parameters() if n.startswith("backbone.")]; hd = [p for n, p in m.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": LR_BB}, {"params": hd, "lr": LR_HEAD}], weight_decay=WD)
    steps_per_epoch = int(np.ceil(len(tr_idx) / BATCH))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps_per_epoch, eta_min=1e-6)
    n1 = y_win[tr_idx].sum(); n0 = len(tr_idx) - n1
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n0 / max(n1, 1), device=dev))
    yt = torch.from_numpy(y_win.astype(np.float32))
    best_auc, best_state, best_ep = -1, None, -1
    for ep in range(EPOCHS):
        m.train(); t0 = time.time(); order = rng.permutation(tr_idx)
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            x = torch.from_numpy(W[b].astype(np.float32)).to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = m(x)
            loss = lossf(out.float(), yt[b].to(dev))
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        sc = unit_scores(predict(m, W, va_idx), va_units)
        auc = roc_auc_score(va_y_unit.loc[sc.index].values, sc.values)
        if auc > best_auc:
            best_auc, best_ep = auc, ep; best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        log(f"  epoch {ep + 1:2d}: val AUROC {auc:.3f} (best {best_auc:.3f} @ {best_ep + 1}), {time.time() - t0:.0f}s")
    m.load_state_dict(best_state)
    return m, best_auc, best_ep + 1

def log_to(path):
    def log(msg):
        print(msg, flush=True)
        with open(path, "a") as f:
            f.write(msg + "\n")
    return log

def run_p4(seed):
    out_path = f"{OUT}/p4_seed{seed}.json"
    if os.path.exists(out_path):
        return
    meta = meta_table(); test_subj = ek.eegbench_test_subjects(); keep = (meta.duration_s >= 60).values
    subj, yr = meta.subject.values, meta.label.values
    tr_rec = np.where(keep & ~np.isin(subj, list(test_subj)))[0]; te_rec = np.where(keep & np.isin(subj, list(test_subj)))[0]
    itr, iva = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(tr_rec, yr[tr_rec], subj[tr_rec]))
    W, wrec = load_windows(meta, np.concatenate([tr_rec, te_rec])); y_win = yr[wrec]
    log = log_to(f"{OUT}/p4_seed{seed}.log"); log(f"P4 seed {seed}: windows {len(W)}")
    tr_idx = np.where(np.isin(wrec, tr_rec[itr]))[0]; va_idx = np.where(np.isin(wrec, tr_rec[iva]))[0]; te_idx = np.where(np.isin(wrec, te_rec))[0]
    m, vauc, vep = finetune(W, tr_idx, y_win, va_idx, wrec[va_idx], pd.Series(yr), seed, log)
    va = unit_scores(predict(m, W, va_idx), wrec[va_idx]); thr, vbac = ts.best_threshold(yr[va.index.values], va.values)
    te = unit_scores(predict(m, W, te_idx), wrec[te_idx])
    json.dump(dict(val_auc=vauc, best_epoch=vep, val_threshold=float(thr), val_bac=float(vbac), test={int(k): float(v) for k, v in te.items()}), open(out_path, "w"))
    log(f"P4 seed {seed} done: best epoch {vep}, val AUROC {vauc:.3f}, val threshold {thr:.3f}")

def run_p3(r, k):
    out_path = f"{OUT}/p3_r{r}_f{k}.json"
    if os.path.exists(out_path):
        return
    meta = meta_table(); F = ek.folds(); Fr = F[(F.repeat == r) & (F.fold == k)]
    te_p = set(Fr.subject); subj, yr = meta.subject.values, meta.label.values
    recs = np.where(meta["first"].values)[0]
    tr_rec = recs[~np.isin(subj[recs], list(te_p))]; te_rec = recs[np.isin(subj[recs], list(te_p))]
    itr, iva = next(GroupShuffleSplit(1, test_size=0.2, random_state=100 * r + k).split(tr_rec, yr[tr_rec], subj[tr_rec]))
    W, wrec = load_windows(meta, recs); y_win = yr[wrec]; wsub = subj[wrec]
    log = log_to(f"{OUT}/p3_r{r}_f{k}.log"); log(f"P3 repeat {r} fold {k}: windows {len(W)}")
    tr_idx = np.where(np.isin(wrec, tr_rec[itr]))[0]; va_idx = np.where(np.isin(wrec, tr_rec[iva]))[0]; te_idx = np.where(np.isin(wrec, te_rec))[0]
    ypat = meta.groupby("subject").label.first()
    m, vauc, vep = finetune(W, tr_idx, y_win, va_idx, wsub[va_idx], ypat, 1000 + 10 * r + k, log)
    va = unit_scores(predict(m, W, va_idx), wsub[va_idx]); thr, _ = ts.best_threshold(ypat.loc[va.index].values, va.values)
    te = unit_scores(predict(m, W, te_idx), wsub[te_idx])
    json.dump(dict(val_auc=vauc, best_epoch=vep, val_threshold=float(thr), test=te.to_dict()), open(out_path, "w"))
    log(f"P3 r{r} f{k} done: best epoch {vep}, val AUROC {vauc:.3f}, test AUC {roc_auc_score(ypat.loc[te.index].values, te.values):.3f}")

if __name__ == "__main__":
    if sys.argv[1] == "smoke":
        meta = meta_table(); recs = np.where(meta["first"].values)[0][:40]
        W, wrec = load_windows(meta, recs); idx = np.arange(min(2000, len(W)))
        m = Model().to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-4); t0 = time.time()
        for s in range(0, len(idx), BATCH):
            x = torch.from_numpy(W[idx[s:s + BATCH]].astype(np.float32)).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = m(x).float().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize(); dt = time.time() - t0
        print(f"train {len(idx)} windows: {dt:.1f}s -> {len(idx) / dt:.0f} windows/s; GPU mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    elif sys.argv[1] == "p4":
        run_p4(int(sys.argv[2]))
    elif sys.argv[1] == "p3":
        run_p3(int(sys.argv[2]), int(sys.argv[3]))
