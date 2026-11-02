"""Tang et al. 2024 reproduction: PS(order 2) features -> 60-step Bi-LSTM(2x128) + attention -> FC(2).
Usage: python tang_train.py smoke | r1 <fs> | r2 <fs>
r1 = 5-fold CV over 60-s sequences (paper's protocol); r2 = subject left out (LOOCV, or grouped 10-fold by the rule
fixed in reports/tang_repro_prediction.md). Results per fold in data/tang/<run>/fold<k>.npz (resumable)."""
import os, sys, json, time, numpy as np, pandas as pd, h5py, torch, torch.nn as nn
from sklearn.model_selection import KFold, GroupKFold
D = "/home/miftah_f/RTX-5090/tuep_fresh/data/tang"
SEQ, EPOCHS, BATCH, LR = 60, 100, 64, 0.005
LOSO_MAX_SECONDS = 120          # pre-registered: LOOCV only if one 100-epoch training takes <= 2 min


class TangNet(nn.Module):
    def __init__(self, d_in=506, hidden=128, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, num_layers=layers, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hidden, 2 * hidden)                       # u_t = tanh(W_w h_t + b_w)
        self.u_w = nn.Parameter(torch.randn(2 * hidden) * 0.1)             # context vector u_w
        self.fc = nn.Linear(2 * hidden, 2)

    def forward(self, x):
        h, _ = self.lstm(x)                                                 # (B, T, 2H), h_t = [fh_t, bh_t]
        a = torch.softmax(torch.tanh(self.att(h)) @ self.u_w, dim=1)       # alpha_t
        return self.fc((a.unsqueeze(-1) * h).sum(1))                       # s = sum_t alpha_t h_t


class SeqView:
    """Overlapping 60-step sequences gathered on the fly from per-second features (post hoc R1b)."""
    def __init__(self, F, starts):
        self.F, self.starts, self.ar = F, starts, np.arange(SEQ)
    def __len__(self):
        return len(self.starts)
    def __getitem__(self, idx):
        return self.F[self.starts[idx][:, None] + self.ar]


def load_overlap(fs, stride):
    idx = pd.read_csv(f"{D}/index_{fs}.csv"); F = np.empty((idx.n_sec.sum(), 506), np.float32); starts, rows, off = [], [], 0
    with h5py.File(f"{D}/feat_{fs}.h5") as f:
        for r in idx.itertuples():
            F[off:off + r.n_sec] = f[r.file][:]
            st = np.arange(0, r.n_sec - SEQ + 1, stride) + off if r.n_sec >= SEQ else np.zeros(0, int)
            starts.append(st); rows += [(r.file, r.subj, r.label)] * len(st); off += r.n_sec
    return SeqView(F, np.concatenate(starts)), pd.DataFrame(rows, columns=["file", "subj", "label"])


def load(fs):
    idx = pd.read_csv(f"{D}/index_{fs}.csv"); idx["n_seq"] = idx.n_sec // SEQ
    X = np.empty((idx.n_seq.sum(), SEQ, 506), np.float32); rows = []
    with h5py.File(f"{D}/feat_{fs}.h5") as f:
        k = 0
        for r in idx.itertuples():
            if r.n_seq == 0:
                continue
            F = f[r.file][: r.n_seq * SEQ]
            X[k:k + r.n_seq] = F.reshape(r.n_seq, SEQ, 506); k += r.n_seq
            rows += [(r.file, r.subj, r.label)] * r.n_seq
    meta = pd.DataFrame(rows, columns=["file", "subj", "label"])
    return X, meta


class GpuSeqView:
    """SeqView held on the GPU: sequences are gathered on the device (same values as SeqView.__getitem__)."""
    def __init__(self, sv, dev):
        self.F, self.starts, self.ar, self.dev = torch.from_numpy(sv.F).to(dev), torch.from_numpy(sv.starts).to(dev), torch.arange(SEQ, device=dev), dev
    def batch(self, idx):
        return self.F[self.starts[torch.as_tensor(idx, device=self.dev)][:, None] + self.ar]


def standardise_gpu(get, tr, dev):
    s = torch.zeros(506, dtype=torch.float64, device=dev); ss = torch.zeros_like(s); n = 0
    for i in range(0, len(tr), 2048):
        b = get(tr[i:i + 2048]).reshape(-1, 506).double(); s += b.sum(0); ss += (b ** 2).sum(0); n += len(b)
    mu = s / n; sd = torch.sqrt(torch.clamp(ss / n - mu ** 2, min=1e-12))
    return mu.float(), sd.float()


def standardise(X, tr):
    s = np.zeros(506); ss = np.zeros(506); n = 0
    for i in range(0, len(tr), 2048):
        b = X[tr[i:i + 2048]].reshape(-1, 506).astype(np.float64); s += b.sum(0); ss += (b ** 2).sum(0); n += len(b)
    mu = s / n; sd = np.sqrt(np.maximum(ss / n - mu ** 2, 1e-12))
    return mu.astype(np.float32), sd.astype(np.float32)


def train_eval(X, y, tr, te, seed, dev, curve=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    if isinstance(X, SeqView) and dev.type == "cuda":                     # R1b: features live on the GPU, batches gathered there
        get = GpuSeqView(X, dev).batch; mu_t, sd_t = standardise_gpu(get, tr, dev)
    else:                                                                  # R1/R2: unchanged path
        get = None; mu, sd = standardise(X, tr); mu_t, sd_t = torch.from_numpy(mu).to(dev), torch.from_numpy(sd).to(dev)
    m = TangNet().to(dev); opt = torch.optim.Adam(m.parameters(), lr=LR); lossf = nn.CrossEntropyLoss()
    yt = torch.from_numpy(y).long(); t0 = time.time(); losses = []
    for ep in range(EPOCHS):
        m.train(); order = rng.permutation(tr); tot = 0.0
        for i in range(0, len(order), BATCH):
            b = np.sort(order[i:i + BATCH])
            x = ((torch.from_numpy(X[b]).to(dev) if get is None else get(b)) - mu_t) / sd_t
            loss = lossf(m(x), yt[b].to(dev)); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        losses.append(tot / len(tr))
        if curve is not None:                                              # post hoc R1b only: test accuracy after every epoch
            curve.append(float(((predict(m, X, te, mu_t, sd_t, dev, get) > 0.5) == y[te]).mean())); m.train()
    secs = time.time() - t0
    return predict(m, X, te, mu_t, sd_t, dev, get), secs, losses


def predict(m, X, te, mu_t, sd_t, dev, get=None):
    m.eval(); p = []
    with torch.no_grad():
        for i in range(0, len(te), 512):
            x = ((torch.from_numpy(X[te[i:i + 512]]).to(dev) if get is None else get(te[i:i + 512])) - mu_t) / sd_t
            p.append(torch.softmax(m(x), 1)[:, 1].cpu().numpy())
    return np.concatenate(p)


def run(proto, fs):
    dev = torch.device(os.environ.get("TANG_DEVICE", "cuda")); torch.set_num_threads(8)
    X, meta = load_overlap(fs, 10) if proto == "r1b" else load(fs); y = meta.label.values; out = f"{D}/{proto}_{fs}"; os.makedirs(out, exist_ok=True)
    if proto in ("r1", "r1b"):
        folds = list(KFold(5, shuffle=True, random_state=0).split(np.arange(len(X))))
    else:
        subj = meta.subj.values; us = np.unique(subj)
        if not os.path.exists(f"{out}/probe.json"):                        # GPU timing probe = first LOOCV fold, timing only
            tr0, te0 = np.where(subj != us[0])[0], np.where(subj == us[0])[0]
            _, t, _ = train_eval(X, y, tr0, te0, 999, dev); json.dump(dict(train_seconds=t, device=str(dev)), open(f"{out}/probe.json", "w"))
        t = json.load(open(f"{out}/probe.json"))["train_seconds"]
        if t <= LOSO_MAX_SECONDS:
            folds = [(np.where(subj != s)[0], np.where(subj == s)[0]) for s in us]; mode = "LOOCV"
        else:
            folds = list(GroupKFold(10).split(X, y, subj)); mode = "grouped 10-fold"
        json.dump(dict(mode=mode, probe_train_seconds=t, n_folds=len(folds)), open(f"{out}/mode.json", "w"))
    print(f"{proto} fs {fs}: {len(X)} sequences, {meta.subj.nunique()} subjects, {len(folds)} folds", flush=True)
    for k, (tr, te) in enumerate(folds):
        if os.path.exists(f"{out}/fold{k}.json"):
            continue
        overlap = len(set(meta.subj.values[tr]) & set(meta.subj.values[te]))
        curve = [] if proto == "r1b" else None
        p, secs, losses = train_eval(X, y, tr, te, 1000 * (proto == "r2") + k, dev, curve)
        np.savez(f"{out}/fold{k}.npz", te=te, p=p)
        json.dump(dict(train_seconds=secs, subject_overlap=overlap, n_train=len(tr), n_test=len(te), final_loss=losses[-1], loss_curve=losses[::10], test_acc_curve=curve), open(f"{out}/fold{k}.json", "w"))
        print(f"  fold {k}: {secs:.0f}s, train loss {losses[0]:.3f} -> {losses[-1]:.4f}, subjects shared train/test {overlap}, test acc {((p > 0.5) == y[te]).mean():.3f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "smoke":                                             # CPU only: shapes, parameter count, a few steps
        m = TangNet(); n_lstm = sum(p.numel() for p in m.lstm.parameters()); n_all = sum(p.numel() for p in m.parameters())
        print(f"LSTM params {n_lstm} (expected 1,046,528 -> {n_lstm * 4 / 1e6:.2f} MB float32), all params {n_all}")
        macs = sum(p.numel() for n_, p in m.lstm.named_parameters() if "weight" in n_) * SEQ
        print(f"LSTM multiply-accumulates for one 60-step sequence: {macs / 1e9:.4f} G (paper: 0.063G)")
        X, meta = load(250); print("250 Hz sequences", X.shape, "labels", meta.label.value_counts().to_dict(), "subjects", meta.subj.nunique())
        idx = np.arange(0, len(X), 40); y = meta.label.values
        EPOCHS = 3                                                         # module-level: train_eval reads it at call time
        p, secs, losses = train_eval(X, y, idx[: len(idx) // 2], idx[len(idx) // 2:], 0, torch.device("cpu"))
        print(f"smoke: 3 epochs on {len(idx) // 2} sequences in {secs:.0f}s, losses {np.round(losses, 4)}, probs range {p.min():.3f}-{p.max():.3f}")
    else:
        run(sys.argv[1], int(sys.argv[2]))
