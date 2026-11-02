"""Step 2b: attention-based multiple-instance learning over window embeddings (pre-specified).

Bag = all clean 10 s windows of a patient's first session (P3) or of one recording (P4).
Model: gated attention MIL (Ilse et al., 2018): Linear(d,128)-ReLU-Dropout(0.25) -> gated attention
(64 hidden) -> bag vector -> [optional age+sex] -> Linear -> logit.
Training: Adam lr 1e-3, weight decay 1e-2, batch 16 bags, <= 100 epochs, early stopping (patience 15)
on an inner 20% validation split of the training patients, instance subsampling 50-100% per bag per
epoch, 5 seeds averaged. Instance features are standardized with training-fold statistics.
Nothing here was tuned on the outer test folds.
Change log: BENDR window embeddings (time-mean of the encoder output) were dropped from MIL and from
the fusion after they failed the patient-identification sanity check (top-1 0.019 vs 0.66-0.69 for
LaBraM/bandpower): GroupNorm makes their time-average nearly input-independent.
"""
import sys, json, numpy as np, pandas as pd, torch, torch.nn as nn, warnings
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
import evalkit as ek
warnings.filterwarnings("ignore")
D, R = ek.D, "/home/miftah_f/RTX-5090/tuep_fresh/reports"
dev = torch.device("cuda")
SEEDS, MAX_EP, PATIENCE, BATCH = 5, 100, 15, 16

class GatedMIL(nn.Module):
    def __init__(self, d, n_cov=0, h=128, a=64):
        super().__init__()
        self.inst = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.25))
        self.V, self.U, self.w = nn.Linear(h, a), nn.Linear(h, a), nn.Linear(a, 1)
        self.cls = nn.Linear(h + n_cov, 1)
    def forward(self, x, mask, cov=None):                 # x [B, N, d], mask [B, N]
        h = self.inst(x)
        s = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
        s = s.masked_fill(~mask, -1e9); att = torch.softmax(s, 1)
        z = (att.unsqueeze(-1) * h).sum(1)
        if cov is not None:
            z = torch.cat([z, cov], 1)
        return self.cls(z).squeeze(-1), att

def pad(bags, idx, rng=None):
    sel = []
    for i in idx:
        b = bags[i]
        if rng is not None and len(b) > 4:                    # instance subsampling (training only)
            k = max(2, int(len(b) * rng.uniform(0.5, 1.0))); b = b[rng.choice(len(b), k, replace=False)]
        sel.append(b)
    n = max(len(b) for b in sel); d = sel[0].shape[1]
    X = np.zeros((len(sel), n, d), np.float32); M = np.zeros((len(sel), n), bool)
    for j, b in enumerate(sel):
        X[j, :len(b)] = b; M[j, :len(b)] = True
    return torch.from_numpy(X).to(dev), torch.from_numpy(M).to(dev)

def train_predict(bags, y, cov, tr, te, groups=None, seed=0):
    """Train one model on bags[tr] (with inner validation split), return probabilities for bags[te]."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    if groups is None:
        inner = StratifiedShuffleSplit(1, test_size=0.2, random_state=seed)
        itr, iva = next(inner.split(tr, y[tr]))
    else:
        inner = GroupShuffleSplit(1, test_size=0.2, random_state=seed)
        itr, iva = next(inner.split(tr, y[tr], groups[tr]))
    itr, iva = tr[itr], tr[iva]
    allx = np.concatenate([bags[i] for i in itr]); mu, sd = allx.mean(0), allx.std(0) + 1e-6
    B = [(b - mu) / sd for b in bags]
    C = None
    if cov is not None:
        cm, cs = cov[itr].mean(0), cov[itr].std(0) + 1e-6; C = torch.tensor((cov - cm) / cs, dtype=torch.float32, device=dev)
    model = GatedMIL(B[0].shape[1], 0 if cov is None else cov.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
    pw = torch.tensor((y[itr] == 0).sum() / max(1, (y[itr] == 1).sum()), device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    best, best_state, wait = 1e9, None, 0
    for ep in range(MAX_EP):
        model.train(); order = rng.permutation(itr)
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]; X, M = pad(B, b, rng)
            out, _ = model(X, M, None if C is None else C[b]); loss = lossf(out, yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            X, M = pad(B, iva); out, _ = model(X, M, None if C is None else C[iva])
            vl = lossf(out, yt[iva]).item()
        if vl < best - 1e-4:
            best, best_state, wait = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        probs = []
        for s in range(0, len(te), 64):
            b = te[s:s + 64]; X, M = pad(B, b); out, _ = model(X, M, None if C is None else C[b])
            probs.append(torch.sigmoid(out).cpu().numpy())
    return np.concatenate(probs)

def ensemble(bags, y, cov, tr, te, groups=None):
    return np.mean([train_predict(bags, y, cov, tr, te, groups, seed) for seed in range(SEEDS)], axis=0)

if __name__ == "__main__":
    meta = pd.read_csv(f"{D}/features_bandpower_recordings.csv"); meta = meta[meta.n_clean > 0].reset_index(drop=True)
    hdr = pd.read_csv(f"{D}/edf_headers.csv"); meta = meta.merge(hdr[["file", "duration_s"]], on="file", how="left")
    reps = {}
    z = np.load(f"{D}/features_bandpower_windows.npz", allow_pickle=True)
    pos = {h: i for i, h in enumerate(meta.h5_path)}; wr = np.array([pos.get(h, -1) for h in z["h5_path"]])
    reps["bandpower"] = (z["X"][wr >= 0].astype(np.float32), wr[wr >= 0])
    e = np.load(f"{D}/emb_labram_windows.npz", allow_pickle=True); reps["labram"] = (e["E"].astype(np.float32), e["rec"])
    reps["fusion"] = (np.hstack([reps[k][0] for k in ["bandpower", "labram"]]), reps["bandpower"][1])
    which = sys.argv[1] if len(sys.argv) > 1 else "p3"
    T = np.load(f"{D}/p3_patient_table.npz", allow_pickle=True)
    subjects, y, agesex = T["subjects"], T["y"], T["agesex"]
    first = (meta.groupby("subject").session_no.transform("min") == meta.session_no).values
    rows, fold_aucs = [], {}
    if which == "p3":
        print("== MIL, P3: first session, per patient, canonical 10x5 folds, 5-seed ensembles ==", flush=True)
        base = pd.read_csv(f"{R}/step2a_p3.csv")
        for k, (E, rec) in reps.items():
            bags = [E[np.isin(rec, np.where(first & (meta.subject.values == s))[0])] for s in subjects]
            for with_as in [False, True]:
                name = f"MIL {k}" + (" + age+sex" if with_as else "")
                fp = lambda Xtr, ytr, Xte, _b=bags, _c=(agesex if with_as else None): None
                F = ek.folds(); posi = {s: i for i, s in enumerate(subjects)}; reps_m, fa = [], []
                for r in range(10):
                    Fr = F[F.repeat == r]; p = np.full(len(y), np.nan)
                    for f in range(5):
                        te = np.array([posi[s] for s in Fr[Fr.fold == f].subject]); tr = np.setdiff1d(np.arange(len(y)), te)
                        p[te] = ensemble(bags, y, agesex if with_as else None, tr, te)
                        fa.append(ek.roc_auc_score(y[te], p[te]))
                    reps_m.append(dict(repeat=r, **ek.metrics(y, p)))
                m = pd.DataFrame(reps_m); rows.append(ek.summarize(name, m)); fold_aucs[name] = np.array(fa)
        np.savez(f"{D}/mil_p3_foldaucs.npz", **{k.replace(' ', '_').replace('+', 'p'): v for k, v in fold_aucs.items()})
        pd.DataFrame(rows).to_csv(f"{R}/step2b_mil_p3.csv", index=False)
    else:
        test_subj = ek.eegbench_test_subjects(); keep = (meta.duration_s >= 60).values
        subj_r, y_r = meta.subject.values, meta.label.values
        trm, tem = np.where(keep & ~np.isin(subj_r, list(test_subj)))[0], np.where(keep & np.isin(subj_r, list(test_subj)))[0]
        print(f"== MIL, P4: EEG-Bench split, bags = recordings, train {len(trm)} / test {len(tem)} files ==", flush=True)
        for k, (E, rec) in reps.items():
            bags = [E[rec == i] for i in range(len(meta))]
            p = ensemble(bags, y_r, None, trm, tem, groups=subj_r)
            yt, st = y_r[tem], subj_r[tem]; m = ek.metrics(yt, p)
            rng = np.random.default_rng(0); us = np.unique(st); rows_of = {s: np.where(st == s)[0] for s in us}; boots = []
            for _ in range(2000):
                idx = np.concatenate([rows_of[s] for s in rng.choice(us, len(us))])
                if len(np.unique(yt[idx])) == 2:
                    boots.append(ek.balanced_accuracy_score(yt[idx], p[idx] > 0.5))
            lo, hi = np.percentile(boots, [2.5, 97.5]); rows.append(dict(name=f"MIL {k}", **m, BAC_lo=lo, BAC_hi=hi))
            print(f"MIL {k:10s} AUC {m['AUC']:.3f}  BAC {m['BAC']:.3f} (95% CI {lo:.3f}-{hi:.3f})  SENS {m['SENS']:.3f}  SPEC {m['SPEC']:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(f"{R}/step2b_mil_p4.csv", index=False)
