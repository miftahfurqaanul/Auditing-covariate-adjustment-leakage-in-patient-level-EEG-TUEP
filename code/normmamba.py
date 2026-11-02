"""Round 16 (reports/round16_normmamba_prediction.md, commit e2b8b14): NormMamba.
Bidirectional Mamba (pure PyTorch selective scan) over 4-s segment tokens; tokens are control-referenced normative deviations
(I2 design fitted on the segments of NON-epilepsy training patients only, inside each fold); fusion with the patient normative summary.
Variants: F (fusion, main), S (sequence only), raw (no normative layer). Patient-level labels; canonical 10x5 folds.
Usage: python normmamba.py --test | --dry-run | --run"""
import os, sys, json, time, argparse, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from sklearn.metrics import roc_auc_score
import evalkit as ek, round4 as r4, tang_fixed as tf
D = "/home/miftah_f/RTX-5090/tuep_fresh/data"; OUT = f"{D}/round16"; os.makedirs(OUT, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Z = np.load(f"{D}/seg4s_features.npz", allow_pickle=True); assert list(Z["subjects"]) == list(r4.SUBJ)
XSEG, PID = Z["X"].astype(np.float64), Z["pid"]; Y, AS, DES = r4.Y, r4.AGESEX, r4.DESIGN["I2"]; N = len(Y)
POS = {s: i for i, s in enumerate(r4.SUBJ)}; F = ek.folds()
JOBS = [(r, k, np.setdiff1d(np.arange(N), te), te) for r in range(10) for k in range(5)
        for te in [np.array([POS[s] for s in F[(F.repeat == r) & (F.fold == k)].subject])]]
GRID = [(32, 32), (32, 96), (64, 32), (64, 96)]                 # (d_model, crop length L)
SEEDS = {"F": 5, "S": 3, "raw": 3}; MAX_EPOCHS, PATIENCE, CROPS, BATCH = 60, 10, 8, 32


# ------------------------------------------------------------------ tokens (inside each fold, no labels except "control" selection on training patients)
def make_tokens(tr, variant):
    """Returns per-patient token arrays (list of (n_seg, 114)) and summary (N, 116)."""
    if variant == "raw":
        tm = np.isin(PID, tr); mu, sd = XSEG[tm].mean(0), XSEG[tm].std(0)
        T = (XSEG - mu) / sd
    else:
        c = tr[Y[tr] == 0]; cm = np.isin(PID, c); Dseg = DES[PID]
        B = np.linalg.lstsq(Dseg[cm], XSEG[cm], rcond=None)[0]; sd = (XSEG[cm] - Dseg[cm] @ B).std(0, ddof=np.linalg.matrix_rank(Dseg[cm]))
        T = (XSEG - Dseg @ B) / sd
    toks = [T[PID == i].astype(np.float32) for i in range(N)]
    summ = np.column_stack([np.vstack([np.median(t, 0) for t in toks]), (AS - AS[tr].mean(0)) / AS[tr].std(0)]).astype(np.float32)
    return toks, summ, T


# ------------------------------------------------------------------ model
class Mamba(nn.Module):
    """Single-direction Mamba block (Gu & Dao 2023): in_proj -> causal depthwise conv -> selective SSM -> gate -> out_proj."""
    def __init__(self, d, d_state=16, expand=2, d_conv=4):
        super().__init__(); di = expand * d; self.di, self.ds = di, d_state; self.dt_rank = math.ceil(d / 16)
        self.in_proj = nn.Linear(d, 2 * di); self.conv = nn.Conv1d(di, di, d_conv, groups=di, padding=d_conv - 1)
        self.x_proj = nn.Linear(di, self.dt_rank + 2 * d_state, bias=False); self.dt_proj = nn.Linear(self.dt_rank, di)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(di, 1)); self.Dp = nn.Parameter(torch.ones(di))
        self.out_proj = nn.Linear(di, d)
        # dt initialisation as in the reference Mamba implementation: dt ~ log-uniform[1e-3, 1e-1], bias = softplus^-1(dt)
        with torch.no_grad():
            nn.init.uniform_(self.dt_proj.weight, -self.dt_rank ** -0.5, self.dt_rank ** -0.5)
            dt = torch.exp(torch.rand(di) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)).clamp(min=1e-4)
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x):                                              # x: (B, L, d)
        Bsz, L, _ = x.shape; xz = self.in_proj(x); u, z = xz.chunk(2, -1)
        u = Fn.silu(self.conv(u.transpose(1, 2))[..., :L].transpose(1, 2))
        dt, Bm, Cm = torch.split(self.x_proj(u), [self.dt_rank, self.ds, self.ds], -1); dt = Fn.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log); h = torch.zeros(Bsz, self.di, self.ds, device=x.device); ys = []
        dA = torch.exp(dt.unsqueeze(-1) * A); dBu = dt.unsqueeze(-1) * Bm.unsqueeze(2) * u.unsqueeze(-1)
        for t in range(L):                                             # selective scan (sequential; L <= 96)
            h = dA[:, t] * h + dBu[:, t]; ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, 1) + u * self.Dp
        return self.out_proj(y * Fn.silu(z))


class BiMambaLayer(nn.Module):
    """EEGMamba-style bidirectional layer: forward and backward SSM outputs summed; pre-norm residual."""
    def __init__(self, d):
        super().__init__(); self.norm = nn.LayerNorm(d); self.fwd, self.bwd = Mamba(d), Mamba(d)

    def forward(self, x):
        h = self.norm(x); return x + self.fwd(h) + self.bwd(h.flip(1)).flip(1)


class NormMamba(nn.Module):
    def __init__(self, d, fusion=True, n_in=114, n_sum=116, layers=2, p=0.2):
        super().__init__(); self.fusion = fusion
        self.tok = nn.Sequential(nn.Linear(n_in, d), nn.LayerNorm(d), nn.Dropout(p))
        self.layers = nn.ModuleList([BiMambaLayer(d) for _ in range(layers)]); self.norm = nn.LayerNorm(d); self.att = nn.Linear(d, 1)
        self.summ = nn.Sequential(nn.Linear(n_sum, d), nn.GELU(), nn.Dropout(p)) if fusion else None
        self.head = nn.Sequential(nn.Linear(2 * d if fusion else d, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, 1))

    def forward(self, tok, summ):
        h = self.tok(tok)
        for l in self.layers:
            h = l(h)
        h = self.norm(h); w = torch.softmax(self.att(h), 1); v = (w * h).sum(1)
        if self.fusion:
            v = torch.cat([v, self.summ(summ)], -1)
        return self.head(v).squeeze(-1)


# ------------------------------------------------------------------ training
def crops_eval(t, L):
    n = len(t); st = list(range(0, max(n - L, 0) + 1, max(L // 2, 1)))
    if st[-1] != max(n - L, 0):
        st.append(max(n - L, 0))
    return np.stack([t[s:s + L] for s in st])


@torch.no_grad()
def predict(model, toks, summ, idx, L):
    model.eval(); out = []
    for i in idx:
        c = torch.tensor(crops_eval(toks[i], L), device=DEV); s = torch.tensor(summ[i], device=DEV).expand(len(c), -1)
        out.append(torch.sigmoid(model(c, s)).mean().item())
    return np.array(out)


def train_one(toks, summ, tr_i, va_i, d, L, seed, fusion, max_epochs=MAX_EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    model = NormMamba(d, fusion=fusion).to(DEV); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    best, best_state, wait, ep_used = -1, None, 0, 0
    for ep in range(max_epochs):
        model.train(); items = np.repeat(tr_i, CROPS); rng.shuffle(items)
        for b in range(0, len(items), BATCH):
            ids = items[b:b + BATCH]; st = [rng.integers(0, len(toks[i]) - L + 1) for i in ids]
            x = torch.tensor(np.stack([toks[i][s:s + L] for i, s in zip(ids, st)]), device=DEV)
            x = x * (torch.rand(x.shape[:2], device=DEV) > 0.1).unsqueeze(-1).float() + 0.1 * torch.randn_like(x)   # token dropout + noise
            loss = Fn.binary_cross_entropy_with_logits(model(x, torch.tensor(summ[ids], device=DEV)), torch.tensor(Y[ids], dtype=torch.float32, device=DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        auc = roc_auc_score(Y[va_i], predict(model, toks, summ, va_i, L)); ep_used = ep + 1
        if auc > best:
            best, wait, best_state = auc, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state); return model, best, ep_used


def run_fold(r, k, tr, te, max_epochs=MAX_EPOCHS, variants=("F", "S", "raw")):
    ylab = {s: int(Y[POS[s]]) for s in r4.SUBJ}; tr_p, va_p = tf.split_val(list(r4.SUBJ[tr]), ylab, 100 * r + k)
    tr_i, va_i = np.array([POS[s] for s in tr_p]), np.array([POS[s] for s in va_p])
    assert not (set(tr_i) | set(va_i)) & set(te) and set(tr_i) | set(va_i) == set(tr)
    res = {"r": r, "k": k}
    for var in variants:
        toks, summ, _ = make_tokens(tr, "raw" if var == "raw" else "norm"); fusion = var != "S"; t0 = time.time()
        g = {f"{d}_{L}": train_one(toks, summ, tr_i, va_i, d, L, 1000 * r + 10 * k, fusion, max_epochs)[1] for d, L in GRID}
        bd, bL = map(int, max(g, key=g.get).split("_")); preds, eps = [], []
        for s in range(SEEDS[var]):
            m, va, ep = train_one(toks, summ, tr_i, va_i, bd, bL, 7919 * (s + 1) + 1000 * r + 10 * k, fusion, max_epochs); preds.append(predict(m, toks, summ, te, bL)); eps.append(ep)
        res[var] = dict(grid_val_auc=g, chosen=[bd, bL], epochs=eps, test={r4.SUBJ[i]: float(p) for i, p in zip(te, np.mean(preds, 0))},
                        seconds=time.time() - t0, test_auc=float(roc_auc_score(Y[te], np.mean(preds, 0))))
    return res


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--test", action="store_true"); g.add_argument("--dry-run", action="store_true"); g.add_argument("--run", action="store_true"); a = ap.parse_args()
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    print(f"device {DEV} ({torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'cpu'}); segments {len(XSEG)}; patients {N}", flush=True)
    if a.test:
        tr0 = JOBS[0][2]; toks, summ, T = make_tokens(tr0, "norm"); c = tr0[Y[tr0] == 0]
        print(("PASS" if np.abs(T[np.isin(PID, c)].mean(0)).max() < 1e-8 else "FAIL") + f" normative z of training-control segments has mean 0 (max {np.abs(T[np.isin(PID, c)].mean(0)).max():.1e})")
        _, _, Tr = make_tokens(tr0, "raw"); print(("PASS" if np.abs(Tr[np.isin(PID, tr0)].mean(0)).max() < 1e-8 else "FAIL") + " raw tokens standardised on training segments only")
        torch.manual_seed(0); m = NormMamba(32).to(DEV); x = torch.randn(4, 50, 114, device=DEV); s = torch.randn(4, 116, device=DEV); o = m(x, s)
        print(("PASS" if o.shape == (4,) else "FAIL") + f" output shape {tuple(o.shape)}"); o.sum().backward()
        print(("PASS" if all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad) else "FAIL") + " gradients flow to all parameters and are finite")
        blk = Mamba(32).to(DEV).eval(); xx = torch.randn(1, 40, 32, device=DEV); xx2 = xx.clone(); xx2[:, -1] += 5 * torch.randn(32, device=DEV)   # non-constant: LayerNorm removes a constant shift
        print(("PASS" if torch.allclose(blk(xx)[:, :39], blk(xx2)[:, :39], atol=1e-6) else "FAIL") + " forward Mamba is causal (changing the last token leaves earlier outputs unchanged)")
        bi = BiMambaLayer(32).to(DEV).eval()
        print(("PASS" if not torch.allclose(bi(xx)[:, 0], bi(xx2)[:, 0], atol=1e-6) else "FAIL") + " bidirectional layer: first output depends on the last token (backward path works)")
        m.eval(); torch.manual_seed(1); o1 = m(x, s); torch.manual_seed(1); o2 = m(x, s)
        print(("PASS" if torch.equal(o1, o2) else "FAIL") + " deterministic forward in eval mode")
        ylab = {s_: int(Y[POS[s_]]) for s_ in r4.SUBJ}
        ok = all(not (set(tf.split_val(list(r4.SUBJ[tr]), ylab, 100 * r + k)[0]) | set(tf.split_val(list(r4.SUBJ[tr]), ylab, 100 * r + k)[1])) & set(r4.SUBJ[te]) for r, k, tr, te in JOBS)
        print(("PASS" if ok else "FAIL") + " in all 50 folds, test patients never appear in training or validation")
        print("params (d=32 / d=64):", sum(p.numel() for p in NormMamba(32).parameters()), "/", sum(p.numel() for p in NormMamba(64).parameters()))
        sys.exit()
    if a.dry_run:
        r, k, tr, te = JOBS[0]; t0 = time.time(); toks, summ, _ = make_tokens(tr, "norm")
        ylab = {s: int(Y[POS[s]]) for s in r4.SUBJ}; tr_p, va_p = tf.split_val(list(r4.SUBJ[tr]), ylab, 0); tr_i, va_i = np.array([POS[s] for s in tr_p]), np.array([POS[s] for s in va_p])
        for d, L in GRID:
            t1 = time.time(); m, va, ep = train_one(toks, summ, tr_i, va_i, d, L, 0, True, max_epochs=3)
            print(f"   d={d} L={L}: 3 epochs in {time.time() - t1:.1f} s (val AUC {va:.3f}) -> {(time.time() - t1) / 3:.1f} s/epoch", flush=True)
        sys.exit()
    for r, k, tr, te in JOBS:
        path = f"{OUT}/r{r}_k{k}.json"
        if os.path.exists(path):
            continue
        t0 = time.time(); res = run_fold(r, k, tr, te); json.dump(res, open(path, "w"))
        print(f"fold r{r}k{k} done in {(time.time() - t0) / 60:.1f} min: test AUC F {res['F']['test_auc']:.3f} S {res['S']['test_auc']:.3f} raw {res['raw']['test_auc']:.3f}; "
              f"chosen F {res['F']['chosen']}", flush=True)
    print("ALL FOLDS DONE", flush=True)
