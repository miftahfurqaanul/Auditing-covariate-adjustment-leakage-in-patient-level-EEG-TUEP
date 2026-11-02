"""Round 25 (reports/round25_prediction.md, commit ec52215): fine-tuned foundation models on the 50 canonical folds, strict P3 protocol.
CBraMod: the 2026-10-07 recipe, reusing cbramod_finetune.Model / finetune / predict unchanged.
LaBraM: official run_class_finetuning.py recipe (README TUAB command, commit c431221); get_parameter_groups, LayerDecayValueAssigner and
cosine_scheduler copied verbatim below because optim_factory/utils cannot be imported here (need pyhealth and an older timm).
Deviations for LaBraM (pre-registered): pos_weight in the loss and epoch selection by patient-level validation AUROC, as for CBraMod.
GPU, one process. Does NOT import round17_common/round10 (they hide the GPU).
Usage: python round25_fm.py --cache | --dry-run | --fold <model> <r> <k>     (model: cbramod | labram)"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys, re, ast, json, math, time, argparse, numpy as np, pandas as pd, h5py, torch, torch.nn as nn, warnings
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
warnings.filterwarnings("ignore")
torch.use_deterministic_algorithms(True, warn_only=True); torch.backends.cudnn.benchmark = False
import evalkit as ek, features as fe, threshold_select as ts
import cbramod_finetune as cb

D = ek.D; OUT = f"{D}/round25"; os.makedirs(OUT, exist_ok=True); CACHE = f"{OUT}/windows_first_session.npy"; CMETA = f"{OUT}/windows_first_session_meta.npz"
LABRAM_DIR = "/home/miftah_f/RTX-5090/LaBraM"; dev = torch.device("cuda")
CH = fe.CH                                                                     # cache channel order (features.py); CBraMod re-ordered with cb.PERM
EPOCHS, BATCH = 50, 64
LB = dict(lr=5e-4, min_lr=1e-6, warmup_epochs=5, weight_decay=0.05, layer_decay=0.65, drop_path=0.1, opt_eps=1e-8)


# ------------------------------------------------------------------ windows (label-free cache, first session only)
def rec_windows(h5):
    with h5py.File(f"{fe.D}/{h5}") as f:
        x = f["eeg"][()].astype(np.float32)
    x = x[:, fe.EDGE:x.shape[1] - fe.EDGE]; n = x.shape[1] // fe.WIN
    if n == 0:
        return np.zeros((0, 19, fe.WIN), np.float16)
    w = x[:, :n * fe.WIN].reshape(19, n, fe.WIN).transpose(1, 0, 2)
    return (w[(w.max(2) - w.min(2)).max(1) <= fe.PTP_MAX] / 100.0).astype(np.float16)


def build_cache():
    meta = cb.meta_table(); recs = np.where(meta["first"].values)[0]
    with Pool(12) as pool:
        ws = pool.map(rec_windows, meta.h5_path.values[recs], chunksize=4)
    W = np.concatenate(ws); wrec = np.concatenate([np.full(len(w), r, np.int32) for r, w in zip(recs, ws)])
    np.save(CACHE, W); np.savez(CMETA, wsub=meta.subject.values[wrec], wlab=meta.label.values[wrec], wrec=wrec)
    # identity with the 2026-10-07 pipeline (cbramod_finetune.rec_windows) on the first recording
    old = cb.rec_windows(meta.h5_path.values[recs[0]]); new = W[wrec == recs[0]][:, cb.PERM].reshape(-1, 19, 10, 200)
    print(("PASS" if np.array_equal(old, new) else "FAIL") + f" cache equals cbramod_finetune.rec_windows on recording {recs[0]} ({len(old)} windows)")
    print(f"cache: {W.shape} windows, {len(np.unique(meta.subject.values[wrec]))} patients, {W.nbytes / 1e9:.2f} GB")


def load_cache():
    m = np.load(CMETA, allow_pickle=True); return np.load(CACHE), m["wsub"], m["wlab"], m["wrec"]


def split(r, k, wsub, wrec, wlab):
    """Exactly the 2026-10-07 split (cbramod_finetune.run_p3): first-session recordings of meta_table, GroupShuffleSplit by patient."""
    meta = cb.meta_table(); F = ek.folds(); te_p = set(F[(F.repeat == r) & (F.fold == k)].subject); subj, yr = meta.subject.values, meta.label.values
    recs = np.where(meta["first"].values)[0]; assert set(np.unique(wrec)) <= set(recs)
    tr_rec = recs[~np.isin(subj[recs], list(te_p))]; te_rec = recs[np.isin(subj[recs], list(te_p))]
    itr, iva = next(GroupShuffleSplit(1, test_size=0.2, random_state=100 * r + k).split(tr_rec, yr[tr_rec], subj[tr_rec]))
    tr_idx = np.where(np.isin(wrec, tr_rec[itr]))[0]; va_idx = np.where(np.isin(wrec, tr_rec[iva]))[0]; te_idx = np.where(np.isin(wrec, te_rec))[0]
    s_tr, s_va, s_te = set(wsub[tr_idx]), set(wsub[va_idx]), set(wsub[te_idx])
    assert not (s_tr & s_te) and not (s_va & s_te) and not (s_tr & s_va) and s_te == te_p and len(s_tr | s_va) == 160, "split leakage"
    return tr_idx, va_idx, te_idx


# ------------------------------------------------------------------ LaBraM official pieces (copied verbatim from LaBraM optim_factory.py / utils.py)
def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None
        if group_name not in parameter_group_names:
            scale = get_layer_scale(layer_id) if get_layer_scale is not None else 1.
            parameter_group_names[group_name] = {"weight_decay": this_weight_decay, "params": [], "lr_scale": scale}
            parameter_group_vars[group_name] = {"weight_decay": this_weight_decay, "params": [], "lr_scale": scale}
        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values()), parameter_group_names


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array([final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def labram_model():
    sys.path.insert(0, LABRAM_DIR)
    import modeling_finetune
    m = modeling_finetune.labram_base_patch200_200(num_classes=1, drop_rate=0.0, drop_path_rate=LB["drop_path"], attn_drop_rate=0.0, use_mean_pooling=True,
                                                   init_scale=0.001, use_rel_pos_bias=False, use_abs_pos_emb=True, init_values=0.1, qkv_bias=False)
    ck = torch.load(f"{LABRAM_DIR}/checkpoints/labram-base.pth", map_location="cpu", weights_only=False)["model"]
    sd = {k[8:]: v for k, v in ck.items() if k.startswith("student.") and "relative_position_index" not in k}
    st = m.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in sd and sd[k].shape != st[k].shape:
            del sd[k]
    res = m.load_state_dict(sd, strict=False)
    src = open(f"{LABRAM_DIR}/utils.py").read()
    std1020 = ast.literal_eval(re.search(r"standard_1020 = (\[.*?\])", src, re.S).group(1).replace("\\\n", ""))
    chans = [0] + [std1020.index(c) + 1 for c in CH]                         # utils.get_input_chans
    return m, chans, res


@torch.no_grad()
def labram_predict(m, W, idx, chans):
    m.eval(); out = []
    for s in range(0, len(idx), 96):                                          # official eval batch = 1.5 x 64
        x = torch.from_numpy(W[np.sort(idx[s:s + 96])].astype(np.float32)).to(dev).reshape(-1, 19, 10, 200)
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.sigmoid(m(x, input_chans=chans).float()).squeeze(-1).cpu().numpy())
    return np.concatenate(out)


def labram_finetune(W, tr_idx, wlab, va_idx, va_units, ypat, seed, log, epochs=EPOCHS):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m, chans, _ = labram_model(); m = m.to(dev)
    nl = m.get_num_layers(); assigner = LayerDecayValueAssigner([LB["layer_decay"] ** (nl + 1 - i) for i in range(nl + 2)])
    groups, _ = get_parameter_groups(m, LB["weight_decay"], m.no_weight_decay(), assigner.get_layer_id, assigner.get_scale)
    opt = torch.optim.AdamW(groups, lr=LB["lr"], weight_decay=0.0, eps=LB["opt_eps"])
    spe = len(tr_idx) // BATCH; lr_s = cosine_scheduler(LB["lr"], LB["min_lr"], epochs, spe, warmup_epochs=min(LB["warmup_epochs"], epochs - 1))     # = 5 for the 50-epoch runs; shorter only in the 2-epoch dry-run
    n1 = wlab[tr_idx].sum(); n0 = len(tr_idx) - n1
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n0 / max(n1, 1), device=dev)); scaler = torch.amp.GradScaler("cuda")
    best_auc, best_state, best_ep, curve = -1, None, -1, []
    for ep in range(epochs):
        m.train(); t0 = time.time(); order = rng.permutation(tr_idx)
        for st in range(spe):
            it = ep * spe + st
            for g in opt.param_groups:
                g["lr"] = lr_s[it] * g["lr_scale"]
            b = np.sort(order[st * BATCH:(st + 1) * BATCH])
            x = torch.from_numpy(W[b].astype(np.float32)).to(dev).reshape(-1, 19, 10, 200); yb = torch.from_numpy(wlab[b].astype(np.float32)).to(dev).unsqueeze(-1)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(m(x, input_chans=chans), yb)
            scaler.scale(loss).backward(); scaler.unscale_(opt); scaler.step(opt); scaler.update(); opt.zero_grad()
        sc = pd.Series(labram_predict(m, W, np.sort(va_idx), chans)).groupby(va_units[np.sort(va_idx)]).mean()
        auc = roc_auc_score(ypat.loc[sc.index].values, sc.values); curve.append(auc)
        if auc > best_auc:
            best_auc, best_ep = auc, ep; best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        log(f"  epoch {ep + 1:2d}: val AUROC {auc:.3f} (best {best_auc:.3f} @ {best_ep + 1}), lr {lr_s[it]:.2e}, {time.time() - t0:.0f}s")
    m.load_state_dict(best_state)
    return m, chans, best_auc, best_ep + 1, curve


# ------------------------------------------------------------------ one fold
def run_fold(model, r, k, W, wsub, wlab, wrec, epochs=EPOCHS, save=True):
    out_path = f"{OUT}/{model}/r{r}_k{k}.json"; os.makedirs(f"{OUT}/{model}", exist_ok=True)
    if save and os.path.exists(out_path):
        return None
    tr_idx, va_idx, te_idx = split(r, k, wsub, wrec, wlab); ypat = pd.Series(wlab).groupby(wsub).first(); seed = 1000 + 10 * r + k
    log = cb.log_to(f"{OUT}/{model}/r{r}_k{k}.log") if save else (lambda s: print(s, flush=True)); t0 = time.time()
    log(f"{model} r{r} k{k}: windows train {len(tr_idx)}, val {len(va_idx)}, test {len(te_idx)}; seed {seed}")
    if model == "cbramod":
        cb.EPOCHS = epochs; Wc = W[:, cb.PERM].reshape(len(W), 19, 10, 200)      # CBraMod channel order, as in cbramod_finetune.rec_windows
        m, vauc, vep = cb.finetune(Wc, tr_idx, wlab, va_idx, wsub[va_idx], ypat, seed, log)
        pv, pt = cb.predict(m, Wc, va_idx), cb.predict(m, Wc, te_idx)
    else:
        m, chans, vauc, vep, _ = labram_finetune(W, tr_idx, wlab, va_idx, wsub, ypat, seed, log, epochs)
        pv, pt = labram_predict(m, W, np.sort(va_idx), chans), labram_predict(m, W, np.sort(te_idx), chans); va_idx, te_idx = np.sort(va_idx), np.sort(te_idx)
    va = pd.Series(pv).groupby(wsub[va_idx]).mean(); thr, _ = ts.best_threshold(ypat.loc[va.index].values, va.values)
    te = pd.Series(pt).groupby(wsub[te_idx]).mean(); tauc = roc_auc_score(ypat.loc[te.index].values, te.values)
    res = dict(model=model, repeat=r, fold=k, seed=seed, val_auc=float(vauc), best_epoch=int(vep), val_threshold=float(thr), seconds=time.time() - t0,
               test={s: float(v) for s, v in te.items()})
    log(f"{model} r{r} k{k} done: best epoch {vep}, val AUROC {vauc:.3f}, test AUC {tauc:.3f}, {(time.time() - t0) / 60:.1f} min")
    if save:
        json.dump(res, open(out_path, "w"))
    return res


def dry_run(W, wsub, wlab, wrec):
    ok = True
    m, chans, res = labram_model(); exp_unexp = {"norm.weight", "norm.bias", "mask_token", "lm_head.weight", "lm_head.bias"}          # pretraining-only (masked-token prediction)
    print(f"LaBraM missing keys {res.missing_keys} | unexpected {res.unexpected_keys}")
    ok &= set(res.missing_keys) <= {"head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"} and set(res.unexpected_keys) <= exp_unexp
    print(("PASS" if ok else "FAIL") + " LaBraM pretrained weights loaded (missing only head/fc_norm; unexpected only pretraining-only mask_token/lm_head and the final norm replaced by fc_norm under mean pooling)")
    nl = m.get_num_layers(); a = LayerDecayValueAssigner([LB["layer_decay"] ** (nl + 1 - i) for i in range(nl + 2)])
    _, names = get_parameter_groups(m, LB["weight_decay"], m.no_weight_decay(), a.get_layer_id, a.get_scale)
    sc = {g: round(v["lr_scale"], 6) for g, v in names.items()}; print("layer-decay lr scales:", {g: sc[g] for g in sorted(sc)[:3]}, "...", f"{len(sc)} groups")
    c1 = abs(names["layer_0_no_decay"]["lr_scale"] - 0.65 ** 13) < 1e-12 and abs(names["layer_13_decay"]["lr_scale"] - 1.0) < 1e-12 and "pos_embed" in names["layer_0_no_decay"]["params"]
    print(("PASS" if c1 else "FAIL") + " layer decay: embeddings 0.65^13, head 1.0; pos_embed without weight decay"); ok &= c1
    s = cosine_scheduler(5e-4, 1e-6, 50, 100, warmup_epochs=5); c2 = s[0] == 0 and abs(s[499] - 5e-4) < 1e-12 and abs(s[-1] - 1e-6) < 1e-8 and abs(s[500] - 5e-4) < 1e-12
    print(("PASS" if c2 else "FAIL") + " cosine_scheduler: warmup 0 -> 5e-4 over 5 epochs, cosine to 1e-6"); ok &= c2
    tr, va, te = split(0, 0, wsub, wrec, wlab); print(f"PASS split r0k0: train {len(set(wsub[tr]))} / val {len(set(wsub[va]))} / test {len(set(wsub[te]))} patients, disjoint")
    for model in ("cbramod", "labram"):
        t0 = time.time(); a1 = run_fold(model, 0, 0, W, wsub, wlab, wrec, epochs=2, save=False); t1 = time.time() - t0
        a2 = run_fold(model, 0, 0, W, wsub, wlab, wrec, epochs=2, save=False); same = a1["test"] == a2["test"]
        dmax = max(abs(a1["test"][s_] - a2["test"][s_]) for s_ in a1["test"])
        print(("PASS" if same else "WARN") + f" {model}: two 2-epoch runs bitwise identical={same} (max |diff| {dmax:.2e}); {t1:.0f} s per 2 epochs incl. eval -> "
              f"about {t1 / 2 * EPOCHS / 60:.0f} min per 50-epoch fold; GPU peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cache", action="store_true"); g.add_argument("--dry-run", action="store_true"); g.add_argument("--fold", nargs=3)
    a = ap.parse_args(); print(f"round25_fm | {time.strftime('%Y-%m-%d %H:%M:%S')} | torch {torch.__version__} | {torch.cuda.get_device_name(0)}", flush=True)
    if a.cache:
        build_cache(); sys.exit()
    W, wsub, wlab, wrec = load_cache()
    if a.dry_run:
        sys.exit(0 if dry_run(W, wsub, wlab, wrec) else "STOP")
    run_fold(a.fold[0], int(a.fold[1]), int(a.fold[2]), W, wsub, wlab, wrec)
