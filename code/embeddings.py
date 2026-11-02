"""Frozen foundation-model embeddings for every clean 10 s window of the rebuilt recordings.

Encoders (official weights, verified against their public releases):
  LaBraM-base  (LaBraM/checkpoints/labram-base.pth, git blob 221148b5...)
      input: uV / 100, shape [B, 19, 10 patches, 200], channel ids from LaBraM's standard_1020,
      config as in the official TUAB fine-tuning recipe (abs pos emb, no rel-pos bias, no qkv bias)
      embedding: mean of last-layer patch tokens after the pretrained final LayerNorm (200-d)
  BENDR encoder (EEG-Bench/chkpt/encoder.pt, md5 0a0a293a... == GitHub release v0.1-alpha)
      input: uV, 19 channels in BENDR's 10-20 order + one constant channel (=1, the initial value
      of EEG-Bench's learnable scale channel); embedding: mean over the encoder time steps (512-d)
Windows are identical to features.py (10 s, 10 s edges skipped, peak-to-peak <= 800 uV).
"""
import ast, os, re, sys, time, numpy as np, pandas as pd, h5py, torch, torch.nn as nn
D = "/home/miftah_f/RTX-5090/tuep_fresh/data"
LABRAM_DIR = "/home/miftah_f/RTX-5090/LaBraM"
BENDR_W = "/home/miftah_f/RTX-5090/EEG-Bench/chkpt/encoder.pt"
FS, WIN, EDGE, PTP_MAX = 200, 2000, 2000, 800.0
CH = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ", "C4", "T4", "T5", "P3", "PZ", "P4", "T6", "O1", "O2"]
dev = torch.device("cuda")

def windows(h5_path):
    with h5py.File(os.path.join(D, h5_path)) as f:
        x = f["eeg"][()].astype(np.float32)
    x = x[:, EDGE:x.shape[1] - EDGE]; n = x.shape[1] // WIN
    if n == 0:
        return np.zeros((0, 19, WIN), np.float32)
    w = x[:, :n * WIN].reshape(19, n, WIN).transpose(1, 0, 2)
    return w[(w.max(2) - w.min(2)).max(1) <= PTP_MAX]

# ------------------------------------------------------------------ LaBraM
def load_labram():
    sys.path.insert(0, LABRAM_DIR)
    import modeling_finetune  # noqa: registers labram_base_patch200_200
    src = open(f"{LABRAM_DIR}/utils.py").read()
    std1020 = ast.literal_eval(re.search(r"standard_1020 = (\[.*?\])", src, re.S).group(1).replace("\\\n", ""))
    m = modeling_finetune.labram_base_patch200_200(num_classes=0, use_mean_pooling=False, init_values=0.1,
                                                   qkv_bias=False, use_rel_pos_bias=False, use_abs_pos_emb=True)
    ck = torch.load(f"{LABRAM_DIR}/checkpoints/labram-base.pth", map_location="cpu", weights_only=False)["model"]
    sd = {k[8:]: v for k, v in ck.items() if k.startswith("student.") and "relative_position_index" not in k}
    res = m.load_state_dict(sd, strict=False)
    print("LaBraM missing keys:", res.missing_keys, "| unexpected:", res.unexpected_keys)
    assert all(k.startswith(("head", "fc_norm")) for k in res.missing_keys), "pretrained weights not fully loaded"
    input_chans = [0] + [std1020.index(c) + 1 for c in CH]
    return m.eval().to(dev), torch.tensor(input_chans, device=dev)

@torch.no_grad()
def labram_embed(model, chans, w):
    x = torch.from_numpy(w).to(dev) / 100.0
    x = x.reshape(x.shape[0], 19, WIN // 200, 200)
    tok = model.forward_features(x, input_chans=chans, return_patch_tokens=True)   # [B, 190, 200], after final norm
    return tok.mean(1).float().cpu().numpy()

# ------------------------------------------------------------------ BENDR (encoder copied from EEG-Bench / dn3)
class ConvEncoderBENDR(nn.Module):
    def __init__(self, in_features, encoder_h=512, enc_width=(3, 2, 2, 2, 2, 2), enc_downsample=(3, 2, 2, 2, 2, 2)):
        super().__init__()
        enc_width = [e if e % 2 else e + 1 for e in enc_width]
        self.encoder = nn.Sequential()
        for i, (width, ds) in enumerate(zip(enc_width, enc_downsample)):
            self.encoder.add_module(f"Encoder_{i}", nn.Sequential(
                nn.Conv1d(in_features, encoder_h, width, stride=ds, padding=width // 2),
                nn.Dropout1d(0.0), nn.GroupNorm(encoder_h // 2, encoder_h), nn.GELU()))
            in_features = encoder_h
    def forward(self, x):
        return self.encoder(x)

def load_bendr():
    m = ConvEncoderBENDR(20)
    m.load_state_dict(torch.load(BENDR_W, map_location="cpu"), strict=True)
    return m.eval().to(dev)

@torch.no_grad()
def bendr_embed(model, w):
    x = torch.from_numpy(w).to(dev)
    x = torch.cat([x, torch.ones(x.shape[0], 1, x.shape[2], device=dev)], dim=1)
    return model(x).mean(2).float().cpu().numpy()

# ------------------------------------------------------------------ run
if __name__ == "__main__":
    meta = pd.read_csv(f"{D}/features_bandpower_recordings.csv")
    meta = meta[meta.n_clean > 0].reset_index(drop=True)
    lab, chans = load_labram(); ben = load_bendr()
    # sanity: embeddings must depend on the input and must not collapse to a constant
    w0 = windows(meta.h5_path[0])[:8]
    e1, e2 = labram_embed(lab, chans, w0), labram_embed(lab, chans, w0 * 0.5)
    print("LaBraM sanity: per-dim std across windows %.4f | change when input halved %.4f" % (e1.std(0).mean(), np.abs(e1 - e2).mean()))
    b1 = bendr_embed(ben, w0); print("BENDR  sanity: per-dim std across windows %.4f" % b1.std(0).mean())
    out = {"labram": [], "bendr": []}; rec_id = []; t0 = time.time()
    for i, h in enumerate(meta.h5_path):
        w = windows(h)
        for s in range(0, len(w), 256):
            out["labram"].append(labram_embed(lab, chans, w[s:s + 256]).astype(np.float16))
            out["bendr"].append(bendr_embed(ben, w[s:s + 256]).astype(np.float16))
        rec_id.append(np.full(len(w), i, np.int32))
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(meta)} recordings, {time.time() - t0:.0f}s", flush=True)
    rec_id = np.concatenate(rec_id)
    for k in out:
        E = np.concatenate(out[k])
        np.savez(f"{D}/emb_{k}_windows.npz", E=E, rec=rec_id, h5_path=meta.h5_path.values)
        cos = E[np.random.default_rng(0).choice(len(E), 2000, replace=False)].astype(np.float32)
        cos /= np.linalg.norm(cos, axis=1, keepdims=True) + 1e-8
        print(f"{k}: {E.shape}, mean pairwise cosine {float((cos @ cos.T).mean()):.3f} (1.0 would mean collapsed)")
