"""Frozen CBraMod (official weights, SHA-256 0792cb80...) on the same clean 10 s windows as the other backbones.
Channels are reordered to CBraMod's pretraining layout; input = uV / 100 as [B, 19, 10 patches, 200].
Feature per layer = mean over channels x patches of that encoder layer's output (proj_out is not used)."""
import sys, time, numpy as np, pandas as pd, torch
from embeddings import windows, D, WIN, CH, dev
sys.path.insert(0, "/home/miftah_f/RTX-5090/CBraMod")
from models.cbramod import CBraMod
CBRAMOD_ORDER = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]
PERM = [CH.index(c) for c in CBRAMOD_ORDER]

if __name__ == "__main__":
    meta = pd.read_csv(f"{D}/features_bandpower_recordings.csv"); meta = meta[meta.n_clean > 0].reset_index(drop=True)
    m = CBraMod()
    m.load_state_dict(torch.load("/home/miftah_f/RTX-5090/CBraMod/pretrained_weights/pretrained_weights.pth", map_location="cpu"), strict=True)
    m = m.eval().to(dev)
    feats = {}
    hooks = [lay.register_forward_hook(lambda mod, i, o, k=k: feats.__setitem__(k, o.mean((1, 2)))) for k, lay in enumerate(m.encoder.layers)]
    @torch.no_grad()
    def run(w):
        x = torch.from_numpy(np.ascontiguousarray(w[:, PERM])).to(dev) / 100.0
        m.patch_embedding(x.reshape(len(x), 19, WIN // 200, 200))       # warm check below uses the full encoder
        m.encoder(m.patch_embedding(x.reshape(len(x), 19, WIN // 200, 200)))
        return torch.stack([feats[k] for k in range(len(m.encoder.layers))], 1)
    w0 = windows(meta.h5_path[0])[:8]
    a, b = run(w0).float().cpu().numpy(), run(w0 * 0.5).float().cpu().numpy()
    print(f"sanity: layers {a.shape[1]}, per-dim std across windows (last layer) {a[:, -1].std(0).mean():.4f}, change when input halved {np.abs(a - b).mean():.4f}")
    out, rec = [], []; t0 = time.time()
    for i, h in enumerate(meta.h5_path):
        w = windows(h)
        for s in range(0, len(w), 256):
            out.append(run(w[s:s + 256]).half().cpu().numpy())
        rec.append(np.full(len(w), i, np.int32))
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(meta)} recordings, {time.time() - t0:.0f}s", flush=True)
    np.savez(f"{D}/emb_cbramod_layers_windows.npz", E=np.concatenate(out), rec=np.concatenate(rec), h5_path=meta.h5_path.values)
    print("saved", np.concatenate(rec).shape)
