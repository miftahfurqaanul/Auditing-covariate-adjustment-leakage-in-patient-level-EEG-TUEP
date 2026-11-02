"""C3: frozen LaBraM, mean over all patch tokens at the output of each of the 12 transformer blocks.
Same windows and model loading as embeddings.py; one GPU pass over every clean window."""
import numpy as np, pandas as pd, torch, time
from embeddings import load_labram, windows, D, WIN, dev

if __name__ == "__main__":
    meta = pd.read_csv(f"{D}/features_bandpower_recordings.csv"); meta = meta[meta.n_clean > 0].reset_index(drop=True)
    lab, chans = load_labram()
    feats = {}
    hooks = [blk.register_forward_hook(lambda m, i, o, k=k: feats.__setitem__(k, o[:, 1:].mean(1))) for k, blk in enumerate(lab.blocks)]
    out, rec = [], []; t0 = time.time()
    with torch.no_grad():
        for i, h in enumerate(meta.h5_path):
            w = windows(h)
            for s in range(0, len(w), 256):
                x = torch.from_numpy(w[s:s + 256]).to(dev) / 100.0
                lab.forward_features(x.reshape(len(x), 19, WIN // 200, 200), input_chans=chans)
                out.append(torch.stack([feats[k] for k in range(12)], 1).half().cpu().numpy())   # [B, 12, 200]
            rec.append(np.full(len(w), i, np.int32))
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(meta)} recordings, {time.time() - t0:.0f}s", flush=True)
    for hk in hooks:
        hk.remove()
    np.savez(f"{D}/emb_labram_layers_windows.npz", E=np.concatenate(out), rec=np.concatenate(rec), h5_path=meta.h5_path.values)
    print("saved", np.concatenate(rec).shape)
