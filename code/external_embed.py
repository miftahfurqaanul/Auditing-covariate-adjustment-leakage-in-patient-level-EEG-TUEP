"""Frozen LaBraM embeddings on the shared 10-channel montage, for the Emotiv windows (closed, whole) and the
TUEP first-session windows. Channel ids use the modern 10-20 names for both datasets
(T3=T7, T4=T8, T5=P7, T6=P8 are the same positions)."""
import ast, re, sys, numpy as np, pandas as pd, torch
from multiprocessing import Pool
import external_prep as ep
D = ep.D
LABRAM_DIR = "/home/miftah_f/RTX-5090/LaBraM"

if __name__ == "__main__":
    meta = pd.read_csv(f"{D}/features_bandpower_recordings.csv"); meta = meta[meta.n_clean > 0].reset_index(drop=True)
    first = (meta.groupby("subject").session_no.transform("min") == meta.session_no).values
    fm = meta[first].reset_index(drop=True)
    with Pool(12) as pool:                                   # CPU windowing before CUDA is initialised
        tw = pool.map(ep.prep_tuep, fm.h5_path)
    sys.path.insert(0, LABRAM_DIR)
    import modeling_finetune
    src = open(f"{LABRAM_DIR}/utils.py").read()
    std1020 = ast.literal_eval(re.search(r"standard_1020 = (\[.*?\])", src, re.S).group(1).replace("\\\n", ""))
    m = modeling_finetune.labram_base_patch200_200(num_classes=0, use_mean_pooling=False, init_values=0.1,
                                                   qkv_bias=False, use_rel_pos_bias=False, use_abs_pos_emb=True)
    ck = torch.load(f"{LABRAM_DIR}/checkpoints/labram-base.pth", map_location="cpu", weights_only=False)["model"]
    res = m.load_state_dict({k[8:]: v for k, v in ck.items() if k.startswith("student.")}, strict=False)
    assert all(k.startswith(("head", "fc_norm")) for k in res.missing_keys)
    dev = torch.device("cuda"); m = m.eval().to(dev)
    chans = torch.tensor([0] + [std1020.index(c) + 1 for c in ep.COMMON], device=dev)

    @torch.no_grad()
    def emb(W):
        out = []
        for s in range(0, len(W), 256):
            x = torch.from_numpy(np.asarray(W[s:s + 256], np.float32)).to(dev) / 100.0
            out.append(m.forward_features(x.reshape(len(x), 10, 10, 200), input_chans=chans, return_patch_tokens=True).mean(1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 200), np.float32)

    for seg in ["closed", "whole"]:
        z = np.load(f"{D}/ext_zenodo_{seg}_windows.npz")
        np.savez(f"{D}/ext_zenodo_{seg}_labram.npz", X=emb(z["W"]), sid=z["sid"])
        print(seg, "Emotiv windows embedded:", len(z["sid"]), flush=True)
    S = np.concatenate([[s] * len(w) for s, w in zip(fm.subject, tw)])
    np.savez(f"{D}/ext_tuep10_first_labram.npz", X=emb(np.concatenate(tw)), sid=S)
    print("TUEP first-session windows embedded:", len(S))
