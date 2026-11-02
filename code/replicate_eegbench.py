"""Faithful replication of the EEG-Bench epilepsy recipe, plus two controlled variants.

EEG-Bench (LaBraM/utils_2.py process_bendr / process_filter), reproduced as written:
  crop to the first 30 min; CAR over ALL channels in the file (MNE default), band-pass 0.1-75 Hz,
  notch 50 Hz, resample 200 Hz; 19 channels in BENDR's 10-20 order; 1600-sample chunks (8 s), no
  artifact rejection; BENDR representation = flattened encoder output (512 x 17 = 8704), averaged over
  chunks; the linear probe is replaced by nested-CV logistic regression on the same features.
Variants:
  A  EEG-Bench as written (notch 50 Hz -> 60 Hz US mains noise passes)
  B  identical but notch 60 Hz
  C  our clean recordings (19-ch CAR, 0.5-45 Hz), same chunking and representation
LaBraM embeddings (8-patch chunks, mean patch tokens) are computed for every variant as well.
"""
import os, sys, glob, numpy as np, pandas as pd, h5py, mne, torch, warnings
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from embeddings import load_labram, load_bendr, CH, dev
warnings.filterwarnings("ignore"); mne.set_log_level("ERROR")
D, RAW = "/home/miftah_f/RTX-5090/tuep_fresh/data", "/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1"
CHUNK, MAX_S = 1600, 30 * 60

def eegbench_variants(edf):
    raw = mne.io.read_raw_edf(edf, preload=False)
    if raw.times[-1] > MAX_S:
        raw.crop(tmax=MAX_S)
    raw.load_data(); raw.set_eeg_reference("average")
    raw.filter(l_freq=0.1, h_freq=75.0 if 75.0 < 0.5 * raw.info["sfreq"] else None)
    names = [next(n for n in (f"EEG {c}-REF", f"EEG {c}-LE") if n in raw.ch_names) for c in CH]
    out = {}
    for tag, notch in [("A", 50.0), ("B", 60.0)]:
        r = raw.copy()
        if 0.5 * r.info["sfreq"] > notch:
            r.notch_filter(notch)
        r.resample(200)
        out[tag] = r.get_data(picks=names, units="uV").astype(np.float32)
    return out

def work(args):
    i, edf, h5 = args
    try:
        out = eegbench_variants(edf)
    except Exception as e:
        return i, None, repr(e)
    with h5py.File(os.path.join(D, h5)) as f:
        out["C"] = f["eeg"][:, :MAX_S * 200].astype(np.float32)
    return i, out, None

@torch.no_grad()
def embed(x, lab, chans, ben):
    n = x.shape[1] // CHUNK
    if n == 0:
        return None, None
    c = torch.from_numpy(x[:, :n * CHUNK].reshape(19, n, CHUNK).transpose(1, 0, 2).copy()).to(dev)
    b = ben(torch.cat([c, torch.ones(n, 1, CHUNK, device=dev)], 1)).flatten(1).mean(0)
    l = lab.forward_features((c / 100.0).reshape(n, 19, CHUNK // 200, 200), input_chans=chans, return_patch_tokens=True).mean(1).mean(0)
    return b.float().cpu().numpy(), l.float().cpu().numpy()

if __name__ == "__main__":
    meta = pd.read_csv(f"{D}/features_bandpower_recordings.csv"); meta = meta[meta.n_clean > 0].reset_index(drop=True)
    idx = pd.read_csv(f"{D}/index.csv")[["h5_path", "orig_duration_s"]]
    meta = meta.merge(idx, on="h5_path", how="left")
    edfs = {os.path.basename(p): p for p in glob.glob(f"{RAW}/0[01]_*/*/*/*/*.edf")}
    jobs = [(i, edfs[f], h) for i, (f, h, d) in enumerate(zip(meta.file, meta.h5_path, meta.orig_duration_s)) if d >= 60]
    if len(sys.argv) > 1:
        jobs = jobs[:int(sys.argv[1])]
    feats = {v: {"bendr": {}, "labram": {}} for v in "ABC"}; errs = []
    with Pool(8) as pool:                       # fork workers before CUDA is initialised
        lab, chans = load_labram(); ben = load_bendr()
        for k, (i, out, err) in enumerate(pool.imap_unordered(work, jobs, chunksize=2), 1):
            if err:
                errs.append((meta.file[i], err)); continue
            for v, x in out.items():
                b, l = embed(x, lab, chans, ben)
                if b is not None:
                    feats[v]["bendr"][i] = b.astype(np.float16); feats[v]["labram"][i] = l.astype(np.float16)
            if k % 200 == 0:
                print(f"{k}/{len(jobs)}", flush=True)
    print("errors:", len(errs), errs[:3])
    for v in "ABC":
        ids = sorted(feats[v]["bendr"])
        np.savez(f"{D}/eegbench_rep_{v}{'_test' if len(sys.argv) > 1 else ''}.npz", rec=np.array(ids), bendr=np.vstack([feats[v]["bendr"][i] for i in ids]),
                 labram=np.vstack([feats[v]["labram"][i] for i in ids]))
        print(v, "recordings:", len(ids))
