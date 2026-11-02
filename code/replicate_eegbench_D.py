"""EEG-Bench-F, variant D (reports/eegbench_fixed_prediction.md): the EEG-Bench epilepsy recipe exactly as in variant B
(crop 30 min, 0.1-75 Hz, notch 60 Hz, resample 200 Hz, 19 channels, 8-s chunks, BENDR flatten / LaBraM mean tokens),
except that the common average reference is computed over the 19 EEG channels only instead of every channel in the file.
Usage: python replicate_eegbench_D.py check | run"""
import os, sys, glob, numpy as np, pandas as pd, mne, warnings
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
import replicate_eegbench as rb
from embeddings import CH
warnings.filterwarnings("ignore"); mne.set_log_level("ERROR")


def names_of(raw):
    return [next(n for n in (f"EEG {c}-REF", f"EEG {c}-LE") if n in raw.ch_names) for c in CH]


def variant_D(edf):
    raw = mne.io.read_raw_edf(edf, preload=False)
    if raw.times[-1] > rb.MAX_S:
        raw.crop(tmax=rb.MAX_S)
    raw.load_data(); names = names_of(raw)
    raw.pick(names)                                         # the fix: CAR over the 19 EEG channels only
    raw.set_eeg_reference("average")
    raw.filter(l_freq=0.1, h_freq=75.0 if 75.0 < 0.5 * raw.info["sfreq"] else None)
    if 0.5 * raw.info["sfreq"] > 60.0:
        raw.notch_filter(60.0)
    raw.resample(200)
    return raw.get_data(picks=names, units="uV").astype(np.float32)


def work(args):
    i, edf = args
    try:
        return i, variant_D(edf), None
    except Exception as e:
        return i, None, repr(e)


def jobs():
    meta = pd.read_csv(f"{rb.D}/features_bandpower_recordings.csv"); meta = meta[meta.n_clean > 0].reset_index(drop=True)
    idx = pd.read_csv(f"{rb.D}/index.csv")[["h5_path", "orig_duration_s"]]; meta = meta.merge(idx, on="h5_path", how="left")
    edfs = {os.path.basename(p): p for p in glob.glob(f"{rb.RAW}/0[01]_*/*/*/*/*.edf")}
    return meta, [(i, edfs[f]) for i, (f, d) in enumerate(zip(meta.file, meta.orig_duration_s)) if d >= 60]


if __name__ == "__main__":
    meta, J = jobs()
    if sys.argv[1] == "check":
        ref = np.load(f"{rb.D}/eegbench_rep_B.npz")["rec"]
        print(f"jobs {len(J)}; recordings in the saved variant B file: {len(ref)}")
        for i, edf in [J[0], J[len(J) // 2], J[-1]]:                    # epilepsy and non-epilepsy files
            xD = variant_D(edf); xB = rb.eegbench_variants(edf)["B"]
            zD = np.abs(xD.mean(0)).max() / xD.std(); zB = np.abs(xB.mean(0)).max() / xB.std()
            print(("PASS" if zD < 1e-4 and zB > 1e-2 and xD.shape == xB.shape else "FAIL") +
                  f" {os.path.basename(edf)} (label {meta.label[i]}): shape {xD.shape}; max |mean over 19 ch| / std: D {zD:.1e} (must be ~0), B {zB:.1e} (CAR over all channels)")
        sys.exit()
    lab, chans = None, None; feats = {"bendr": {}, "labram": {}}; errs = []
    with Pool(8) as pool:                                                 # fork workers before CUDA is initialised
        lab, chans = rb.load_labram(); ben = rb.load_bendr()
        for k, (i, x, err) in enumerate(pool.imap_unordered(work, J, chunksize=2), 1):
            if err:
                errs.append((meta.file[i], err)); continue
            b, l = rb.embed(x, lab, chans, ben)
            if b is not None:
                feats["bendr"][i] = b.astype(np.float16); feats["labram"][i] = l.astype(np.float16)
            if k % 200 == 0:
                print(f"{k}/{len(J)}", flush=True)
    ids = sorted(feats["bendr"]); print("errors:", len(errs), errs[:3]); print("D recordings:", len(ids))
    np.savez(f"{rb.D}/eegbench_rep_D.npz", rec=np.array(ids), bendr=np.vstack([feats["bendr"][i] for i in ids]), labram=np.vstack([feats["labram"][i] for i in ids]))
