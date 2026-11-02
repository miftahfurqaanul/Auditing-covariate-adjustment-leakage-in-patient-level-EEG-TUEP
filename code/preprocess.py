"""Rebuild TUEP v2.0.1 from raw EDF into one HDF5 file per recording.

Per recording:
  1. read the 19 standard 10-20 channels (-REF or -LE names), in microvolts
  2. trim leading/trailing calibration (every channel carries the same waveform)
  3. common average reference over channels that are not flat
  4. resample to 200 Hz (polyphase), band-pass 0.5-45 Hz (zero-phase FIR)
  5. store float16 [19, T] plus provenance and QC attributes
Recordings longer than 2 h (long-term monitoring) keep only their first 60 min.
Nothing here uses labels, so it cannot leak them into the signal.
"""
import argparse, glob, hashlib, json, os, re, sys, time
from fractions import Fraction
from multiprocessing import Pool
import h5py, mne, numpy as np, pandas as pd
from scipy.signal import resample_poly

RAW_ROOT = "/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1"
OUT_ROOT = "/home/miftah_f/RTX-5090/tuep_fresh/data"
CHANNELS = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ",
            "C4", "T4", "T5", "P3", "PZ", "P4", "T6", "O1", "O2"]
FS_OUT = 200
BAND = (0.5, 45.0)
LTM_THRESHOLD_S = 7200      # longer recordings are treated as long-term monitoring
LTM_KEEP_S = 3600           # ...and only their first hour is kept
CALIB_TOL_UV = 1.0          # max spread across channels for a 1 s window to count as calibration
FLAT_STD_UV = 0.5
PIPELINE_VERSION = "1.0"

def parse_path(path):
    parts = path.split("/")
    cls, subj, sess, montage, fname = parts[-5:]
    return dict(cls=cls, label=int(cls == "00_epilepsy"), subject=subj, session=sess,
                session_no=int(sess.split("_")[0][1:]), year=int(sess.split("_")[1]),
                montage=montage, file=fname, token=int(re.search(r"_t(\d+)", fname).group(1)))

def header_info(path):
    with open(path, "rb") as f:
        h = f.read(256)
    patient, recording = h[8:88].decode("latin1"), h[88:168].decode("latin1")
    age = re.search(r"Age:(-?\d+)", patient); sex = re.search(r"^\S+\s+([MF])\s", patient)
    dev = re.search(r"XXX\s+(.*)$", recording.strip())
    return dict(age=int(age.group(1)) if age else None, sex=sex.group(1) if sex else None,
                device=dev.group(1).strip() if dev else None)

def identical_windows(x, fs):
    """Boolean per 1 s window: all channels carry the same waveform (calibration or flat)."""
    w = int(round(fs)); n = x.shape[1] // w
    if n == 0:
        return np.zeros(0, bool)
    spread = (x[:, :n * w].max(0) - x[:, :n * w].min(0)).reshape(n, w).max(1)
    return spread < CALIB_TOL_UV

def process(path):
    row = parse_path(path); row.update(header_info(path)); row["status"] = "ok"
    out_path = os.path.join(OUT_ROOT, "eeg", row["cls"], row["subject"], row["file"].replace(".edf", ".h5"))
    row["h5_path"] = os.path.relpath(out_path, OUT_ROOT)
    try:
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        fs = float(raw.info["sfreq"]); row["orig_fs"] = fs
        row["orig_duration_s"] = raw.n_times / fs
        picks = []
        for ch in CHANNELS:
            name = next((n for n in (f"EEG {ch}-REF", f"EEG {ch}-LE") if n in raw.ch_names), None)
            picks.append(name)
        missing = [c for c, p in zip(CHANNELS, picks) if p is None]
        if missing:
            row.update(status="skipped_missing_channels", missing_channels="|".join(missing)); return row
        truncated = row["orig_duration_s"] > LTM_THRESHOLD_S
        stop = raw.n_times if not truncated else min(raw.n_times, int((LTM_KEEP_S + 300) * fs))
        x = raw.get_data(picks=picks, stop=stop) * 1e6
        # calibration trim (leading / trailing runs of identical-across-channel windows)
        ident = identical_windows(x, fs); w = int(round(fs))
        lead = int(np.argmin(ident)) if not ident.all() else len(ident)
        trail = int(np.argmin(ident[::-1])) if not ident.all() else 0
        row["calib_lead_s"], row["calib_trail_s"] = lead, trail
        row["interior_identical_s"] = int(ident[lead:len(ident) - trail].sum())
        x = x[:, lead * w: x.shape[1] - trail * w]
        if truncated:
            x = x[:, :int(LTM_KEEP_S * fs)]
        row["truncated_ltm"] = bool(truncated)
        if x.shape[1] < 10 * fs:
            row.update(status="skipped_too_short_after_trim"); return row
        row["md5_first30s"] = hashlib.md5(np.round(x[:, :int(30 * fs)], 1).astype(np.float32).tobytes()).hexdigest()
        # common average reference over non-flat channels
        good = x.std(1) >= FLAT_STD_UV
        row["flat_channels"] = "|".join(c for c, g in zip(CHANNELS, good) if not g)
        x = x - x[good].mean(0, keepdims=True)
        x[~good] = 0.0
        # resample then band-pass
        ratio = Fraction(FS_OUT, int(round(fs))).limit_denominator(1000)
        if ratio != 1:
            x = resample_poly(x, ratio.numerator, ratio.denominator, axis=1)
        x = mne.filter.filter_data(x, FS_OUT, BAND[0], BAND[1], method="fir", phase="zero",
                                   fir_design="firwin", verbose="ERROR")
        # QC on the processed signal (1 s windows)
        n = x.shape[1] // FS_OUT; xw = x[:, :n * FS_OUT].reshape(19, n, FS_OUT)
        ptp = xw.max(2) - xw.min(2)
        row.update(duration_s=x.shape[1] / FS_OUT, std_uV_median=float(np.median(x[good].std(1))),
                   frac_win_ptp_gt_500uV=float((ptp.max(0) > 500).mean()),
                   frac_win_ptp_gt_1000uV=float((ptp.max(0) > 1000).mean()),
                   max_abs_uV=float(np.abs(x).max()))
        if row["max_abs_uV"] > 60000:
            x = np.clip(x, -60000, 60000); row["clipped_to_fp16"] = True
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with h5py.File(out_path, "w") as f:
            f.create_dataset("eeg", data=x.astype(np.float16), chunks=(19, min(x.shape[1], 4000)), compression="lzf")
            f.create_dataset("channel_good", data=good.astype(np.uint8))
            f.attrs["channels"] = CHANNELS
            for k, v in row.items():
                if v is not None and k not in ("h5_path",):
                    f.attrs[k] = v
            f.attrs.update(fs=FS_OUT, band_hz=BAND, reference="average", unit="uV",
                           pipeline_version=PIPELINE_VERSION, source_edf=os.path.relpath(path, RAW_ROOT))
    except Exception as e:
        row.update(status="error", error=repr(e)[:300])
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--files", nargs="*")
    a = ap.parse_args()
    files = a.files or sorted(glob.glob(f"{RAW_ROOT}/0[01]_*/*/*/*/*.edf"))
    if a.limit:
        files = files[:a.limit]
    t0 = time.time(); rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process, files, chunksize=2), 1):
            rows.append(r)
            if i % 100 == 0 or i == len(files):
                print(f"{i}/{len(files)}  {time.time() - t0:.0f}s", flush=True)
    idx = pd.DataFrame(rows).sort_values(["cls", "subject", "session_no", "token"])
    # within-subject duplicates: same first 30 s after calibration trim -> keep the longest copy
    idx["duplicate_of"] = None
    for (subj, md5), g in idx[idx.status == "ok"].groupby(["subject", "md5_first30s"]):
        if len(g) > 1:
            keep = g.sort_values(["duration_s", "session_no"], ascending=[False, True]).index[0]
            idx.loc[g.index.drop(keep), "duplicate_of"] = idx.loc[keep, "file"]
    idx.to_csv(os.path.join(OUT_ROOT, "index.csv" if not (a.limit or a.files) else "index_partial.csv"), index=False)
    print(idx.status.value_counts().to_string(), flush=True)

if __name__ == "__main__":
    main()
