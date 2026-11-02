"""Check processed recordings against an independent MNE implementation, plus PSD and a visual."""
import sys, h5py, mne, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import welch
from preprocess import CHANNELS, RAW_ROOT, OUT_ROOT, FS_OUT, BAND
idx = pd.read_csv(f"{OUT_ROOT}/{sys.argv[1] if len(sys.argv) > 1 else 'index_partial.csv'}")
idx = idx[idx.status == "ok"]
if len(sys.argv) > 2:
    idx = idx.sample(int(sys.argv[2]), random_state=0)
res = []
for _, r in idx.iterrows():
    with h5py.File(f"{OUT_ROOT}/{r.h5_path}") as f:
        y = f["eeg"][()].astype(np.float64); src = f.attrs["source_edf"]; lead = f.attrs["calib_lead_s"]
    raw = mne.io.read_raw_edf(f"{RAW_ROOT}/{src}", preload=False, verbose="ERROR")
    names = [next(n for n in (f"EEG {c}-REF", f"EEG {c}-LE") if n in raw.ch_names) for c in CHANNELS]
    raw.pick(names).crop(tmin=float(lead), tmax=min(raw.times[-1], lead + y.shape[1] / FS_OUT)).load_data(verbose="ERROR")
    raw.set_eeg_reference("average", verbose="ERROR")
    raw.resample(FS_OUT, method="polyphase", verbose="ERROR").filter(*BAND, verbose="ERROR")
    ref = raw.get_data() * 1e6
    E = 10 * FS_OUT                                          # ignore filter transients at both edges
    n = min(ref.shape[1], y.shape[1]) - 2 * E
    a, b = ref[:, E:E + n], y[:, E:E + n]
    corr = [np.corrcoef(a[i], b[i])[0, 1] for i in range(19)]
    f_, P = welch(y, fs=FS_OUT, nperseg=4 * FS_OUT, axis=1); Pm = P.mean(0)
    band = lambda lo, hi: Pm[(f_ >= lo) & (f_ < hi)].sum()
    res.append(dict(file=r.file, fs=r.orig_fs, montage=r.montage, min_corr=min(corr),
                    rel_rms_err=float(np.sqrt(((a - b) ** 2).mean()) / np.sqrt((a ** 2).mean())),
                    p55_65_over_1_45=band(55, 65) / band(1, 45), p45_100_over_1_45=band(45, 100) / band(1, 45)))
print(pd.DataFrame(res).round(5).to_string(index=False))
# visual: raw vs processed, 10 s, for the first two files
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
for k, (_, r) in enumerate(idx.head(2).iterrows()):
    with h5py.File(f"{OUT_ROOT}/{r.h5_path}") as f:
        y = f["eeg"][:, 60 * FS_OUT:70 * FS_OUT].astype(float); lead = f.attrs["calib_lead_s"]
    raw = mne.io.read_raw_edf(f"{RAW_ROOT}/{f.filename if False else src}", preload=False, verbose="ERROR") if False else None
    rr = mne.io.read_raw_edf(f"{RAW_ROOT}/{h5py.File(f'{OUT_ROOT}/{r.h5_path}').attrs['source_edf']}", preload=False, verbose="ERROR")
    names = [next(n for n in (f"EEG {c}-REF", f"EEG {c}-LE") if n in rr.ch_names) for c in CHANNELS]
    fs = rr.info["sfreq"]; s0 = int((lead + 60) * fs)
    x = rr.get_data(picks=names, start=s0, stop=s0 + int(10 * fs)) * 1e6
    for ax, sig, sr, title in [(axes[k, 0], x, fs, f"RAW {r.file} ({fs:.0f} Hz, {r.montage})"), (axes[k, 1], y, FS_OUT, "PROCESSED (CAR, 200 Hz, 0.5-45 Hz)")]:
        t = np.arange(sig.shape[1]) / sr
        for i in range(19):
            ax.plot(t, sig[i] - sig[i].mean() - i * 150, lw=0.5, color="k")
        ax.set_yticks([-i * 150 for i in range(19)]); ax.set_yticklabels(CHANNELS, fontsize=7); ax.set_title(title, fontsize=9); ax.set_xlabel("s")
plt.tight_layout(); plt.savefig("/home/miftah_f/RTX-5090/tuep_fresh/reports/validate_example.png", dpi=80)
