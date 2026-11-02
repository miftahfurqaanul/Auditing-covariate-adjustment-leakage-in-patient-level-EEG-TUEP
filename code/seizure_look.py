"""Look at 5 epilepsy recordings and rank 5-s windows by "seizure-likeness" (reports/seizure_look_prediction.md). CPU only, no labels, no model."""
import os, numpy as np, pandas as pd, h5py, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
D = "/home/miftah_f/RTX-5090/tuep_fresh/data"; OUT = "/home/miftah_f/RTX-5090/tuep_fresh/reports/seizure_look"; os.makedirs(OUT, exist_ok=True)
FS, WIN = 200, 1000
CH = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ", "C4", "T4", "T5", "P3", "PZ", "P4", "T6", "O1", "O2"]
BIP = [("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"), ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
       ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"), ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"), ("FZ", "CZ"), ("CZ", "PZ")]

idx = pd.read_csv(f"{D}/index.csv"); idx = idx[(idx.status == "ok") & (idx.label == 1) & (idx.duration_s >= 1200)]
best = idx.sort_values("duration_s", ascending=False).groupby("subject").head(1)
pick = best.sample(5, random_state=0).reset_index(drop=True)
print(f"eligible epilepsy patients: {len(best)}; chosen (seed 0):"); print(pick[["subject", "file", "age", "sex", "year", "device", "duration_s"]].to_string())

def robust_z(v):
    m = np.median(v); s = 1.4826 * np.median(np.abs(v - m)) + 1e-9
    return (v - m) / s

def window_features(x):
    n = x.shape[1] // WIN; w = x[:, :n * WIN].reshape(19, n, WIN).transpose(1, 0, 2).astype(np.float32)
    ll = np.median(np.abs(np.diff(w, axis=2)).mean(2), axis=1)
    f, P = welch(w, fs=FS, nperseg=1000, axis=2); P = P.mean(1)                           # [n, freq]
    band = (f >= 2) & (f <= 25); fb = f[band]; Pb = P[:, band]
    pk = fb[np.argmax(Pb, 1)]; rhythm = np.array([P[i][(f >= pk[i] - 1) & (f <= pk[i] + 1)].sum() / P[i][(f >= 1) & (f <= 45)].sum() for i in range(n)])
    emg = P[:, (f >= 30) & (f <= 45)].sum(1) / P[:, (f >= 1) & (f <= 45)].sum(1)
    amp = np.sqrt((w ** 2).mean(2)).max(1)
    return w, ll, rhythm, emg, pk, amp

def bipolar(w):
    ix = {c: i for i, c in enumerate(CH)}
    return np.stack([w[ix[a]] - w[ix[b]] for a, b in BIP])

def pick_far(order, k, gap=6):
    chosen = []
    for i in order:
        if all(abs(i - j) >= gap for j in chosen):
            chosen.append(i)
        if len(chosen) == k:
            break
    return chosen

def draw(ax, seg, title, scale=100):
    t = np.arange(seg.shape[1]) / FS
    for i, tr in enumerate(seg):
        ax.plot(t, tr / scale + (len(seg) - 1 - i), lw=0.5, color="k")
    ax.set_yticks(range(len(seg))); ax.set_yticklabels([f"{a}-{b}" for a, b in BIP][::-1], fontsize=5)
    ax.set_xlim(0, t[-1]); ax.set_title(title, fontsize=7); ax.tick_params(axis="x", labelsize=6)

rows = []
for _, r in pick.iterrows():
    with h5py.File(f"{D}/{r.h5_path}") as h:
        x = h["eeg"][()].astype(np.float32)
    w, ll, rh, emg, pk, amp = window_features(x); n = len(ll)
    score = robust_z(np.log(ll)) + robust_z(rh); score[emg > np.percentile(emg, 90)] = -np.inf
    cand = pick_far(np.argsort(-score), 3)
    bg_ok = np.where(emg < np.median(emg))[0]; bg = pick_far(bg_ok[np.argsort(np.abs(np.log(ll[bg_ok]) - np.median(np.log(ll))))], 3, gap=12)
    fig = plt.figure(figsize=(15, 13)); gs = fig.add_gridspec(3, 3, height_ratios=[0.9, 2, 2], hspace=0.35, wspace=0.15)
    a0 = fig.add_subplot(gs[0, :]); tt = np.arange(n) * 5 / 60
    a0.plot(tt, robust_z(np.log(ll)), lw=0.7, label="line length (z)"); a0.plot(tt, robust_z(rh), lw=0.7, label="ritmisitas (z)")
    for c in cand:
        a0.axvline(c * 5 / 60, color="r", lw=1)
    for b in bg:
        a0.axvline(b * 5 / 60, color="g", lw=1)
    a0.set_xlabel("menit"); a0.legend(fontsize=6, loc="upper right"); a0.tick_params(labelsize=6)
    a0.set_title(f"{r.subject}  umur {r.age} {r.sex}  tahun {r.year}  {r.device}  {r.duration_s / 60:.0f} menit   |  merah = kandidat mirip kejang, hijau = latar normal", fontsize=8)
    for j, c in enumerate(cand):
        s0 = max(0, c - 1); seg = bipolar(w[s0:c + 2].transpose(1, 0, 2).reshape(19, -1))
        draw(fig.add_subplot(gs[1, j]), seg, f"KANDIDAT {j + 1}: menit {c * 5 / 60:.1f} (tengah = jendela 5 s; 15 s total)\nline length z={robust_z(np.log(ll))[c]:.1f}, ritmisitas {rh[c]:.2f} @ {pk[c]:.1f} Hz, otot {emg[c]:.2f}")
    for j, b in enumerate(bg):
        s0 = max(0, b - 1); seg = bipolar(w[s0:b + 2].transpose(1, 0, 2).reshape(19, -1))
        draw(fig.add_subplot(gs[2, j]), seg, f"LATAR {j + 1}: menit {b * 5 / 60:.1f}\nline length z={robust_z(np.log(ll))[b]:.1f}, ritmisitas {rh[b]:.2f} @ {pk[b]:.1f} Hz, otot {emg[b]:.2f}")
    fig.savefig(f"{OUT}/{r.subject}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    zl, zr = robust_z(np.log(ll)), robust_z(rh)
    for kind, ids in [("kandidat", cand), ("latar", bg)]:
        for i in ids:
            rows.append(dict(subject=r.subject, kind=kind, minute=round(i * 5 / 60, 2), window=int(i), ll_z=round(float(zl[i]), 2), rhythm=round(float(rh[i]), 3), peak_hz=float(pk[i]), emg=round(float(emg[i]), 3), max_rms_uV=round(float(amp[i]), 1)))
    print(f"{r.subject}: {n} windows, line-length z max {zl.max():.1f}, rhythmicity z max {zr.max():.1f}, windows with LL z>5: {(zl > 5).sum()}")
pd.DataFrame(rows).to_csv(f"{OUT}/windows.csv", index=False); print(pd.DataFrame(rows).to_string(index=False))
