"""Double checks for tang_ps.py before any feature extraction (montage, signature maths, real-file loading, filter)."""
import re, numpy as np, pandas as pd, mne
from scipy.signal import freqz
import tang_ps as tp

ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)

# 1. montage constants vs the official TUH definition files
DOCS = "/home/langgeng/Documents/TUSZ/DOCS"
def parse(f):
    return [tuple(re.findall(r"EEG (\w+)-", l)) for l in open(f) if l.startswith("montage")]
for m, suf in [("01_tcp_ar", "REF"), ("02_tcp_le", "LE")]:
    ok(parse(f"{DOCS}/{m}_montage.txt") == tp.TCP22, f"{m}: 22 pairs identical to official file, same order")
ar_a = parse(f"{DOCS}/03_tcp_ar_a_montage.txt")
ok(ar_a == [p for p in tp.TCP22 if "A1" not in p and "A2" not in p], "03_tcp_ar_a = TCP22 without A1-T3 and T4-A2 (20 ch) -> cannot give 22 channels")

# 2. order-2 signature maths
rng = np.random.default_rng(0)
def loop_sig(path):                                   # independent algorithm: Chen's identity one increment at a time
    d = path.shape[1]; s1 = np.zeros(d); s2 = np.zeros((d, d))
    for k in range(1, len(path)):
        dx = path[k] - path[k - 1]; s2 += np.outer(s1, dx) + 0.5 * np.outer(dx, dx); s1 += dx
    return np.concatenate([s1, s2.ravel()])
P = rng.normal(size=(5, 22, 250)).cumsum(2)
S = tp.signature2(P)
ok(S.shape == (5, 506), f"feature dimension {S.shape[1]} = 22 + 22^2 = 506")
ok(np.allclose(S, np.stack([loop_sig(p.T) for p in P])), "vectorised == incremental Chen-identity loop (5 random 22-ch paths)")
s1, s2 = S[:, :22], S[:, 22:].reshape(5, 22, 22)
ok(np.allclose(s2 + s2.transpose(0, 2, 1), np.einsum("ni,nj->nij", s1, s1)), "shuffle identity: S2 + S2^T = S1 (x) S1")
a, b = P[0][:, :120], P[0][:, 119:]
Sa, Sb, Sab = tp.signature2(a[None])[0], tp.signature2(b[None])[0], tp.signature2(P[0][None])[0]
chen2 = Sa[22:].reshape(22, 22) + Sb[22:].reshape(22, 22) + np.outer(Sa[:22], Sb[:22])
ok(np.allclose(Sab[22:].reshape(22, 22), chen2), "Chen identity for concatenated paths")
q = rng.normal(size=(3, 40)).cumsum(1); fine = np.concatenate([np.linspace(q[:, k], q[:, k + 1], 2001)[:-1] for k in range(39)] + [q[:, -1:].T])
dx = np.diff(fine, axis=0); rs = (fine[:-1] - fine[0]).T @ dx    # left Riemann sum of iterated integral on a 2000x finer grid
ok(np.allclose(rs, tp.signature2(q[None])[0, 3:].reshape(3, 3), rtol=0, atol=2e-3 * np.abs(rs).max()), "matches numerical iterated integral (fine grid, tolerance 0.2%)")

# 3. real files: bipolar derivation and loading
h = pd.read_csv("/home/miftah_f/RTX-5090/tuep_fresh/data/edf_headers.csv")
root = "/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1"
for m in ["01_tcp_ar", "02_tcp_le"]:
    r = h[(h.fs_eeg == 250) & (h.montage == m)].iloc[0]
    path = f"{root}/{r.cls}/{r.subj}/{r.sess}/{r.montage}/{r.file}"
    x, fs = tp.load_tcp(path, m)
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR"); suf = tp.SUFFIX[m]
    manual = (raw.get_data(picks=[f"EEG A1-{suf}"])[0] - raw.get_data(picks=[f"EEG T3-{suf}"])[0]) * 1e6
    ok(np.allclose(x[8], manual) and x.shape[0] == 22 and fs == 250, f"{m} {r.file}: shape {x.shape}, fs {fs}, channel 8 == A1-T3 computed by hand")
    print(f"     raw bipolar std (uV) per channel: median {np.median(x.std(1)):.1f}, min {x.std(1).min():.1f}, max {x.std(1).max():.1f}")
    seg = tp.segments(tp.bandpass(x, fs), fs); F = tp.signature2(seg)
    ok(seg.shape[1:] == (22, 250) and np.isfinite(F).all(), f"segments {seg.shape}, features {F.shape}, all finite")
    print(f"     feature scale before standardisation: |S1| median {np.median(np.abs(F[:, :22])):.2f}, |S2| median {np.median(np.abs(F[:, 22:])):.2f}")

# 4. FIR band-pass response
for fs in [250.0, 256.0]:
    h_ = mne.filter.create_filter(None, fs, 0.5, 50.0, method="fir", fir_design="firwin", phase="zero", h_trans_bandwidth=5.0, verbose="ERROR")
    f, g = freqz(h_, worN=2 ** 16, fs=fs); g = np.abs(g)
    at = {q: g[np.argmin(np.abs(f - q))] for q in [0.1, 0.25, 0.5, 1, 10, 45, 50, 55, 60, 100]}
    print(f"     fs {fs}: taps {len(h_)}, gain " + ", ".join(f"{k} Hz {v:.3f}" for k, v in at.items()))
    ok(min(at[0.5], at[1], at[10], at[45], at[50]) > 0.99 and max(at[55], at[60], at[100]) < 0.005 and g[0] < 1e-3, f"fs {fs}: design spec met (passband 0.5-50 Hz >= 0.99, stopband >= 55 Hz < 0.005, DC {g[0]:.1e}); 0.1 Hz gain reported as is")
