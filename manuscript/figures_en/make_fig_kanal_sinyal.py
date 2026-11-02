#!/usr/bin/env python3
"""Channel and example-signal figures. CPU only; EDF files are only read (read-only).
Output: figures/gambar_kanal.pdf (channels and preprocessing workflow) and figures/gambar_sinyal.pdf (example 10 s windows).

Selection of examples (fixed before looking at the signals; abnormal appearance was not used):
- First session only (as in the study data); montage 01_tcp_ar; 250 Hz; duration 15--30 minutes; status ok.
- Epilepsy--non-epilepsy pair with the same recording year, device (X), and sex, age difference <= 3 years.
- The pair was drawn at random with numpy default_rng(20261024) from all eligible pairs.
- Preprocessing as in the study: average reference (CAR) over 19 channels, resampling to 200 Hz, zero-phase FIR 0.5--45 Hz; non-overlapping 10 s windows,
  first and last 10 s skipped, windows with channel peak-to-peak > 800 uV rejected.
- Example window = clean window whose line length (mean |difference| per channel) is closest to the median of that recording.
TUEP v2.0.1 contains no seizure annotations; nothing is marked as a seizure."""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import mne
mne.set_log_level("ERROR")
HERE = Path(__file__).resolve().parent
DATA = Path("/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1")
IDX = Path("/home/miftah_f/RTX-5090/tuep_fresh/data/index.csv")
CH19 = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]
SHOW = ["F3", "C3", "P3", "O1", "F4", "C4", "P4", "O2"]
REG = {"Frontal": ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8"], "Central": ["C3", "CZ", "C4"], "Temporal": ["T3", "T4", "T5", "T6"],
       "Parietal": ["P3", "PZ", "P4"], "Occipital": ["O1", "O2"]}
COL = {"Frontal": "#4c78a8", "Central": "#59a14f", "Temporal": "#f28e2b", "Parietal": "#b07aa1", "Occipital": "#e15759"}
POS = {"FP1": (-.28, .82), "FP2": (.28, .82), "F7": (-.70, .52), "F3": (-.36, .50), "FZ": (0, .46), "F4": (.36, .50), "F8": (.70, .52),
       "T3": (-.88, 0), "C3": (-.46, 0), "CZ": (0, 0), "C4": (.46, 0), "T4": (.88, 0), "T5": (-.70, -.52), "P3": (-.36, -.50), "PZ": (0, -.46),
       "P4": (.36, -.50), "T6": (.70, -.52), "O1": (-.28, -.82), "O2": (.28, -.82)}
assert sorted(sum(REG.values(), [])) == sorted(CH19)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "pdf.fonttype": 42})


def pilih_pasangan():
    ix = pd.read_csv(IDX)
    f = ix[ix.groupby("subject")["session_no"].transform("min") == ix["session_no"]].copy()
    f = f[(f.status == "ok") & (f.montage == "01_tcp_ar") & (f.orig_fs == 250) & (f.duration_s.between(900, 1800))]
    e = f[f.cls.str.startswith("00")]; n = f[f.cls.str.startswith("01")]
    pairs = [(a, b) for _, a in e.iterrows() for _, b in n.iterrows() if a.year == b.year and a.device == b.device and a.sex == b.sex and abs(a.age - b.age) <= 3]
    rng = np.random.default_rng(20261024)
    a, b = pairs[int(rng.integers(len(pairs)))]
    return a, b, len(pairs)


def jendela_contoh(row):
    path = DATA / row.cls / row.subject / row.session / "01_tcp_ar" / row.file
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
    names = {c: c.replace("EEG ", "").replace("-REF", "") for c in raw.ch_names}
    raw.rename_channels(names)
    raw.pick([c for c in CH19]); raw.reorder_channels(CH19)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    raw.resample(200.0, verbose=False); raw.filter(0.5, 45.0, fir_design="firwin", verbose=False)
    X = raw.get_data() * 1e6; W = 2000; EDGE = 2000
    starts = list(range(EDGE, X.shape[1] - EDGE - W + 1, W))
    cand = []
    for s in starts:
        w = X[:, s:s + W]
        if (w.max(1) - w.min(1)).max() > 800.0: continue
        ll = np.abs(np.diff(w, axis=1)).mean()
        cand.append((s, ll))
    med = np.median([c[1] for c in cand]); s = min(cand, key=lambda c: abs(c[1] - med))[0]
    w = X[:, s:s + W]
    return w[[CH19.index(c) for c in SHOW]], s / 200.0, len(cand), float(med)


def gambar_kanal(out):
    import textwrap
    fig = plt.figure(figsize=(7.0, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.92, 1.3], left=0.01, right=0.995, top=0.91, bottom=0.02, wspace=0.03)
    ax = fig.add_subplot(gs[0]); ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.18, 1.3)
    ax.add_patch(Circle((0, 0), 1.0, ec="#222", fc="#fafbfc", lw=1.4))
    ax.add_patch(Polygon([(-.11, .985), (0, 1.14), (.11, .985)], closed=True, ec="#222", fc="#fafbfc", lw=1.4))
    for sx in (-1, 1):
        ax.add_patch(Polygon([(sx * 1.0, .12), (sx * 1.08, .10), (sx * 1.10, -.10), (sx * 1.0, -.12)], closed=False, ec="#222", fc="#fafbfc", lw=1.2))
    for r, chs in REG.items():
        for c in chs:
            x, y = POS[c]; sel = c in SHOW
            ax.add_patch(Circle((x, y), .118, ec="#111" if sel else COL[r], fc=COL[r] + "55", lw=2.2 if sel else 1.0))
            ax.text(x, y, c.capitalize() if c in ("FP1", "FP2") else c.title() if len(c) == 2 else c, ha="center", va="center", fontsize=7.2, fontweight="bold")
    ax.text(0, 1.21, "Front", ha="center", fontsize=8); ax.text(0, -1.13, "Back", ha="center", fontsize=8)
    ax.text(-1.30, 0, "Left", ha="center", va="center", fontsize=8, rotation=90); ax.text(1.30, 0, "Right", ha="center", va="center", fontsize=8, rotation=270)
    ax.set_title("(a) 19 channels of the 10–20 system", fontsize=9, fontweight="bold", loc="left")
    bx = fig.add_subplot(gs[1]); bx.axis("off"); bx.set_xlim(0, 1); bx.set_ylim(0, 1)
    bx.set_title("(b) From recording to features", fontsize=9, fontweight="bold", loc="left")
    steps = [("Recording", "Clinical first-session EDF; 19 channels 10–20; 250–400 Hz"),
             ("Preprocessing", "Average reference (CAR); resampling to 200 Hz; zero-phase FIR 0.5–45 Hz"),
             ("Windows", "Non-overlapping 10 s; first and last 10 s skipped; peak-to-peak > 800 µV rejected"),
             ("Features per patient", "I2: relative power of five bands and total power per channel (114 features); P3: 18×18 covariance of five bands; NM: 4 s segments"),
             ("Adjustment and model", "Age, sex, and era removed from the features (fitted on training data only); one prediction per patient")]
    y = 0.97
    for i, (t, d) in enumerate(steps):
        lines = textwrap.wrap(d, 62); h = 0.056 + 0.050 * len(lines)
        bx.add_patch(plt.Rectangle((0.0, y - h), 1.0, h, fc="#eef3f8" if i % 2 == 0 else "#f7f7f7", ec="#9aa7b4", lw=0.7))
        bx.text(0.02, y - 0.038, f"{i + 1}. {t}", fontsize=7.6, fontweight="bold", va="center")
        bx.text(0.02, y - 0.080, "\n".join(lines), fontsize=6.9, va="top", color="#222", linespacing=1.2)
        y -= h + 0.014
    hs = [plt.Line2D([], [], marker="o", ls="", ms=6, mfc=COL[r] + "55", mec=COL[r], label=f"{r} ({len(c)}): {', '.join(x.title() if len(x) == 2 else x for x in c)}") for r, c in REG.items()]
    hs.append(plt.Line2D([], [], marker="o", ls="", ms=6, mfc="white", mec="#111", mew=2, label="Example channels (signal figure)"))
    bx.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.0, -0.03), fontsize=6.0, frameon=False, handletextpad=0.2, labelspacing=0.3, ncol=2, columnspacing=0.8)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def gambar_sinyal(out, info):
    (a, b, wa, wb) = info
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.8), sharex=True)
    t = np.arange(2000) / 200.0
    off = 110.0; ylim = (-off * 0.8, off * (len(SHOW) - 1) + off * 0.9)
    for ax, (row, (w, t0, nwin, med)), col, ttl in [(axes[0], (b, wb), "#1f4e9c", "(a) Non-epilepsy"), (axes[1], (a, wa), "#7a1f3d", "(b) Epilepsy")]:
        for i, ch in enumerate(SHOW):
            ax.plot(t, w[i] + (len(SHOW) - 1 - i) * off, color=col, lw=0.6)
        ax.set_yticks([(len(SHOW) - 1 - i) * off for i in range(len(SHOW))]); ax.set_yticklabels(SHOW, fontsize=8)
        ax.set_ylim(*ylim); ax.set_xlim(0, 10)
        ax.set_title(f"{ttl}: age {int(row.age)} years, {'female' if row.sex == 'F' else 'male'}, year {int(row.year)}, device {row.device}; 10 s window at minute {t0 / 60:.1f}", fontsize=8.2, loc="left", fontweight="bold")
        ax.plot([9.0, 9.0], [ylim[0] + 20, ylim[0] + 70], color="k", lw=1.5); ax.text(8.93, ylim[0] + 45, "50 µV", ha="right", va="center", fontsize=7.5)
        ax.spines["left"].set_visible(False)
    axes[1].set_xlabel("Time within window (s)", fontsize=8.5)
    fig.tight_layout(h_pad=1.0)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    a, b, npairs = pilih_pasangan()
    print("pasangan memenuhi syarat:", npairs, "| epilepsi:", a.subject, a.file, "| non-epilepsi:", b.subject, b.file)
    wa = jendela_contoh(a); wb = jendela_contoh(b)
    print("epilepsi: mulai jendela %.0f s, jendela bersih %d; non-epilepsi: mulai %.0f s, jendela bersih %d" % (wa[1], wa[2], wb[1], wb[2]))
    meta = dict(epilepsi=dict(subject=a.subject, file=a.file, usia=int(a.age), jk=a.sex, tahun=int(a.year), perangkat=a.device, mulai_s=wa[1], jendela_bersih=wa[2]),
                kontrol=dict(subject=b.subject, file=b.file, usia=int(b.age), jk=b.sex, tahun=int(b.year), perangkat=b.device, mulai_s=wb[1], jendela_bersih=wb[2]), pasangan_memenuhi_syarat=npairs)
    json.dump(meta, open(HERE / "gambar_sinyal_meta.json", "w"), indent=1, default=str)
    gambar_kanal(HERE / "gambar_kanal.pdf"); gambar_sinyal(HERE / "gambar_sinyal.pdf", (a, b, wa, wb))
