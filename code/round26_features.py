"""Round 26 label-free features (reports/round26_prediction.md, commit 69df653). CPU only; EDF files are only read.
1. Session-level band-power features for ALL sessions (median of the clean windows of the session's recordings, as for the first session),
   with per-session age (EDF header) and era (<=2009, 2010, 2011, 2012, 2013, 2014-15, >=2016)  -> E1.
2. First-session windows: photic-stimulation blocks from the EDF 'PHOTIC' channel (active second = per-second sd > 0; gaps <= 30 s merged;
   block extended 10 s after its end; window j of an H5 file = EDF seconds [lead + 10 + 10j, lead + 20 + 10j), lead = calib_lead_s),
   posterior alpha index per window (mean over O1, O2, P3, P4, PZ of log rel alpha - log rel theta)  -> E2 patient features.
Usage: python round26_features.py"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import glob, time, numpy as np, pandas as pd, h5py, mne
from multiprocessing import Pool
from scipy.signal import welch
import evalkit as ek, features as fe

mne.set_log_level("ERROR")
D = ek.D; OUT = f"{D}/round26"; os.makedirs(OUT, exist_ok=True); RAW = "/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1"
POST = ["O1", "O2", "P3", "P4", "PZ"]; MIN_HALF = 10; GAP_S, TAIL_S = 30, 10


def era_of(y):
    return "<=2009" if y <= 2009 else (str(int(y)) if y <= 2013 else ("2014-15" if y <= 2015 else ">=2016"))


def photic_blocks(edf_path):
    """Returns (status, list of (start_s, end_s) EDF-time blocks, n active seconds)."""
    raw = mne.io.read_raw_edf(edf_path, preload=False); ch = [c for c in raw.ch_names if "PHOTIC" in c.upper()]
    if not ch:
        return "no_channel", [], 0
    fs = raw.info["sfreq"]; x = raw.get_data(picks=[ch[0]])[0]; n = int(len(x) // fs)
    act = x[:int(n * fs)].reshape(n, -1).std(1) > 0
    if not act.any():
        return "flat", [], 0
    sec = np.where(act)[0]; blocks = []; s0 = p = sec[0]
    for t in sec[1:]:
        if t - p - 1 > GAP_S:
            blocks.append((s0, p + 1 + TAIL_S)); s0 = t
        p = t
    blocks.append((s0, p + 1 + TAIL_S)); return "active", blocks, int(act.sum())


def window_positions(h5_path):
    """Index j of the clean windows of an H5 file, in the order of features.extract (same EDGE/WIN/PTP rule)."""
    with h5py.File(f"{fe.D}/{h5_path}") as f:
        x = f["eeg"][()].astype(np.float32)
    x = x[:, fe.EDGE:x.shape[1] - fe.EDGE]; n = x.shape[1] // fe.WIN
    w = x[:, :n * fe.WIN].reshape(19, n, fe.WIN).transpose(1, 0, 2); keep = (w.max(2) - w.min(2)).max(1) <= fe.PTP_MAX
    return np.where(keep)[0], w[keep]


def first_session_job(args):
    h5_path, edf_path, lead = args
    pos, w = window_positions(h5_path); status, blocks, nact = photic_blocks(edf_path)
    t0 = lead + fe.EDGE / fe.FS + pos * fe.WIN / fe.FS; t1 = t0 + fe.WIN / fe.FS
    ph = np.zeros(len(pos), bool)
    for a, b in blocks:
        ph |= (t0 < b) & (t1 > a)
    peak = np.full(len(pos), np.nan)
    if ph.any():                                                       # occipital spectral peakiness, for the detection check only
        f_, P = welch(w[:, [fe.CH.index("O1"), fe.CH.index("O2")]], fs=fe.FS, nperseg=400, axis=2); P = P.mean(1); band = (f_ >= 3) & (f_ <= 30)
        peak = P[:, band].max(1) / np.median(P[:, band], 1)
    return h5_path, status, len(blocks), nact, ph, peak, len(pos)


if __name__ == "__main__":
    t_start = time.time()
    z = np.load(f"{D}/features_bandpower_windows.npz", allow_pickle=True); Xw, win_h5, names = z["X"], z["h5_path"], list(z["names"])
    rec = pd.read_csv(f"{D}/features_bandpower_recordings.csv"); rec = rec[rec.n_clean > 0].reset_index(drop=True)
    hdr = pd.read_csv(f"{D}/edf_headers.csv"); hdr["age"] = hdr.patient.str.extract(r"Age:(-?\d+)")[0].astype(float)
    rec = rec.merge(hdr[["file", "age", "year"]], on="file"); idx = pd.read_csv(f"{D}/index.csv").set_index("file")
    pos = {h: i for i, h in enumerate(rec.h5_path)}; wr = np.array([pos.get(h, -1) for h in win_h5]); Xw, wr = Xw[wr >= 0], wr[wr >= 0]
    T = np.load(f"{D}/p3_patient_table.npz", allow_pickle=True); SUBJ = list(T["subjects"]); BP_ref = T["bandpower"]
    first = (rec.groupby("subject").session_no.transform("min") == rec.session_no).values
    # --- reproduce the I2 patient features exactly (sanity) -------------------------------------------------------------
    fw = {s: np.where(np.isin(wr, np.where(first & (rec.subject.values == s))[0]))[0] for s in SUBJ}
    BP = np.vstack([np.median(Xw[fw[s]], 0) for s in SUBJ])
    print(("PASS" if np.array_equal(BP, BP_ref) else "FAIL") + " first-session patient features reproduce p3_patient_table bandpower exactly")
    # --- 1. session table (all sessions) ---------------------------------------------------------------------------------
    rows, SX = [], []
    for (s, sn), g in rec.groupby(["subject", "session_no"]):
        wi = np.where(np.isin(wr, g.index.values))[0]
        rows.append(dict(subject=s, session_no=sn, year=int(g.year.iloc[0]), era=era_of(g.year.iloc[0]), age=float(g.age.iloc[0]), first=bool(first[g.index[0]]), n_win=len(wi)))
        SX.append(np.median(Xw[wi], 0))
    ses = pd.DataFrame(rows); SX = np.vstack(SX)
    f_idx = ses[ses["first"]].set_index("subject").loc[SUBJ].index; ok = np.allclose(SX[ses["first"].values][[list(ses[ses['first']].subject).index(s) for s in SUBJ]], BP)
    print(("PASS" if ok else "FAIL") + f" session table: first-session rows equal the I2 patient features; {len(ses)} sessions, {ses.subject.nunique()} patients")
    # --- 2. first-session windows: photic + alpha index -----------------------------------------------------------------
    edf_map = {os.path.basename(p): p for p in glob.glob(f"{RAW}/0*_*epilepsy/**/*.edf", recursive=True)}
    fr = rec[first]; jobs = [(h, edf_map[f], float(idx.loc[f, "calib_lead_s"])) for h, f in zip(fr.h5_path, fr.file)]
    with Pool(12) as pool:
        res = {r[0]: r for r in pool.map(first_session_job, jobs, chunksize=2)}
    PH = np.zeros(len(Xw), bool); PEAK = np.full(len(Xw), np.nan); stat = []
    for h, status, nb, nact, ph, peak, npos in (res[h] for h in fr.h5_path):
        wi = np.where(wr == pos[h])[0]; assert npos == len(wi), f"window count mismatch {h}: {npos} vs {len(wi)}"
        PH[wi] = ph; PEAK[wi] = peak; stat.append(dict(h5_path=h, photic_status=status, n_blocks=nb, active_s=nact, photic_windows=int(ph.sum()), windows=len(wi)))
    stat = pd.DataFrame(stat).merge(fr[["h5_path", "subject", "file"]], on="h5_path")
    ci = lambda c: names.index(c)
    AIDX = np.mean([Xw[:, ci(f"rel_alpha_{c}")] - Xw[:, ci(f"rel_theta_{c}")] for c in POST], 0)
    # explicit second implementation of the photic rule for one active file
    act_file = stat[stat.photic_status == "active"].iloc[0]
    raw = mne.io.read_raw_edf(edf_map[act_file.file], preload=False); chn = [c for c in raw.ch_names if "PHOTIC" in c.upper()][0]
    fs = raw.info["sfreq"]; x = raw.get_data(picks=[chn])[0]; secs = [t for t in range(int(len(x) // fs)) if np.std(x[int(t * fs):int((t + 1) * fs)]) > 0]
    blocks2 = []
    for t in secs:
        if blocks2 and t - blocks2[-1][1] <= GAP_S:
            blocks2[-1][1] = t
        else:
            blocks2.append([t, t])
    blocks2 = [(a, b + 1 + TAIL_S) for a, b in blocks2]; lead = float(idx.loc[act_file.file, "calib_lead_s"])
    p2, _ = window_positions(act_file.h5_path); ph2 = np.array([any((lead + 10 + 10 * j) < b and (lead + 20 + 10 * j) > a for a, b in blocks2) for j in p2])
    print(("PASS" if np.array_equal(ph2, PH[wr == pos[act_file.h5_path]]) else "FAIL") + f" photic rule re-implemented with explicit loops on {act_file.file} ({ph2.sum()} photic windows)")
    # --- E2 patient feature variants --------------------------------------------------------------------------------------
    def half(wi, top=True):
        if len(wi) < 2 * MIN_HALF:
            return wi
        o = wi[np.argsort(-AIDX[wi] if top else AIDX[wi], kind="stable")]; k = int(np.ceil(len(wi) / 2)) if top else len(wi) // 2
        return o[:k]
    var = {"SM": [], "PHOTIC_ONLY": [], "STATE_ONLY": [], "ALERT": [], "DROWSY": []}; nsel = []
    for s in SUBJ:
        wi = fw[s]; wn = wi[~PH[wi]]
        if len(wn) == 0:
            wn = wi
        var["SM"].append(np.median(Xw[half(wn)], 0)); var["PHOTIC_ONLY"].append(np.median(Xw[wn], 0)); var["STATE_ONLY"].append(np.median(Xw[half(wi)], 0))
        var["ALERT"].append(np.median(Xw[half(wn)], 0)); var["DROWSY"].append(np.median(Xw[half(wn, top=False)], 0)); nsel.append((len(wi), len(wn), len(half(wn))))
    var = {k: np.vstack(v) for k, v in var.items()}; nsel = np.array(nsel)
    print(f"windows per patient (first session): all median {np.median(nsel[:, 0]):.0f}; after photic exclusion {np.median(nsel[:, 1]):.0f}; SM selection {np.median(nsel[:, 2]):.0f} (min {nsel[:, 2].min()})")
    print("photic status of first-session recordings:", stat.photic_status.value_counts().to_dict(), "| blocks per active file:", stat[stat.photic_status == 'active'].n_blocks.value_counts().to_dict())
    print(f"active seconds per active file: median {stat[stat.photic_status == 'active'].active_s.median():.0f}, max {stat.active_s.max()}; photic windows total {int(PH.sum())} of {sum(len(fw[s]) for s in SUBJ)}")
    # detection check: occipital peakiness inside vs outside blocks, per recording with photic windows
    chk = []
    for h in stat[stat.photic_windows > 0].h5_path:
        wi = np.where(wr == pos[h])[0]; a, b = PEAK[wi][PH[wi]], PEAK[wi][~PH[wi]]
        if len(a) and len(b) and np.isfinite(b).any():
            chk.append(np.median(a) > np.median(b))
    print(f"photic detection check: occipital spectral peakiness higher inside the blocks in {sum(chk)}/{len(chk)} recordings")
    first_win_patient = np.full(len(Xw), -1)
    for i, s in enumerate(SUBJ):
        first_win_patient[fw[s]] = i
    fwin = np.where(first_win_patient >= 0)[0]
    # session-level alert-half features (for the secondary WP+SM variant): photic exclusion is only possible for first sessions
    PHall = PH.copy(); SX_SM = []
    for (sname, sn), g in rec.groupby(["subject", "session_no"]):
        wi = np.where(np.isin(wr, g.index.values))[0]; wn = wi[~PHall[wi]] if len(wi[~PHall[wi]]) else wi; SX_SM.append(np.median(Xw[half(wn)], 0))
    SX_SM = np.vstack(SX_SM); fm = ses["first"].values
    okc = np.allclose(SX_SM[fm][[list(ses[fm].subject).index(s) for s in SUBJ]], var["SM"])
    print(("PASS" if okc else "FAIL") + " session alert-half features: first-session rows equal the SM patient features")
    np.savez(f"{OUT}/features.npz", SUBJ=np.array(SUBJ), BP=BP, ses_subject=ses.subject.values, ses_session=ses.session_no.values, ses_year=ses.year.values,
             ses_era=ses.era.values, ses_age=ses.age.values, ses_first=ses["first"].values, ses_nwin=ses.n_win.values, SX=SX, SX_SM=SX_SM,
             win_patient=first_win_patient[fwin], win_X=Xw[fwin], win_photic=PH[fwin], win_aidx=AIDX[fwin], win_peak=PEAK[fwin], **{f"BP_{k}": v for k, v in var.items()})
    stat.to_csv(f"{OUT}/photic_status.csv", index=False)
    print(f"saved {OUT}/features.npz | {(time.time() - t_start) / 60:.2f} min")
