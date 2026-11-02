"""Audit of the Zenodo 1252141 Emotiv dataset (Guinea-Bissau + Nigeria): file/subject mapping,
signal quality, and label confounds. Read-only; writes an index CSV next to the raw data."""
import os, glob, gzip, re, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats
Z = "/home/miftah_f/RTX-5090/tuep_fresh/external/zenodo_1252141"
EEG = ["AF3", "AF4", "F3", "F4", "F7", "F8", "FC5", "FC6", "O1", "O2", "P7", "P8", "T7", "T8"]
FS = 128

def load(path):
    d = pd.read_csv(path, compression="gzip")
    return d

def audit_file(path):
    d = load(path)
    x = d[EEG].values.astype(float)
    cnt = d["COUNTER"].values
    steps = (np.diff(cnt) % 129)                       # Emotiv counter runs 0..128 (128 = gyro packet) or 0..127
    cq = d[[f"CQ_{c}" for c in EEG]].values
    xc = x - np.median(x, 0)
    win = xc[: len(xc) // FS * FS].reshape(-1, FS, 14)
    ptp = win.max(1) - win.min(1)
    return dict(path=os.path.relpath(path, Z), n=len(d), dur_s=len(d) / FS,
                interp_frac=float(d["INTERPOLATED"].mean()), counter_max=int(cnt.max()),
                counter_jumps=int(((steps != 1) & (steps != 0)).sum()),
                cq_mean=float(cq.mean()), cq_frac_good=float((cq >= 4).mean()), cq_frac_bad=float((cq <= 1).mean()),
                dc_median=float(np.median(x)), std_med=float(np.median(xc.std(0))), std_max=float(xc.std(0).max()),
                n_flat=int((xc.std(0) < 1).sum()), frac_win_ptp_gt_200=float((ptp.max(1) > 200).mean()),
                gyro_sd=float(d[["GYROX", "GYROY"]].values.std(0).mean()))

if __name__ == "__main__":
    gb = pd.read_csv(f"{Z}/metadata_guineabissau.csv"); ng = pd.read_csv(f"{Z}/metadata_nigeria.csv")
    print("GB metadata rows", len(gb), "| unique subjects", gb["subject.id"].nunique(), "| groups", gb.Group.value_counts().to_dict())
    print("NG metadata rows", len(ng), "| unique subjects", ng["subject.id"].nunique(), "| groups", ng.Group.value_counts().to_dict())
    print("NG sessions per subject:", ng.groupby("subject.id").size().value_counts().to_dict())
    gb["file"] = gb["subject.id"].map(lambda s: f"raw/gb/EEGs_Guinea-Bissau/signal-{s}.csv.gz")
    ng["file"] = "raw/ng/EEGs_Nigeria/" + ng["csv.file"].astype(str)
    gb_exist = gb.file.map(lambda f: os.path.exists(f"{Z}/{f}")); ng_exist = ng.file.map(lambda f: os.path.exists(f"{Z}/{f}"))
    print("GB metadata rows with a file:", gb_exist.sum(), "| NG:", ng_exist.sum())
    print("NG rows without file:", ng[~ng_exist][["subject.id", "session.id", "csv.file", "Group", "remarks"]].to_string(index=False))
    all_ng_files = set(os.path.relpath(p, Z) for p in glob.glob(f"{Z}/raw/ng/EEGs_Nigeria/*.gz"))
    print("NG files not referenced by metadata:", sorted(all_ng_files - set(ng.file))[:20], len(all_ng_files - set(ng.file)))
    with Pool(12) as pool:
        q = pd.DataFrame(pool.map(audit_file, sorted(glob.glob(f"{Z}/raw/*/*/*.gz"))))
    gb2 = gb.rename(columns={"Eyes.condition": "eyes", "Remarks": "remarks"}).assign(country="GB", group=gb.Group.str.lower())
    ng2 = ng.rename(columns={"first_condition": "eyes"}).assign(country="NG", group=ng.Group.str.lower())
    meta = pd.concat([gb2[["subject.id", "country", "group", "eyes", "remarks", "recordedPeriod", "startTime", "file"]],
                      ng2[["subject.id", "country", "group", "eyes", "remarks", "recordedPeriod", "startTime", "file", "session.id"]]])
    meta = meta.merge(q, left_on="file", right_on="path", how="left")
    meta.to_csv(f"{Z}/zenodo_index.csv", index=False)
    print("\nfiles audited:", len(q), "| sampling check: counter max", q.counter_max.value_counts().to_dict())
    print("\n== group x country ==\n", pd.crosstab(meta.group, meta.country))
    print("\n== eyes condition x group ==\n", pd.crosstab([meta.country, meta.eyes], meta.group))
    num = ["dur_s", "interp_frac", "counter_jumps", "cq_mean", "cq_frac_bad", "std_med", "frac_win_ptp_gt_200", "gyro_sd", "n_flat"]
    print("\n== signal quality by country x group (median) ==")
    print(meta.groupby(["country", "group"])[num].median().round(3).T)
    print("\n== Mann-Whitney p (epilepsy vs control) within country ==")
    for c in ["GB", "NG"]:
        m = meta[meta.country == c]
        print(c, {k: f"{stats.mannwhitneyu(m[m.group=='epilepsy'][k].dropna(), m[m.group=='control'][k].dropna()).pvalue:.2g}" for k in num})
    print("\nremarks (non-empty) by group:", meta[meta.remarks.notna() & (meta.remarks.astype(str) != "NA")].groupby("group").size().to_dict())
