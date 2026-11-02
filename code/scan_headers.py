import os, re, csv, glob
ROOT = "/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1"
OUT = "/home/miftah_f/RTX-5090/tuep_fresh/data/edf_headers.csv"

def parse_edf(path):
    with open(path, "rb") as f:
        h = f.read(256)
        ns = int(h[252:256].decode().strip())
        rest = f.read(ns * 256)
    patient = h[8:88].decode("latin1").strip()
    recording = h[88:168].decode("latin1").strip()
    startdate = h[168:176].decode().strip()
    nrec = int(h[236:244].decode().strip())
    recdur = float(h[244:252].decode().strip())
    def field(off, width):
        base = off * ns
        return [rest[base + i*width: base + (i+1)*width].decode("latin1").strip() for i in range(ns)]
    labels = field(0, 16)
    # offsets in units of ns: label16, transducer80, physdim8, physmin8, physmax8, digmin8, digmax8, prefilt80, nsamp8
    o = 16*ns + 80*ns + 8*ns + 8*ns + 8*ns + 8*ns + 8*ns
    prefilt = [rest[o + i*80: o + (i+1)*80].decode("latin1").strip() for i in range(ns)]
    o2 = o + 80*ns
    nsamp = [int(rest[o2 + i*8: o2 + (i+1)*8].decode().strip()) for i in range(ns)]
    return dict(patient=patient, recording=recording, startdate=startdate, nrec=nrec, recdur=recdur,
                ns=ns, labels=labels, prefilt=prefilt, nsamp=nsamp)

rows = []
for cls in ["00_epilepsy", "01_no_epilepsy"]:
    for p in sorted(glob.glob(f"{ROOT}/{cls}/*/*/*/*.edf")):
        parts = p.split("/")
        subj, sess, mont, fn = parts[-4], parts[-3], parts[-2], parts[-1]
        d = parse_edf(p)
        eeg_idx = [i for i, l in enumerate(d["labels"]) if l.upper().startswith("EEG")]
        fs_eeg = sorted(set(round(d["nsamp"][i] / d["recdur"], 3) for i in eeg_idx)) if d["recdur"] > 0 else []
        rows.append(dict(cls=cls, subj=subj, sess=sess, year=sess.split("_")[1], montage=mont, file=fn,
                         token=re.search(r"_t(\d+)", fn).group(1), duration_s=d["nrec"] * d["recdur"],
                         n_sig=d["ns"], n_eeg=len(eeg_idx), fs_eeg="|".join(map(str, fs_eeg)),
                         labels="|".join(d["labels"]), prefilt_first=d["prefilt"][0] if d["prefilt"] else "",
                         patient=d["patient"], recording=d["recording"], startdate=d["startdate"],
                         size_mb=os.path.getsize(p) / 1e6))
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("rows", len(rows), "->", OUT)
