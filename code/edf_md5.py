"""MD5 checksum of every TUEP v2.0.1 EDF file used in this project -> data/edf_md5.csv (tracked in git)."""
import glob, hashlib, os, pandas as pd
from multiprocessing import Pool
ROOT = "/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1"
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return os.path.relpath(p, ROOT), os.path.getsize(p), h.hexdigest()
if __name__ == "__main__":
    files = sorted(glob.glob(f"{ROOT}/0[01]_*/*/*/*/*.edf"))
    with Pool(6) as pool:
        rows = pool.map(md5, files, chunksize=8)
    pd.DataFrame(rows, columns=["path", "bytes", "md5"]).to_csv("/home/miftah_f/RTX-5090/tuep_fresh/data/edf_md5.csv", index=False)
    print(len(rows), "files")
