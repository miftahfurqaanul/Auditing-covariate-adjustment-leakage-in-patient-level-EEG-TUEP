"""Confound-controlled subset for the strict protocol (reports/matched_prediction.md).
First session per patient, 250 Hz, montage 01_tcp_ar (all device "X", years 2009-2013), then 1:1 epilepsy/non-epilepsy pairs
with the same recording year and |age difference| <= 5 years (Hungarian assignment per year). Uses metadata and class labels only."""
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
D = "/home/miftah_f/RTX-5090/tuep_fresh/data"; CALIPER = 5


def subject_table():
    T = np.load(f"{D}/p3_patient_table.npz", allow_pickle=True); subj, y = T["subjects"], T["y"]
    i = pd.read_csv(f"{D}/index.csv"); i = i[i.status == "ok"]
    first = i[i.session_no == i.groupby("subject").session_no.transform("min")]
    s = first.groupby("subject").agg(fs=("orig_fs", lambda x: x.mode()[0]), montage=("montage", lambda x: x.mode()[0]),
                                     device=("device", lambda x: x.mode()[0]), year=("year", "first"), age=("age", "first"),
                                     male=("sex", lambda x: float(x.iloc[0] == "M")), dur_min=("duration_s", lambda x: x.sum() / 60),
                                     n_files=("file", "count")).loc[subj]
    s["y"] = y
    return s


def smd(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)


if __name__ == "__main__":
    s = subject_table(); c = s[(s.fs == 250) & (s.montage == "01_tcp_ar")]
    pairs = []
    for yr, g in c.groupby("year"):
        e, n = g[g.y == 1], g[g.y == 0]
        if len(e) == 0 or len(n) == 0:
            continue
        cost = np.abs(e.age.values[:, None] - n.age.values[None, :]).astype(float); cost[cost > CALIPER] = 1e6
        r, k = linear_sum_assignment(cost)
        pairs += [(e.index[a], n.index[b], yr, cost[a, b]) for a, b in zip(r, k) if cost[a, b] <= CALIPER]
    P = pd.DataFrame(pairs, columns=["epilepsy", "non_epilepsy", "year", "age_diff"]); P["pair"] = range(len(P))
    P.to_csv(f"{D}/matched_pairs.csv", index=False)
    m = pd.concat([s.loc[P.epilepsy].assign(pair=P.pair.values), s.loc[P.non_epilepsy].assign(pair=P.pair.values)])
    print(f"candidates (250 Hz, 01_tcp_ar): {len(c)} ({int(c.y.sum())} epilepsy / {int((1 - c.y).sum())} non); matched pairs: {len(P)}")
    print(f"pairs per year: {P.year.value_counts().sort_index().to_dict()}; mean |age diff| {P.age_diff.mean():.2f} y")
    print(f"devices in matched set: {m.device.value_counts().to_dict()}")
    print(f"{'covariate':10s} {'SMD before (all 200)':>22s} {'SMD 250Hz/01_tcp_ar':>21s} {'SMD matched':>12s}")
    for v in ["age", "year", "male", "dur_min", "n_files"]:
        print(f"{v:10s} {smd(s[s.y == 1][v], s[s.y == 0][v]):22.3f} {smd(c[c.y == 1][v], c[c.y == 0][v]):21.3f} {smd(m[m.y == 1][v], m[m.y == 0][v]):12.3f}")
