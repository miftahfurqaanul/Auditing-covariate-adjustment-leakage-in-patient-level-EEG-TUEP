import os, glob, json, re, subprocess, numpy as np, pandas as pd
from scipy import stats
os.environ["CUDA_VISIBLE_DEVICES"] = ""
RAW = "/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1"; T = "/home/miftah_f/RTX-5090/tuep_fresh"; FAIL = []
def chk(ok, m): print(("PASS " if ok else "FAIL ") + m); (None if ok else FAIL.append(m))
fl = json.load(open(f"{T}/reports/tripod_ai/flow_counts.json"))
print("== flow from the raw directory tree")
for cdir, name in (("00_epilepsy", "Epilepsy"), ("01_no_epilepsy", "No epilepsy")):
    edf = glob.glob(f"{RAW}/{cdir}/*/*/*/*.edf"); subj = {p.split("/")[-4] for p in edf}; ses = {(p.split("/")[-4], p.split("/")[-3]) for p in edf}
    first = {}
    for s_, se in ses:
        n = int(se.split("_")[0][1:]); first[s_] = min(first.get(s_, 10 ** 9), n)
    fe = [p for p in edf if int(p.split("/")[-3].split("_")[0][1:]) == first[p.split("/")[-4]]]
    f = fl[name]
    chk(len(edf) == f["edf"] and len(subj) == f["patients"] and len(ses) == f["sessions"] and len(fe) == f["first_session_edf"],
        f"{name}: {len(edf)} EDF, {len(subj)} patients, {len(ses)} sessions, {len(fe)} first-session EDF (disk) == flow_counts")
    chk(f["first_session_edf"] - f["excl_missing_channels"] - f["excl_too_short"] - f["excl_duplicate"] - f["excl_no_clean_window"] == f["analysed_recordings"]
        and f["windows_before_ptp"] - f["windows_rejected_ptp"] == f["clean_windows"], f"{name}: exclusion arithmetic consistent ({f['analysed_recordings']} recordings, {f['clean_windows']} windows)")
chk(fl["Epilepsy"]["analysed_patients"] + fl["No epilepsy"]["analysed_patients"] == 200 and fl["sensitivity_without_le2009"]["remaining"] == 169, "200 analysed patients; 169 in the sensitivity set")
print("== Table 1 from EDF headers (independent of matched_design)")
h = pd.read_csv(f"{T}/data/edf_headers.csv"); h["age"] = h.patient.str.extract(r"Age:(-?\d+)")[0].astype(float); h["sesno"] = h.sess.str.extract(r"s(\d+)_")[0].astype(int)
P = np.load(f"{T}/data/p3_patient_table.npz", allow_pickle=True); S = list(P["subjects"]); y = pd.Series(P["y"], index=S)
first = h[h.groupby("subj").sesno.transform("min") == h.sesno].groupby("subj").first().loc[S]
for lab, nm in ((1, "Epilepsy"), (0, "No epilepsy")):
    a = first.age[y == lab]; print(f"   {nm}: age median {a.median():.1f} [{a.quantile(.25):.1f}-{a.quantile(.75):.1f}]; range {a.min():.0f}-{a.max():.0f}")
T1 = pd.read_csv(f"{T}/reports/tripod_ai/table1.csv"); row = T1[T1.Characteristic.str.startswith("Age")].iloc[0]
chk(row.Epilepsy.startswith(f"{first.age[y == 1].median():.1f}") and row["No epilepsy"].startswith(f"{first.age[y == 0].median():.1f}"), "age medians in Table 1 match the EDF headers")
era = np.where(first.year <= 2009, "<=2009", first.year.astype(int).astype(str)); ct = pd.crosstab(era, y.values)
ok = all(T1[T1.Characteristic == f"  {e}"].Epilepsy.iloc[0].startswith(f"{ct.loc[e, 1]} (") and T1[T1.Characteristic == f"  {e}"]["No epilepsy"].iloc[0].startswith(f"{ct.loc[e, 0]} (") for e in ct.index)
chk(ok and abs(stats.chi2_contingency(ct.values)[1] - float(T1[T1.Characteristic.str.startswith('Recording era')].p.iloc[0])) < 5e-4, f"era counts and chi-square p match ({ct.to_dict()})")
male = first.patient.str.extract(r"^\w+\s+([MF])\s")[0] == "M"; chk(int(male[y == 1].sum()) == 41 and int(male[y == 0].sum()) == 50, f"sex from headers: male {int(male[y == 1].sum())} / {int(male[y == 0].sum())}")
print("== sample-size numbers recomputed")
ss = json.load(open(f"{T}/reports/tripod_ai/sample_size.json"))
A, n1, n0 = 0.727, 100, 100; Q1, Q2 = A / (2 - A), 2 * A ** 2 / (1 + A); se = np.sqrt((A * (1 - A) + (n1 - 1) * (Q1 - A ** 2) + (n0 - 1) * (Q2 - A ** 2)) / (n1 * n0))
chk(abs(1.96 * se - ss["hanley_mcneil_ci_halfwidth_n200"]) < 1e-9, f"Hanley-McNeil half-width {1.96 * se:.4f}")
import statsmodels.api as sm
rng = np.random.default_rng(123); n = 300000; mu = np.sqrt(2) * stats.norm.ppf(A); yy = rng.random(n) < 0.5; x = rng.normal(np.where(yy, mu, 0), 1)
fit = sm.Logit(yy.astype(float), sm.add_constant(x)).fit(disp=0); r2 = 1 - np.exp(-2 * (fit.llf - fit.llnull) / n)
chk(abs(r2 - ss["riley2020_r2cs_from_auc"]) < 0.005, f"R2_CS from AUC 0.727 (statsmodels, other seed): {r2:.4f} vs {ss['riley2020_r2cs_from_auc']:.4f}")
chk(abs(116 / ((0.9 - 1) * np.log(1 - r2 / 0.9)) - ss["riley2020_criterion1_n_by_parameters"]["116"]) / ss["riley2020_criterion1_n_by_parameters"]["116"] < 0.05, "Riley criterion 1 for 116 parameters within 5%")
fa = np.load(f"{T}/reports/round25_fold_aucs.npz"); d = fa["F2"] - fa["I2"]; s_ = np.sqrt(d.var(ddof=1) * (1 / 50 + 40 / 160))
chk(abs(stats.t.ppf(0.975, 49) * s_ - ss["min_significant_difference_p05"]["F2 - I2"]) < 1e-9 and abs((stats.t.ppf(0.975, 49) + stats.t.ppf(0.8, 49)) * s_ - ss["mde_80pct_power_corrected_t"]["F2 - I2"]) < 1e-9, f"F2 - I2: significance threshold {stats.t.ppf(0.975, 49) * s_:.4f}, MDE80 {(stats.t.ppf(0.975, 49) + stats.t.ppf(0.8, 49)) * s_:.4f}")
print("== README: scripts, flags and referenced files exist")
rd = open(f"{T}/README.md").read(); cmds = re.findall(r"python ([\w]+\.py)([^\n#&]*)", rd)
miss = [c for c, _ in cmds if not os.path.exists(f"{T}/code/{c}")]; chk(not miss and os.path.exists(f"{T}/code/run_round25_queue.sh"), f"{len(cmds)} commands, all scripts exist {miss}")
bad = []
for c, flags in cmds:
    fl_ = flags.split()
    if not fl_: continue
    src = open(f"{T}/code/{c}").read(); uses_c17 = "c17.args()" in src
    for f_ in [x for x in fl_ if x.startswith("--")]:
        if f_ not in src and not (uses_c17 and f_ in ("--run", "--dry-run", "--jobs")): bad.append(f"{c} {f_}")
chk(not bad, f"every flag used in README is defined by its script {bad}")
refs = re.findall(r"`((?:reports|data|code)/[^`*<> ]+)`", rd); missing = [r for r in refs if not glob.glob(f"{T}/{r.rstrip('/')}")]
chk(not missing, f"{len(refs)} referenced paths exist {missing}")
print(f"\n{'ALL CHECKS PASSED' if not FAIL else str(len(FAIL)) + ' FAILED: ' + str(FAIL)}")
