"""Round 22 parts A and B (reports/round22_prediction.md, commit 95ad5a4): NormMamba-F (round-16 procedure, F variant only, same seeds)
  det   : fold r0k0 with the true design, re-run to measure GPU run-to-run noise against round 16
  j169  : canonical folds without the <=2009 group (169 patients), true design
  shufI : 200 patients, year groups permuted exactly as round17_common.shuffled_year(I) (seed 17000 + I), I = 0, 1, 2
One GPU process; resumable (one JSON per fold per condition in data/round22/<condition>/).
Does NOT import round17_common or round10: both set CUDA_VISIBLE_DEVICES="" on import. Their fold/shuffle definitions are re-implemented here
and checked for identity in round22_eval.py.  Usage: python round22_nm.py --dry-run | --run"""
import os, sys, json, time, argparse, numpy as np, torch
import normmamba as nm, round4 as r4

OUT = f"{nm.D}/round22"; Y, SUBJ, POS, F = nm.Y, r4.SUBJ, nm.POS, nm.F; N = len(Y)
KEEP = r4.YEAR_G != "<=2009"


def jobs(keep):                                         # = round10.jobs(keep)
    out = []
    for r in range(10):
        for k in range(5):
            te = np.array([POS[s] for s in F[(F.repeat == r) & (F.fold == k)].subject if keep[POS[s]]])
            out.append((r, k, np.setdiff1d(np.where(keep)[0], te), te))
    return out


def year_design(yg):                                    # = round17_common.year_design
    return np.column_stack([np.ones(N), r4.AGE, r4.MALE, r4.dummies(yg)])


def shuffled_year(i):                                   # = round17_common.shuffled_year
    return np.random.default_rng(17000 + i).permutation(r4.YEAR_G)


J200, J169 = jobs(np.ones(N, bool)), jobs(KEEP)
CONDITIONS = [("det", r4.DESIGN["I2"], J200[:1]), ("j169", r4.DESIGN["I2"], J169)] + [(f"shuf{i}", year_design(shuffled_year(i)), J200) for i in range(3)]


def run_condition(name, design, J, max_epochs=nm.MAX_EPOCHS):
    os.makedirs(f"{OUT}/{name}", exist_ok=True); nm.DES = design                    # make_tokens reads the module-level DES at call time
    for r, k, tr, te in J:
        path = f"{OUT}/{name}/r{r}_k{k}.json"
        if os.path.exists(path):
            continue
        t0 = time.time(); res = nm.run_fold(r, k, tr, te, max_epochs=max_epochs, variants=("F",)); json.dump(res, open(path, "w"))
        print(f"{name} fold r{r}k{k}: {(time.time() - t0) / 60:.1f} min, test AUC F {res['F']['test_auc']:.3f}, chosen {res['F']['chosen']}", flush=True)
    nm.DES = r4.DESIGN["I2"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--dry-run", action="store_true"); g.add_argument("--run", action="store_true")
    ap.add_argument("--only", default=None, help="run a single condition (parallel GPU processes, approved by the user 2026-10-12)")
    a = ap.parse_args(); torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    print(f"device {nm.DEV} ({torch.cuda.get_device_name(0) if nm.DEV.type == 'cuda' else 'cpu'}); torch {torch.__version__}", flush=True)
    if nm.DEV.type != "cuda":
        sys.exit("STOP: no GPU visible")
    ok = all(KEEP[np.concatenate([tr, te])].all() for _, _, tr, te in J169) and len(J169) == 50
    ok &= all(not np.array_equal(d, r4.DESIGN["I2"]) for n_, d, _ in CONDITIONS if n_.startswith("shuf"))
    ok &= all(sorted(shuffled_year(i).tolist()) == sorted(r4.YEAR_G.tolist()) for i in range(3))
    print(("PASS" if ok else "FAIL") + " J169 has no <=2009 patient; shuffled designs differ from the true design; shuffles preserve group sizes", flush=True)
    if not ok:
        sys.exit("STOP")
    if a.dry_run:
        t0 = time.time(); nm.DES = year_design(shuffled_year(0)); r, k, tr, te = J169[0]
        toks, summ, T = nm.make_tokens(tr, "norm"); c = tr[Y[tr] == 0]
        print(f"dry-run: tokens built with a shuffled design on J169 fold r{r}k{k}; training-control segment z mean max {np.abs(T[np.isin(nm.PID, c)].mean(0)).max():.1e}; "
              f"{time.time() - t0:.1f} s", flush=True); nm.DES = r4.DESIGN["I2"]; sys.exit()
    for name, design, J in CONDITIONS:
        if a.only and name != a.only:
            continue
        print(f"== condition {name}: {len(J)} folds", flush=True); run_condition(name, design, J)
    print("ALL CONDITIONS DONE", flush=True)
