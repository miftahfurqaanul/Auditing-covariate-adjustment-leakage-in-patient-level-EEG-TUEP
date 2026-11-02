"""Round 19 (reports/round19_prediction.md, commit e0f242a): inner out-of-fold NormMamba-F predictions for the 160 training patients of every
outer fold. Inside each outer fold: StratifiedKFold(5, shuffle, random_state=100r+k) over the training patients (as rounds 10/14); for every
inner fold (a, b) the full round-16 NormMamba-F procedure is run on a only (normative tokens from the controls of a, 20% validation of a,
grid on validation with 1 seed, chosen config retrained with 3 seeds) and b is predicted. One GPU process; resumable (one JSON per outer fold).
Usage: python round19_nm_inner.py --dry-run | --run"""
import os, sys, json, time, argparse, numpy as np, torch
from sklearn.model_selection import StratifiedKFold
import normmamba as nm, round4 as r4, tang_fixed as tf

OUT = f"{nm.D}/round19"; os.makedirs(OUT, exist_ok=True); Y, SUBJ, POS = nm.Y, r4.SUBJ, nm.POS
N_SEEDS = 3


def inner_folds(r, k, tr):
    return list(StratifiedKFold(5, shuffle=True, random_state=100 * r + k).split(tr, Y[tr]))


def run_outer(r, k, tr, te, max_epochs=nm.MAX_EPOCHS, only_first=False):
    ylab = {s: int(Y[POS[s]]) for s in SUBJ}; oof, info = {}, []
    for j, (ia, ib) in enumerate(inner_folds(r, k, tr)):
        a, b = tr[ia], tr[ib]; t0 = time.time()
        tr_p, va_p = tf.split_val(list(SUBJ[a]), ylab, 100 * r + k + 7 * (j + 1))
        tr_i, va_i = np.array([POS[s] for s in tr_p]), np.array([POS[s] for s in va_p])
        assert not (set(tr_i) | set(va_i)) & (set(b) | set(te)) and set(tr_i) | set(va_i) == set(a)
        toks, summ, _ = nm.make_tokens(a, "norm"); base = 1000 * r + 10 * k + 500 + j
        g = {f"{d}_{L}": nm.train_one(toks, summ, tr_i, va_i, d, L, base, True, max_epochs)[1] for d, L in nm.GRID}
        bd, bL = map(int, max(g, key=g.get).split("_")); preds = []
        for s in range(N_SEEDS):
            m, _, _ = nm.train_one(toks, summ, tr_i, va_i, bd, bL, 7919 * (s + 1) + base, True, max_epochs); preds.append(nm.predict(m, toks, summ, b, bL))
        for i, p in zip(b, np.mean(preds, 0)):
            oof[SUBJ[i]] = float(p)
        info.append(dict(j=j, chosen=[bd, bL], grid_val_auc=g, seconds=time.time() - t0))
        if only_first:
            break
    return dict(r=r, k=k, oof=oof, inner=info)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--dry-run", action="store_true"); g.add_argument("--run", action="store_true")
    a = ap.parse_args(); torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    print(f"device {nm.DEV} ({torch.cuda.get_device_name(0) if nm.DEV.type == 'cuda' else 'cpu'}); torch {torch.__version__}; N_SEEDS {N_SEEDS}", flush=True)
    # check: inner splits partition the training patients and never touch the test patients
    ok = all(sorted(np.concatenate([tr[ib] for _, ib in inner_folds(r, k, tr)]).tolist()) == sorted(tr.tolist()) and not set(tr) & set(te) for r, k, tr, te in nm.JOBS)
    print(("PASS" if ok else "FAIL") + " in all 50 outer folds the 5 inner held-out sets partition the 160 training patients and exclude the test patients", flush=True)
    if not ok:
        sys.exit("STOP")
    if a.dry_run:
        r, k, tr, te = nm.JOBS[0]; t0 = time.time(); res = run_outer(r, k, tr, te, max_epochs=3, only_first=True)
        print(f"dry-run (3 epochs, first inner fold only): {len(res['oof'])} inner predictions, {time.time() - t0:.0f} s; chosen {res['inner'][0]['chosen']}", flush=True); sys.exit()
    for r, k, tr, te in nm.JOBS:
        path = f"{OUT}/nm_inner_r{r}_k{k}.json"
        if os.path.exists(path):
            continue
        t0 = time.time(); res = run_outer(r, k, tr, te); assert len(res["oof"]) == len(tr); json.dump(res, open(path, "w"))
        print(f"outer fold r{r}k{k}: {len(res['oof'])} inner OOF predictions in {(time.time() - t0) / 60:.1f} min; chosen {[i['chosen'] for i in res['inner']]}", flush=True)
    print("ALL OUTER FOLDS DONE", flush=True)
