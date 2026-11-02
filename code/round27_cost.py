"""Round 27 A9 (reports/round27_prediction.md, commit 23f825b): parameters, training time and CPU inference time of I2, P3, NM (NormMamba-F),
LaBraM-base and CBraMod. Training times of NM and the FMs are read from their per-fold JSON files (RTX 4080 SUPER); I2/P3 are timed on CPU here.
Inference is timed on CPU (1 thread) for one patient. Weights do not affect timing, so untrained NormMamba weights are used. CPU only."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["OMP_NUM_THREADS"] = "1"
import json, time, glob, collections, numpy as np, pandas as pd, torch
torch.set_num_threads(1)
import round17_common as c17, round4 as r4, evalkit as ek

J, Y, N, SUBJ = c17.JOBS, c17.Y, c17.N, np.asarray(c17.SUBJ); rows = []
tr, te = J[0][2], J[0][3]
# I2
t0 = time.time(); X = np.hstack([r4.normative_z(r4.DESIGN["I2"], tr), c17.AS]); m = ek.nested_lr().fit(X[tr], Y[tr]); t_fit = time.time() - t0
t0 = time.time(); [m.predict_proba(X[[i]]) for i in te]; t_inf = (time.time() - t0) / len(te)
rows.append(dict(model="I2", trainable_parameters=X.shape[1] + 1, train_time_per_fold_s=t_fit, train_hardware="CPU (1 thread)", inference_ms_per_patient_cpu=1000 * t_inf))
# P3
import round17_p3 as p3
C, pid = p3.load_cache(1); t0 = time.time(); p3.p3_predict(C, pid, tr, te, r4.YEAR_G, "era"); t_p3 = time.time() - t0
rows.append(dict(model="P3", trainable_parameters=p3.NPC * 855 + 855 + p3.NPC + 2 + 1, train_time_per_fold_s=t_p3, train_hardware="CPU (1 thread), incl. reference + tangent + PCA",
                 inference_ms_per_patient_cpu=np.nan))
# NM
js = [json.load(open(f)) for f in sorted(glob.glob(f"{ek.D}/round16/r*_k*.json"))]
chosen = collections.Counter(tuple(d["F"]["chosen"]) for d in js).most_common(1)[0][0]; d_, L_ = chosen
import normmamba as nmm
mdl = nmm.NormMamba(d_, fusion=True); n_nm = sum(p.numel() for p in mdl.parameters())
toks, summ, _ = nmm.make_tokens(tr, "norm"); i0 = te[0]; mdl.eval()
with torch.no_grad():
    c = torch.tensor(nmm.crops_eval(toks[i0], L_)); s = torch.tensor(summ[i0]).expand(len(c), -1); t0 = time.time(); mdl(c, s); t_nm = time.time() - t0
rows.append(dict(model=f"NM (NormMamba-F, d={d_}, L={L_}, most frequent choice)", trainable_parameters=n_nm, train_time_per_fold_s=np.mean([d["F"]["seconds"] for d in js]),
                 train_hardware="GPU RTX 4080 SUPER (grid 9 configs + 5 seeds)", inference_ms_per_patient_cpu=1000 * t_nm))
# FMs
import round25_fm as fm, cbramod_finetune as cb
lab, chans, _ = fm.labram_model(); cbm = cb.Model()
W, wsub, wlab, wrec = fm.load_cache(); wi = np.where(wsub == SUBJ[i0])[0]; x = torch.from_numpy(W[wi].astype(np.float32))
lab.eval(); cbm.eval()
with torch.no_grad():
    t0 = time.time(); lab(x.reshape(-1, 19, 10, 200), input_chans=chans); t_lab = time.time() - t0
    t0 = time.time(); cbm(x[:, cb.PERM].reshape(-1, 19, 10, 200)); t_cb = time.time() - t0
for name, model, key, t in [("LaBraM-base fine-tuned", lab, "labram", t_lab), ("CBraMod fine-tuned", cbm, "cbramod", t_cb)]:
    secs = [json.load(open(f))["seconds"] for f in sorted(glob.glob(f"{ek.D}/round25/{key}/r*_k*.json"))]
    rows.append(dict(model=name, trainable_parameters=sum(p.numel() for p in model.parameters()), train_time_per_fold_s=float(np.mean(secs)),
                     train_hardware="GPU RTX 4080 SUPER (50 epochs; excludes the authors' pretraining)", inference_ms_per_patient_cpu=1000 * t))
T = pd.DataFrame(rows); T.to_csv(f"{c17.R}/round27_A9_cost.csv", index=False)
print(f"patient used for inference timing: {SUBJ[i0]} ({len(wi)} windows of 10 s; NM {len(c)} crops)")
print(T.round(3).to_string(index=False))
