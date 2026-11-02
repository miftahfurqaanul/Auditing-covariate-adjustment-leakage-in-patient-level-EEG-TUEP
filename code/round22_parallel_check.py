"""Round 22 check: the three year-shuffle conditions were run as parallel GPU processes (approved by the user 2026-10-12).
Re-run fold r0k0 of shuf1 (it ran while two other processes shared the GPU) ALONE on the GPU and compare with the saved JSON bit for bit."""
import json, numpy as np, torch
import normmamba as nm, round22_nm as r22

if __name__ == "__main__":
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    assert nm.DEV.type == "cuda"
    nm.DES = r22.year_design(r22.shuffled_year(1)); r, k, tr, te = r22.J200[0]
    res = nm.run_fold(r, k, tr, te, variants=("F",))["F"]; ref = json.load(open(f"{r22.OUT}/shuf1/r0_k0.json"))["F"]
    d = np.array([res["test"][s] - ref["test"][s] for s in ref["test"]])
    same = res["chosen"] == ref["chosen"] and res["epochs"] == ref["epochs"] and np.abs(d).max() == 0.0
    print(("PASS" if same else "FAIL") + f" shuf1 r0k0 re-run alone == parallel run: chosen {res['chosen']} vs {ref['chosen']}, epochs {res['epochs']} vs {ref['epochs']}, "
          f"max |prediction diff| {np.abs(d).max():.2e}")
