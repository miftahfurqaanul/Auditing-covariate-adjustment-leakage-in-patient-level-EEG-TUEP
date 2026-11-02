"""Post hoc sensitivity analysis (requested 2026-10-14, not pre-registered): does adding the two exploratory Emotiv tests
(Zenodo 1252141, analysis Z4, reports/external_results.csv) to the 89-test project registry (round 27 A1) change any conclusion?
The Emotiv validation was run on 2026-10-05, before the git repository existed. Its prediction file entered git together with
its results (snapshot 6708292), so the order cannot be verified. Z1-Z3 have no p-values (descriptive AUC / bootstrap CI only).
Output: reports/registry_sensitivity_emotiv.{csv,txt}. CPU only, read-only on inputs."""
import numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests

R = "/home/miftah_f/RTX-5090/tuep_fresh/reports"
G = pd.read_csv(f"{R}/round27_registry.csv")
E = pd.read_csv(f"{R}/external_results.csv").query("analysis == 'Z4'")
assert len(G) == 89 and len(E) == 2 and E.p.notna().all()
q0 = multipletests(G.p, method="fdr_bh")[1]
assert np.allclose(q0, G.q_bh_global, atol=1e-12), "stored registry q-values not reproduced"
p_all = np.r_[G.p.values, E.p.values]
q1 = multipletests(p_all, method="fdr_bh")[1]
h0, h1 = multipletests(G.p, method="holm")[1], multipletests(p_all, method="holm")[1]
out = G[["round", "family", "test", "p", "p_holm_family"]].assign(q_bh_89=q0, q_bh_91=q1[:89], holm_global_89=h0, holm_global_91=h1[:89])
emo = pd.DataFrame(dict(round="ext", family="Emotiv Z4 (exploratory)", test=["Z4 TUEP+Emotiv vs TUEP (" + m + ")" for m in E.model],
                        p=E.p.values, p_holm_family=multipletests(E.p, method="holm")[1], q_bh_89=np.nan, q_bh_91=q1[89:], holm_global_89=np.nan, holm_global_91=h1[89:]))
pd.concat([out, emo]).to_csv(f"{R}/registry_sensitivity_emotiv.csv", index=False)

dq = q1[:89] - q0
lines = [
    "Sensitivity of the project-wide multiplicity registry to the two exploratory Emotiv tests (post hoc, 2026-10-14)",
    f"tests: 89 registry + {len(E)} Emotiv Z4 = {len(p_all)}",
    f"BH q < 0.05: {(q0 < 0.05).sum()} -> {(q1[:89] < 0.05).sum()} | BH q < 0.10: {(q0 < 0.10).sum()} -> {(q1[:89] < 0.10).sum()}",
    f"global Holm p < 0.05: {(h0 < 0.05).sum()} -> {(h1[:89] < 0.05).sum()}",
    f"max increase in q of a registry test: {dq.max():.4f}",
    f"tests changing side of 0.05 or 0.10: {int(((q0 < 0.05) != (q1[:89] < 0.05)).sum() + ((q0 < 0.10) != (q1[:89] < 0.10)).sum())}",
    "Emotiv tests in the enlarged registry: " + "; ".join(f"{t}: p {p:.3f}, q {q:.3f}" for t, p, q in zip(emo.test, emo.p, emo.q_bh_91)),
    "",
    "Key method claims (q_bh 89 -> 91):",
]
for t in ["C1 combined normative-layer effect (mean of 3 views per fold)", "F2 - LaBraM-FT", "F2 - CBraMod-FT"]:
    r = out[out.test == t]; assert len(r) == 1, t
    lines.append(f"  {t}: {r.q_bh_89.iloc[0]:.4f} -> {r.q_bh_91.iloc[0]:.4f}")
open(f"{R}/registry_sensitivity_emotiv.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
