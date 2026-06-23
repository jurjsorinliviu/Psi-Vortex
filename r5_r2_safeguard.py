"""Tier-2: R^2 acceptance safeguard for the coupling-recovery readout.

Reviewer 2 noted the R^2-based selector can fail SILENTLY: if no seed yields an
interpretable (high-R^2) decomposition, the pipeline would still emit a number.
This integrates a hard min-R^2 acceptance gate into the recovery step (Algorithm 1,
post-training readout): the framework returns an alpha ONLY when the best held-out
R^2 clears the gate, otherwise it ABSTAINS ("coupling present but not interpretable"
/ "no coupling") instead of reporting an unreliable value.

Three scenarios show the gate behaves correctly:
  (S1) genuine coupling + recurrent model   -> ACCEPT (interpretable alpha).
  (S2) genuine coupling + memoryless MLP     -> ABSTAIN (coupling encoded implicitly;
       the exact "all-seeds-implicit" silent-failure case the reviewer described).
  (S3) alpha=0 negative control              -> ABSTAIN (no false coupling invented).

Budget is env-overridable (EPOCHS, M, SEEDS).
"""
import os

import numpy as np
import pandas as pd

import supplement_sound as SS

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_SEEDS = [42, 123, 456, 7, 99][:SS.M]
HELD_SEEDS = [777, 888, 555]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
R2_MIN = SS.R2_ACCEPT  # 0.8


def recover_seeds(alpha_gt, kind, mode):
    """Return [(alpha, r2), ...] over training seeds for one (model, init)."""
    train = [SS.realization(alpha_gt, s)[0] for s in TRAIN_SEEDS]
    held = [SS.realization(alpha_gt, s)[1] for s in HELD_SEEDS]
    gb = SS.realization(alpha_gt, TRAIN_SEEDS[0])[2]
    out = []
    for s in SEEDS:
        a, r2, _ = SS._train_track(kind, mode, train, held, gb, seed=s)
        out.append((a, r2))
    return out


def safeguard(recoveries, r2_min=R2_MIN):
    """Hard min-R^2 gate. ACCEPT best-R^2 seed if it clears the gate, else ABSTAIN.
    Reports the naive (un-gated) number that WOULD have been emitted, to show the
    silent failure the gate prevents."""
    best_a, best_r2 = max(recoveries, key=lambda x: x[1])
    n_pass = sum(r >= r2_min for _, r in recoveries)
    naive = float(np.mean([a for a, _ in recoveries]))   # what un-gated reporting emits
    if best_r2 >= r2_min:
        return dict(status="ACCEPT", alpha=best_a, best_r2=best_r2, n_pass=n_pass, naive_alpha=naive)
    return dict(status="ABSTAIN", alpha=None, best_r2=best_r2, n_pass=0, naive_alpha=naive)


def main():
    print(f"R^2 safeguard (gate={R2_MIN}) | EPOCHS={SS.EPOCHS} M={SS.M} seeds={SEEDS}")
    scenarios = [
        ("S1 genuine + recurrent", 0.08, "psi", "physics"),
        ("S2 genuine + MLP", 0.08, "mlp", "random"),
        ("S3 alpha=0 control", 0.0, "psi", "physics"),
    ]
    rows = []
    for label, agt, kind, mode in scenarios:
        rec = recover_seeds(agt, kind, mode)
        g = safeguard(rec)
        rows.append(dict(scenario=label, alpha_gt=agt, **g))
        alpha_str = "--" if g["alpha"] is None else f"{g['alpha']:.4f}"
        print(f"  {label:24s} gt={agt:.2f}  best_R2={g['best_r2']:.2f}  "
              f"pass={g['n_pass']}/{len(rec)}  -> {g['status']:7s} "
              f"alpha={alpha_str}  (un-gated would emit {g['naive_alpha']:.4f})")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "r5_r2_safeguard.csv"), index=False)
    with open(os.path.join(HERE, "r5_r2_safeguard.tex"), "w") as f:
        f.write("\\begin{tabular}{lccccc}\n\\toprule\n")
        f.write("Scenario & $\\alpha_{gt}$ & best held-out $R^2$ & seeds passing & "
                "decision & un-gated $\\alpha$ \\\\\n\\midrule\n")
        for _, r in df.iterrows():
            dec = "ACCEPT ($\\alpha$=%.3f)" % r["alpha"] if r["status"] == "ACCEPT" else "ABSTAIN"
            f.write(f"{r['scenario']} & {r['alpha_gt']:.2f} & {r['best_r2']:.2f} & "
                    f"{int(r['n_pass'])}/{len(SEEDS)} & {dec} & {r['naive_alpha']:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("% Gate = min held-out R^2 for an interpretable readout. ABSTAIN prevents "
                "the silent emission of an unreliable coupling value.\n")
    print("\nSaved: r5_r2_safeguard.csv / .tex")


if __name__ == "__main__":
    main()
