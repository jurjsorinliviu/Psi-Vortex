"""Tier-2: multi-layer / multi-path coupling recovery (beyond the pairwise case).

Reviewers 2/4/6 noted the two-layer pairwise demo is simplified -- real stacks have
multiple victims per driver with distance-decaying coupling. We build a multi-victim
stack (one driver heating K victims with couplings alpha_k that decay with distance)
and test whether the framework recovers the per-layer coupling PROFILE on held-out
drivers, not just a single scalar.

This uses the data generator's multi-victim support (generate_thermal_data with
victim_alphas) + the validated held-out-driver recovery. It demonstrates multi-path
recovery from one source; cumulative, nonlinear multi-source heat paths in dense
crossbars remain explicitly future work.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import supplement_sound as SS
import supplementary_experiments as SE

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHAS_VEC = [0.12, 0.08, 0.05]            # victims at increasing distance from driver
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
TRAIN_SEEDS = [42, 123, 456, 7, 99][:SS.M]
HELD_SEEDS = [777, 888, 555]


def realization_victim(avec, seed, k, stride=None):
    """Mirror supplement_sound.realization but select victim k of a multi-victim stack."""
    stride = SS.STRIDE if stride is None else stride
    ds = SE.generate_thermal_data(alpha_gt=avec[0], seed=seed,
                                  n_victims=len(avec), victim_alphas=avec)
    gb = ds["g_base"]
    n = ds["n_train"]
    V = ds["full_V"][:n][::stride]
    dT = ds["dT"][:n][::stride]
    Ik = ds["I_victims"][k][:n][::stride]
    Vt = torch.tensor(V, dtype=torch.float32, device=SS.DEV).view(-1, 1)
    y = torch.tensor(np.log(np.maximum(Ik, 1e-30) / (SS.V_READ * gb)),
                     dtype=torch.float32, device=SS.DEV).view(-1, 1)
    return (Vt, y), (Vt, dT), gb


def main():
    print(f"Multi-layer recovery | couplings={ALPHAS_VEC} EPOCHS={SS.EPOCHS} M={SS.M} seeds={SEEDS}")
    rows = []
    for k, a_true in enumerate(ALPHAS_VEC):
        train = [realization_victim(ALPHAS_VEC, s, k)[0] for s in TRAIN_SEEDS]
        held = [realization_victim(ALPHAS_VEC, s, k)[1] for s in HELD_SEEDS]
        gb = realization_victim(ALPHAS_VEC, TRAIN_SEEDS[0], k)[2]
        recs = [SS._train_track("psi", "physics", train, held, gb, seed=s)[:2] for s in SEEDS]
        a_s = np.array([a for a, _ in recs])
        r2_s = np.array([r for _, r in recs])
        rows.append(dict(layer=k + 1, distance=k + 1, alpha_true=a_true,
                         alpha_rec=float(a_s.mean()), alpha_std=float(a_s.std()),
                         r2=float(r2_s.mean()), err_pct=100 * abs(a_s.mean() - a_true) / a_true))
        print(f"  victim {k+1} (alpha_true={a_true:.3f}): recovered {a_s.mean():.4f}"
              f"+/-{a_s.std():.4f}  R2={r2_s.mean():.2f}  err={rows[-1]['err_pct']:.1f}%")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "r8_multilayer.csv"), index=False)
    with open(os.path.join(HERE, "r8_multilayer.tex"), "w") as f:
        f.write("\\begin{tabular}{ccccc}\n\\toprule\n")
        f.write("Victim (distance) & $\\alpha_{true}$ & recovered $\\alpha$ & held-out $R^2$ & err \\% \\\\\n\\midrule\n")
        for _, r in df.iterrows():
            f.write(f"{int(r['layer'])} & {r['alpha_true']:.3f} & "
                    f"{r['alpha_rec']:.3f}$\\pm${r['alpha_std']:.3f} & {r['r2']:.2f} & "
                    f"{r['err_pct']:.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("% Multi-victim stack: per-layer coupling recovered on held-out drivers.\n")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = df["layer"].values
    ax.bar(x - 0.18, df["alpha_true"], 0.36, label="true $\\alpha$", color="0.6")
    ax.bar(x + 0.18, df["alpha_rec"], 0.36, yerr=df["alpha_std"], capsize=4,
           label="recovered $\\alpha$", color="C3")
    ax.set_xticks(x); ax.set_xlabel("victim layer (distance from driver)")
    ax.set_ylabel(r"thermal coupling $\alpha$")
    ax.set_title("Multi-layer coupling profile (held-out recovery)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "r8_multilayer.png"), dpi=150)
    print("Saved: r8_multilayer.csv / .tex / .png")


if __name__ == "__main__":
    main()
