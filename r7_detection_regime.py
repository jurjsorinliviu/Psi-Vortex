"""Tier-2: detection-regime boundary for latent coupling as f(dataset size, noise).

Reviewers 2/4 asked for a quantitative boundary of the practical detection regime
rather than the single weak-coupling anecdote. Using the sound held-out-driver
protocol, we sweep training-sequence length (proxy for dataset size N) x measurement
noise at fixed alpha=0.08 and map where the coupling is identifiable (held-out
R^2 >= 0.8). This yields an empirical rule: the binding constraint is sample size,
not noise (recovery is noise-robust once N clears a threshold).

Budget env-overridable (EPOCHS, M, SEEDS).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import supplement_sound as SS

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = 0.08
N_DRIVERS = [1, 2, 3, 5]                    # number of training driver realizations -> dataset size
NOISE = [0, 2, 5, 10]                       # measurement noise (%)
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
TRAIN_SEEDS = [42, 123, 456, 7, 99][:SS.M]
HELD_SEEDS = [777, 888, 555]
R2A = SS.R2_ACCEPT


def cell(n_drivers, noise, seed):
    drv = TRAIN_SEEDS[:n_drivers]
    train = [SS.realization(ALPHA, s, noise_pct=noise)[0] for s in drv]   # default stride/n_steps (short seqs)
    held = [SS.realization(ALPHA, s)[1] for s in HELD_SEEDS]
    gb = SS.realization(ALPHA, drv[0])[2]
    a, r2, _ = SS._train_track("psi", "physics", train, held, gb, seed=seed)
    return abs(a - ALPHA) / ALPHA * 100.0, r2


def main():
    print(f"Detection regime (alpha={ALPHA}) | EPOCHS={SS.EPOCHS} M={SS.M} stride={SS.STRIDE} seeds={SEEDS}")
    rows = []
    pts_per = int(0.667 * 3000 / SS.STRIDE)        # ~250 points per driver realization
    for nd in N_DRIVERS:
        n_train = nd * pts_per
        for noise in NOISE:
            errs, r2s = zip(*[cell(nd, noise, s) for s in SEEDS])
            rows.append(dict(n_drivers=nd, n_train_pts=n_train, noise_pct=noise,
                             err_pct=float(np.mean(errs)), r2=float(np.mean(r2s)),
                             identifiable=bool(np.mean(r2s) >= R2A)))
            print(f"  drivers={nd} (n_train={n_train:4d})  noise={noise:2d}%  "
                  f"err={rows[-1]['err_pct']:6.1f}%  R2={rows[-1]['r2']:.2f}  "
                  f"{'IDENTIFIABLE' if rows[-1]['identifiable'] else 'not'}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "r7_detection_regime.csv"), index=False)

    # boundary: smallest n_train identifiable at every tested noise level
    ident = df[df.identifiable]
    rule = (f"identifiable for n_train >= {int(ident.n_train_pts.min())} across noise up to "
            f"{int(df.noise_pct.max())}%" if len(ident) else "not identifiable in tested grid")
    print("\nEmpirical boundary:", rule)

    # heatmap of err% with identifiable cells outlined
    piv = df.pivot(index="noise_pct", columns="n_train_pts", values="err_pct")
    idn = df.pivot(index="noise_pct", columns="n_train_pts", values="identifiable")
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    im = ax.imshow(piv.values, origin="lower", aspect="auto", cmap="viridis_r",
                   vmin=0, vmax=min(150, np.nanmax(piv.values)))
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_xlabel("training points $N_{train}$")
    ax.set_ylabel("measurement noise (%)")
    ax.set_title(f"Coupling-recovery error (%) at $\\alpha$={ALPHA}")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            mark = "*" if idn.values[i, j] else ""
            ax.text(j, i, f"{piv.values[i, j]:.0f}{mark}", ha="center", va="center",
                    color="w", fontsize=8)
    fig.colorbar(im, label="alpha error (%)")
    ax.text(0.02, -0.22, "* = identifiable (held-out $R^2\\geq0.8$)", transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "r7_detection_regime.png"), dpi=150)

    with open(os.path.join(HERE, "r7_detection_regime.tex"), "w") as f:
        f.write("\\begin{tabular}{rrrrc}\n\\toprule\n")
        f.write("$N_{train}$ & noise \\% & $\\alpha$ err \\% & held-out $R^2$ & identifiable \\\\\n\\midrule\n")
        for _, r in df.iterrows():
            f.write(f"{int(r['n_train_pts'])} & {int(r['noise_pct'])} & {r['err_pct']:.1f} & "
                    f"{r['r2']:.2f} & {'yes' if r['identifiable'] else 'no'} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write(f"% Detection regime at alpha={ALPHA}. Boundary: {rule}.\n")
    print("Saved: r7_detection_regime.csv / .tex / .png")


if __name__ == "__main__":
    main()
