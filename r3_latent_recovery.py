"""Stage 2: latent thermal-coupling recovery -- Psi-Vortex vs contemporary baselines.

This is where Psi-Vortex is *expected* to differentiate (Stage 1 showed every method
fits a simple I-V sweep). The task: recover the latent thermal coupling alpha from
victim current observations on HELD-OUT driver realizations (train on drivers
{42,123,456,...}, recover on unseen drivers {777,888,555}). Because the coupling acts
through a latent thermal state that depends on the driver's *history*, only a model
that integrates over time can recover alpha on a new driver; a memoryless map cannot.

Methods (identical observed input = instantaneous driver V, identical estimator):
  * Psi-Vortex  : true-sequence mLSTM/sLSTM with physics-aware init (recurrent).
  * MLP         : pointwise feed-forward (no temporal integration).
  * SINDy       : sparse polynomial in V (static library) -- the interpretable baseline.

All three are scored with the SAME validated free-intercept OLS estimator
(supplement_sound.free_intercept_alpha) on the SAME held-out drivers. Recovery is
accepted when held-out R^2 >= 0.8 (the manuscript's gate).

Budget is env-overridable (EPOCHS, M, ALPHAS, SEEDS); final numbers should use the
frozen config (EPOCHS=350, M=5, STRIDE=8).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import supplement_sound as SS
from baselines import SINDyBaseline

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_SEEDS = [42, 123, 456, 7, 99][:SS.M]
HELD_SEEDS = [777, 888, 555]
ALPHAS = [float(a) for a in os.environ.get("ALPHAS", "0.05,0.08,0.20").split(",")]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
R2A = SS.R2_ACCEPT


def sindy_recover(train, held, gb, degree=3):
    """Fit a static sparse polynomial y=log(I/(V_read*gb)) ~ poly(V) on the training
    drivers, then recover alpha on held-out drivers with the shared estimator."""
    V = np.concatenate([p[0].cpu().numpy().ravel() for p in train])
    y = np.concatenate([p[1].cpu().numpy().ravel() for p in train])
    m = SINDyBaseline(degree=degree).fit(V, np.zeros_like(V), y)
    al, r2 = [], []
    for Vt, dT in held:
        Vh = Vt.cpu().numpy().ravel()
        yp = m.predict(Vh, np.zeros_like(Vh))
        I_pred = np.exp(yp) * (SS.V_READ * gb)
        a, r = SS.free_intercept_alpha(I_pred, dT, gb)
        al.append(a)
        r2.append(r)
    return float(np.mean(al)), float(np.mean(r2)), m.n_params


def main():
    print(f"Stage 2 latent recovery | EPOCHS={SS.EPOCHS} M={SS.M} STRIDE={SS.STRIDE} "
          f"alphas={ALPHAS} seeds={SEEDS}")
    rows = []
    for alpha in ALPHAS:
        train = [SS.realization(alpha, s)[0] for s in TRAIN_SEEDS]
        held = [SS.realization(alpha, s)[1] for s in HELD_SEEDS]
        gb = SS.realization(alpha, TRAIN_SEEDS[0])[2]

        # SINDy is deterministic (one fit per alpha)
        a, r2, _ = sindy_recover(train, held, gb)
        rows.append(dict(method="SINDy", alpha_gt=alpha, seed=-1, alpha_rec=a, r2=r2))
        print(f"  a={alpha:.2f} SINDy        rec={a:.4f}  R2={r2:.2f}")

        for seed in SEEDS:
            a, r2, _ = SS._train_track("psi", "physics", train, held, gb, seed=seed)
            rows.append(dict(method="Psi-Vortex", alpha_gt=alpha, seed=seed, alpha_rec=a, r2=r2))
            print(f"  a={alpha:.2f} Psi-Vortex s{seed} rec={a:.4f}  R2={r2:.2f}")
            a, r2, _ = SS._train_track("mlp", "random", train, held, gb, seed=seed)
            rows.append(dict(method="MLP", alpha_gt=alpha, seed=seed, alpha_rec=a, r2=r2))
            print(f"  a={alpha:.2f} MLP        s{seed} rec={a:.4f}  R2={r2:.2f}")

    df = pd.DataFrame(rows)
    df["err_pct"] = 100.0 * np.abs(df.alpha_rec - df.alpha_gt) / df.alpha_gt
    df["recovers"] = df.r2 >= R2A
    df.to_csv(os.path.join(HERE, "r3_latent_recovery.csv"), index=False)

    summ = (df.groupby(["method", "alpha_gt"])
              .agg(alpha_rec=("alpha_rec", "mean"), err_pct=("err_pct", "mean"),
                   r2=("r2", "mean"), recov_rate=("recovers", "mean")).reset_index())
    print("\nSUMMARY (mean over seeds)")
    print(summ.to_string(index=False))

    # TeX: method x alpha -> err% (R2)
    order = ["Psi-Vortex", "MLP", "SINDy"]
    al_cols = sorted(df.alpha_gt.unique())
    with open(os.path.join(HERE, "r3_latent_recovery.tex"), "w") as f:
        f.write("\\begin{tabular}{l" + "c" * len(al_cols) + "}\n\\toprule\n")
        f.write("Method & " + " & ".join(f"$\\alpha={a:g}$" for a in al_cols) + " \\\\\n\\midrule\n")
        for m in order:
            cells = []
            for a in al_cols:
                r = summ[(summ.method == m) & (summ.alpha_gt == a)]
                cells.append(f"{r.err_pct.values[0]:.0f}\\% ({r.r2.values[0]:.2f})" if len(r) else "--")
            f.write(f"{m} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("% Held-out-driver recovery: cell = alpha error %% (held-out R^2). "
                "R^2>=0.8 = identifiable.\n")

    # Figure: recovered alpha vs ground truth
    fig, ax = plt.subplots(figsize=(6, 5))
    lim = max(al_cols) * 1.15
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="ideal (y=x)")
    colors = {"Psi-Vortex": "C3", "MLP": "C0", "SINDy": "C1"}
    for m in order:
        sub = summ[summ.method == m].sort_values("alpha_gt")
        ax.plot(sub.alpha_gt, sub.alpha_rec, "o-", c=colors[m], label=m)
    ax.set_xlabel(r"ground-truth coupling $\alpha$")
    ax.set_ylabel(r"recovered $\alpha$ (held-out drivers)")
    ax.set_title("Latent thermal-coupling recovery")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "r3_latent_recovery.png"), dpi=150)
    print("\nSaved: r3_latent_recovery.csv / .tex / .png")


if __name__ == "__main__":
    main()
