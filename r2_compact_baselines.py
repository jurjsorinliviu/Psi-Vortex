"""Head-to-head: Psi-Vortex vs contemporary baselines on compact modeling.

Compares the Psi-Vortex teacher against MLP, PIKAN (Chebyshev-KAN) and SINDy on the
SAME held-out protocol, on both the synthetic benchmark (printed-memristor data) and
the independent measured datasets (figshare GO-PE). Reports accuracy (held-out corr,
NRMSE) AND compactness (parameter / active-term count) so the comparison is on the
axes Psi-Vortex actually claims advantage on -- not just MSE of an easy fit.

Honest expectation: on a single-amplitude (V,t)->I sweep all methods fit well; the
point of the table is the trade-off across accuracy, compactness, interpretability
and deployability, not a marginal MSE win.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_go_pe import make_split
from baselines import MLPBaseline, KANBaseline, SINDyBaseline
from r1_real_data_validation import train_teacher, predict as t_predict, _load_pipeline_module

HERE = os.path.dirname(os.path.abspath(__file__))


def corr(a, b):
    return float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])


def nrmse(p, t):
    t = np.ravel(t)
    rng = float(t.max() - t.min())
    return float(np.sqrt(np.mean((np.ravel(p) - t) ** 2)) / (rng + 1e-12))


def synth_split(device_id=0):
    """Held-out split of the synthetic printed-memristor benchmark (train cyc0/test cyc1)."""
    df = pd.read_csv(os.path.join(HERE, "printed_memristor_training_data.csv"))
    df = df[df.device_id == device_id]
    vs = float(np.abs(df.voltage.values).max())
    isc = float(np.abs(df.current.values).max())

    def cyc(c):
        d = df[df.cycle_id == c]
        return (d.voltage.values / vs, np.linspace(0, 1, len(d)), d.current.values / isc)

    return {"train": cyc(0), "test": cyc(1), "i_scale": isc}


def real_split(device):
    sp = make_split(device)
    tr = tuple(x.numpy().ravel() for x in sp["train"])
    te = tuple(x.numpy().ravel() for x in sp["test"])
    return {"train": tr, "test": te, "i_scale": sp["i_scale"]}


def run_methods(ds, mod, seed):
    """One seed: returns [(method, corr, nrmse, n_params)] + the SINDy model.
    All methods use the SAME seed -> equal-seed comparison (addresses R7)."""
    Vtr, ttr, Itr = ds["train"]
    Vte, tte, Ite = ds["test"]
    rows = []

    # Psi-Vortex teacher (reuse the framework model + training settings)
    to_col = lambda a: torch.tensor(a, dtype=torch.float32).view(-1, 1)
    teacher = train_teacher(mod, to_col(Vtr), to_col(ttr), to_col(Itr), seed=seed)
    p = t_predict(teacher, to_col(Vte), to_col(tte)).ravel()
    rows.append(("Psi-Vortex (teacher)", corr(p, Ite), nrmse(p, Ite),
                 sum(pp.numel() for pp in teacher.parameters())))

    # Contemporary baselines (fresh instance per seed)
    sindy = SINDyBaseline()
    for B in [MLPBaseline(seed=seed), KANBaseline(seed=seed), sindy]:
        B.fit(Vtr, ttr, Itr)
        p = B.predict(Vte, tte)
        rows.append((B.name, corr(p, Ite), nrmse(p, Ite), B.n_params))
    return rows, sindy


def main():
    mod = _load_pipeline_module()
    datasets = {
        "synthetic (printed memristor)": synth_split(0),
        "real GO-PDADMAC4": real_split("GO-PDADMAC4"),
        "real GO-PEI3": real_split("GO-PEI3"),
        "real GO-PEI4": real_split("GO-PEI4"),
    }

    SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
    all_rows, sindy_eq = [], None
    print(f"\nHEAD-TO-HEAD: compact modeling (held-out, {len(SEEDS)} equal seeds)")
    for dname, ds in datasets.items():
        for seed in SEEDS:
            rows, sindy = run_methods(ds, mod, seed)
            if dname == "real GO-PEI4" and seed == SEEDS[0]:
                sindy_eq = sindy.equation()
            for method, c, nr, npar in rows:
                all_rows.append({"dataset": dname, "method": method, "seed": seed,
                                 "corr": c, "nrmse": nr, "n_params": npar})

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(HERE, "r2_compact_baselines.csv"), index=False)

    # Aggregate over the (equal) seeds -> mean +/- std
    agg = (df.groupby(["dataset", "method"])
             .agg(corr=("corr", "mean"), nrmse=("nrmse", "mean"),
                  nrmse_std=("nrmse", "std"), n_params=("n_params", "first"))
             .reset_index())
    agg["nrmse_std"] = agg["nrmse_std"].fillna(0.0)   # SINDy is deterministic
    agg.to_csv(os.path.join(HERE, "r2_compact_baselines_summary.csv"), index=False)
    print(agg.to_string(index=False))
    print("\nSINDy model (real GO-PEI4):", sindy_eq)

    order = ["Psi-Vortex (teacher)", "MLP", "PIKAN", "SINDy"]
    piv_m = agg.pivot(index="method", columns="dataset", values="nrmse").reindex(order)
    piv_s = agg.pivot(index="method", columns="dataset", values="nrmse_std").reindex(order)
    par = agg.pivot(index="method", columns="dataset", values="n_params").reindex(order)
    with open(os.path.join(HERE, "r2_compact_baselines.tex"), "w") as f:
        cols = list(piv_m.columns)
        f.write("\\begin{tabular}{l" + "c" * (len(cols) + 1) + "}\n\\toprule\n")
        f.write("Method & " + " & ".join(c.replace("_", " ") for c in cols) + " & Params \\\\\n")
        f.write("\\midrule\n")
        for m in order:
            vals = " & ".join(f"{piv_m.loc[m, c]:.3f}$\\pm${piv_s.loc[m, c]:.3f}" for c in cols)
            f.write(f"{m} & {vals} & {int(par.loc[m, cols[0]])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write(f"% Held-out NRMSE (mean$\\pm$std over {len(SEEDS)} equal seeds); lower is better. "
                "Params = native model size (SINDy = active terms).\n")

    # Figure: accuracy vs compactness trade-off (mean params vs mean held-out NRMSE)
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"synthetic (printed memristor)": "o", "real GO-PDADMAC4": "s",
               "real GO-PEI3": "^", "real GO-PEI4": "D"}
    colors = {"Psi-Vortex (teacher)": "C3", "MLP": "C0", "PIKAN": "C2", "SINDy": "C1"}
    for _, r in agg.iterrows():
        ax.scatter(r["n_params"], r["nrmse"], marker=markers[r["dataset"]],
                   c=colors[r["method"]], s=70, edgecolor="k", linewidth=0.4)
    for meth, col in colors.items():
        ax.scatter([], [], c=col, label=meth, s=70, edgecolor="k", linewidth=0.4)
    for dname, mk in markers.items():
        ax.scatter([], [], marker=mk, c="0.5", label=dname, s=70, edgecolor="k", linewidth=0.4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("parameters / active terms (log)")
    ax.set_ylabel("held-out NRMSE (log)")
    ax.set_title(f"Accuracy vs compactness (held-out, {len(SEEDS)} seeds)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "r2_compact_baselines.png"), dpi=150)
    print("Saved: r2_compact_baselines.csv / _summary.csv / .tex / .png")


if __name__ == "__main__":
    main()
