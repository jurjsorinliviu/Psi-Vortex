"""Stage 3: compression-fidelity trade-off on real measured data.

Turns the single 98%-compression Verilog-A point (which lost fidelity on real data)
into a usable Pareto curve: sweep the deployable student size and report held-out
fidelity on the GO-PEI4 measured dataset, against the full teacher as reference.
A reader can then pick an operating point for their accuracy/footprint budget.

Students are the pipeline's own CompactStudent (the Verilog-A export model), trained
on the real training cycles and evaluated on held-out cycles -- same protocol as the
rest of the real-data validation.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data_go_pe import make_split
from r1_real_data_validation import (_load_pipeline_module, train_teacher,
                                     predict as run_predict, corr, nrmse)

HERE = os.path.dirname(os.path.abspath(__file__))
HIDDEN_SIZES = [2, 4, 8, 16, 32, 64]


def train_model(model, V, t, I, epochs=500, lr=5e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lf = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        pred, _ = model(V, t)
        loss = lf(pred, I)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def main():
    mod = _load_pipeline_module()
    sp = make_split("GO-PEI4")
    Vtr, ttr, Itr = sp["train"]
    Vte, tte, Ite = sp["test"]
    Ite_np = Ite.numpy()

    teacher = train_teacher(mod, Vtr, ttr, Itr, seed=0)
    tparams = sum(p.numel() for p in teacher.parameters())
    pte = run_predict(teacher, Vte, tte)
    rows = [dict(model="teacher", hidden=64, params=tparams, compression_pct=0.0,
                 corr=corr(pte, Ite_np), nrmse=nrmse(pte, Ite_np))]
    print(f"teacher: {tparams} params  corr {rows[0]['corr']:.4f}  NRMSE {rows[0]['nrmse']:.4f}")

    for h in HIDDEN_SIZES:
        s = mod.CompactStudent(input_size=2, hidden_size=h, output_size=1)
        train_model(s, Vtr, ttr, Itr, seed=0)
        p = run_predict(s, Vte, tte)
        spar = sum(pp.numel() for pp in s.parameters())
        rows.append(dict(model=f"student-h{h}", hidden=h, params=spar,
                         compression_pct=100.0 * (1 - spar / tparams),
                         corr=corr(p, Ite_np), nrmse=nrmse(p, Ite_np)))
        print(f"student h={h:3d}: {spar:6d} params  ({rows[-1]['compression_pct']:.1f}% compressed)"
              f"  corr {rows[-1]['corr']:.4f}  NRMSE {rows[-1]['nrmse']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "r4_compression_fidelity.csv"), index=False)

    with open(os.path.join(HERE, "r4_compression_fidelity.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrcc}\n\\toprule\n")
        f.write("Model & Params & Compression & Held-out corr & Held-out NRMSE \\\\\n\\midrule\n")
        for _, r in df.iterrows():
            f.write(f"{r['model']} & {int(r['params'])} & {r['compression_pct']:.1f}\\% "
                    f"& {r['corr']:.4f} & {r['nrmse']:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    stu = df[df.model != "teacher"]
    ax.plot(stu["params"], stu["nrmse"], "o-", c="C0", label="compact student")
    ax.scatter(df[df.model == "teacher"]["params"], df[df.model == "teacher"]["nrmse"],
               marker="*", s=200, c="C3", label="teacher", zorder=5)
    for _, r in stu.iterrows():
        ax.annotate(f"{r['compression_pct']:.0f}%", (r["params"], r["nrmse"]),
                    textcoords="offset points", xytext=(4, 5), fontsize=7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("deployable parameters (log)")
    ax.set_ylabel("held-out NRMSE (log)")
    ax.set_title("Compression-fidelity trade-off (GO-PEI4 measured)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "r4_compression_fidelity.png"), dpi=150)
    print("Saved: r4_compression_fidelity.csv / .tex / .png")


if __name__ == "__main__":
    main()
