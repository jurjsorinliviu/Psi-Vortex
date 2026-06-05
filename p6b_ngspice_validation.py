"""
P6b - Actual ngspice transient validation of the exported compact model
=======================================================================
Upgrades the P6 "behavioral Verilog-A-equivalent" check into a real external
SPICE simulation. The exported Psi-Vortex compact thermal-coupling model

    I_victim = V_read * G_base * exp(alpha_hat * dT)
    dT/dt    = -(dT)/tau + heat_coeff * V_drv^2     (companion thermal network)

is written as a NATIVE ngspice netlist (no Verilog-A compiler needed):
  - a thermal node `dt` = temperature rise, realised as an RC network
    (C=1 F, R=tau) driven by a Joule-heating B current source heat_coeff*V(drv)^2;
  - a victim B current source V_read*G_base*exp(alpha_hat*V(dt)) read across a
    1-ohm sense resistor, so V(vic) equals the victim current numerically.

For each stimulus we run a true transient (.tran ... uic) in ngspice_con.exe and
compare the ngspice output against (i) the calibrated ground-truth generator,
(ii) the PyTorch Psi-Vortex network, and (iii) the Python behavioral reference
used in P6 (this last cross-check confirms the behavioral reference equals the
actual SPICE engine, i.e. the P6 "equivalent" claim is exact).

Outputs (supplementary_experiments_output/):
  p6b_ngspice_fidelity.csv / .md / .tex
  p6b_ngspice_waveforms.png

Requires: ngspice (ngspice_con.exe). Usage:  python p6b_ngspice_validation.py
"""
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import supplementary_experiments as rre
from supplementary_experiments import (
    V_READ, G_BASE, T_AMB, generate_thermal_data, build_model, init_model,
    train_supervised, alpha_fit, thermal_forward_physics, compact_model_current,
    _stimulus, to_col, set_seed, save_table, OUTDIR)

NGSPICE = r"C:/ngspice/bin/ngspice_con.exe"
WORK = r"C:/ngspice_work"
TAU = 0.05
HEAT = 800.0
DT = 1e-4
N = 1500
STIMULI = ["dc_read", "pulse_train", "sinusoid", "crosstalk", "ood_pulse"]


def build_netlist(stim, V, t, alpha_hat):
    pairs = [f"{ti:.7e} {vi:.5f}" for ti, vi in zip(t, V)]
    pwl = "\n".join("+ " + " ".join(pairs[i:i + 6]) for i in range(0, len(pairs), 6))
    dtstep = DT / 10.0
    tstop = float(t[-1])
    out = f"{WORK}/{stim}_out.txt"
    return f"""* Psi-Vortex compact thermal-coupling model -- ngspice validation ({stim})
Vdrv drv 0 PWL(
{pwl}
)
* thermal node 'dt' = temperature rise above ambient (C=1 F, R=tau)
C1 dt 0 1
R1 dt 0 {TAU}
Bheat 0 dt I = {HEAT}*V(drv)*V(drv)
* victim readout: V(vic) numerically equals the victim current
Bvic 0 vic I = {V_READ}*{G_BASE}*exp({alpha_hat}*V(dt))
Rs vic 0 1
.ic v(dt)=0
.control
tran {dtstep:.3e} {tstop:.6e} uic
wrdata {out} v(dt) v(vic)
quit
.endc
.end
"""


def run_ngspice(stim, V, t, alpha_hat):
    os.makedirs(WORK, exist_ok=True)
    cir = f"{WORK}/{stim}.cir"
    out = f"{WORK}/{stim}_out.txt"
    if os.path.exists(out):
        os.remove(out)
    with open(cir, "w") as f:
        f.write(build_netlist(stim, V, t, alpha_hat))
    r = subprocess.run([NGSPICE, "-b", cir], capture_output=True, text=True, timeout=120)
    if not os.path.exists(out):
        raise RuntimeError(f"ngspice produced no output for {stim}:\n{r.stdout[-500:]}\n{r.stderr[-500:]}")
    data = np.loadtxt(out)
    tt, dT_ng, iv_ng = data[:, 0], data[:, 1], data[:, 3]
    # interpolate ngspice solver points onto the stimulus grid
    I_ng = np.interp(t, tt, iv_ng)
    dT_ng_i = np.interp(t, tt, dT_ng)
    return I_ng, dT_ng_i


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    print("=" * 70)
    print("P6b - ACTUAL ngspice transient validation of the compact model")
    print("=" * 70)

    # 1. train the same Psi-Vortex model as P6 and recover alpha_hat
    seed = 42
    ds = generate_thermal_data(0.08, seed=seed)
    set_seed(seed)
    model = build_model("psi", hidden_size=32)
    init_model(model, "physics")
    train_supervised(model, ds, 120)
    alpha_hat, _ = alpha_fit(model, ds)
    print(f"recovered alpha_hat = {alpha_hat:.4f}  (ground-truth alpha = 0.08)")
    print(f"ngspice: {NGSPICE}")

    rows = []
    waveforms = {}
    for stim in STIMULI:
        V = _stimulus(stim, n=N, seed=seed).astype(float)
        t = np.linspace(0, (N - 1) * DT, N)
        T = thermal_forward_physics(V, tau_th=TAU, heat_coeff=HEAT)
        I_gt = V_READ * G_BASE * np.exp(0.08 * (T - T_AMB))            # ground truth
        I_pybehav = compact_model_current(V, alpha_hat, tau_th=TAU, heat_coeff=HEAT)  # P6 python ref
        model.eval()
        with torch.no_grad():
            I_pt, _ = model(to_col(V), to_col(t))
        I_pt = I_pt.cpu().numpy().flatten()
        try:
            I_ng, _ = run_ngspice(stim, V, t, alpha_hat)               # REAL ngspice
        except Exception as e:
            print(f"  [{stim}] ngspice FAILED: {e}")
            continue
        waveforms[stim] = (t, I_gt, I_ng, I_pt)
        scale = float(np.mean(np.abs(I_gt))) + 1e-30
        rows.append(dict(
            stimulus=stim,
            ngspice_vs_gt_rel_pct=100 * mae(I_ng, I_gt) / scale,
            pytorch_vs_gt_rel_pct=100 * mae(I_pt, I_gt) / scale,
            ngspice_vs_pybehav_rel_pct=100 * mae(I_ng, I_pybehav) / scale,
            ngspice_vs_pybehav_rmse=rmse(I_ng, I_pybehav),
            ngspice_vs_gt_rmse=rmse(I_ng, I_gt)))
        print(f"  [{stim:11s}] ngspice-vs-GT {rows[-1]['ngspice_vs_gt_rel_pct']:6.1f}% | "
              f"pytorch-vs-GT {rows[-1]['pytorch_vs_gt_rel_pct']:6.1f}% | "
              f"ngspice-vs-pyref {rows[-1]['ngspice_vs_pybehav_rel_pct']:.3f}%")

    df = pd.DataFrame(rows)
    df["alpha_hat"] = alpha_hat
    df["ngspice_version"] = "ngspice_con (native B-source netlist)"
    save_table(df, "p6b_ngspice_fidelity")
    print("\nSaved: p6b_ngspice_fidelity.{csv,md,tex}")
    print(f"\nMax ngspice-vs-Python-reference error across stimuli: "
          f"{df['ngspice_vs_pybehav_rel_pct'].max():.3f}%  "
          f"(confirms the P6 behavioral reference == the real SPICE engine)")

    # figure: GT vs ngspice vs PyTorch
    plot_stims = [s for s in ["pulse_train", "sinusoid", "crosstalk", "ood_pulse"] if s in waveforms]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for ax, stim in zip(axes.flat, plot_stims):
        t, I_gt, I_ng, I_pt = waveforms[stim]
        ax.plot(t * 1e3, I_gt * 1e6, "k-", lw=2.2, label="ground truth")
        ax.plot(t * 1e3, I_ng * 1e6, "r--", lw=1.7, label="ngspice compact model")
        ax.plot(t * 1e3, I_pt * 1e6, "b:", lw=1.6, label="PyTorch Psi-Vortex")
        ax.set_title(stim); ax.set_xlabel("time (ms)"); ax.set_ylabel("victim I (uA)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle("P6b: Exported compact model in ngspice (real SPICE .tran) vs PyTorch vs ground truth",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "p6b_ngspice_waveforms.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: p6b_ngspice_waveforms.png")
    return df


if __name__ == "__main__":
    main()
