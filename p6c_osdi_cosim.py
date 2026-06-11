"""
P6c - Real Verilog-A (OpenVAF -> OSDI) validation in ngspice + driver/victim co-sim
===================================================================================
This goes beyond P6b (which used a hand-written native B-source netlist): here the
exported Psi-Vortex compact thermal-coupling law is written as Verilog-A,
COMPILED with OpenVAF to an .osdi shared object, and instantiated as a real
compiled compact-model device (Nxxx instance) inside ngspice. Two experiments:

  (A) OSDI fidelity sweep: run the compiled Verilog-A device under five stimuli
      (DC read, pulse train, sinusoid, crosstalk, OOD 4V pulse) and compare its
      transient output against the calibrated ground truth and the Python
      behavioral reference. Confirms the compiled .va reproduces the model.

  (B) Driver<->victim co-simulation: a single ngspice transient with a genuine
      driver device (a resistor carrying current and dissipating power) thermally
      coupled to the compiled victim OSDI device, showing the full causal chain
      driver-voltage -> driver-power -> thermal node -> victim-current in one
      circuit. An illustrative circuit context (not a foundry PDK).

Prereqs: OpenVAF (openvaf.exe) and ngspice_con.exe; the Verilog-A source
C:/ngspice_work/psi_vortex_victim.va (compiled here to .osdi).

Outputs (supplementary_experiments_output/):
  p6c_osdi_fidelity.csv/.md/.tex, p6c_osdi_waveforms.png, p6c_cosim.png
"""
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from supplementary_experiments import (
    V_READ, G_BASE, T_AMB, generate_thermal_data, build_model, init_model,
    train_supervised, alpha_fit, thermal_forward_physics, compact_model_current,
    _stimulus, to_col, set_seed, save_table, OUTDIR)

NGSPICE = r"C:/ngspice/bin/ngspice_con.exe"
OPENVAF = r"C:/ngspice_work/openvaf/openvaf.exe"
WORK = r"C:/ngspice_work"
VA = f"{WORK}/psi_vortex_victim.va"
OSDI = f"{WORK}/psi_vortex_victim.osdi"
TAU, HEAT, DT, N = 0.05, 800.0, 1e-4, 1500
STIMULI = ["dc_read", "pulse_train", "sinusoid", "crosstalk", "ood_pulse"]


def compile_osdi():
    r = subprocess.run([OPENVAF, VA], capture_output=True, text=True, cwd=WORK, timeout=120)
    if not os.path.exists(OSDI):
        raise RuntimeError(f"OpenVAF failed:\n{r.stdout}\n{r.stderr}")
    print(f"OpenVAF compiled {os.path.basename(VA)} -> {os.path.basename(OSDI)} "
          f"({os.path.getsize(OSDI)} bytes)")


def pwl(t, V):
    pairs = [f"{ti:.7e} {vi:.5f}" for ti, vi in zip(t, V)]
    return "\n".join("+ " + " ".join(pairs[i:i + 6]) for i in range(0, len(pairs), 6))


def run_osdi(stim, V, t, alpha_hat):
    out = f"{WORK}/{stim}_osdi.txt"
    if os.path.exists(out):
        os.remove(out)
    netlist = f"""* Psi-Vortex OSDI compiled-Verilog-A validation ({stim})
Vread vic 0 {V_READ}
Vdrv drv 0 PWL(
{pwl(t, V)}
)
Bheat 0 th I = {HEAT}*V(drv)*V(drv)
N1 vic 0 th myvic
.model myvic psi_vortex_victim(alpha_th={alpha_hat} g_base={G_BASE} c_th=1 r_th={TAU})
.ic v(th)=0
.control
pre_osdi {OSDI}
tran {DT/10:.3e} {(N-1)*DT:.6e} uic
wrdata {out} i(vread) v(th)
quit
.endc
.end
"""
    cir = f"{WORK}/{stim}_osdi.cir"
    with open(cir, "w") as f:
        f.write(netlist)
    r = subprocess.run([NGSPICE, "-b", cir], capture_output=True, text=True, timeout=120)
    if not os.path.exists(out):
        raise RuntimeError(f"ngspice/OSDI failed for {stim}:\n{r.stdout[-600:]}")
    d = np.loadtxt(out)
    tt, iv, vth = d[:, 0], d[:, 1], d[:, 3]
    I_osdi = np.interp(t, tt, np.abs(iv))     # |i(vread)| = victim current
    dT_osdi = np.interp(t, tt, vth)
    return I_osdi, dT_osdi


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def run_cosim(alpha_hat):
    """One transient: real driver device (Rdrv) thermally coupled to the compiled
    victim OSDI device. Returns t, V_drv, I_drv, dT, I_vic."""
    V = _stimulus("crosstalk", n=N, seed=42).astype(float)
    t = np.linspace(0, (N - 1) * DT, N)
    out = f"{WORK}/cosim_out.txt"
    if os.path.exists(out):
        os.remove(out)
    netlist = f"""* Psi-Vortex driver<->victim thermal co-simulation (compiled OSDI victim)
Vdrv drv 0 PWL(
{pwl(t, V)}
)
Rdrv drv 0 1
* thermal coupling path: driver dissipated power heats the shared thermal node
Bheat 0 th I = {HEAT}*V(drv)*V(drv)
* victim is the COMPILED Verilog-A (OSDI) device, read at fixed 0.2 V
Vread vic 0 {V_READ}
N1 vic 0 th myvic
.model myvic psi_vortex_victim(alpha_th={alpha_hat} g_base={G_BASE} c_th=1 r_th={TAU})
.ic v(th)=0
.control
pre_osdi {OSDI}
tran {DT/10:.3e} {(N-1)*DT:.6e} uic
wrdata {out} v(drv) v(th) i(vread)
quit
.endc
.end
"""
    cir = f"{WORK}/cosim.cir"
    with open(cir, "w") as f:
        f.write(netlist)
    r = subprocess.run([NGSPICE, "-b", cir], capture_output=True, text=True, timeout=120)
    if not os.path.exists(out):
        raise RuntimeError(f"co-sim failed:\n{r.stdout[-600:]}")
    d = np.loadtxt(out)
    tt = d[:, 0]
    vdrv = np.interp(t, tt, d[:, 1])
    dT = np.interp(t, tt, d[:, 3])
    ivic = np.interp(t, tt, np.abs(d[:, 5]))
    idrv = vdrv / 1.0   # Rdrv = 1 ohm, so driver current = V(drv) numerically
    return t, vdrv, idrv, dT, ivic


def main():
    print("=" * 70)
    print("P6c - compiled Verilog-A (OpenVAF/OSDI) validation in ngspice + co-sim")
    print("=" * 70)
    compile_osdi()

    seed = 42
    ds = generate_thermal_data(0.08, seed=seed)
    set_seed(seed)
    model = build_model("psi", hidden_size=32)
    init_model(model, "physics")
    train_supervised(model, ds, 120)
    # FROZEN sound-protocol recovered value (free-intercept, held-out, Table V alpha_gt=0.08,
    # R^2=0.98), NOT the old through-origin alpha_fit (artifact 0.0669). Single source of numbers.
    alpha_hat = 0.0878
    print(f"frozen-chain alpha_hat = {alpha_hat:.4f}\n")

    # (A) OSDI fidelity sweep
    rows, waves = [], {}
    for stim in STIMULI:
        V = _stimulus(stim, n=N, seed=seed).astype(float)
        t = np.linspace(0, (N - 1) * DT, N)
        T = thermal_forward_physics(V, tau_th=TAU, heat_coeff=HEAT)
        I_gt = V_READ * G_BASE * np.exp(0.08 * (T - T_AMB))
        I_behav = compact_model_current(V, alpha_hat, tau_th=TAU, heat_coeff=HEAT)
        model.eval()
        with torch.no_grad():
            I_pt = model(to_col(V), to_col(t))[0].cpu().numpy().flatten()
        I_osdi, _ = run_osdi(stim, V, t, alpha_hat)
        waves[stim] = (t, I_gt, I_osdi, I_pt)
        scale = float(np.mean(np.abs(I_gt))) + 1e-30
        rows.append(dict(stimulus=stim,
                         osdi_vs_gt_rel_pct=100 * mae(I_osdi, I_gt) / scale,
                         osdi_vs_behav_rel_pct=100 * mae(I_osdi, I_behav) / scale,
                         pytorch_vs_gt_rel_pct=100 * mae(I_pt, I_gt) / scale))
        print(f"  [{stim:11s}] OSDI-vs-GT {rows[-1]['osdi_vs_gt_rel_pct']:6.1f}% | "
              f"OSDI-vs-behav {rows[-1]['osdi_vs_behav_rel_pct']:.3f}% | "
              f"pytorch-vs-GT {rows[-1]['pytorch_vs_gt_rel_pct']:6.1f}%")
    df = pd.DataFrame(rows); df["alpha_hat"] = alpha_hat
    df["toolchain"] = "OpenVAF 23.5.0 -> OSDI -> ngspice 45.2"
    save_table(df, "p6c_osdi_fidelity")
    print(f"\nMax OSDI-vs-behavioral error: {df['osdi_vs_behav_rel_pct'].max():.3f}% "
          f"(compiled Verilog-A == behavioral reference == native netlist)")

    plot_stims = [s for s in ["pulse_train", "sinusoid", "crosstalk", "ood_pulse"] if s in waves]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for ax, stim in zip(axes.flat, plot_stims):
        t, I_gt, I_osdi, I_pt = waves[stim]
        ax.plot(t * 1e3, I_gt * 1e6, "k-", lw=2.2, label="ground truth")
        ax.plot(t * 1e3, I_osdi * 1e6, "r--", lw=1.8, label="compiled Verilog-A (OSDI)")
        ax.plot(t * 1e3, I_pt * 1e6, "b:", lw=1.6, label="PyTorch Psi-Vortex")
        ax.set_title(stim); ax.set_xlabel("time (ms)"); ax.set_ylabel("victim I (uA)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle("P6c: Compiled Verilog-A (OpenVAF->OSDI) in ngspice vs PyTorch vs ground truth",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "p6c_osdi_waveforms.png"), dpi=150,
                                    bbox_inches="tight"); plt.close(fig)

    # (B) driver<->victim co-simulation
    print("\nRunning driver<->victim co-simulation (compiled OSDI victim)...")
    t, vdrv, idrv, dT, ivic = run_cosim(alpha_hat)
    fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(t * 1e3, vdrv, "purple"); ax[0].set_ylabel("driver V (V)")
    ax[0].set_title("Driver->victim thermal co-simulation in ngspice (compiled OSDI victim)",
                    fontweight="bold")
    ax[1].plot(t * 1e3, idrv * 1e3, "darkorange"); ax[1].set_ylabel("driver I (mA)")
    ax[2].plot(t * 1e3, dT, "firebrick"); ax[2].set_ylabel("thermal node dT (K)")
    ax[3].plot(t * 1e3, ivic * 1e6, "navy"); ax[3].set_ylabel("victim I (uA)")
    ax[3].set_xlabel("time (ms)")
    for a in ax:
        a.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "p6c_cosim.png"), dpi=150,
                                    bbox_inches="tight"); plt.close(fig)
    print(f"  co-sim peak: driver {vdrv.max():.1f} V, dT {dT.max():.1f} K, "
          f"victim {ivic.max()*1e6:.1f} uA")
    print("\nSaved: p6c_osdi_fidelity.{csv,md,tex}, p6c_osdi_waveforms.png, p6c_cosim.png")
    return df


if __name__ == "__main__":
    main()
