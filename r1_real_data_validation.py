"""Independent real-data validation of Psi-Vortex compact-model extraction.

Dataset: Szuwarzynski et al. (2026), figshare doi:10.6084/m9.figshare.31407306 --
real measured GO-polyelectrolyte memristor I-V sweeps (loaded via data_go_pe.py).

Four sub-experiments, all reusing the framework's own teacher/student, physics-aware
init, and training settings from 08_end_to_end_pipeline.py (no re-implementation):

  (A) Held-out-cycle fidelity per device (multi-seed, equal protocol).
  (B) Cross-device transfer matrix (train on one material, test held-out cycles of
      every material).
  (C) Cross-sweep-rate generalization (train at 400 mV/s, test at slower rates) --
      probes whether the model captures rate-dependent dynamics.
  (D) Verilog-A round-trip fidelity on real data (full EndToEndPipeline -> .va),
      the measured-data analog of the synthetic P6 experiment.

Scope guard: this validates compact-model extraction + dynamic fidelity on real
data only. It carries NO thermal-coupling ground truth; the 3D thermal-crosstalk
alpha-recovery remains an explicitly synthetic proof-of-concept.

Fidelity metrics: Pearson correlation (scale-invariant) and range-normalized RMSE
(NRMSE). The current scale (uA) is reported so physical RMSE can be recovered.
"""
import contextlib
import io
import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_go_pe import load_iv_sheet, make_split, to_tensors

HERE = os.path.dirname(os.path.abspath(__file__))

# NOTE: in the source dataset the GO-PDADMAC3 and GO-PDADMAC4 I-V workbooks carry
# identical numeric content (verified bit-identical across sheets), so only one
# PDADMAC representative is used -> three genuinely distinct measured datasets.
DEVICES = ["GO-PDADMAC4", "GO-PEI3", "GO-PEI4"]
SHEET = "1p0_400mvs"                       # +/-1.0 V, 400 mV/s
RATE_SHEETS = ["1p0_400mvs", "1p0_200mvs", "1p0_100mvs", "1p0_50mvs"]
SEEDS = [0, 1, 2]


@contextlib.contextmanager
def silence():
    """Suppress the framework's chatty stdout during training / pipeline runs."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _load_pipeline_module():
    spec = importlib.util.spec_from_file_location(
        "e2e", os.path.join(HERE, "08_end_to_end_pipeline.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def nrmse(pred, tgt):
    rng = float(tgt.max() - tgt.min())
    return float(np.sqrt(np.mean((pred - tgt) ** 2)) / (rng + 1e-12))


def train_teacher(mod, V, t, I, epochs=500, lr=0.005, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    teacher = mod.PSI_xLSTM_Teacher(
        input_size=2, hidden_size=64, output_size=1, memory_size=32
    )
    with silence():
        mod.apply_auto_vortex_init(teacher, V.flatten(), I.flatten(), verbose=False)
    opt = torch.optim.Adam(teacher.parameters(), lr=lr)
    lf = torch.nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        pred, _ = teacher(V, t)
        loss = lf(pred, I)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
        opt.step()
    return teacher


@torch.no_grad()
def predict(model, V, t):
    model.eval()
    dev = next(model.parameters()).device
    pred, _ = model(V.to(dev), t.to(dev))
    return pred.cpu().numpy()


def main():
    mod = _load_pipeline_module()
    splits = {d: make_split(d, SHEET) for d in DEVICES}

    # ---- (A) Held-out fidelity per device (multi-seed) -------------------
    print("\n[A] HELD-OUT-CYCLE FIDELITY (per device, %d seeds)" % len(SEEDS))
    dev_rows, models = [], {}
    for d in DEVICES:
        Vtr, ttr, Itr = splits[d]["train"]
        Vte, tte, Ite = splits[d]["test"]
        cs, ns = [], []
        for s in SEEDS:
            m = train_teacher(mod, Vtr, ttr, Itr, seed=s)
            p = predict(m, Vte, tte)
            cs.append(corr(p, Ite.numpy()))
            ns.append(nrmse(p, Ite.numpy()))
            if s == 0:
                models[d] = m
        dev_rows.append({
            "device": d,
            "holdout_corr_mean": np.mean(cs), "holdout_corr_std": np.std(cs),
            "holdout_nrmse_mean": np.mean(ns), "holdout_nrmse_std": np.std(ns),
            "i_scale_uA": splits[d]["i_scale"],
        })
        print(f"    {d:12s} corr {np.mean(cs):.4f}+/-{np.std(cs):.4f}   "
              f"NRMSE {np.mean(ns):.4f}+/-{np.std(ns):.4f}")
    df_dev = pd.DataFrame(dev_rows)

    # ---- (B) Cross-device transfer matrix -------------------------------
    print("\n[B] CROSS-DEVICE TRANSFER  (row = trained on, col = held-out of)")
    n = len(DEVICES)
    cmat = np.zeros((n, n))
    nmat = np.zeros((n, n))
    for i, dx in enumerate(DEVICES):
        for j, dy in enumerate(DEVICES):
            Vte, tte, Ite = splits[dy]["test"]
            p = predict(models[dx], Vte, tte)
            cmat[i, j] = corr(p, Ite.numpy())
            nmat[i, j] = nrmse(p, Ite.numpy())
    df_mat = pd.DataFrame(cmat, index=DEVICES, columns=DEVICES)
    print(df_mat.round(4).to_string())
    n_other = max(1, len(DEVICES) - 1)
    cross_avg = {d: (cmat[i].sum() - cmat[i, i]) / n_other for i, d in enumerate(DEVICES)}

    # ---- (C) Cross-sweep-rate generalization ----------------------------
    print("\n[C] CROSS-SWEEP-RATE  (model = GO-PEI4 @ 400 mV/s, applied to slower)")
    ref = models["GO-PEI4"]
    rate_rows = []
    for rs in RATE_SHEETS:
        df = load_iv_sheet("GO-PEI4", rs)
        v = float(np.abs(df["V"].values).max())
        ic = float(np.abs(df["I"].values).max())
        V, t, I = to_tensors(df, sorted(df["cycle"].unique()), v, ic)
        p = predict(ref, V, t)
        rate_rows.append({"rate_sheet": rs, "corr": corr(p, I.numpy()),
                          "nrmse": nrmse(p, I.numpy()), "i_scale_uA": ic})
        tag = "  (train rate)" if rs == SHEET else ""
        print(f"    {rs:12s} corr {corr(p, I.numpy()):.4f}  NRMSE {nrmse(p, I.numpy()):.4f}{tag}")
    df_rate = pd.DataFrame(rate_rows)

    # ---- (D) Verilog-A round-trip fidelity on real data -----------------
    print("\n[D] VERILOG-A ROUND-TRIP (full EndToEndPipeline on GO-PEI4 real data)")
    Vtr, ttr, Itr = splits["GO-PEI4"]["train"]
    Vte, tte, Ite = splits["GO-PEI4"]["test"]
    va_path = os.path.join(HERE, "psi_vortex_GO-PEI4_real.va")
    with silence():
        pipe = mod.EndToEndPipeline()
        res = pipe.run(Vtr, ttr, Itr, output_va_file=va_path)
    student = res["student"]
    p = predict(student, Vte, tte)
    va_corr, va_nrmse = corr(p, Ite.numpy()), nrmse(p, Ite.numpy())
    comp = res["metrics"]["stage4"]["compression_ratio"]
    sym = res["metrics"]["stage1"]["symmetry_type"]
    print(f"    detected symmetry={sym}  compression={comp:.1f}%")
    print(f"    compact-student held-out  corr {va_corr:.4f}  NRMSE {va_nrmse:.4f}")
    print(f"    Verilog-A written to {os.path.basename(va_path)}")

    # ---- Save tables ----------------------------------------------------
    df_dev["cross_device_corr_avg"] = [cross_avg[d] for d in df_dev["device"]]
    df_dev.to_csv(os.path.join(HERE, "r1_real_data_results.csv"), index=False)
    df_mat.to_csv(os.path.join(HERE, "r1_cross_device_matrix.csv"))
    df_rate.to_csv(os.path.join(HERE, "r1_cross_rate_results.csv"), index=False)

    with open(os.path.join(HERE, "r1_real_data_validation.tex"), "w") as f:
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Device & Held-out corr & Held-out NRMSE & Cross-device corr & $I$ scale ($\\mu$A) \\\\\n")
        f.write("\\midrule\n")
        for _, r in df_dev.iterrows():
            f.write(f"{r['device']} & {r['holdout_corr_mean']:.4f}$\\pm${r['holdout_corr_std']:.4f} "
                    f"& {r['holdout_nrmse_mean']:.4f}$\\pm${r['holdout_nrmse_std']:.4f} "
                    f"& {r['cross_device_corr_avg']:.4f} & {r['i_scale_uA']:.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---- Figure: measured vs predicted I-V loops (held-out, cross-device)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # held-out, GO-PEI4
    s = splits["GO-PEI4"]
    Vte, tte, Ite = s["test"]
    p = predict(models["GO-PEI4"], Vte, tte)
    ax[0].scatter(Vte.numpy() * s["v_scale"], Ite.numpy() * s["i_scale"],
                  s=6, c="0.6", label="measured")
    ax[0].scatter(Vte.numpy() * s["v_scale"], p * s["i_scale"],
                  s=6, c="C3", alpha=0.6, label="Psi-Vortex")
    ax[0].set_title(f"GO-PEI4 held-out cycles\ncorr={corr(p, Ite.numpy()):.3f}")
    ax[0].set_xlabel("Voltage (V)"); ax[0].set_ylabel("Current ($\\mu$A)"); ax[0].legend()
    # cross-device, GO-PEI4 -> GO-PDADMAC4
    s2 = splits["GO-PDADMAC4"]
    Vte2, tte2, Ite2 = s2["test"]
    p2 = predict(models["GO-PEI4"], Vte2, tte2)
    ax[1].scatter(Vte2.numpy() * s2["v_scale"], Ite2.numpy() * s2["i_scale"],
                  s=6, c="0.6", label="measured")
    ax[1].scatter(Vte2.numpy() * s2["v_scale"], p2 * s2["i_scale"],
                  s=6, c="C0", alpha=0.6, label="Psi-Vortex (GO-PEI4 model)")
    ax[1].set_title(f"Cross-device GO-PEI4$\\rightarrow$GO-PDADMAC4\ncorr={corr(p2, Ite2.numpy()):.3f}")
    ax[1].set_xlabel("Voltage (V)"); ax[1].set_ylabel("Current ($\\mu$A)"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "r1_real_data_validation.png"), dpi=150)
    print("\nSaved: r1_real_data_results.csv, r1_cross_device_matrix.csv, "
          "r1_cross_rate_results.csv, r1_real_data_validation.tex, "
          "r1_real_data_validation.png")


if __name__ == "__main__":
    main()
