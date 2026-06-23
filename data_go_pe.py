"""Loader for the INDEPENDENT measured memristor I-V dataset.

Szuwarzynski, Kruk, Kostecka, Chrzaszcz, Mazur, Wieczorek (2026),
"Interfacial Organization of Graphene Oxide-Polyelectrolyte Multilayers with
Tunable Memristive Behavior", figshare, doi:10.6084/m9.figshare.31407306.

Real measured GO-polyelectrolyte memristor sweeps used for independent real-data
validation of Psi-Vortex compact-model extraction. Each xlsx has 28 sheets named
<amplitude>_<rate> (e.g. "1p0_400mvs" = +/-1.0 V at 400 mV/s); every sheet holds
several measured cycles with columns [cycle_id, voltage (V), current (uA)].
"""
import os
import numpy as np
import pandas as pd
import torch

# Folder name contains an en-dash (U+2013); spell it with an escape for safety.
DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Interfacial Organization of Graphene Oxide–Polyelectrolyte "
    "Multilayers with Tunable Memristive Behavior",
)

DEVICES = {
    "GO-PDADMAC3": "IV-plot_GO-PDADMAC3.xlsx",
    "GO-PDADMAC4": "IV-plot_GO-PDADMAC4.xlsx",
    "GO-PEI3": "IV-plot_GO-PEI3.xlsx",
    "GO-PEI4": "IV-plot_GO-PEI4.xlsx",
}


def load_iv_sheet(device, sheet="1p0_400mvs"):
    """Tidy DataFrame [cycle, V, I] of measured I-V for one device/sheet."""
    path = os.path.join(DATA_DIR, DEVICES[device])
    df = pd.read_excel(path, sheet_name=sheet, header=None)
    df = df.iloc[:, :3].copy()
    df.columns = ["cycle", "V", "I"]
    df = df.dropna()
    df["cycle"] = df["cycle"].astype(int)
    return df


def to_tensors(df, cycles, v_scale, i_scale):
    """Build [N,1] max-abs-normalized (V, t, I) tensors for the given cycle ids.

    t runs 0..1 within each cycle so repeated cycles are phase-aligned.
    """
    Vs, ts, Is = [], [], []
    for c in cycles:
        d = df[df["cycle"] == c]
        Vs.append(d["V"].values / v_scale)
        Is.append(d["I"].values / i_scale)
        ts.append(np.linspace(0.0, 1.0, len(d)))
    V = torch.tensor(np.concatenate(Vs), dtype=torch.float32).view(-1, 1)
    t = torch.tensor(np.concatenate(ts), dtype=torch.float32).view(-1, 1)
    I = torch.tensor(np.concatenate(Is), dtype=torch.float32).view(-1, 1)
    return V, t, I


def make_split(device, sheet="1p0_400mvs", train_frac=0.7):
    """Split measured cycles into train/held-out, max-abs normalized.

    Scales are computed over the whole sheet so train and test share one
    physical normalization. Returns a dict with tensors and the scales
    (i_scale in uA) needed to report fidelity in physical units.
    """
    df = load_iv_sheet(device, sheet)
    cycles = sorted(df["cycle"].unique())
    n_tr = max(1, int(round(len(cycles) * train_frac)))
    train_cycles, test_cycles = cycles[:n_tr], cycles[n_tr:]
    v_scale = float(np.abs(df["V"].values).max())
    i_scale = float(np.abs(df["I"].values).max())
    return {
        "device": device,
        "sheet": sheet,
        "train_cycles": train_cycles,
        "test_cycles": test_cycles,
        "v_scale": v_scale,
        "i_scale": i_scale,
        "train": to_tensors(df, train_cycles, v_scale, i_scale),
        "test": to_tensors(df, test_cycles, v_scale, i_scale),
    }


if __name__ == "__main__":
    sp = make_split("GO-PEI4")
    Vtr, ttr, Itr = sp["train"]
    Vte, tte, Ite = sp["test"]
    print(f"{sp['device']} / {sp['sheet']}")
    print(f"  train cycles {sp['train_cycles']} -> {len(Vtr)} pts")
    print(f"  test  cycles {sp['test_cycles']} -> {len(Vte)} pts")
    print(f"  V scale {sp['v_scale']:.3g} V, I scale {sp['i_scale']:.3g} uA")
