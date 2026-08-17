"""Authoritative trajectory loaders and manuscript physical generators."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from .data import Trajectory


AMBIENT_TEMPERATURE = 298.0
THERMAL_DT = 1e-4


@dataclass(frozen=True)
class ThermalSample:
    trajectory: Trajectory
    delta_temperature: np.ndarray
    voltage: np.ndarray
    alpha: tuple[float, ...]


def pulse_driver(
    n_steps: int,
    source_seed: int,
    *,
    amplitude: float = 2.0,
    width: int = 60,
    pulses: int | None = None,
) -> np.ndarray:
    if n_steps < 4 or width < 1:
        raise ValueError("invalid driver length or pulse width")
    rng = np.random.RandomState(source_seed)
    signal = np.zeros(n_steps, dtype=float)
    pulse_count = pulses if pulses is not None else max(3, round(6 * n_steps / 3000))
    margin = min(200, max(1, n_steps // 10))
    for _ in range(pulse_count):
        start = rng.randint(margin, max(margin + 1, n_steps - margin))
        signal[start : min(start + width, n_steps)] = amplitude
    return signal


def thermal_response(
    voltage: np.ndarray,
    *,
    tau: float = 0.05,
    heat_coefficient: float = 800.0,
    delta_t: float = THERMAL_DT,
) -> np.ndarray:
    if tau <= 0 or heat_coefficient <= 0 or delta_t <= 0:
        raise ValueError("thermal parameters must be positive")
    signal = np.asarray(voltage, dtype=float)
    temperature = np.empty(signal.size, dtype=float)
    temperature[0] = AMBIENT_TEMPERATURE
    for index in range(1, signal.size):
        derivative = (
            -(temperature[index - 1] - AMBIENT_TEMPERATURE) / tau
            + heat_coefficient * signal[index - 1] ** 2
        )
        temperature[index] = temperature[index - 1] + derivative * delta_t
    return temperature - AMBIENT_TEMPERATURE


def _apply_artifact(
    voltage: np.ndarray,
    targets: np.ndarray,
    source_seed: int,
    artifact: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = np.ones(len(voltage), dtype=bool)
    if artifact in (None, "none"):
        return voltage, targets, keep
    rng = np.random.RandomState(source_seed + 4103)
    position = np.linspace(0, 1, len(voltage))[:, None]
    if artifact == "contact_drift":
        targets = targets + 0.03 * position
    elif artifact == "read_noise":
        targets = targets + 0.03 * np.std(targets, axis=0, keepdims=True) * rng.randn(*targets.shape)
    elif artifact == "amplitude_jitter":
        voltage = voltage * (1 + 0.03 * rng.randn(len(voltage)))
    elif artifact == "baseline_drift":
        targets = targets + 0.04 * np.sin(2 * np.pi * position)
    elif artifact == "device_variation":
        targets = targets * 1.05
    elif artifact == "aging":
        targets = targets * (1 - 0.08 * position)
    elif artifact == "missing_samples":
        keep[rng.choice(len(voltage), size=max(1, len(voltage) // 20), replace=False)] = False
    else:
        raise ValueError(f"unknown artifact: {artifact}")
    return voltage, targets, keep


def thermal_trajectory(
    alpha: float | Iterable[float],
    source_seed: int,
    *,
    n_steps: int = 3000,
    stride: int = 8,
    noise_pct: float = 0.0,
    tau: float = 0.05,
    heat_coefficient: float = 800.0,
    amplitude: float = 2.0,
    pulse_width: int = 60,
    include_time_feature: bool = False,
    artifact: str | None = None,
    source_prefix: str = "driver",
) -> ThermalSample:
    if stride < 1:
        raise ValueError("stride must be positive")
    alphas = (float(alpha),) if np.isscalar(alpha) else tuple(float(item) for item in alpha)
    voltage = pulse_driver(n_steps, source_seed, amplitude=amplitude, width=pulse_width)
    delta_temperature = thermal_response(
        voltage, tau=tau, heat_coefficient=heat_coefficient
    )
    targets = np.stack([value * delta_temperature for value in alphas], axis=1)
    if noise_pct:
        rng = np.random.RandomState(source_seed + 777)
        current_ratio = np.exp(targets) * (
            1 + noise_pct / 100 * rng.randn(*targets.shape)
        )
        targets = np.log(np.maximum(current_ratio, 1e-30))
    voltage, targets, keep = _apply_artifact(voltage, targets, source_seed, artifact)
    time = np.arange(n_steps, dtype=float) * THERMAL_DT
    end = max(2, int(0.667 * n_steps))
    selected = np.arange(0, end, stride)
    selected = selected[keep[selected]]
    if selected.size < 2:
        raise ValueError("artifact/stride configuration leaves fewer than two timesteps")
    feature_columns = [voltage[selected, None]]
    if include_time_feature:
        feature_columns.append((time[selected] / max(time[selected].max(), THERMAL_DT))[:, None])
    features = np.concatenate(feature_columns, axis=1)
    condition = f"tau{tau:g}-heat{heat_coefficient:g}-artifact{artifact or 'none'}"
    source_id = f"{source_prefix}-{source_seed}-{condition}"
    trajectory = Trajectory(
        source_id,
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(targets[selected], dtype=torch.float32),
        torch.tensor(time[selected, None], dtype=torch.float32),
        source_id,
    )
    return ThermalSample(
        trajectory, delta_temperature[selected], voltage[selected], alphas
    )


def thermal_split(
    alpha: float | Iterable[float],
    train_sources: Iterable[int],
    validation_sources: Iterable[int],
    test_sources: Iterable[int],
    **kwargs,
) -> tuple[list[Trajectory], list[Trajectory], list[Trajectory], list[ThermalSample]]:
    train_samples = [thermal_trajectory(alpha, source, **kwargs) for source in train_sources]
    validation_samples = [thermal_trajectory(alpha, source, **kwargs) for source in validation_sources]
    test_samples = [thermal_trajectory(alpha, source, **kwargs) for source in test_sources]
    return (
        [item.trajectory for item in train_samples],
        [item.trajectory for item in validation_samples],
        [item.trajectory for item in test_samples],
        test_samples,
    )


MEASURED_FILES = {
    "GO-PDADMAC3": "IV-plot_GO-PDADMAC3.xlsx",
    "GO-PDADMAC4": "IV-plot_GO-PDADMAC4.xlsx",
    "GO-PEI3": "IV-plot_GO-PEI3.xlsx",
    "GO-PEI4": "IV-plot_GO-PEI4.xlsx",
}


def load_measured_cycles(
    data_directory: str | Path,
    device: str,
    *,
    sheet: str = "1p0_400mvs",
) -> tuple[list[Trajectory], dict[str, float]]:
    if device not in MEASURED_FILES:
        raise ValueError(f"unknown measured device: {device}")
    path = Path(data_directory) / MEASURED_FILES[device]
    frame = pd.read_excel(path, sheet_name=sheet, header=None).iloc[:, :3]
    frame.columns = ["cycle", "voltage", "current"]
    frame = frame.dropna()
    frame["cycle"] = frame["cycle"].astype(int)
    voltage_scale = float(np.abs(frame.voltage).max())
    current_scale = float(np.abs(frame.current).max())
    trajectories: list[Trajectory] = []
    for cycle in sorted(frame.cycle.unique()):
        part = frame[frame.cycle == cycle]
        length = len(part)
        source_id = f"{device}-{sheet}-cycle-{cycle}"
        features = np.stack(
            (
                part.voltage.to_numpy(dtype=float) / voltage_scale,
                np.linspace(0, 1, length),
            ),
            axis=1,
        )
        targets = part.current.to_numpy(dtype=float)[:, None] / current_scale
        trajectories.append(
            Trajectory(
                source_id,
                torch.tensor(features, dtype=torch.float32),
                torch.tensor(targets, dtype=torch.float32),
                torch.arange(length, dtype=torch.float32)[:, None],
                source_id,
            )
        )
    return trajectories, {
        "voltage_scale": voltage_scale,
        "current_scale": current_scale,
        "sheet": sheet,
    }


def split_measured_cycles(
    trajectories: list[Trajectory],
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> tuple[list[Trajectory], list[Trajectory], list[Trajectory]]:
    if len(trajectories) < 3:
        raise ValueError("at least three measured cycles are required")
    first = max(1, int(train_fraction * len(trajectories)))
    second = max(first + 1, int((train_fraction + validation_fraction) * len(trajectories)))
    second = min(second, len(trajectories) - 1)
    return trajectories[:first], trajectories[first:second], trajectories[second:]


def load_printed_memristor(path: str | Path, *, use_noisy: bool = False) -> list[Trajectory]:
    frame = pd.read_csv(path)
    required = {"device_id", "cycle_id", "voltage", "current"}
    if not required.issubset(frame.columns):
        raise ValueError(f"printed dataset is missing columns: {sorted(required - set(frame.columns))}")
    current_column = "current_noisy" if use_noisy and "current_noisy" in frame else "current"
    current_scale = float(frame[current_column].std())
    if not np.isfinite(current_scale) or current_scale <= 0:
        raise ValueError("printed current scale must be finite and positive")
    trajectories: list[Trajectory] = []
    for (device, cycle), part in frame.groupby(["device_id", "cycle_id"], sort=True):
        source_id = f"printed-device-{device}"
        trajectory_id = f"{source_id}-cycle-{cycle}"
        voltage = part["voltage_noisy"] if use_noisy and "voltage_noisy" in part else part["voltage"]
        current = part["current_noisy"] if use_noisy and "current_noisy" in part else part["current"]
        features = np.stack((voltage.to_numpy(), np.linspace(0, 1, len(part))), axis=1)
        trajectories.append(
            Trajectory(
                trajectory_id,
                torch.tensor(features, dtype=torch.float32),
                torch.tensor((current.to_numpy() / current_scale)[:, None], dtype=torch.float32),
                torch.arange(len(part), dtype=torch.float32)[:, None],
                source_id,
            )
        )
    return trajectories


def split_printed_memristor_sources(
    trajectories: list[Trajectory],
    train_devices: Iterable[int],
    validation_devices: Iterable[int],
    test_devices: Iterable[int],
) -> tuple[list[Trajectory], list[Trajectory], list[Trajectory]]:
    """Split complete cycles by persistent physical-device identity."""
    requested = [
        {f"printed-device-{int(device)}" for device in devices}
        for devices in (train_devices, validation_devices, test_devices)
    ]
    if any(requested[left] & requested[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("printed train, validation, and test device IDs must be disjoint")
    available = {item.source_trajectory_id for item in trajectories}
    missing = set.union(*requested) - available
    if missing:
        raise ValueError(f"printed split requests unavailable devices: {sorted(missing)}")
    splits = [
        [item for item in trajectories if item.source_trajectory_id in identities]
        for identities in requested
    ]
    if any(not split for split in splits):
        raise ValueError("printed source split cannot be empty")
    return splits[0], splits[1], splits[2]


def load_3d_thermal_csv(path: str | Path) -> Trajectory:
    frame = pd.read_csv(path)
    required = {"time", "V_driver", "I_victim"}
    if not required.issubset(frame.columns):
        raise ValueError(f"3D thermal dataset is missing columns: {sorted(required - set(frame.columns))}")
    source_id = "3d-thermal-crosstalk-record"
    features = np.stack((frame.V_driver.to_numpy(), frame.time.to_numpy()), axis=1)
    return Trajectory(
        source_id,
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(frame.I_victim.to_numpy()[:, None], dtype=torch.float32),
        torch.tensor(frame.time.to_numpy()[:, None], dtype=torch.float32),
        source_id,
    )
