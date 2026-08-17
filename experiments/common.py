"""Shared provenance, persistence, and prediction helpers."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import pandas as pd
import matplotlib
import openpyxl
import jinja2

from psi_vortex import Trajectory, __version__


ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_model(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def source_tree_hash() -> str:
    digest = hashlib.sha256()
    paths = sorted(
        [*ROOT.glob("psi_vortex/*.py"), *ROOT.glob("experiments/*.py"), *ROOT.glob("configs/*.json")]
    )
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def model_architecture(model: torch.nn.Module) -> dict[str, Any]:
    base = model.base if hasattr(model, "base") else model
    record: dict[str, Any] = {"class": type(base).__name__}
    for name in (
        "input_size",
        "hidden_size",
        "output_size",
        "num_layers",
        "rank",
        "block_widths",
        "materialized_cluster_count",
    ):
        owner = model if name == "materialized_cluster_count" else base
        if hasattr(owner, name):
            record[name] = getattr(owner, name)
    if hasattr(base, "network"):
        linear = [module for module in base.network if isinstance(module, torch.nn.Linear)]
        if linear:
            record["linear_widths"] = [linear[0].in_features] + [
                layer.out_features for layer in linear
            ]
    if hasattr(base, "recurrent") and isinstance(
        base.recurrent, (torch.nn.GRU, torch.nn.LSTM)
    ):
        record.setdefault("input_size", base.recurrent.input_size)
        record.setdefault("hidden_size", base.recurrent.hidden_size)
        record.setdefault("num_layers", base.recurrent.num_layers)
    if hasattr(base, "first") and hasattr(base.first, "coefficients"):
        first_shape = list(base.first.coefficients.shape)
        second_shape = list(base.second.coefficients.shape)
        record["chebyshev_widths"] = [first_shape[1], first_shape[0], second_shape[0]]
        record["chebyshev_degree"] = first_shape[2] - 1
    record["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    return record


def predict(model: torch.nn.Module, trajectory: Trajectory) -> np.ndarray:
    device = next(model.parameters()).device
    with torch.no_grad():
        prediction, _, _ = model(trajectory.features.unsqueeze(0).to(device), None)
    return prediction[0].detach().cpu().numpy()


def predict_reset_chunks(
    model: torch.nn.Module, trajectory: Trajectory, chunk_length: int
) -> np.ndarray:
    if chunk_length < 2:
        raise ValueError("control chunk length must be greater than one")
    device = next(model.parameters()).device
    pieces: list[torch.Tensor] = []
    with torch.no_grad():
        start = 0
        while start < len(trajectory.time):
            stop = min(start + chunk_length, len(trajectory.time))
            if len(trajectory.time) - stop == 1:
                stop = len(trajectory.time)
            part = trajectory.features[start:stop].unsqueeze(0).to(device)
            pieces.append(model(part, None)[0])
            start = stop
    return torch.cat(pieces, dim=1)[0].cpu().numpy()


def predict_shuffled(model: torch.nn.Module, trajectory: Trajectory, seed: int) -> np.ndarray:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(trajectory.time), generator=generator)
    device = next(model.parameters()).device
    with torch.no_grad():
        prediction, _, _ = model(trajectory.features[order].unsqueeze(0).to(device), None)
    inverse = torch.argsort(order)
    return prediction[0, inverse].cpu().numpy()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class RunContext:
    config: dict[str, Any]
    output: Path

    def __post_init__(self) -> None:
        for name in (
            "configs",
            "manifests",
            "checkpoints",
            "raw_results",
            "figures",
            "tables",
            "logs",
            "environment",
            "artifacts",
        ):
            (self.output / name).mkdir(parents=True, exist_ok=True)
        environment = {
            "package": "psi-vortex-reproducible",
            "package_version": __version__,
            "source_tree_sha256": source_tree_hash(),
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "openpyxl": openpyxl.__version__,
            "jinja2": jinja2.__version__,
            "platform": platform.platform(),
            "device_requested": self.config["device"],
            "cuda_available": torch.cuda.is_available(),
        }
        (self.output / "environment" / "runtime.json").write_text(
            json.dumps(environment, indent=2), encoding="utf-8"
        )

    def write_rows(self, group: str, rows: list[dict[str, Any]]) -> Path:
        for row in rows:
            # ``device`` is already the physical measured-device identifier in
            # several public tables.  Keep that scientific meaning and record
            # the compute backend separately and uniformly on every row.
            row.setdefault("execution_device", self.config["device"])
            encoded = row.get("model_architecture")
            if encoded and not row.get("cluster_count"):
                try:
                    architecture = json.loads(encoded)
                except (TypeError, json.JSONDecodeError):
                    architecture = {}
                count = architecture.get("materialized_cluster_count")
                if count is not None:
                    row["cluster_count"] = count
        path = self.output / "raw_results" / f"{group}.csv"
        write_csv(path, rows)
        return path

    def checkpoint(self, model: torch.nn.Module, label: str) -> dict[str, str]:
        """Persist and hash the exact trained state associated with a result row."""
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._")
        if not safe_label:
            raise ValueError("checkpoint label must contain a filename-safe character")
        path = self.output / "checkpoints" / f"{safe_label}.pt"
        architecture = model_architecture(model)
        config_hash = hashlib.sha256(
            json.dumps(self.config, sort_keys=True).encode("utf-8")
        ).hexdigest()
        torch.save(
            {
                "package_version": __version__,
                "source_tree_sha256": source_tree_hash(),
                "config_sha256": config_hash,
                "architecture": architecture,
                "state_dict": model.state_dict(),
            },
            path,
        )
        return {
            "checkpoint_path": path.relative_to(self.output).as_posix(),
            "checkpoint_hash": hash_model(model),
            "checkpoint_file_sha256": sha256_file(path),
            "model_class": type(model).__name__,
            "model_architecture": json.dumps(architecture, sort_keys=True),
            "model_parameter_count": str(
                sum(parameter.numel() for parameter in model.parameters())
            ),
        }

    def checkpoint_payload(self, payload: Any, label: str) -> dict[str, str]:
        """Persist a non-module fitted state, such as SINDy coefficients."""
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._")
        if not safe_label:
            raise ValueError("checkpoint label must contain a filename-safe character")
        path = self.output / "checkpoints" / f"{safe_label}.pt"
        torch.save(
            {
                "package_version": __version__,
                "source_tree_sha256": source_tree_hash(),
                "config_sha256": hashlib.sha256(
                    json.dumps(self.config, sort_keys=True).encode("utf-8")
                ).hexdigest(),
                "architecture": {"class": "serialized_payload"},
                "payload": payload,
            },
            path,
        )
        digest = sha256_file(path)
        return {
            "checkpoint_path": path.relative_to(self.output).as_posix(),
            "checkpoint_hash": digest,
            "checkpoint_file_sha256": digest,
            "model_class": "serialized_payload",
        }

    def provenance(self, group: str, started: float, rows: int) -> None:
        result_path = self.output / "raw_results" / f"{group}.csv"
        record = {
            "group": group,
            "profile": self.config["profile"],
            "seeds": self.config["seeds"],
            "device": self.config["device"],
            "git_revision": git_revision(),
            "package_version": __version__,
            "source_tree_sha256": source_tree_hash(),
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "runtime_s": time.perf_counter() - started,
            "rows": rows,
            "raw_result_path": (
                result_path.relative_to(self.output).as_posix()
                if result_path.is_file()
                else None
            ),
            "raw_result_sha256": (
                sha256_file(result_path) if result_path.is_file() else None
            ),
            "config_sha256": hashlib.sha256(
                json.dumps(self.config, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        }
        (self.output / "manifests" / f"run_{group}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )


def source_record(
    train: Iterable[Trajectory], validation: Iterable[Trajectory], test: Iterable[Trajectory]
) -> dict[str, str]:
    def identities(items: Iterable[Trajectory]) -> str:
        return ";".join(
            dict.fromkeys(
                item.source_trajectory_id or item.trajectory_id for item in items
            )
        )

    return {
        "train_source_ids": identities(train),
        "validation_source_ids": identities(validation),
        "test_source_ids": identities(test),
    }
