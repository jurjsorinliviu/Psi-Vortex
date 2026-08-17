"""Measured-device, cross-domain, compression, and operating-regime experiments."""
from __future__ import annotations

import random
import time
from typing import Any

import numpy as np
import torch

from psi_vortex import (
    PIKAN,
    SINDyRegressor,
    SequenceTrainer,
    StaticMLP,
    compression_report,
    correlation,
    evaluate,
    export_torchscript,
    iter_batches,
    load_measured_cycles,
    nrmse,
    select_cluster_count,
    split_measured_cycles,
    train_static_model,
)

from .common import ROOT, RunContext, predict, source_record
from .train import fit_pipeline, make_pipeline


DATA = ROOT / "data" / "measured"


def _predictions(model, trajectories):
    return np.concatenate([predict(model, item) for item in trajectories])


def _targets(trajectories):
    return np.concatenate([item.targets.detach().cpu().numpy() for item in trajectories])


def measured_fidelity(context: RunContext) -> list[dict[str, Any]]:
    config = context.config
    rows: list[dict[str, Any]] = []
    for device in config["real_devices"]:
        cycles, scales = load_measured_cycles(DATA, device, sheet=config["real_sheets"][0])
        train, validation, test = split_measured_cycles(cycles)
        target = _targets(test)
        for seed in config["seeds"]:
            pipeline = make_pipeline(config, seed, input_size=2, real=True)
            started = time.perf_counter()
            metrics = fit_pipeline(pipeline, train, validation, test, config, real=True)
            prediction = _predictions(pipeline.student, test)
            checkpoint = context.checkpoint(
                pipeline.student, f"measured_{device}_seed{seed}"
            )
            rows.append(
                {
                    "device": device,
                    "sheet": config["real_sheets"][0],
                    "method": "psi_vortex",
                    "seed": seed,
                    "correlation": correlation(prediction, target),
                    "nrmse": nrmse(prediction, target),
                    "test_mse": metrics["test"]["mse"],
                    "cluster_count": pipeline.selected_cluster_count,
                    **checkpoint,
                    "runtime_s": time.perf_counter() - started,
                    "current_scale": scales["current_scale"],
                    **source_record(train, validation, test),
                    "profile": config["profile"],
                }
            )
    return rows


def measured_baselines(context: RunContext) -> list[dict[str, Any]]:
    config = context.config
    rows: list[dict[str, Any]] = []
    for device in config["real_devices"]:
        cycles, _ = load_measured_cycles(DATA, device, sheet=config["real_sheets"][0])
        train, validation, test = split_measured_cycles(cycles)
        target = _targets(test)
        for seed in config["seeds"]:
            for name, model in (
                ("mlp", StaticMLP(2, 32, 1)),
                ("pikan", PIKAN(2, 8, 1, degree=3)),
            ):
                train_static_model(
                    model,
                    train,
                    epochs=max(1, config["real_epochs"]),
                    seed=seed,
                    batch_size=min(config["batch_size"], len(train)),
                    device=config["device"],
                )
                with torch.no_grad():
                    prediction = np.concatenate(
                        [
                            model(item.features.unsqueeze(0).to(config["device"]))[0]
                            .cpu()
                            .numpy()
                            for item in test
                        ]
                    )
                rows.append(
                    {
                        "device": device,
                        "method": name,
                        "seed": seed,
                        "correlation": correlation(prediction, target),
                        "nrmse": nrmse(prediction, target),
                        "parameters": sum(parameter.numel() for parameter in model.parameters()),
                        **context.checkpoint(
                            model, f"measured_baseline_{device}_{name}_seed{seed}"
                        ),
                        **source_record(train, validation, test),
                        "profile": config["profile"],
                    }
                )
            sindy = SINDyRegressor(degree=3).fit(train)
            prediction = np.concatenate([sindy.predict(item) for item in test])
            rows.append(
                {
                    "device": device,
                    "method": "sindy",
                    "seed": seed,
                    "correlation": correlation(prediction, target),
                    "nrmse": nrmse(prediction, target),
                    "parameters": sindy.active_terms,
                    **context.checkpoint_payload(
                        {
                            "coefficients": torch.as_tensor(sindy.coefficients),
                            "powers": [list(power) for power in sindy.powers],
                        },
                        f"measured_baseline_{device}_sindy_seed{seed}",
                    ),
                    **source_record(train, validation, test),
                    "profile": config["profile"],
                }
            )
    return rows


def compression_fidelity(context: RunContext) -> list[dict[str, Any]]:
    """Directly train compact recurrent models on the declared GO-PEI4 split.

    This is the clean chronological replacement for Supplement Note 10.  The
    compact students are supervised by measured targets rather than distilled;
    the separate export experiment continues to test the canonical distillation
    path.  Cluster count is selected using validation trajectories only.
    """
    config = context.config
    maximum_rank = 2 + config["student_hidden"]
    invalid_ranks = [
        rank
        for rank in config["student_rank_sweep"]
        if rank < 1 or rank > maximum_rank
    ]
    if invalid_ranks:
        raise ValueError(
            "student_rank_sweep contains ranks outside the valid interval "
            f"[1, {maximum_rank}] for input_size=2 and "
            f"student_hidden={config['student_hidden']}: {invalid_ranks}"
        )
    measured_device = config.get("compression_device")
    if not isinstance(measured_device, str) or not measured_device:
        raise ValueError("compression_device must explicitly name a measured device")
    if measured_device not in config.get("real_devices", []):
        raise ValueError(
            "compression_device must be included in real_devices; "
            f"got {measured_device!r}"
        )
    if config.get("compression_training_mode") != "direct_supervised":
        raise ValueError(
            "compression_training_mode must be 'direct_supervised' for the "
            "manuscript compression-fidelity experiment"
        )
    learning_rate = float(config.get("compression_learning_rate", 0.0))
    if learning_rate <= 0:
        raise ValueError("compression_learning_rate must be positive")

    sheet = config["real_sheets"][0]
    cycles, _ = load_measured_cycles(DATA, measured_device, sheet=sheet)
    train, validation, test = split_measured_cycles(cycles)
    target = _targets(test)
    rows: list[dict[str, Any]] = []
    candidates = [
        ("gru", hidden, None)
        for hidden in config["student_hidden_sweep"]
    ] + [
        ("low_rank", config["student_hidden"], rank)
        for rank in config["student_rank_sweep"]
    ]

    def train_direct(model: torch.nn.Module, seed: int) -> float:
        trainer = SequenceTrainer(
            model,
            torch.optim.Adam(model.parameters(), lr=learning_rate),
            config["chunk_length"],
            config.get("real_gradient_clip"),
        )
        started = time.perf_counter()
        for epoch in range(config["real_epochs"]):
            trainer.train_epoch(
                iter_batches(
                    train,
                    min(config["batch_size"], len(train)),
                    shuffle_trajectories=True,
                    generator=random.Random(seed + epoch),
                )
            )
        return time.perf_counter() - started

    def validation_loss(candidate: torch.nn.Module) -> torch.Tensor:
        model_device = next(candidate.parameters()).device
        squared_error: torch.Tensor | None = None
        count = 0
        for batch in iter_batches(validation, 1, shuffle_trajectories=False):
            batch = batch.to(model_device)
            with torch.no_grad():
                prediction, _, _ = candidate(batch.features, None)
            batch_sse = (prediction - batch.targets).square().sum()
            squared_error = (
                batch_sse if squared_error is None else squared_error + batch_sse
            )
            count += batch.targets.numel()
        if squared_error is None or count < 2:
            raise ValueError("compression validation requires scalar target elements")
        return squared_error / count

    observations = sum(item.targets.numel() for item in validation)
    split_record = source_record(train, validation, test)
    for seed in config["seeds"]:
        reference = make_pipeline(config, seed, input_size=2, real=True)
        teacher_runtime = train_direct(reference.teacher, seed)
        teacher_prediction = _predictions(reference.teacher, test)
        rows.append(
            {
                "student_type": "teacher",
                "hidden_size": config["real_teacher_hidden"],
                "rank": None,
                "seed": seed,
                "measured_device": measured_device,
                "sheet": sheet,
                "training_mode": "direct_supervised_reference",
                "correlation": correlation(teacher_prediction, target),
                "nrmse": nrmse(teacher_prediction, target),
                "validation_mse": evaluate(
                    reference.teacher,
                    iter_batches(validation, 1, shuffle_trajectories=False),
                )["mse"],
                "test_mse": float(np.mean((teacher_prediction - target) ** 2)),
                "cluster_count": None,
                **compression_report(reference.teacher),
                **context.checkpoint(
                    reference.teacher, f"compression_direct_teacher_seed{seed}"
                ),
                "runtime_s": teacher_runtime,
                **split_record,
                "profile": config["profile"],
            }
        )

        for student_type, hidden, rank in candidates:
            pipeline = make_pipeline(
                config,
                seed,
                input_size=2,
                real=True,
                student_hidden=hidden,
                student_type=student_type,
                student_rank=rank or config["student_rank"],
            )
            student_runtime = train_direct(pipeline.student, seed)
            bic_score, requested_count, clustered = select_cluster_count(
                pipeline.student,
                list(config["cluster_candidates"]),
                validation_loss,
                observations,
            )
            pipeline.student = clustered
            pipeline.selected_cluster_count = clustered.materialized_cluster_count
            if pipeline.selected_cluster_count > requested_count:
                raise AssertionError(
                    "materialized cluster count cannot exceed requested count"
                )
            prediction = _predictions(clustered, test)
            artifact = (
                context.output
                / "artifacts"
                / f"direct_{student_type}_h{hidden}_r{rank or 0}_seed{seed}.pt"
            )
            export_torchscript(clustered, test[0].features.unsqueeze(0), artifact)
            rows.append(
                {
                    "student_type": student_type,
                    "hidden_size": hidden,
                    "rank": rank,
                    "seed": seed,
                    "measured_device": measured_device,
                    "sheet": sheet,
                    "training_mode": "direct_supervised",
                    "correlation": correlation(prediction, target),
                    "nrmse": nrmse(prediction, target),
                    "validation_mse": float(validation_loss(clustered)),
                    "test_mse": float(np.mean((prediction - target) ** 2)),
                    "cluster_count": pipeline.selected_cluster_count,
                    "bic_requested_cluster_count": requested_count,
                    "bic_score": bic_score,
                    **compression_report(clustered, artifact),
                    **context.checkpoint(
                        clustered,
                        f"compression_direct_{student_type}_h{hidden}_r{rank or 0}_seed{seed}",
                    ),
                    "runtime_s": student_runtime,
                    **split_record,
                    "profile": config["profile"],
                }
            )
    return rows


def cross_device(context: RunContext) -> list[dict[str, Any]]:
    config = context.config
    loaded = {}
    for device in config["real_devices"]:
        cycles, _ = load_measured_cycles(DATA, device, sheet=config["real_sheets"][0])
        loaded[device] = split_measured_cycles(cycles)
    rows: list[dict[str, Any]] = []
    for source_device, (train, validation, own_test) in loaded.items():
        for seed in config["seeds"]:
            pipeline = make_pipeline(config, seed, input_size=2, real=True)
            fit_pipeline(pipeline, train, validation, own_test, config, real=True)
            checkpoint = context.checkpoint(
                pipeline.student, f"cross_device_{source_device}_seed{seed}"
            )
            for target_device, (_, _, target_test) in loaded.items():
                prediction = _predictions(pipeline.student, target_test)
                target = _targets(target_test)
                rows.append(
                    {
                        "source_device": source_device,
                        "target_device": target_device,
                        "seed": seed,
                        "correlation": correlation(prediction, target),
                        "nrmse": nrmse(prediction, target),
                        **checkpoint,
                        **source_record(train, validation, target_test),
                        "profile": config["profile"],
                    }
                )
    return rows


def cross_rate(context: RunContext) -> list[dict[str, Any]]:
    config = context.config
    rows: list[dict[str, Any]] = []
    for device in config["real_devices"]:
        sheets = config["real_sheets"]
        source_cycles, _ = load_measured_cycles(DATA, device, sheet=sheets[0])
        train, validation, source_test = split_measured_cycles(source_cycles)
        for seed in config["seeds"]:
            pipeline = make_pipeline(config, seed, input_size=2, real=True)
            fit_pipeline(pipeline, train, validation, source_test, config, real=True)
            checkpoint = context.checkpoint(
                pipeline.student, f"cross_rate_{device}_seed{seed}"
            )
            for target_sheet in sheets:
                if target_sheet == sheets[0]:
                    target_cycles = source_test
                else:
                    target_cycles, _ = load_measured_cycles(DATA, device, sheet=target_sheet)
                prediction = _predictions(pipeline.student, target_cycles)
                target = _targets(target_cycles)
                rows.append(
                    {
                        "device": device,
                        "train_sheet": sheets[0],
                        "validation_sheet": sheets[0],
                        "test_sheet": target_sheet,
                        "seed": seed,
                        "correlation": correlation(prediction, target),
                        "nrmse": nrmse(prediction, target),
                        **checkpoint,
                        **source_record(train, validation, target_cycles),
                        "profile": config["profile"],
                    }
                )
    return rows
