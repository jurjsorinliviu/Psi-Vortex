"""Chronological synthetic, latent, robustness, and ablation experiments."""
from __future__ import annotations

import time
from typing import Any
import random
from dataclasses import replace

import numpy as np
import torch

from psi_vortex import (
    PIKAN,
    PsiXLSTMTeacher,
    SINDyRegressor,
    SequenceTrainer,
    StaticMLP,
    Trajectory,
    VanillaLSTM,
    fit_coupling,
    evaluate,
    iter_batches,
    load_printed_memristor,
    physics_aware_initialize,
    random_xavier_initialize,
    split_printed_memristor_sources,
    symmetry_orthogonal_initialize,
    relative_error,
    thermal_split,
    thermal_trajectory,
    train_static_model,
)

from .common import ROOT, RunContext, predict, predict_reset_chunks, predict_shuffled, source_record
from .train import coupling_scores, fit_pipeline, make_pipeline, mean_estimate, thermal_data


def _base_record(config: dict[str, Any], seed: int, train, validation, test):
    return {
        "profile": config["profile"],
        "seed": seed,
        "device": config["device"],
        **source_record(train, validation, test),
    }


def latent_recovery(context: RunContext) -> list[dict[str, Any]]:
    config = context.config
    rows: list[dict[str, Any]] = []
    for alpha in config["alphas"]:
        train, validation, test, samples = thermal_data(config, alpha)
        for seed in config["seeds"]:
            pipeline = make_pipeline(config, seed)
            started = time.perf_counter()
            metrics = fit_pipeline(pipeline, train, validation, test, config)
            estimates = coupling_scores(
                pipeline.student, samples, threshold=config["r2_accept"]
            )
            summary = mean_estimate(estimates)
            checkpoint = context.checkpoint(
                pipeline.student, f"latent_alpha{alpha:g}_seed{seed}"
            )
            rows.append(
                {
                    "method": "psi_vortex",
                    "alpha_true": alpha,
                    **summary,
                    "alpha_error": relative_error(summary["alpha_rec"], alpha),
                    "validation_mse": metrics["validation"]["mse"],
                    "test_mse": metrics["test"]["mse"],
                    "cluster_count": pipeline.selected_cluster_count,
                    **checkpoint,
                    "runtime_s": time.perf_counter() - started,
                    **_base_record(config, seed, train, validation, test),
                }
            )
    return rows


def recurrence_controls(context: RunContext) -> list[dict[str, Any]]:
    config = context.config
    alpha = 0.08
    train, validation, test, samples = thermal_data(config, alpha)
    rows: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        pipeline = make_pipeline(config, seed)
        fit_pipeline(pipeline, train, validation, test, config)
        recurrent_checkpoint = context.checkpoint(
            pipeline.student, f"recurrence_controls_recurrent_seed{seed}"
        )
        controls: dict[str, list[np.ndarray]] = {
            "chronological": [predict(pipeline.student, sample.trajectory) for sample in samples],
            "state_reset": [
                predict_reset_chunks(pipeline.student, sample.trajectory, 2) for sample in samples
            ],
            "shuffled_order": [
                predict_shuffled(pipeline.student, sample.trajectory, seed + 991)
                for sample in samples
            ],
        }
        for name, predictions in controls.items():
            estimates = []
            for prediction, sample in zip(predictions, samples):
                length = min(len(prediction), len(sample.delta_temperature))
                estimates.append(
                    fit_coupling(
                        prediction[:length, 0],
                        sample.delta_temperature[:length],
                        r2_threshold=config["r2_accept"],
                        driver_band=0.5,
                    )
                )
            summary = mean_estimate(estimates)
            squared = sum(
                float(
                    np.square(
                        prediction[: len(sample.trajectory.targets)]
                        - sample.trajectory.targets.detach().cpu().numpy()
                    ).sum()
                )
                for prediction, sample in zip(predictions, samples)
            )
            elements = sum(sample.trajectory.targets.numel() for sample in samples)
            rows.append(
                {
                    "control": name,
                    "alpha_true": alpha,
                    **summary,
                    "control_mse": squared / elements,
                    **recurrent_checkpoint,
                    **_base_record(config, seed, train, validation, test),
                }
            )
        # Explicit absent-memory/pointwise control through a matched static MLP.
        pointwise = train_static_model(
            StaticMLP(1, 16, 1),
            train,
            epochs=max(1, config["student_epochs"]),
            seed=seed,
            batch_size=min(config["batch_size"], len(train)),
            device=config["device"],
        )
        pointwise_predictions = [
            pointwise(sample.trajectory.features.unsqueeze(0).to(config["device"]))[0]
            .detach()
            .cpu()
            .numpy()
            for sample in samples
        ]
        estimates = [
            fit_coupling(
                prediction[:, 0],
                sample.delta_temperature,
                r2_threshold=config["r2_accept"],
                driver_band=0.5,
            )
            for prediction, sample in zip(pointwise_predictions, samples)
        ]
        pointwise_squared = sum(
            float(
                np.square(
                    prediction - sample.trajectory.targets.detach().cpu().numpy()
                ).sum()
            )
            for prediction, sample in zip(pointwise_predictions, samples)
        )
        rows.append(
            {
                "control": "absent_memory_pointwise",
                "alpha_true": alpha,
                **mean_estimate(estimates),
                "control_mse": pointwise_squared
                / sum(sample.trajectory.targets.numel() for sample in samples),
                **context.checkpoint(
                    pointwise, f"recurrence_controls_pointwise_seed{seed}"
                ),
                **_base_record(config, seed, train, validation, test),
            }
        )
    return rows


def baseline_comparison(context: RunContext) -> list[dict[str, Any]]:
    config = context.config
    rows: list[dict[str, Any]] = []
    for alpha in config["alphas"]:
        train, validation, test, samples = thermal_data(config, alpha)
        for seed in config["seeds"]:
            models = {
                "mlp": StaticMLP(1, 32, 1),
                "pikan": PIKAN(1, 8, 1, degree=3),
            }
            for name, model in models.items():
                train_static_model(
                    model,
                    train,
                    epochs=max(1, config["student_epochs"]),
                    seed=seed,
                    batch_size=min(config["batch_size"], len(train)),
                    device=config["device"],
                )
                estimates = []
                for sample in samples:
                    with torch.no_grad():
                        prediction = model(
                            sample.trajectory.features.unsqueeze(0).to(config["device"])
                        )[0, :, 0].cpu().numpy()
                    estimates.append(
                        fit_coupling(
                            prediction,
                            sample.delta_temperature,
                            r2_threshold=config["r2_accept"],
                            driver_band=0.5,
                        )
                    )
                rows.append(
                    {
                        "method": name,
                        "alpha_true": alpha,
                        "parameters": sum(parameter.numel() for parameter in model.parameters()),
                        **mean_estimate(estimates),
                        **context.checkpoint(
                            model,
                            f"synthetic_baseline_{name}_alpha{alpha:g}_seed{seed}",
                        ),
                        **_base_record(config, seed, train, validation, test),
                    }
                )
            sindy = SINDyRegressor(degree=3).fit(train)
            estimates = [
                fit_coupling(
                    sindy.predict(sample.trajectory)[:, 0],
                    sample.delta_temperature,
                    r2_threshold=config["r2_accept"],
                    driver_band=0.5,
                )
                for sample in samples
            ]
            rows.append(
                {
                    "method": "sindy",
                    "alpha_true": alpha,
                    "parameters": sindy.active_terms,
                    **mean_estimate(estimates),
                    **context.checkpoint_payload(
                        {
                            "coefficients": torch.as_tensor(sindy.coefficients),
                            "powers": [list(power) for power in sindy.powers],
                        },
                        f"synthetic_baseline_sindy_alpha{alpha:g}_seed{seed}",
                    ),
                    **_base_record(config, seed, train, validation, test),
                }
            )
            torch.manual_seed(seed)
            vanilla = VanillaLSTM(1, config["student_hidden"], 1).to(config["device"])
            trainer = SequenceTrainer(
                vanilla,
                torch.optim.Adam(vanilla.parameters(), lr=config["learning_rate"]),
                config["chunk_length"],
            )
            for epoch in range(config["student_epochs"]):
                trainer.train_epoch(
                    iter_batches(
                        train,
                        min(config["batch_size"], len(train)),
                        shuffle_trajectories=True,
                        generator=random.Random(seed + epoch),
                    )
                )
            estimates = coupling_scores(vanilla, samples, threshold=config["r2_accept"])
            rows.append(
                {
                    "method": "vanilla_lstm",
                    "alpha_true": alpha,
                    "parameters": sum(parameter.numel() for parameter in vanilla.parameters()),
                    **mean_estimate(estimates),
                    **context.checkpoint(
                        vanilla,
                        f"synthetic_baseline_vanilla_lstm_alpha{alpha:g}_seed{seed}",
                    ),
                    **_base_record(config, seed, train, validation, test),
                }
            )
    return rows


def rrad_ablation(context: RunContext) -> list[dict[str, Any]]:
    config = context.config
    alpha = 0.08
    train, validation, test, samples = thermal_data(config, alpha)
    variants = {
        "full": ((1.0, 0.5, 1.0, 0.5), config["lambda_bic"]),
        "output_only": ((1.0, 0.0, 0.0, 0.0), config["lambda_bic"]),
        "no_temporal": ((1.0, 0.0, 1.0, 0.0), config["lambda_bic"]),
        "no_hidden": ((1.0, 0.5, 0.0, 0.0), config["lambda_bic"]),
        "no_bic": ((1.0, 0.5, 1.0, 0.5), 0.0),
    }
    rows: list[dict[str, Any]] = []
    for variant, (weights, bic_weight) in variants.items():
        for seed in config["seeds"]:
            pipeline = make_pipeline(config, seed)
            metrics = fit_pipeline(
                pipeline,
                train,
                validation,
                test,
                config,
                lambda_bic=bic_weight,
                rrad_weights=weights,
                cluster_candidates=() if variant == "no_bic" else None,
            )
            estimates = coupling_scores(
                pipeline.student, samples, threshold=config["r2_accept"]
            )
            summary = mean_estimate(estimates)
            rows.append(
                {
                    "variant": variant,
                    **summary,
                    "test_mse": metrics["test"]["mse"],
                    "cluster_count": pipeline.selected_cluster_count,
                    **context.checkpoint(
                        pipeline.student, f"rrad_{variant}_seed{seed}"
                    ),
                    **_base_record(config, seed, train, validation, test),
                }
            )
    return rows


def initialization_ablation(context: RunContext) -> list[dict[str, Any]]:
    config = context.config
    rows: list[dict[str, Any]] = []
    printed = load_printed_memristor(
        ROOT / "data" / "printed_memristor_training_data.csv"
    )
    printed_train, printed_validation, printed_test = split_printed_memristor_sources(
        printed,
        config["printed_train_devices"],
        config["printed_validation_devices"],
        config["printed_test_devices"],
    )
    for seed in config.get("initialization_seeds", config["seeds"]):
        runs: dict[str, dict[str, Any]] = {}
        for mode in ("baseline_random", "physics_odd_corrected"):
            torch.manual_seed(seed)
            model = PsiXLSTMTeacher(
                2,
                config.get("initialization_teacher_hidden", 64),
                1,
                num_blocks=config.get("initialization_teacher_blocks", 2),
            ).to(config["device"])
            if mode == "baseline_random":
                random_xavier_initialize(model)
            else:
                symmetry_orthogonal_initialize(
                    model, "odd", preserve_recurrence=True
                )
            trainer = SequenceTrainer(
                model,
                torch.optim.Adam(
                    model.parameters(),
                    lr=config.get("initialization_learning_rate", config["learning_rate"]),
                ),
                config["chunk_length"],
                config.get("gradient_clip"),
            )
            curve = []
            started = time.perf_counter()
            for epoch in range(config.get("initialization_epochs", config["teacher_epochs"])):
                curve.append(
                    trainer.train_epoch(
                        iter_batches(
                            printed_train,
                            min(config["batch_size"], len(printed_train)),
                            shuffle_trajectories=True,
                            generator=random.Random(seed + epoch),
                        )
                    ).loss
                )
            runs[mode] = {
                "curve": curve,
                "runtime_s": time.perf_counter() - started,
                "validation_mse": evaluate(model, iter_batches(printed_validation, 1))["mse"],
                "test_mse": evaluate(model, iter_batches(printed_test, 1))["mse"],
                "checkpoint": context.checkpoint(
                    model, f"initialization_printed_{mode}_seed{seed}"
                ),
            }
        shared_target = min(runs["baseline_random"]["curve"])
        epochs_to_target: dict[str, int | None] = {}
        for mode, record in runs.items():
            epochs_to_target[mode] = next(
                (index + 1 for index, loss in enumerate(record["curve"]) if loss <= shared_target),
                None,
            )
        baseline_epoch = epochs_to_target["baseline_random"]
        for mode, record in runs.items():
            reached = epochs_to_target[mode]
            rows.append(
                {
                    "benchmark": "printed_memristor_sequence_convergence",
                    "initialization": mode,
                    "seed": seed,
                    "shared_target_mse": shared_target,
                    "epochs_to_target": reached,
                    "target_reached": reached is not None,
                    "speedup_vs_random": (
                        baseline_epoch / reached
                        if baseline_epoch is not None and reached is not None
                        else None
                    ),
                    "final_train_loss": record["curve"][-1],
                    "validation_mse": record["validation_mse"],
                    "test_mse": record["test_mse"],
                    "runtime_s": record["runtime_s"],
                    **source_record(printed_train, printed_validation, printed_test),
                    "profile": config["profile"],
                    "device": config["device"],
                    **record["checkpoint"],
                }
            )

    train, validation, test, samples = thermal_data(config, 0.08)
    for mode in ("random", "physical_retention"):
        for seed in config["seeds"]:
            pipeline = make_pipeline(config, seed)
            if mode == "physical_retention":
                physics_aware_initialize(
                    pipeline.teacher, 0.05, 1e-4 * config["stride"]
                )
            started = time.perf_counter()
            metrics = fit_pipeline(pipeline, train, validation, test, config)
            estimates = coupling_scores(
                pipeline.student, samples, threshold=config["r2_accept"]
            )
            rows.append(
                {
                    "benchmark": "thermal_recovery",
                    "initialization": mode,
                    **mean_estimate(estimates),
                    "test_mse": metrics["test"]["mse"],
                    "runtime_s": time.perf_counter() - started,
                    **context.checkpoint(
                        pipeline.student, f"initialization_thermal_{mode}_seed{seed}"
                    ),
                    **_base_record(config, seed, train, validation, test),
                }
            )
    return rows


def detection_regime(context: RunContext) -> list[dict[str, Any]]:
    config = context.config
    rows: list[dict[str, Any]] = []
    alpha = 0.08
    for count in config["source_counts"]:
        for noise in config["noise_pct"]:
            train_sources = config["driver_train_sources"][:count]
            train, validation, test, samples = thermal_split(
                alpha,
                train_sources,
                config["driver_validation_sources"],
                config["driver_test_sources"],
                n_steps=config["n_steps"],
                stride=config["stride"],
                noise_pct=noise,
            )
            for seed in config["seeds"]:
                pipeline = make_pipeline(config, seed)
                fit_pipeline(pipeline, train, validation, test, config)
                estimates = coupling_scores(
                    pipeline.student, samples, threshold=config["r2_accept"]
                )
                summary = mean_estimate(estimates)
                rows.append(
                    {
                        "source_count": count,
                        "noise_pct": noise,
                        "alpha_true": alpha,
                        **summary,
                        "alpha_error": relative_error(summary["alpha_rec"], alpha),
                        **context.checkpoint(
                            pipeline.student,
                            f"detection_sources{count}_noise{noise}_seed{seed}",
                        ),
                        **_base_record(config, seed, train, validation, test),
                    }
                )
    return rows


def artifact_stress(context: RunContext) -> list[dict[str, Any]]:
    config = context.config
    rows: list[dict[str, Any]] = []
    for artifact in config["artifacts"]:
        train, validation, test, samples = thermal_data(config, 0.08, artifact=artifact)
        for seed in config["seeds"]:
            pipeline = make_pipeline(config, seed)
            metrics = fit_pipeline(pipeline, train, validation, test, config)
            estimates = coupling_scores(
                pipeline.student, samples, threshold=config["r2_accept"]
            )
            summary = mean_estimate(estimates)
            rows.append(
                {
                    "artifact": artifact,
                    **summary,
                    "alpha_error": relative_error(summary["alpha_rec"], 0.08),
                    "alpha_error_pct": 100.0
                    * relative_error(summary["alpha_rec"], 0.08),
                    "latent_slope": summary["alpha_rec"],
                    "cluster_count": pipeline.selected_cluster_count,
                    **context.checkpoint(
                        pipeline.student, f"artifact_{artifact}_seed{seed}"
                    ),
                    **_base_record(config, seed, train, validation, test),
                }
            )
    clean_by_seed = {
        row["seed"]: row["alpha_error_pct"]
        for row in rows
        if row["artifact"] == "none"
    }
    for row in rows:
        row["degradation_pp"] = row["alpha_error_pct"] - clean_by_seed[row["seed"]]
    return rows


def geometry_transfer(context: RunContext) -> list[dict[str, Any]]:
    config = context.config
    rows: list[dict[str, Any]] = []
    for geometry in config["geometries"]:
        geometry_stride = 4 if geometry["name"] == "tau_fast" else config["stride"]
        train, validation, test, samples = thermal_data(
            config,
            0.08,
            tau=geometry["tau"],
            heat_coefficient=geometry["heat_coefficient"],
            stride=geometry_stride,
        )
        for seed in config["seeds"]:
            pipeline = make_pipeline(config, seed)
            metrics = fit_pipeline(pipeline, train, validation, test, config)
            estimates = coupling_scores(
                pipeline.student, samples, threshold=config["r2_accept"]
            )
            summary = mean_estimate(estimates)
            rows.append(
                {
                    "geometry": geometry["name"],
                    "tau": geometry["tau"],
                    "heat_coefficient": geometry["heat_coefficient"],
                    "stride": geometry_stride,
                    **summary,
                    "alpha_error": relative_error(summary["alpha_rec"], 0.08),
                    **context.checkpoint(
                        pipeline.student,
                        f"geometry_{geometry['name']}_seed{seed}",
                    ),
                    **_base_record(config, seed, train, validation, test),
                }
            )
    return rows


def multilayer(context: RunContext) -> list[dict[str, Any]]:
    config = context.config
    truths = (0.12, 0.08, 0.05)
    train, validation, test, samples = thermal_data(config, truths)
    rows: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        pipeline = make_pipeline(config, seed, output_size=len(truths))
        fit_pipeline(pipeline, train, validation, test, config)
        checkpoint = context.checkpoint(pipeline.student, f"multilayer_seed{seed}")
        for layer, truth in enumerate(truths):
            estimates = coupling_scores(
                pipeline.student,
                samples,
                output_index=layer,
                threshold=config["r2_accept"],
            )
            summary = mean_estimate(estimates)
            rows.append(
                {
                    "layer": layer + 1,
                    "alpha_true": truth,
                    **summary,
                    "alpha_error": relative_error(summary["alpha_rec"], truth),
                    **checkpoint,
                    **_base_record(config, seed, train, validation, test),
                }
            )
    return rows


def negative_controls(context: RunContext) -> list[dict[str, Any]]:
    config = context.config
    rows: list[dict[str, Any]] = []
    cases = (
        ("genuine", 0.08, None, "none"),
        ("zero_coupling", 0.0, None, "none"),
        ("unrelated_slow_drift", 0.0, "baseline_drift", "none"),
        ("shuffled_driver", 0.08, None, "shuffle_driver_values"),
        ("victim_only_no_driver", 0.08, None, "zero_driver"),
    )
    for name, alpha, artifact, transformation in cases:
        train, validation, test, samples = thermal_data(config, alpha, artifact=artifact)

        def transform(items, offset):
            transformed = []
            for index, item in enumerate(items):
                if transformation == "shuffle_driver_values":
                    generator = torch.Generator().manual_seed(31415 + offset + index)
                    order = torch.randperm(len(item.time), generator=generator)
                    features = item.features[order]
                elif transformation == "zero_driver":
                    features = torch.zeros_like(item.features)
                else:
                    features = item.features
                transformed.append(
                    Trajectory(
                        item.trajectory_id,
                        features,
                        item.targets,
                        item.time,
                        item.source_trajectory_id,
                    )
                )
            return transformed

        train = transform(train, 0)
        validation = transform(validation, 1000)
        test = transform(test, 2000)
        test_by_id = {item.trajectory_id: item for item in test}
        samples = [
            replace(sample, trajectory=test_by_id[sample.trajectory.trajectory_id])
            for sample in samples
        ]
        for seed in config["seeds"]:
            pipeline = make_pipeline(config, seed)
            metrics = fit_pipeline(pipeline, train, validation, test, config)
            estimates = coupling_scores(
                pipeline.student, samples, threshold=config["r2_accept"]
            )
            summary = mean_estimate(estimates)
            if name == "genuine":
                expected_signature = "accepted_and_within_relative_error"
                signature_passed = bool(
                    summary["accepted"]
                    and relative_error(summary["alpha_rec"], alpha)
                    <= config.get("positive_control_relative_error", 0.2)
                )
            elif name in {"zero_coupling", "unrelated_slow_drift"}:
                expected_signature = "near_zero_slope"
                signature_passed = bool(
                    abs(summary["alpha_rec"])
                    <= config.get("null_slope_tolerance", 0.01)
                )
            else:
                expected_signature = "r2_below_acceptance_gate"
                signature_passed = not summary["accepted"]
            rows.append(
                {
                    "control": name,
                    "control_protocol": transformation,
                    "alpha_true": alpha,
                    **summary,
                    "expected_signature": expected_signature,
                    "signature_passed": signature_passed,
                    "test_mse": metrics["test"]["mse"],
                    **context.checkpoint(
                        pipeline.student, f"negative_{name}_seed{seed}"
                    ),
                    **_base_record(config, seed, train, validation, test),
                }
            )
    return rows
