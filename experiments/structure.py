"""Structure selection, symmetry, scalability, sensitivity, and runtime experiments."""
from __future__ import annotations

import time
from typing import Any
from itertools import product
import random

import numpy as np
import torch

from psi_vortex import (
    ArchitectureCandidate,
    AutoSymmetryDetector,
    PsiXLSTMTeacher,
    RecurrentStudent,
    SequenceTrainer,
    compare_estimators,
    fit_coupling,
    load_3d_thermal_csv,
    load_printed_memristor,
    split_printed_memristor_sources,
    evaluate,
    iter_batches,
    minibatch_effective_dof,
    rff_effective_dof,
    random_xavier_initialize,
    select_architecture,
    symmetry_orthogonal_initialize,
    thermal_split,
    thermal_trajectory,
)
from psi_vortex.bic import DifferentiableBIC, exact_effective_dof

from .common import ROOT, RunContext, predict, source_record
from .train import coupling_scores, fit_pipeline, make_pipeline, mean_estimate, thermal_data


def bic_structural_ablation(context: RunContext) -> list[dict[str, Any]]:
    config = context.config
    train, validation, test, samples = thermal_data(config, 0.08)
    methods = (
        ("full_bic", {}, None),
        ("no_bic", {"lambda_bic": 0.0, "cluster_candidates": ()}, None),
        ("l1", {"lambda_bic": 0.0, "cluster_candidates": ()}, ("l1", 1e-5)),
        ("l2", {"lambda_bic": 0.0, "cluster_candidates": ()}, ("l2", 1e-5)),
        ("fixed_k4", {"lambda_bic": 0.0, "cluster_candidates": (4,)}, None),
        ("fixed_rank4", {"lambda_bic": 0.0, "cluster_candidates": ()}, None),
    )
    rows: list[dict[str, Any]] = []
    for method, options, regularizer in methods:
        for seed in config["seeds"]:
            pipeline = make_pipeline(
                config,
                seed,
                student_type="low_rank" if method == "fixed_rank4" else config["student_type"],
                student_rank=4 if method == "fixed_rank4" else config["student_rank"],
            )
            metrics = fit_pipeline(
                pipeline,
                train,
                validation,
                test,
                config,
                parameter_regularizer=regularizer,
                **options,
            )
            estimates = coupling_scores(
                pipeline.student, samples, threshold=config["r2_accept"]
            )
            trainable = sum(
                parameter.numel()
                for parameter in pipeline.student.parameters()
                if parameter.requires_grad
            )
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    **mean_estimate(estimates),
                    "validation_mse": metrics["validation"]["mse"],
                    "test_mse": metrics["test"]["mse"],
                    "effective_values": getattr(
                        pipeline.student, "materialized_cluster_count", trainable
                    ),
                    "cluster_count": pipeline.selected_cluster_count,
                    "manual_choices": 0 if method == "full_bic" else 1,
                    **context.checkpoint(
                        pipeline.student, f"bic_structural_{method}_seed{seed}"
                    ),
                    **source_record(train, validation, test),
                    "profile": config["profile"],
                }
            )
    return rows


def architecture_search(context: RunContext) -> list[dict[str, Any]]:
    config = context.config
    trajectories = load_printed_memristor(ROOT / "data" / "printed_memristor_training_data.csv")
    train, validation, test = split_printed_memristor_sources(
        trajectories,
        config["printed_train_devices"],
        config["printed_validation_devices"],
        config["printed_test_devices"],
    )
    if "architecture_candidates" in config:
        candidates = [ArchitectureCandidate(**record) for record in config["architecture_candidates"]]
    else:
        candidates = [
            ArchitectureCandidate(hidden, blocks, memory)
            for hidden, blocks, memory in product(
                config["architecture_hidden_sizes"],
                config["architecture_block_counts"],
                config["architecture_memory_sizes"],
            )
        ]
    rows: list[dict[str, Any]] = []
    for seed in config.get("architecture_seeds", config["seeds"]):
        selected_model, selected, scores = select_architecture(
            candidates,
            train,
            validation,
            test,
            seed=seed,
            teacher_epochs=config.get("architecture_epochs", config["teacher_epochs"]),
            batch_size=min(config["batch_size"], len(train)),
            chunk_length=config["chunk_length"],
            device=config["device"],
            input_size=2,
            output_size=1,
            learning_rate=config.get("architecture_learning_rate", config["learning_rate"]),
            max_grad_norm=config.get("gradient_clip"),
        )
        selected_checkpoint = context.checkpoint(
            selected_model, f"architecture_selected_seed{seed}"
        )
        for score in scores:
            rows.append(
                {
                    "seed": seed,
                    "teacher_hidden": score.candidate.teacher_hidden,
                    "teacher_blocks": score.candidate.teacher_blocks,
                    "teacher_memory_size": score.candidate.teacher_memory_size,
                    "validation_mse": score.validation_mse,
                    "selected": score.candidate == selected.candidate,
                    "selected_test_mse": selected.test_mse if score.candidate == selected.candidate else None,
                    **(selected_checkpoint if score.candidate == selected.candidate else {}),
                    **source_record(train, validation, test),
                    "profile": config["profile"],
                }
            )
    return rows


def automatic_symmetry(context: RunContext) -> list[dict[str, Any]]:
    config = context.config
    rows: list[dict[str, Any]] = []
    trajectories = load_printed_memristor(
        ROOT / "data" / "printed_memristor_training_data.csv"
    )
    train, validation, test = split_printed_memristor_sources(
        trajectories,
        config["printed_train_devices"],
        config["printed_validation_devices"],
        config["printed_test_devices"],
    )
    detected = AutoSymmetryDetector().detect_trajectories(train, 0, 0)
    for mode in ("baseline_random", "identity", "expert_odd", "automatic"):
        for seed in config.get("symmetry_seeds", config["seeds"]):
            torch.manual_seed(seed)
            teacher = PsiXLSTMTeacher(
                2,
                config.get("symmetry_teacher_hidden", 64),
                1,
                num_blocks=config.get("symmetry_teacher_blocks", 2),
            ).to(config["device"])
            encoded = "none"
            if mode == "baseline_random":
                random_xavier_initialize(teacher)
            elif mode == "identity":
                symmetry_orthogonal_initialize(teacher, "none", preserve_recurrence=True)
            elif mode == "expert_odd":
                encoded = "odd"
                symmetry_orthogonal_initialize(teacher, encoded, preserve_recurrence=True)
            elif mode == "automatic":
                encoded = detected.symmetry
                symmetry_orthogonal_initialize(teacher, encoded, preserve_recurrence=True)
            trainer = SequenceTrainer(
                teacher,
                torch.optim.Adam(
                    teacher.parameters(),
                    lr=config.get("symmetry_learning_rate", config["learning_rate"]),
                ),
                config["chunk_length"],
                config.get("gradient_clip"),
            )
            training = None
            for epoch in range(config.get("symmetry_epochs", config["teacher_epochs"])):
                training = trainer.train_epoch(
                    iter_batches(
                        train,
                        min(config["batch_size"], len(train)),
                        shuffle_trajectories=True,
                        generator=random.Random(seed + epoch),
                    )
                )
            rows.append(
                {
                    "dataset": "printed_memristor_training_data.csv",
                    "mode": mode,
                    "seed": seed,
                    "detected_symmetry": detected.symmetry,
                    "encoded_symmetry": encoded,
                    "confidence": detected.confidence,
                    "odd_score": detected.odd_score,
                    "even_score": detected.even_score,
                    "matched_pairs": detected.matched_pairs,
                    "final_train_loss": training.loss if training is not None else None,
                    "validation_mse": evaluate(teacher, iter_batches(validation, 1))["mse"],
                    "test_mse": evaluate(teacher, iter_batches(test, 1))["mse"],
                    "symmetry_encoded_in_weights": encoded != "none",
                    **context.checkpoint(
                        teacher, f"symmetry_{mode}_seed{seed}"
                    ),
                    **source_record(train, validation, test),
                    "profile": config["profile"],
                }
            )
    return rows


def scalable_bic(context: RunContext) -> list[dict[str, Any]]:
    config = context.config
    rows: list[dict[str, Any]] = []
    for size in config["scalability_weights"]:
        generator = np.random.RandomState(0)
        centers = np.linspace(-1, 1, 5)
        values = np.concatenate(
            [generator.normal(center, 0.02, int(np.ceil(size / 5))) for center in centers]
        )[:size]
        weights = torch.tensor(values, dtype=torch.float32, device=config["device"])
        record: dict[str, Any] = {
            "weights": size,
            "seed": 0,
            "device": config["device"],
            "profile": config["profile"],
        }
        for name, function in (
            ("minibatch", lambda value: minibatch_effective_dof(value, sample_size=config["bic_sample_size"])),
            ("rff", lambda value: rff_effective_dof(value, features=config["bic_rff_features"])),
        ):
            started = time.perf_counter()
            record[f"{name}_dof"] = float(function(weights))
            record[f"{name}_runtime_s"] = time.perf_counter() - started
        if size <= 8000:
            variable = weights.clone().requires_grad_(True)
            started = time.perf_counter()
            exact = exact_effective_dof(variable, 0.1)
            record["exact_dof"] = float(exact.detach())
            record["exact_runtime_s"] = time.perf_counter() - started
            comparison = compare_estimators(
                weights,
                sample_size=config["bic_sample_size"],
                features=config["bic_rff_features"],
            )
            record.update(comparison)
        else:
            record.update(
                {
                    "exact_dof": None,
                    "exact_runtime_s": None,
                    "minibatch_relative_error": None,
                    "rff_relative_error": None,
                    "minibatch_gradient_cosine": None,
                    "rff_gradient_cosine": None,
                }
            )
        rows.append(record)
    return rows


def frequency_response(context: RunContext) -> list[dict[str, Any]]:
    config = context.config
    rows: list[dict[str, Any]] = []
    for width in config["frequency_widths"]:
        train, validation, test, samples = thermal_split(
            0.08,
            config["driver_train_sources"],
            config["driver_validation_sources"],
            config["driver_test_sources"],
            n_steps=config["n_steps"],
            stride=config["stride"],
            pulse_width=width,
        )
        for seed in config["seeds"]:
            pipeline = make_pipeline(config, seed)
            fit_pipeline(pipeline, train, validation, test, config)
            rows.append(
                {
                    "pulse_width_steps": width,
                    "seed": seed,
                    **mean_estimate(
                        coupling_scores(
                            pipeline.student, samples, threshold=config["r2_accept"]
                        )
                    ),
                    **source_record(train, validation, test),
                    **context.checkpoint(
                        pipeline.student, f"frequency_width{width}_seed{seed}"
                    ),
                    "profile": config["profile"],
                }
            )
    return rows


def long_sequence(context: RunContext) -> list[dict[str, Any]]:
    config = context.config
    train, validation, test, _ = thermal_data(config, 0.08)
    pipeline = make_pipeline(config, config["seeds"][0])
    fit_pipeline(pipeline, train, validation, test, config)
    checkpoint = context.checkpoint(
        pipeline.student, f"long_sequence_seed{config['seeds'][0]}"
    )
    rows: list[dict[str, Any]] = []
    for steps in config["long_sequence_steps"]:
        sample = thermal_trajectory(
            0.08,
            2027 + steps,
            n_steps=steps,
            stride=max(1, config["stride"]),
        )
        started = time.perf_counter()
        prediction = predict(pipeline.student, sample.trajectory)
        runtime = time.perf_counter() - started
        estimate = fit_coupling(
            prediction[:, 0],
            sample.delta_temperature,
            r2_threshold=config["r2_accept"],
            driver_band=0.5,
        )
        rows.append(
            {
                "generated_steps": steps,
                "evaluated_timesteps": len(sample.trajectory.time),
                "seed": config["seeds"][0],
                "device": config["device"],
                "runtime_s": runtime,
                "alpha_rec": estimate.value,
                "r2": estimate.r2,
                "accepted": estimate.accepted,
                "cluster_count": pipeline.selected_cluster_count,
                **checkpoint,
                **source_record(train, validation, [sample.trajectory]),
                "profile": config["profile"],
            }
        )
    return rows


def learning_rate_sensitivity(context: RunContext) -> list[dict[str, Any]]:
    config = context.config
    train, validation, test, samples = thermal_data(config, 0.08)
    rows: list[dict[str, Any]] = []
    for learning_rate in (1e-4, 1e-3, 1e-2):
        # ``make_pipeline`` gives the stage-specific rates precedence over the
        # shared fallback.  Override all three fields so this experiment truly
        # changes the optimizer rate used by both recurrent training stages.
        modified = {
            **config,
            "learning_rate": learning_rate,
            "teacher_learning_rate": learning_rate,
            "student_learning_rate": learning_rate,
        }
        for seed in config["seeds"]:
            pipeline = make_pipeline(modified, seed)
            metrics = fit_pipeline(pipeline, train, validation, test, modified)
            rows.append(
                {
                    "learning_rate": learning_rate,
                    "seed": seed,
                    **mean_estimate(
                        coupling_scores(
                            pipeline.student, samples, threshold=config["r2_accept"]
                        )
                    ),
                    "validation_mse": metrics["validation"]["mse"],
                    "test_mse": metrics["test"]["mse"],
                    **context.checkpoint(
                        pipeline.student,
                        f"learning_rate_{learning_rate:g}_seed{seed}",
                    ),
                    **source_record(train, validation, test),
                    "profile": config["profile"],
                }
            )
    return rows


def runtime_benchmark(context: RunContext) -> list[dict[str, Any]]:
    """Measure recurrence-safe training components, step overhead, and scaling."""
    config = context.config
    train, validation, test, _ = thermal_data(config, 0.08)
    seed = config["seeds"][0]
    pipeline = make_pipeline(config, seed)
    started = time.perf_counter()
    fit_pipeline(pipeline, train, validation, test, config)
    training_runtime = time.perf_counter() - started
    pipeline_checkpoint = context.checkpoint(
        pipeline.student, f"runtime_pipeline_seed{seed}"
    )
    item = test[0]
    device = torch.device(config["device"])
    features = item.features.unsqueeze(0).to(device)
    targets = item.targets.unsqueeze(0).to(device)
    repeats = config["runtime_repeats"]

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def elapsed(function):
        synchronize()
        begin = time.perf_counter()
        value = function()
        synchronize()
        return value, time.perf_counter() - begin

    torch.manual_seed(seed)
    component_model = RecurrentStudent(1, config["student_hidden"], 1).to(device)
    component_optimizer = torch.optim.Adam(
        component_model.parameters(), lr=config["learning_rate"]
    )
    bic = DifferentiableBIC().to(device)
    component_times = {name: [] for name in ("forward", "mse_loss", "bic_forward", "backward", "optimizer_step")}
    for _ in range(repeats):
        component_optimizer.zero_grad()
        prediction, duration = elapsed(lambda: component_model(features, None)[0])
        component_times["forward"].append(duration)
        mse, duration = elapsed(lambda: torch.nn.functional.mse_loss(prediction, targets))
        component_times["mse_loss"].append(duration)
        bic_value, duration = elapsed(
            lambda: bic(component_model, mse, targets.numel())
        )
        component_times["bic_forward"].append(duration)
        _, duration = elapsed(
            lambda: (mse + config.get("runtime_bic_weight", 0.01) * bic_value).backward()
        )
        component_times["backward"].append(duration)
        _, duration = elapsed(component_optimizer.step)
        component_times["optimizer_step"].append(duration)

    total_component_time = sum(float(np.mean(values)) for values in component_times.values())
    rows: list[dict[str, Any]] = []
    common = {
        "device": config["device"],
        "seed": seed,
        "cluster_count": pipeline.selected_cluster_count,
        "sequence_length": len(item.time),
        "repeats": repeats,
        "profile": config["profile"],
        **pipeline_checkpoint,
        **source_record(train, validation, test),
    }
    for component, values in component_times.items():
        mean = float(np.mean(values))
        rows.append(
            {
                "benchmark": "component_breakdown",
                "configuration": "bic_on",
                "component": component,
                "latency_mean_s": mean,
                "latency_std_s": float(np.std(values)),
                "percentage": 100.0 * mean / total_component_time,
                "training_pipeline_runtime_s": training_runtime,
                **common,
            }
        )

    def step_latency(with_bic: bool) -> list[float]:
        torch.manual_seed(seed)
        model = RecurrentStudent(1, config["student_hidden"], 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        values = []
        for _ in range(repeats):
            def step():
                optimizer.zero_grad()
                prediction = model(features, None)[0]
                mse = torch.nn.functional.mse_loss(prediction, targets)
                objective = (
                    mse + config.get("runtime_bic_weight", 0.01) * bic(model, mse, targets.numel())
                    if with_bic
                    else mse
                )
                objective.backward()
                optimizer.step()
            _, duration = elapsed(step)
            values.append(duration)
        return values

    baseline_times = step_latency(False)
    bic_times = step_latency(True)
    baseline_mean = float(np.mean(baseline_times))
    for configuration, values in (("no_bic", baseline_times), ("bic_on", bic_times)):
        mean = float(np.mean(values))
        rows.append(
            {
                "benchmark": "training_step",
                "configuration": configuration,
                "component": "complete_step",
                "latency_mean_s": mean,
                "latency_std_s": float(np.std(values)),
                "overhead_vs_no_bic": mean / baseline_mean,
                **common,
            }
        )

    inference_times = []
    for _ in range(repeats):
        _, duration = elapsed(lambda: predict(pipeline.student, item))
        inference_times.append(duration)
    rows.append(
        {
            "benchmark": "inference",
            "configuration": "deployed_student",
            "component": "complete_sequence",
            "latency_mean_s": float(np.mean(inference_times)),
            "latency_std_s": float(np.std(inference_times)),
            "latency_per_timestep_s": float(np.mean(inference_times)) / len(item.time),
            **common,
        }
    )

    # True chronological scaling: one complete recurrent trajectory per size.
    for sequence_length in config.get("runtime_sequence_lengths", [64, 256, 1024]):
        generator = torch.Generator(device="cpu").manual_seed(seed + sequence_length)
        values = torch.randn(1, sequence_length, 1, generator=generator).to(device)
        scaling_times = []
        for _ in range(repeats):
            _, duration = elapsed(lambda: component_model(values, None)[0])
            scaling_times.append(duration)
        rows.append(
            {
                **common,
                "benchmark": "chronological_sequence_scaling",
                "configuration": "recurrent_forward",
                "component": "forward",
                "sequence_length": sequence_length,
                "latency_mean_s": float(np.mean(scaling_times)),
                "latency_std_s": float(np.std(scaling_times)),
                "latency_per_timestep_s": float(np.mean(scaling_times)) / sequence_length,
            }
        )
    schedule_epochs = config.get("runtime_schedule_epochs", 100)
    schedule_teacher_epochs = config.get("runtime_schedule_teacher_epochs", schedule_epochs)
    late_start = int(np.ceil(0.7 * schedule_epochs))
    for configuration, bic_weight, bic_start in (
        ("no_bic", 0.0, 0),
        ("full_bic", config.get("runtime_bic_weight", 0.01), 0),
        ("late_bic", config.get("runtime_bic_weight", 0.01), late_start),
    ):
        scheduled = make_pipeline(config, seed)
        begin = time.perf_counter()
        metrics = fit_pipeline(
            scheduled,
            train,
            validation,
            test,
            config,
            lambda_bic=bic_weight,
            bic_start_epoch=bic_start,
            teacher_epochs=schedule_teacher_epochs,
            student_epochs=schedule_epochs,
            cluster_candidates=(),
        )
        duration = time.perf_counter() - begin
        weights = torch.cat(
            [parameter.detach().reshape(-1) for parameter in scheduled.student.parameters()]
        )
        rows.append(
            {
                "benchmark": "bic_schedule_validation",
                "configuration": configuration,
                "component": "complete_training",
                "sequence_length": len(item.time),
                "latency_mean_s": duration,
                "training_pipeline_runtime_s": duration,
                "teacher_epochs": schedule_teacher_epochs,
                "student_epochs": schedule_epochs,
                "bic_start_epoch": bic_start if bic_weight else None,
                "validation_mse": metrics["validation"]["mse"],
                "test_mse": metrics["test"]["mse"],
                "soft_effective_dof": float(exact_effective_dof(weights, 0.1)),
                "device": config["device"],
                "seed": seed,
                "repeats": 1,
                "profile": config["profile"],
                **context.checkpoint(
                    scheduled.student, f"runtime_schedule_{configuration}_seed{seed}"
                ),
                **source_record(train, validation, test),
            }
        )
    return rows


def dataset_integrity(context: RunContext) -> list[dict[str, Any]]:
    trajectory = load_3d_thermal_csv(ROOT / "data" / "3d_thermal_crosstalk_data.csv")
    return [
        {
            "dataset": "3d_thermal_crosstalk_data.csv",
            "source_id": trajectory.source_trajectory_id,
            "timesteps": len(trajectory.time),
            "feature_count": trajectory.features.shape[1],
            "output_count": trajectory.targets.shape[1],
            "chronological": bool(torch.all(trajectory.time[1:] > trajectory.time[:-1])),
            "profile": context.config["profile"],
        }
    ]
