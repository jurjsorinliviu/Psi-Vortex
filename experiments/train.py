"""Common model construction and physical readout helpers."""
from __future__ import annotations

from typing import Any

import numpy as np

from psi_vortex import EndToEndPipeline, fit_coupling, thermal_split

from .common import predict


def make_pipeline(
    config: dict[str, Any],
    seed: int,
    *,
    input_size: int = 1,
    output_size: int = 1,
    student_hidden: int | None = None,
    student_type: str | None = None,
    student_rank: int | None = None,
    teacher_hidden: int | None = None,
    teacher_blocks: int | None = None,
    teacher_memory_size: int | None = None,
    real: bool = False,
) -> EndToEndPipeline:
    teacher_hidden_value = config.get("real_teacher_hidden", config["teacher_hidden"]) if real else config["teacher_hidden"]
    teacher_blocks_value = config.get("real_teacher_blocks", config["teacher_blocks"]) if real else config["teacher_blocks"]
    student_hidden_value = config.get("real_student_hidden", config["student_hidden"]) if real else config["student_hidden"]
    return EndToEndPipeline(
        input_size,
        output_size,
        teacher_hidden or teacher_hidden_value,
        student_hidden or student_hidden_value,
        teacher_blocks or teacher_blocks_value,
        teacher_memory_size=teacher_memory_size,
        student_type=student_type or config["student_type"],
        student_rank=student_rank or config["student_rank"],
        seed=seed,
        device=config["device"],
    )


def fit_pipeline(
    pipeline: EndToEndPipeline,
    train,
    validation,
    test,
    config: dict[str, Any],
    *,
    real: bool = False,
    lambda_bic: float | None = None,
    rrad_weights: tuple[float, float, float, float] = (1.0, 0.5, 1.0, 0.5),
    cluster_candidates: tuple[int, ...] | None = None,
    parameter_regularizer: tuple[str, float] | None = None,
    bic_start_epoch: int = 0,
    teacher_epochs: int | None = None,
    student_epochs: int | None = None,
):
    return pipeline.fit(
        train,
        validation,
        test,
        teacher_epochs=(
            teacher_epochs
            if teacher_epochs is not None
            else (config["real_epochs"] if real else config["teacher_epochs"])
        ),
        student_epochs=(
            student_epochs
            if student_epochs is not None
            else (config["real_epochs"] if real else config["student_epochs"])
        ),
        batch_size=min(config["batch_size"], len(train)),
        chunk_length=config["chunk_length"],
        teacher_lr=(config.get("real_teacher_learning_rate", config["learning_rate"]) if real else config.get("teacher_learning_rate", config["learning_rate"])),
        student_lr=(config.get("real_student_learning_rate", config["learning_rate"]) if real else config.get("student_learning_rate", config["learning_rate"])),
        max_grad_norm=(config.get("real_gradient_clip") if real else config.get("gradient_clip")),
        cluster_candidates=(
            tuple(config["cluster_candidates"])
            if cluster_candidates is None
            else cluster_candidates
        ),
        lambda_bic=config["lambda_bic"] if lambda_bic is None else lambda_bic,
        bic_start_epoch=bic_start_epoch,
        rrad_weights=rrad_weights,
        parameter_regularizer=parameter_regularizer,
    )


def thermal_data(config: dict[str, Any], alpha, **kwargs):
    n_steps = kwargs.pop("n_steps", config["n_steps"])
    stride = kwargs.pop("stride", config["stride"])
    return thermal_split(
        alpha,
        config["driver_train_sources"],
        config["driver_validation_sources"],
        config["driver_test_sources"],
        n_steps=n_steps,
        stride=stride,
        **kwargs,
    )


def coupling_scores(
    model,
    samples,
    output_index: int = 0,
    threshold: float = 0.8,
    driver_band: float | None = 0.5,
):
    estimates = []
    for sample in samples:
        prediction = predict(model, sample.trajectory)[:, output_index]
        estimates.append(
            fit_coupling(
                prediction,
                sample.delta_temperature,
                r2_threshold=threshold,
                driver_band=driver_band,
            )
        )
    return estimates


def mean_estimate(estimates):
    return {
        "alpha_rec": float(np.mean([item.value for item in estimates])),
        "r2": float(np.mean([item.r2 for item in estimates])),
        "accepted": bool(all(item.accepted for item in estimates)),
    }
