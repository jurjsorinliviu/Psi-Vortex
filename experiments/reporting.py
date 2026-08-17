"""Aggregate tables, figures, hashes, and run-completeness records."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import RunContext, sha256_file, write_csv


CLAIM_TABLE_COLUMNS = {
    "circuit_validation": [
        "stimulus",
        "openvaf_version",
        "ngspice_version_label",
        "osdi_max_abs_error",
        "circuit_claim_valid",
    ]
}


def _latex_ready(frame: pd.DataFrame, experiment: str) -> pd.DataFrame:
    columns = CLAIM_TABLE_COLUMNS.get(experiment)
    table = frame.loc[:, [column for column in columns if column in frame]].copy() if columns else frame.copy()
    if experiment == "circuit_validation":
        if "openvaf_version" in table:
            table["openvaf_version"] = table["openvaf_version"].str.replace(
                "OpenVAF-reloaded ", "", regex=False
            )
        table = table.rename(
            columns={
                "openvaf_version": "OpenVAF-r",
                "ngspice_version_label": "ngspice",
                "osdi_max_abs_error": "max_abs_error",
                "circuit_claim_valid": "valid",
            }
        )
    for column in table.select_dtypes(include=["object"]).columns:
        table[column] = (
            table[column]
            .fillna("")
            .astype(str)
            .str.replace(r"[\r\n]+", "; ", regex=True)
        )
    return table


def statistical_summary(context: RunContext) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((context.output / "raw_results").glob("*.csv")):
        if path.name in {"statistical_summary.csv", "artifact_inventory.csv"}:
            continue
        frame = pd.read_csv(path)
        numeric = frame.select_dtypes(include=[np.number]).columns
        for column in numeric:
            values = frame[column].dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "experiment": path.stem,
                    "metric": column,
                    "count": int(values.count()),
                    "mean": float(values.mean()),
                    "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "profile": context.config["profile"],
                }
            )
    return rows


def figures_and_tables(context: RunContext) -> list[dict[str, Any]]:
    """Regenerate claim-facing figures from clean raw results only."""
    specifications = [
        ("latent_recovery", "alpha_true", "alpha_rec", "method"),
        ("recurrence_controls", "control", "r2", "control"),
        ("negative_controls", "control", "r2", "control"),
        ("synthetic_baselines", "alpha_true", "r2", "method"),
        ("rrad_ablation", "variant", "r2", "variant"),
        ("initialization_ablation", "initialization", "validation_mse", "benchmark"),
        ("bic_structural_ablation", "method", "test_mse", "method"),
        ("architecture_search", "teacher_hidden", "validation_mse", "teacher_blocks"),
        ("automatic_symmetry", "mode", "test_mse", "dataset"),
        ("detection_regime", "source_count", "r2", "noise_pct"),
        ("artifact_stress", "artifact", "r2", "artifact"),
        ("geometry_transfer", "geometry", "r2", "geometry"),
        ("multilayer", "layer", "alpha_rec", "layer"),
        ("measured_fidelity", "device", "nrmse", "device"),
        ("measured_baselines", "method", "nrmse", "device"),
        ("cross_device", "target_device", "nrmse", "source_device"),
        ("cross_rate", "test_sheet", "nrmse", "device"),
        ("compression_fidelity", "hidden_size", "nrmse", "student_type"),
        ("frequency_response", "pulse_width_steps", "r2", "pulse_width_steps"),
        ("learning_rate_sensitivity", "learning_rate", "test_mse", "learning_rate"),
        ("long_sequence", "evaluated_timesteps", "runtime_s", "profile"),
        ("scalable_bic", "weights", "minibatch_dof", "profile"),
        ("runtime_benchmark", "sequence_length", "latency_mean_s", "benchmark"),
    ]
    rows: list[dict[str, Any]] = []
    for experiment, x_column, y_column, group_column in specifications:
        source = context.output / "raw_results" / f"{experiment}.csv"
        if not source.exists():
            rows.append({"experiment": experiment, "status": "skipped", "reason": "raw results unavailable"})
            continue
        frame = pd.read_csv(source)
        if not {x_column, y_column, group_column}.issubset(frame.columns):
            rows.append({"experiment": experiment, "status": "skipped", "reason": "required columns unavailable"})
            continue
        figure, axis = plt.subplots(figsize=(6.5, 4.2))
        for label, subset in frame.groupby(group_column, dropna=False, sort=False):
            summary = subset.groupby(x_column, as_index=False, sort=False)[y_column].mean()
            axis.plot(summary[x_column], summary[y_column], marker="o", label=str(label))
        axis.set_xlabel(x_column.replace("_", " "))
        axis.set_ylabel(y_column.replace("_", " "))
        axis.grid(alpha=0.25)
        if frame[group_column].nunique(dropna=False) > 1:
            axis.legend(frameon=False)
        figure.tight_layout()
        destination = context.output / "figures" / f"{experiment}.png"
        figure.savefig(destination, dpi=180)
        plt.close(figure)
        # A compact LaTeX table is generated from seed-aggregated numeric columns.
        numeric = list(frame.select_dtypes(include=[np.number]).columns)
        grouping = list(dict.fromkeys(column for column in (group_column, x_column) if column in frame))
        if grouping:
            aggregate_columns = [column for column in numeric if column not in grouping]
            table = frame.groupby(grouping, dropna=False)[aggregate_columns].mean(numeric_only=True).reset_index()
        else:
            table = frame
        table_path = context.output / "tables" / f"{experiment}.tex"
        table_path.write_text(
            _latex_ready(table, experiment).to_latex(
                index=False, float_format="%.5g", escape=True
            ),
            encoding="utf-8",
        )
        rows.append(
            {
                "experiment": experiment,
                "status": "generated",
                "figure": destination.relative_to(context.output).as_posix(),
                "table": table_path.relative_to(context.output).as_posix(),
            }
        )
    # Every raw experiment receives a machine-regenerated LaTeX table, including
    # categorical deployment and integrity experiments for which a plot would add
    # no information.
    tabled = {row["experiment"] for row in rows if row.get("table")}
    for source in sorted((context.output / "raw_results").glob("*.csv")):
        experiment = source.stem
        if experiment in tabled or experiment in {"figures_and_tables", "statistical_summary"}:
            continue
        frame = pd.read_csv(source)
        table_path = context.output / "tables" / f"{experiment}.tex"
        table_path.write_text(
            _latex_ready(frame, experiment).to_latex(
                index=False, float_format="%.5g", escape=True
            ),
            encoding="utf-8",
        )
        rows.append(
            {
                "experiment": experiment,
                "status": "generated",
                "figure": "",
                "table": table_path.relative_to(context.output).as_posix(),
            }
        )
    return rows


def finalize(context: RunContext, selected_groups: list[str], failures: list[dict[str, str]]) -> None:
    completed = {
        path.stem.removeprefix("run_")
        for path in (context.output / "manifests").glob("run_*.json")
    }
    status_rows = [
        {
            "experiment": group,
            "requested": group in selected_groups,
            "completed": group in completed,
            "status": "failed"
            if any(item["group"] == group for item in failures)
            else ("completed" if group in completed else "not_run"),
            "reason": next(
                (item["error"] for item in failures if item["group"] == group), ""
            ),
        }
        for group in selected_groups
    ]
    write_csv(context.output / "manifests" / "run_completeness.csv", status_rows)
    (context.output / "manifests" / "failures.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8"
    )
    # Write the immutable ledger last so it includes the final completion and
    # failure records and cannot hash stale versions of them.
    inventory: list[dict[str, Any]] = []
    for path in sorted(context.output.rglob("*")):
        if path.is_file() and path.name != "artifact_inventory.csv":
            inventory.append(
                {
                    "relative_path": path.relative_to(context.output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_csv(context.output / "manifests" / "artifact_inventory.csv", inventory)
