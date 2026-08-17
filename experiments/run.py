"""Configuration-driven, failure-recording experiment runner."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from psi_vortex import __version__

from .common import ROOT, RunContext, sha256_file, source_tree_hash
from .registry import EXPERIMENTS
from .reporting import finalize


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "profile",
        "device",
        "seeds",
        "driver_train_sources",
        "driver_validation_sources",
        "driver_test_sources",
        "teacher_hidden",
        "teacher_blocks",
        "student_hidden",
        "teacher_epochs",
        "student_epochs",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"configuration is missing keys: {sorted(missing)}")
    return config


def _config_hash(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_progress(
    output: Path,
    *,
    selected: list[str],
    completed: set[str],
    current_group: str | None,
    status: str,
    run_started_utc: str,
    group_started_utc: str | None = None,
    error: str | None = None,
) -> None:
    record = {
        "status": status,
        "run_started_utc": run_started_utc,
        "updated_utc": _utc_now(),
        "group_started_utc": group_started_utc,
        "current_group": current_group,
        "completed_groups": [group for group in selected if group in completed],
        "completed_count": sum(group in completed for group in selected),
        "total_count": len(selected),
        "package_version": __version__,
        "source_tree_sha256": source_tree_hash(),
        "error": error,
    }
    (output / "manifests" / "progress.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )


def _validate_resume_state(
    output: Path, source_config: Path, config: dict
) -> set[str]:
    """Prove that completed groups belong to this exact executable run.

    A resume never treats mere file presence as completion.  It requires the
    original configuration, package release, complete source tree, execution
    backend, result-row count, and every referenced checkpoint to agree.
    """
    stored_config_path = output / "configs" / source_config.name
    runtime_path = output / "environment" / "runtime.json"
    if not stored_config_path.is_file() or not runtime_path.is_file():
        raise ValueError(
            "resume target is not a canonical run directory: stored config or "
            "runtime provenance is missing"
        )
    stored_config = json.loads(stored_config_path.read_text(encoding="utf-8"))
    if stored_config != config:
        raise ValueError("resume configuration does not match the stored run")

    expected_source = source_tree_hash()
    expected_config = _config_hash(config)
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime_expectations = {
        "package_version": __version__,
        "source_tree_sha256": expected_source,
        "device_requested": config["device"],
    }
    for field, expected in runtime_expectations.items():
        if runtime.get(field) != expected:
            raise ValueError(
                f"resume provenance mismatch for {field}: "
                f"stored={runtime.get(field)!r}, current={expected!r}"
            )

    completed: set[str] = set()
    checked_checkpoints: dict[Path, str] = {}
    for manifest_path in sorted((output / "manifests").glob("run_*.json")):
        group = manifest_path.stem.removeprefix("run_")
        if group not in EXPERIMENTS:
            raise ValueError(f"resume target contains an unknown completed group: {group}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expectations = {
            "group": group,
            "profile": config["profile"],
            "device": config["device"],
            "package_version": __version__,
            "source_tree_sha256": expected_source,
            "config_sha256": expected_config,
        }
        for field, expected in expectations.items():
            if manifest.get(field) != expected:
                raise ValueError(
                    f"resume manifest mismatch for {group}.{field}: "
                    f"stored={manifest.get(field)!r}, current={expected!r}"
                )
        rows = int(manifest.get("rows", -1))
        result_path = output / "raw_results" / f"{group}.csv"
        if rows > 0 and not result_path.is_file():
            raise ValueError(f"completed group {group} has no raw-result table")
        if result_path.is_file() and _csv_row_count(result_path) != rows:
            raise ValueError(f"completed group {group} has a row-count mismatch")
        expected_result_digest = manifest.get("raw_result_sha256")
        if result_path.is_file() and (
            not expected_result_digest
            or sha256_file(result_path) != expected_result_digest
        ):
            raise ValueError(f"completed group {group} has a raw-result hash mismatch")
        if result_path.is_file():
            with result_path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    relative = row.get("checkpoint_path", "").strip()
                    expected_digest = row.get("checkpoint_file_sha256", "").strip()
                    if not relative or not expected_digest:
                        continue
                    checkpoint = (output / relative).resolve()
                    try:
                        checkpoint.relative_to(output.resolve())
                    except ValueError as error:
                        raise ValueError(
                            f"checkpoint for {group} escapes the run directory: {relative}"
                        ) from error
                    if checkpoint in checked_checkpoints:
                        if checked_checkpoints[checkpoint] != expected_digest:
                            raise ValueError(
                                f"conflicting checkpoint hashes for {relative}"
                            )
                        continue
                    if not checkpoint.is_file() or sha256_file(checkpoint) != expected_digest:
                        raise ValueError(
                            f"checkpoint integrity mismatch for completed group {group}: "
                            f"{relative}"
                        )
                    checked_checkpoints[checkpoint] = expected_digest
        completed.add(group)
    return completed


def run_experiments(
    config_path: str | Path,
    groups: list[str],
    *,
    output: str | Path | None = None,
    fail_fast: bool = False,
    resume: bool = False,
) -> tuple[RunContext, list[dict[str, str]]]:
    config = load_config(config_path)
    selected = list(EXPERIMENTS) if groups == ["all"] else groups
    unknown = set(selected) - set(EXPERIMENTS)
    if unknown:
        raise ValueError(f"unknown experiment groups: {sorted(unknown)}")
    output_path = (
        Path(output)
        if output is not None
        else ROOT / "results" / config["profile"]
    )
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    source_config = Path(config_path)
    if not source_config.is_absolute():
        source_config = ROOT / source_config
    nonempty = output_path.exists() and any(output_path.iterdir())
    if nonempty and not resume:
        raise FileExistsError(
            f"refusing to reuse non-empty output directory: {output_path}; "
            "choose a new run directory or pass resume=True after verifying provenance"
        )
    completed = (
        _validate_resume_state(output_path, source_config, config)
        if nonempty and resume
        else set()
    )
    context = RunContext(config, output_path)
    stored_config = context.output / "configs" / source_config.name
    if not stored_config.exists():
        shutil.copy2(source_config, stored_config)
    progress_path = context.output / "manifests" / "progress.json"
    run_started_utc = _utc_now()
    if resume and progress_path.is_file():
        previous_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        run_started_utc = previous_progress.get("run_started_utc", run_started_utc)
    _write_progress(
        context.output,
        selected=selected,
        completed=completed,
        current_group=None,
        status="starting",
        run_started_utc=run_started_utc,
    )
    failures: list[dict[str, str]] = []
    for group in selected:
        if group in completed:
            continue
        # A terminated attempt can leave a result table after write_rows() but
        # before provenance().  Without a completion manifest it is never reused.
        incomplete_result = context.output / "raw_results" / f"{group}.csv"
        if incomplete_result.exists():
            incomplete_result.unlink()
        failure_log = context.output / "logs" / f"{group}_failure.txt"
        if failure_log.exists():
            failure_log.unlink()
        group_started_utc = _utc_now()
        _write_progress(
            context.output,
            selected=selected,
            completed=completed,
            current_group=group,
            status="running",
            run_started_utc=run_started_utc,
            group_started_utc=group_started_utc,
        )
        started = time.perf_counter()
        try:
            rows = EXPERIMENTS[group](context)
            context.write_rows(group, rows)
            context.provenance(group, started, len(rows))
            completed.add(group)
        except Exception as error:  # recorded with full traceback; never mislabeled complete
            failure = {
                "group": group,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            failure_log.write_text(
                failure["traceback"], encoding="utf-8"
            )
            _write_progress(
                context.output,
                selected=selected,
                completed=completed,
                current_group=group,
                status="failed",
                run_started_utc=run_started_utc,
                group_started_utc=group_started_utc,
                error=failure["error"],
            )
            if fail_fast:
                finalize(context, selected, failures)
                raise
    _write_progress(
        context.output,
        selected=selected,
        completed=completed,
        current_group=None,
        status="failed" if failures else "completed",
        run_started_utc=run_started_utc,
        error="; ".join(item["error"] for item in failures) or None,
    )
    finalize(context, selected, failures)
    return context, failures
