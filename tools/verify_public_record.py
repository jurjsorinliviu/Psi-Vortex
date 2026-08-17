"""Verify the immutable public result record."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.registry import EXPERIMENTS


DEFAULT_RECORD = REPOSITORY_ROOT / "results" / "manuscript_record"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_ids(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    return {item for item in str(value).split(";") if item}


def verify(record: Path) -> dict[str, object]:
    record = record.resolve()
    summary_path = record / "manifests" / "record_summary.json"
    inventory_path = record / "manifests" / "file_inventory.csv"
    if not summary_path.is_file() or not inventory_path.is_file():
        raise FileNotFoundError("record summary and file inventory are required")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    expected_groups = set(EXPERIMENTS)
    raw_files = {path.stem for path in (record / "raw_results").glob("*.csv")}
    if raw_files != expected_groups:
        errors.append(
            f"raw-result groups differ: missing={sorted(expected_groups - raw_files)}, "
            f"extra={sorted(raw_files - expected_groups)}"
        )

    frames: dict[str, pd.DataFrame] = {}
    result_rows = 0
    checkpoint_references = 0
    unique_checkpoints: set[str] = set()
    checked_splits = 0
    for group in sorted(expected_groups & raw_files):
        frame = pd.read_csv(record / "raw_results" / f"{group}.csv")
        frames[group] = frame
        if frame.empty:
            errors.append(f"{group} has no result rows")
        result_rows += len(frame)
        split_columns = {"train_source_ids", "validation_source_ids", "test_source_ids"}
        if split_columns <= set(frame.columns):
            for index, row in frame.iterrows():
                train = source_ids(row.train_source_ids)
                validation = source_ids(row.validation_source_ids)
                test = source_ids(row.test_source_ids)
                if train & validation or train & test or validation & test:
                    errors.append(f"{group} row {index} crosses source-level splits")
                checked_splits += 1
        if "checkpoint_path" in frame.columns:
            for index, row in frame.iterrows():
                relative = row.get("checkpoint_path")
                if pd.isna(relative) or not str(relative):
                    continue
                checkpoint_references += 1
                normalized = str(relative).replace("\\", "/")
                unique_checkpoints.add(normalized)
                path = record / normalized
                if not path.is_file():
                    errors.append(f"{group} row {index} is missing {normalized}")
                    continue
                expected_hash = row.get("checkpoint_file_sha256")
                if not pd.isna(expected_hash) and sha256(path) != str(expected_hash):
                    errors.append(f"{group} row {index} has a checkpoint hash mismatch")

    with inventory_path.open(newline="", encoding="utf-8-sig") as stream:
        inventory_rows = list(csv.DictReader(stream))
    listed = {row["relative_path"] for row in inventory_rows}
    actual = {
        path.relative_to(record).as_posix()
        for path in record.rglob("*")
        if path.is_file() and path.resolve() != inventory_path.resolve()
    }
    if listed != actual:
        errors.append(
            f"file inventory differs: missing={sorted(actual - listed)}, "
            f"extra={sorted(listed - actual)}"
        )
    for row in inventory_rows:
        path = record / row["relative_path"]
        if not path.is_file():
            continue
        if path.stat().st_size != int(row["bytes"]):
            errors.append(f"size mismatch for {row['relative_path']}")
        elif sha256(path) != row["sha256"]:
            errors.append(f"SHA-256 mismatch for {row['relative_path']}")

    repository_config = REPOSITORY_ROOT / "configs" / "final.json"
    record_config = record / "configs" / "final.json"
    if repository_config.read_bytes() != record_config.read_bytes():
        errors.append("record and repository final configurations differ")

    recurrence = frames.get("recurrence_controls", pd.DataFrame())
    if not recurrence.empty:
        chronological = recurrence[recurrence.control == "chronological"]
        controls = recurrence[recurrence.control != "chronological"]
        if int(chronological.accepted.sum()) != 3 or int(controls.accepted.sum()) != 0:
            errors.append("recurrence-control acceptance counts do not match the manuscript")

    export = frames.get("export_validation", pd.DataFrame())
    if not export.empty:
        guards = export.length_one_guard.astype(str).str.lower().eq("true")
        maximum_error = max(
            float(export.batch_max_abs_error.max()),
            float(export.streaming_max_abs_error.max()),
        )
        if not guards.all() or maximum_error > 2e-5:
            errors.append("guarded export validation is outside the frozen tolerance")

    circuit = frames.get("circuit_validation", pd.DataFrame())
    if not circuit.empty:
        passed = circuit.circuit_claim_valid.astype(str).str.lower().eq("true")
        if not passed.all() or float(circuit.osdi_max_abs_error.max()) > 1e-3:
            errors.append("compiled circuit validation is outside the declared tolerance")

    observed = {
        "experiment_groups": len(raw_files),
        "result_rows": result_rows,
        "source_split_rows_checked": checked_splits,
        "checkpoint_references": checkpoint_references,
        "unique_checkpoints": len(unique_checkpoints),
    }
    for name, value in observed.items():
        if int(summary.get(name, -1)) != value:
            errors.append(f"summary mismatch for {name}: {summary.get(name)} != {value}")

    return {
        "status": "passed" if not errors else "failed",
        "record": str(record),
        **observed,
        "inventoried_files": len(inventory_rows),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path, nargs="?", default=DEFAULT_RECORD)
    args = parser.parse_args()
    report = verify(args.record)
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
