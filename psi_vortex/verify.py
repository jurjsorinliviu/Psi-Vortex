"""Integrity checks for the source reproducibility bundle."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_repository(root: str | Path | None = None) -> dict[str, object]:
    """Verify immutable inputs and registry/coverage correspondence."""
    repository = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]
    data_manifest = repository / "manifests" / "data_manifest.csv"
    if not data_manifest.is_file():
        raise FileNotFoundError(f"missing data manifest: {data_manifest}")
    verified_files: list[str] = []
    with data_manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            relative = row["relative_path"]
            path = repository / relative
            if not path.is_file():
                raise FileNotFoundError(f"manifested input is missing: {relative}")
            actual_bytes = path.stat().st_size
            if actual_bytes != int(row["bytes"]):
                raise ValueError(f"size mismatch for {relative}: {actual_bytes} != {row['bytes']}")
            if _sha256(path) != row["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {relative}")
            verified_files.append(relative)

    from experiments.registry import EXPERIMENTS

    coverage_path = repository / "manifests" / "experiment_coverage.csv"
    with coverage_path.open(newline="", encoding="utf-8") as stream:
        coverage_rows = list(csv.DictReader(stream))
    covered = [row["experiment_group"] for row in coverage_rows]
    registered = list(EXPERIMENTS)
    if len(covered) != len(set(covered)):
        raise ValueError("experiment coverage manifest contains duplicate groups")
    if set(covered) != set(registered):
        raise ValueError(
            "coverage/registry mismatch: "
            f"missing={sorted(set(registered) - set(covered))}, "
            f"extra={sorted(set(covered) - set(registered))}"
        )
    required_coverage_columns = {
        "experiment_group",
        "manuscript_evidence",
        "canonical_output",
        "external_requirement",
    }
    if not coverage_rows or required_coverage_columns != set(coverage_rows[0]):
        raise ValueError("experiment coverage manifest has an invalid public schema")
    if any(not row["manuscript_evidence"] or not row["canonical_output"] for row in coverage_rows):
        raise ValueError("every experiment group must name its manuscript evidence and output")
    required_configs = ["configs/smoke.json", "configs/final.json"]
    for relative in required_configs:
        if not (repository / relative).is_file():
            raise FileNotFoundError(f"missing required configuration: {relative}")
    return {
        "status": "ok",
        "repository": str(repository),
        "verified_files": len(verified_files),
        "registered_experiment_groups": len(registered),
        "configs": required_configs,
    }
