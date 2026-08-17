"""Command-line interface for the complete reproducibility suite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from experiments.registry import EXPERIMENTS
    from experiments.run import run_experiments

    parser = argparse.ArgumentParser(prog="psi-vortex")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list canonical experiment groups")
    subparsers.add_parser("verify", help="verify bundled data hashes and experiment coverage")
    run_parser = subparsers.add_parser("run", help="run one or more experiment groups")
    run_parser.add_argument("--config", default="configs/smoke.json")
    run_parser.add_argument("--groups", default="all")
    run_parser.add_argument("--output")
    run_parser.add_argument("--fail-fast", action="store_true")
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only provenance-matching completed groups in an existing run",
    )
    args = parser.parse_args(argv)
    if args.command == "list":
        for name in EXPERIMENTS:
            print(name)
        return 0
    if args.command == "verify":
        from .verify import verify_repository

        report = verify_repository()
        print(json.dumps(report, indent=2))
        return 0
    groups = [item.strip() for item in args.groups.split(",") if item.strip()]
    context, failures = run_experiments(
        args.config,
        groups,
        output=args.output,
        fail_fast=args.fail_fast,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "output": str(context.output),
                "requested_groups": groups,
                "failures": failures,
            },
            indent=2,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
