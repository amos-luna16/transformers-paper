from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from transformer_nmt.config import Settings, get_settings
from transformer_nmt.training.trainer import run_training


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line overrides for a training run.

    Every option defaults to the value resolved by the Pydantic settings (``.env`` and
    environment variables), so the script can run with no arguments.
    """
    parser = argparse.ArgumentParser(description="Train the Transformer translator.")
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, help="Per-device batch size.")
    parser.add_argument(
        "--precision", choices=["auto", "bf16", "fp16", "fp32"], help="Numeric precision."
    )
    parser.add_argument(
        "--max-pairs", type=int, help="Number of sentence pairs read from the corpus."
    )
    parser.add_argument("--run-name", type=str, help="MLflow run name and checkpoint directory.")
    parser.add_argument("--experiment", type=str, help="MLflow experiment name.")
    parser.add_argument(
        "--device", type=str, help="Force a device for single-process runs (cpu, mps, cuda)."
    )
    parser.add_argument(
        "--launcher", type=str, default="local", help="Tag describing where the run executes."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast 2-epoch run on a 3k-pair subset to validate the pipeline.",
    )
    parser.add_argument(
        "--summary-json", type=Path, help="Write the training summary to this JSON file."
    )
    return parser.parse_args(argv)


def build_settings(args: argparse.Namespace) -> Settings:
    """Combine environment-driven settings with command-line overrides."""
    settings = get_settings()
    if args.smoke:
        settings = settings.smoke_test_variant()
    training_overrides = {
        key: value
        for key, value in (
            ("epochs", args.epochs),
            ("batch_size", args.batch_size),
            ("precision", args.precision),
        )
        if value is not None
    }
    data_overrides = {"max_pairs": args.max_pairs} if args.max_pairs is not None else {}
    tracking_overrides = {
        key: value
        for key, value in (("run_name", args.run_name), ("experiment_name", args.experiment))
        if value is not None
    }
    return settings.with_overrides(
        training=training_overrides, data=data_overrides, tracking=tracking_overrides
    )


def main(argv: list[str] | None = None) -> int:
    """Script entry point; returns a process exit code."""
    args = parse_args(argv)
    settings = build_settings(args)
    summary = run_training(
        settings,
        run_type="smoke" if args.smoke else "full",
        launcher=args.launcher,
        device=args.device,
    )
    if args.summary_json is not None and int(os.environ.get("RANK", "0")) == 0:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
