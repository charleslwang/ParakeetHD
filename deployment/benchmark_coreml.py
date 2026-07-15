#!/usr/bin/env python3
"""Measure Core ML bundle size and prediction latency."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np


def path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def load_fixture(bundle_dir: Path, fixture: dict) -> dict[str, np.ndarray]:
    with np.load(bundle_dir / "validation" / fixture["inputs"], allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def time_predictions(model, inputs: dict[str, np.ndarray], warmup: int, iterations: int) -> dict:
    for _ in range(warmup):
        model.predict(inputs)

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        model.predict(inputs)
        samples.append(time.perf_counter() - start)

    return {
        "iterations": iterations,
        "mean_ms": statistics.fmean(samples) * 1000.0,
        "median_ms": statistics.median(samples) * 1000.0,
        "min_ms": min(samples) * 1000.0,
        "max_ms": max(samples) * 1000.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    try:
        import coremltools as ct
    except ImportError as exc:
        raise SystemExit("coremltools is required; install requirements-coreml.txt") from exc

    bundle_path = args.bundle.expanduser().resolve()
    bundle_dir = bundle_path.parent
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    manifest = json.loads((bundle_dir / "validation" / "manifest.json").read_text(encoding="utf-8"))

    report = {
        "host": platform.platform(),
        "python": platform.python_version(),
        "coremltools_version": getattr(ct, "__version__", "unknown"),
        "bundle": str(bundle_path),
        "bundle_size_bytes": path_size_bytes(bundle_dir),
        "models": {},
    }

    for fixture_key, model_key in [("encoder", "encoder"), ("decoder_joint", "decoder_joint")]:
        model_cfg = bundle["coreml"][model_key]
        model_path = bundle_dir / model_cfg["path"]
        model = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.ALL)
        inputs = load_fixture(bundle_dir, manifest[fixture_key])
        report["models"][model_key] = {
            "path": model_cfg["path"],
            "size_bytes": path_size_bytes(model_path),
            "inputs": {name: list(value.shape) for name, value in inputs.items()},
            "latency": time_predictions(model, inputs, args.warmup, args.iterations),
        }

    text = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
