#!/usr/bin/env python3
"""Compare exported Core ML subnetworks with saved PyTorch reference outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np


def hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for file_path in files:
        relative = file_path.name if path.is_file() else file_path.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def compare_arrays(actual: np.ndarray, expected: np.ndarray, atol: float, rtol: float) -> dict:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return {"passed": False, "reason": f"shape mismatch: {actual.shape} != {expected.shape}"}
    if np.issubdtype(expected.dtype, np.integer):
        passed = bool(np.array_equal(actual, expected))
        return {"passed": passed, "max_absolute_error": 0.0 if passed else None}
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {
        "passed": bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
        "max_absolute_error": float(difference.max(initial=0.0)),
        "mean_absolute_error": float(difference.mean()) if difference.size else 0.0,
    }


def validate_model(ct, bundle_dir: Path, config: dict, fixture: dict, atol: float, rtol: float) -> dict:
    model_path = bundle_dir / config["path"]
    fixture_dir = bundle_dir / "validation"
    with np.load(fixture_dir / fixture["inputs"], allow_pickle=False) as stored_inputs:
        inputs = {name: stored_inputs[name] for name in stored_inputs.files}
    model = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.ALL)
    prediction = model.predict(inputs)

    comparisons = {}
    with np.load(fixture_dir / fixture["expected_outputs"], allow_pickle=False) as expected:
        for index, output_name in enumerate(fixture["coreml_output_names"]):
            comparisons[output_name] = compare_arrays(
                prediction[output_name],
                expected[f"output_{index}"],
                atol=atol,
                rtol=rtol,
            )
    return {
        "passed": all(result["passed"] for result in comparisons.values()),
        "artifact_sha256": hash_path(model_path),
        "outputs": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if platform.system() != "Darwin":
        raise SystemExit("Core ML prediction validation must run on macOS.")
    try:
        import coremltools as ct
    except ImportError as exc:
        raise SystemExit("coremltools is required; install requirements-coreml.txt") from exc

    bundle_path = args.bundle.expanduser().resolve()
    bundle_dir = bundle_path.parent
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    fixtures = json.loads((bundle_dir / "validation" / "manifest.json").read_text(encoding="utf-8"))

    report = {
        "encoder": validate_model(
            ct,
            bundle_dir,
            bundle["coreml"]["encoder"],
            fixtures["encoder"],
            args.atol,
            args.rtol,
        ),
        "decoder_joint": validate_model(
            ct,
            bundle_dir,
            bundle["coreml"]["decoder_joint"],
            fixtures["decoder_joint"],
            args.atol,
            args.rtol,
        ),
    }
    report["passed"] = report["encoder"]["passed"] and report["decoder_joint"]["passed"]
    report_path = args.report or (bundle_dir / "validation" / "report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
