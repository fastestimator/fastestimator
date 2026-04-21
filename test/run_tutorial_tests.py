# Copyright 2024 The FastEstimator Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Script to run all tutorial notebooks and report their pass/fail status.

Usage:
    python run_tutorial_tests.py [--tutorial-dir PATH] [--output-dir PATH]
                                 [--kernel KERNEL] [--report FILE]
                                 [--timeout SECONDS]

Options:
    --tutorial-dir  Path to the tutorial directory (default: ../tutorial relative to this script).
    --output-dir    Directory to write executed notebook outputs (default: /tmp/tutorial_outputs).
    --kernel        Jupyter kernel name to use (default: python3).
    --report        Path to write the JSON results report (default: tutorial_test_report.json
                    in the output directory).
    --timeout       Per-cell execution timeout in seconds (default: 600).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class NotebookResult:
    path: str
    status: str  # "pass" | "fail" | "skip"
    duration: float = 0.0
    error: Optional[str] = None
    output_path: Optional[str] = None


# Notebooks skipped by default (e.g. known failures, long-running, or work-in-progress).
DEFAULT_SKIP: List[str] = ["t11_model_calibration", "t16_robustness"]


def discover_notebooks(tutorial_dir: Path) -> List[Path]:
    """Return all .ipynb files under *tutorial_dir*, sorted by path."""
    return sorted(tutorial_dir.rglob("*.ipynb"))


def run_notebook(nb_path: Path, output_dir: Path, kernel: str, timeout: int) -> NotebookResult:
    """Execute a single notebook with papermill and return the result."""
    relative = nb_path.name
    output_path = output_dir / nb_path.parent.name / nb_path.stem / (nb_path.stem + "_out.ipynb")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stderr_path = output_path.with_suffix(".stderr.txt")

    cmd = [
        sys.executable,
        "-m",
        "papermill",
        str(nb_path),
        str(output_path),
        "--kernel",
        kernel,
        "--execution-timeout",
        str(timeout),
        "--cwd",
        str(nb_path.parent),
        "--no-progress-bar",
    ]

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        duration = time.monotonic() - start

        # Write stderr regardless of outcome so failures are diagnosable
        stderr_path.write_text(proc.stderr)

        if proc.returncode == 0:
            return NotebookResult(
                path=str(nb_path),
                status="pass",
                duration=duration,
                output_path=str(output_path),
            )
        else:
            # Extract the last meaningful error line for the summary
            error_lines = [l for l in proc.stderr.splitlines() if l.strip()]
            short_error = error_lines[-1] if error_lines else f"exit code {proc.returncode}"
            return NotebookResult(
                path=str(nb_path),
                status="fail",
                duration=duration,
                error=short_error,
                output_path=str(output_path),
            )
    except FileNotFoundError:
        duration = time.monotonic() - start
        return NotebookResult(
            path=str(nb_path),
            status="fail",
            duration=duration,
            error="papermill not found — install it with: pip install papermill",
        )
    except Exception as exc:
        duration = time.monotonic() - start
        return NotebookResult(
            path=str(nb_path),
            status="fail",
            duration=duration,
            error=str(exc),
        )


def print_report(results: List[NotebookResult], tutorial_dir: Path) -> None:
    """Print a human-readable summary table."""
    col_w = max(len(str(Path(r.path).relative_to(tutorial_dir.parent))) for r in results) + 2

    header = f"{'Notebook':<{col_w}}  {'Status':<6}  {'Duration':>10}"
    print("\n" + "=" * len(header))
    print("TUTORIAL NOTEBOOK TEST REPORT")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        rel = str(Path(r.path).relative_to(tutorial_dir.parent))
        status_label = r.status.upper()
        duration_str = f"{r.duration:.1f}s"
        print(f"{rel:<{col_w}}  {status_label:<6}  {duration_str:>10}")
        if r.error:
            print(f"    ERROR: {r.error}")

    print("-" * len(header))

    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    skipped = sum(1 for r in results if r.status == "skip")
    total = len(results)

    print(f"\nTotal: {total}  |  Passed: {passed}  |  Failed: {failed}  |  Skipped: {skipped}")

    if failed:
        print("\nFailed notebooks:")
        for r in results:
            if r.status == "fail":
                rel = Path(r.path).relative_to(tutorial_dir.parent)
                print(f"  - {rel}")
    else:
        print("\nAll notebooks passed.")

    print("=" * len(header) + "\n")


def save_report(results: List[NotebookResult], report_path: Path, tutorial_dir: Path) -> None:
    """Save results as a JSON file for CI consumption."""
    data = {
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.status == "pass"),
            "failed": sum(1 for r in results if r.status == "fail"),
            "skipped": sum(1 for r in results if r.status == "skip"),
        },
        "notebooks": [{
            "notebook": str(Path(r.path).relative_to(tutorial_dir.parent)),
            "status": r.status,
            "duration": round(r.duration, 2),
            "error": r.error,
            "output": r.output_path,
        } for r in results],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(data, indent=2))
    print(f"Report saved to: {report_path}")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_tutorial = script_dir.parent / "tutorial"
    default_output = Path("/tmp/tutorial_outputs")

    parser = argparse.ArgumentParser(
        description="Run all FastEstimator tutorial notebooks and report their status.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tutorial-dir",
        type=Path,
        default=default_tutorial,
        help="Root directory containing tutorial notebooks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory to write executed notebook outputs and stderr logs.",
    )
    parser.add_argument(
        "--kernel",
        default="python3",
        help="Jupyter kernel name to use when executing notebooks.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Path for the JSON results report (default: <output-dir>/tutorial_test_report.json).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-cell execution timeout in seconds.",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=DEFAULT_SKIP,
        metavar="STEM",
        help="Notebook stem(s) to skip (without .ipynb). Defaults to: %(default)s.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    tutorial_dir: Path = args.tutorial_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    report_path: Path = (args.report or (output_dir / "tutorial_test_report.json")).resolve()

    if not tutorial_dir.is_dir():
        print(f"ERROR: tutorial directory not found: {tutorial_dir}", file=sys.stderr)
        return 1

    notebooks = discover_notebooks(tutorial_dir)
    if not notebooks:
        print(f"No .ipynb files found under {tutorial_dir}", file=sys.stderr)
        return 1

    skip_stems = set(args.skip or [])
    if skip_stems:
        print(f"Skipping notebooks: {sorted(skip_stems)}")

    print(f"Found {len(notebooks)} notebook(s) under {tutorial_dir}")
    print(f"Outputs will be written to: {output_dir}\n")

    results: List[NotebookResult] = []
    for nb in notebooks:
        rel = nb.relative_to(tutorial_dir.parent)
        if nb.stem in skip_stems:
            print(f"Skipping: {rel}")
            results.append(NotebookResult(path=str(nb), status="skip"))
            continue
        print(f"Running: {rel} ... ", end="", flush=True)
        result = run_notebook(nb, output_dir, args.kernel, args.timeout)
        print(f"{result.status.upper()}  ({result.duration:.1f}s)")
        if result.error:
            print(f"  -> {result.error}")
        results.append(result)

    print_report(results, tutorial_dir)
    save_report(results, report_path, tutorial_dir)

    # Return non-zero exit code if any notebook failed (useful for CI)
    return 1 if any(r.status == "fail" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
