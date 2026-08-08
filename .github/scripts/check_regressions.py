#!/usr/bin/env python3
"""Regression gate for mypy and ruff against a checked-in baseline.

The codebase carries known, pre-existing mypy errors and ruff findings.
Rather than hard-failing CI on those (which would block every PR) or
silencing the tools entirely (which would let new problems in), this script
compares the current per-file finding counts against a checked-in baseline
(.github/regression_baseline.json) and fails ONLY on regressions:

  * a file whose mypy error count exceeds its baseline count
  * a file whose ruff-check violation count exceeds its baseline count
  * a file that ruff-format would reformat and that is not already listed
    in the baseline

Fixing findings never fails the gate. When the totals drop below the
baseline, the script prints a reminder to ratchet the baseline down.

Usage (from the repo root, inside the dev environment):

    pixi run -e dev check-regressions      # gate (CI runs this)
    pixi run -e dev update-baselines       # regenerate the baseline

Only update the baseline to ratchet DOWN after fixing findings, or in a PR
that deliberately accepts new debt with reviewer sign-off (explain why in
the PR description).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = REPO_ROOT / ".github" / "regression_baseline.json"

# Keep these targets in sync with the pixi `typecheck` and `lint` tasks in
# pyproject.toml.
MYPY_CMD = ["mypy", "src/lidar_tools/"]
RUFF_CHECK_CMD = ["ruff", "check", "src/lidar_tools/", "tests/", "--output-format=json"]
RUFF_FORMAT_CMD = ["ruff", "format", "--check", "src/lidar_tools/", "tests/"]

MYPY_ERROR_RE = re.compile(r"^(?P<path>[^:\n]+):\d+(?::\d+)?: error:")
WOULD_REFORMAT_RE = re.compile(r"^Would reformat: (?P<path>.+)$")


def _relpath(path: str) -> str:
    """Normalize tool-reported paths to repo-relative POSIX form."""
    p = Path(path)
    if p.is_absolute():
        try:
            p = p.relative_to(REPO_ROOT)
        except ValueError:
            pass
    return p.as_posix()


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )


def collect_mypy() -> dict[str, int]:
    proc = _run(MYPY_CMD)
    if proc.returncode not in (0, 1):
        # 2 = fatal error / bad usage: never mask this as "no findings".
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"mypy failed to run (exit {proc.returncode})")
    counts: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        m = MYPY_ERROR_RE.match(line)
        if m:
            path = _relpath(m.group("path"))
            counts[path] = counts.get(path, 0) + 1
    return counts


def collect_ruff_check() -> dict[str, int]:
    proc = _run(RUFF_CHECK_CMD)
    if proc.returncode not in (0, 1):
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"ruff check failed to run (exit {proc.returncode})")
    counts: dict[str, int] = {}
    for violation in json.loads(proc.stdout or "[]"):
        path = _relpath(violation["filename"])
        counts[path] = counts.get(path, 0) + 1
    return counts


def collect_ruff_format() -> list[str]:
    proc = _run(RUFF_FORMAT_CMD)
    if proc.returncode not in (0, 1):
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"ruff format --check failed to run (exit {proc.returncode})")
    files = []
    for line in (proc.stdout + proc.stderr).splitlines():
        m = WOULD_REFORMAT_RE.match(line.strip())
        if m:
            files.append(_relpath(m.group("path")))
    return sorted(files)


def _annotate(kind: str, message: str) -> None:
    """Emit a GitHub Actions annotation when running in CI, else plain text."""
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{kind}::{message}")
    else:
        print(f"{kind.upper()}: {message}")


def compare_counts(
    tool: str, current: dict[str, int], baseline: dict[str, int]
) -> tuple[list[str], bool]:
    """Return (regression messages, improved?) for a per-file count map."""
    regressions = []
    for path in sorted(current):
        base = baseline.get(path, 0)
        if current[path] > base:
            regressions.append(
                f"{tool}: {path} has {current[path]} findings "
                f"(baseline {base}) — fix the new ones introduced by this change"
            )
    improved = sum(current.values()) < sum(baseline.values()) and not regressions
    return regressions, improved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite the baseline file from the current findings",
    )
    args = parser.parse_args()

    current = {
        "mypy": collect_mypy(),
        "ruff_check": collect_ruff_check(),
        "ruff_format_would_reformat": collect_ruff_format(),
    }

    if args.update:
        BASELINE_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print(f"Baseline written to {BASELINE_PATH.relative_to(REPO_ROOT)}")
        for tool, findings in current.items():
            total = (
                len(findings) if isinstance(findings, list) else sum(findings.values())
            )
            print(f"  {tool}: {total} accepted findings")
        return 0

    if not BASELINE_PATH.exists():
        _annotate("error", f"Baseline file missing: {BASELINE_PATH}")
        return 1
    baseline = json.loads(BASELINE_PATH.read_text())

    failures: list[str] = []
    improvements: list[str] = []

    for tool in ("mypy", "ruff_check"):
        regressions, improved = compare_counts(
            tool, current[tool], baseline.get(tool, {})
        )
        failures.extend(regressions)
        if improved:
            improvements.append(tool)

    base_fmt = set(baseline.get("ruff_format_would_reformat", []))
    cur_fmt = set(current["ruff_format_would_reformat"])
    for path in sorted(cur_fmt - base_fmt):
        failures.append(
            f"ruff format: {path} is not formatted — run `pixi run -e dev lint`"
        )
    if cur_fmt < base_fmt and not (cur_fmt - base_fmt):
        improvements.append("ruff_format")

    if failures:
        for msg in failures:
            _annotate("error", msg)
        print(
            f"\n{len(failures)} regression(s) vs {BASELINE_PATH.name}. "
            "Fix the new findings; do not raise the baseline without "
            "reviewer sign-off."
        )
        return 1

    print("No regressions vs baseline.")
    if improvements:
        _annotate(
            "notice",
            "Findings decreased for: "
            + ", ".join(improvements)
            + ". Please ratchet the baseline down with "
            "`pixi run -e dev update-baselines` and commit the result.",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
