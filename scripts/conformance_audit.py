#!/usr/bin/env python
"""
Run the archive-wide conformance checks against the published 3DEP catalog.

Every site we process exercises a handful of workunits, so an assumption that
holds in Arizona and Nevada can still be wrong for a tenth of the archive.
WESM publishes its full attribute table as a few-MB CSV, so this runs offline
in seconds and turns "we will find out when it breaks in Alaska" into a
standing risk register.

Usage
-----
    pixi run -e dev python scripts/conformance_audit.py
    pixi run -e dev python scripts/conformance_audit.py --json report.json
    pixi run -e dev python scripts/conformance_audit.py --wesm local/WESM.csv

WESM grows monthly, so re-run periodically; `--fail-on blocker` makes it a
gate once the current blockers are cleared.
"""

import argparse
import json
import sys

import pandas as pd

from lidar_tools import conformance

WESM_CSV_URL = (
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/metadata/WESM.csv"
)

SEVERITIES = ("blocker", "unsupported", "degraded", "info")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wesm",
        default=WESM_CSV_URL,
        help="WESM attribute table (path or URL); default fetches from USGS",
    )
    parser.add_argument("--json", help="also write the report as JSON")
    parser.add_argument(
        "--fail-on",
        choices=SEVERITIES,
        help="exit non-zero when a finding of this severity or worse exists",
    )
    args = parser.parse_args(argv)

    print(f"Reading WESM attributes: {args.wesm}", file=sys.stderr)
    wesm = pd.read_csv(args.wesm, low_memory=False)
    result = conformance.audit_wesm(wesm)
    print(conformance.format_report(result))

    if args.json:
        payload = {
            "source": args.wesm,
            "n_workunits": result["n_workunits"],
            "findings": [f.as_dict() for f in result["findings"]],
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"\nJSON report: {args.json}")

    if args.fail_on:
        threshold = SEVERITIES.index(args.fail_on)
        worst = [
            f for f in result["findings"] if SEVERITIES.index(f.severity) <= threshold
        ]
        if worst:
            print(
                f"\n{len(worst)} finding(s) at or above '{args.fail_on}'",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
