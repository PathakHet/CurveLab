"""Command-line entry point: ``curvelab run``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from curvelab.config import Config
from curvelab.pipeline import run, write_outputs
from curvelab.report import build_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="curvelab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="run the full pipeline and write the report")
    run_cmd.add_argument("--config", type=Path, default=None, help="YAML config override")
    run_cmd.add_argument("--market-dir", type=str, default=None, help="market data directory")
    run_cmd.add_argument("--journal", action="append", default=None,
                         help="journal workbook (repeatable)")
    run_cmd.add_argument("--root", type=Path, default=Path.cwd(), help="project root")
    run_cmd.add_argument("--no-report", action="store_true", help="write CSVs only")

    args = parser.parse_args(argv)
    config = Config.load(args.config)
    if args.market_dir:
        config.data.market_dir = args.market_dir
    if args.journal:
        config.data.journals = tuple(args.journal)

    result = run(config, args.root)
    results_dir = write_outputs(result, config, args.root)
    print(f"\nwrote {len(result.tables)} tables to {results_dir}\n")
    for finding in result.findings.values():
        print(f"  - {finding}\n")
    if not args.no_report:
        print(f"report: {build_report(result, config, args.root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
