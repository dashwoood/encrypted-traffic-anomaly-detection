#!/usr/bin/env python3
"""
Stream benchmark flows (canonical schema) to flows.csv for detector evaluation.

Models are trained on UNSW-NB15 / CICIDS2017 normalized to the canonical schema.
This replayer streams rows from those datasets into flows.csv so the detector sees
**in-distribution** data. Use this for meaningful end-to-end validation.

Usage:
  python replay_flows.py \
    --source data/datasets/canonical_val.csv \
    --output data/logs/flows.csv \
    --delay-ms 100 \
    --max-rows 5000 \
    --loop

Env:
  SOURCE_PATH   : Path to canonical CSV (default: /data/datasets/canonical_val.csv)
  OUTPUT_PATH   : Path to writes flows.csv (default: /data/logs/flows.csv)
  DELAY_MS      : Delay between rows to simulate live traffic (0 = no delay)
  MAX_ROWS      : Max rows to emit per run (0 = all)
  LOOP          : If set, restart from beginning when done
"""
import argparse
import csv
import os
import signal
import sys
import time
from pathlib import Path

_RUNNING = True


def _shutdown(*_):
    global _RUNNING
    _RUNNING = False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream benchmark flows to flows.csv for detector evaluation"
    )
    parser.add_argument(
        "--source", "-s",
        type=Path,
        default=Path(os.environ.get("SOURCE_PATH", "data/datasets/canonical_val.csv")),
        help="Canonical CSV to read (same schema as UNSW/CICIDS normalized)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path(os.environ.get("OUTPUT_PATH", "data/logs/flows.csv")),
        help="Output flows.csv path (detector reads this)",
    )
    parser.add_argument(
        "--delay-ms",
        type=float,
        default=float(os.environ.get("DELAY_MS", "50")),
        help="Delay between rows in ms (0 = no delay, for batch mode)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=int(os.environ.get("MAX_ROWS", "0")),
        help="Max rows to emit per run (0 = all)",
    )
    parser.add_argument(
        "--loop", "-l",
        action="store_true",
        default=os.environ.get("LOOP", "").lower() in ("1", "true", "yes"),
        help="Loop: restart from beginning when done",
    )
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    if not source.exists():
        print(f"Source not found: {source}", file=sys.stderr)
        print("Run 'make prepare-data' or ensure canonical_val.csv exists.", file=sys.stderr)
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    delay_sec = args.delay_ms / 1000.0
    max_rows = args.max_rows if args.max_rows > 0 else None
    loop = args.loop

    print(
        f"Replayer: {source} -> {output} | delay={args.delay_ms}ms | max_rows={max_rows or 'all'} | loop={loop}"
    )

    total_emitted = 0
    first_run = True

    while _RUNNING:
        with open(source, newline="", encoding="utf-8") as fin:
            reader = csv.DictReader(fin)
            headers = reader.fieldnames
            if not headers:
                print("No header in source CSV.", file=sys.stderr)
                return 1

            write_mode = "w" if first_run else "a"
            first_run = False

            with open(output, write_mode, newline="", encoding="utf-8") as fout:
                writer = csv.DictWriter(fout, fieldnames=headers, extrasaction="ignore")
                if write_mode == "w":
                    writer.writeheader()

                emitted = 0
                for row in reader:
                    if not _RUNNING:
                        break
                    if max_rows and emitted >= max_rows:
                        break
                    writer.writerow(row)
                    emitted += 1
                    total_emitted += 1
                    if delay_sec > 0:
                        time.sleep(delay_sec)

                print(f"Emitted {emitted} rows (total {total_emitted})")

        if not loop:
            break
        print("Looping: restarting from source...")
        time.sleep(1)

    print(f"Done. Total rows written: {total_emitted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
