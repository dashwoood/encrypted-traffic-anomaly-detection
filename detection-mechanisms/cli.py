#!/usr/bin/env python3
"""
CLI for anomaly detection: track traffic, run as daemon, or list models.
Usage:
  detect track --flows data/logs/flows.csv --model baseline
  detect daemon --flows data/logs/flows.csv --model isolation_forest
  detect list-models
"""
import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

_DETECT_ROOT = Path(__file__).resolve().parent
if str(_DETECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DETECT_ROOT))

from flow_reader import read_flows, read_flows_incremental
from models import get, list_models

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"
BOLD = "\033[1m"

LOG = logging.getLogger("detect")
_RUNNING = True


def _fmt(flow: dict) -> str:
    ts = flow.get("timestamp", "?")
    client = flow.get("client_ip", "?")
    method = flow.get("method", "?")
    path = flow.get("path", "?")
    code = flow.get("response_code", "?")
    return f"{ts} | {method} {path} | {client} | {code}"


def _init_detector(model_name: str, model_path: Optional[Path], flows_list: list[dict]):
    """Create detector: load from path or fit on flows."""
    model_cls = get(model_name)
    detector = model_cls()
    if model_path and model_path.exists():
        try:
            detector.load(model_path)
            LOG.info("Loaded model from %s", model_path)
            return detector
        except Exception as e:
            LOG.warning("Could not load model from %s: %s; fitting from flows", model_path, e)
    if flows_list:
        detector.fit(flows_list)
        if model_path and hasattr(detector, "save"):
            try:
                detector.save(model_path)
                LOG.info("Saved model to %s", model_path)
            except NotImplementedError:
                pass
    return detector


def cmd_track(flows_path: Path, model_name: str, follow: bool, model_dir: Optional[Path] = None) -> int:
    """Track flows and output with anomaly highlighting (batch mode)."""
    flows_list = list(read_flows(flows_path))
    if not flows_list:
        print("No flows to process.", file=sys.stderr)
        return 1

    model_path = (Path(model_dir) / f"{model_name}.joblib") if model_dir else None
    detector = _init_detector(model_name, model_path, flows_list)
    labels = detector.predict(flows_list)

    for flow, is_anomaly in zip(flows_list, labels):
        line = _fmt(flow)
        if is_anomaly:
            print(f"{RED}{BOLD}ANOMALY{RESET} {line}")
        else:
            print(f"{GREEN}normal{RESET}   {line}")

    if follow:
        print("(--follow not implemented; use 'daemon' for continuous monitoring)", file=sys.stderr)
    return 0


def cmd_daemon(
    flows_path: Path,
    model_name: str,
    interval: float,
    model_dir: Optional[Path],
    output: Optional[Path],
    min_fit_flows: int,
) -> int:
    """Run detection as daemon: poll flows file and report anomalies."""
    global _RUNNING

    def _shutdown(*_):
        global _RUNNING
        _RUNNING = False

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    model_path = Path(model_dir) / f"{model_name}.joblib" if model_dir else None
    out_file = None
    if output and str(output) != "-":
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        out_file = open(output, "a", encoding="utf-8")

    def _log(msg: dict, anomaly: bool = False):
        """Structured logging of events as JSON plus human-readable console output."""
        import json as _json

        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "anomaly": bool(anomaly),
            "flow": msg,
        }
        line = _json.dumps(payload, sort_keys=True)

        if out_file:
            out_file.write(line + "\n")
            out_file.flush()
        else:
            text = _fmt(msg)
            prefix = f"{RED}{BOLD}ANOMALY{RESET} " if anomaly else f"{GREEN}normal{RESET}   "
            print(prefix + text, flush=True)

    pos = 0
    detector = None
    all_flows: list[dict] = []

    LOG.info("Daemon started: flows=%s model=%s interval=%.1fs", flows_path, model_name, interval)

    while _RUNNING:
        try:
            flows, new_pos = read_flows_incremental(flows_path, pos)
            if new_pos != pos:
                pos = new_pos

            if flows:
                if detector is None:
                    all_flows.extend(flows)
                    if len(all_flows) >= min_fit_flows:
                        detector = _init_detector(model_name, model_path, all_flows)
                    else:
                        LOG.debug("Collecting flows for fit (%d/%d)", len(all_flows), min_fit_flows)
                        time.sleep(interval)
                        continue

                if detector is not None:
                    labels = detector.predict(flows)
                    for flow, is_anomaly in zip(flows, labels):
                        _log(flow, anomaly=is_anomaly)

        except Exception as e:
            LOG.exception("Error processing flows: %s", e)

        time.sleep(interval)

    if out_file:
        out_file.close()
    LOG.info("Daemon stopped")
    return 0


def cmd_list_models() -> int:
    """List available detection models."""
    models = list_models()
    print("Available models:")
    for m in models:
        print(f"  {m}")
    return 0


def _compute_metrics(y_true: list[int], y_pred: list[bool]) -> tuple[float, float, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and not p)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def cmd_train(
    data_dir: Path,
    model_name: str,
    model_dir: Optional[Path],
    skip_download: bool,
    max_cicids: Optional[int],
    max_unsw: Optional[int],
) -> int:
    """Prepare train/val from benchmark datasets, fit model, save weights, evaluate on val."""
    try:
        from datasets.prepare import prepare_datasets
    except ImportError:
        print("datasets module not found. Run from detection-mechanisms or pip install -e .", file=sys.stderr)
        return 1

    data_dir = Path(data_dir)
    model_dir = Path(model_dir) if model_dir else data_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_name}.joblib"

    LOG.info("Preparing datasets (data_dir=%s, download=%s)", data_dir, not skip_download)
    train_path, val_path = prepare_datasets(
        data_dir,
        download_benchmarks=not skip_download,
        max_cicids_rows=max_cicids,
        max_unsw_rows=max_unsw,
    )
    train_flows = list(read_flows(train_path))
    if not train_flows:
        print("No training flows after prepare.", file=sys.stderr)
        return 1

    def _gt(f: dict) -> int:
        v = f.get("ground_truth", f.get("anomaly", 0))
        try:
            return 1 if int(float(v or 0)) else 0
        except (ValueError, TypeError):
            return 0

    model_cls = get(model_name)
    detector = model_cls()
    detector.fit(train_flows)
    if hasattr(detector, "save"):
        try:
            detector.save(model_path)
            LOG.info("Saved model to %s", model_path)
            print(f"Model saved to {model_path}")
        except NotImplementedError:
            print("This model does not support save.", file=sys.stderr)
    else:
        print("Model has no save method.", file=sys.stderr)

    if val_path.exists():
        val_flows = list(read_flows(val_path))
        if val_flows:
            y_true = [_gt(f) for f in val_flows]
            y_pred = detector.predict(val_flows)
            prec, rec, f1 = _compute_metrics(y_true, y_pred)
            print(f"Val: P={prec:.4f} R={rec:.4f} F1={f1:.4f} (n={len(val_flows)}, anomalies={sum(y_true)})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Anomaly detection CLI for flow traffic",
        prog="detect",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    track_p = sub.add_parser("track", help="Track flows and highlight anomalies (batch)")
    track_p.add_argument("--flows", type=Path, required=True, help="Path to flows.csv")
    track_p.add_argument("--model", "-m", default="baseline", help="Model name (default: baseline)")
    track_p.add_argument("--model-dir", type=Path, default=None, help="Load saved model from dir (skip fit)")
    track_p.add_argument("--follow", "-f", action="store_true", help="(Unused; use daemon)")

    train_p = sub.add_parser("train", help="Train on benchmark data (CICIDS2017, UNSW-NB15) and save model weights")
    train_p.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    train_p.add_argument("--model", "-m", default="isolation_forest", help="Model name")
    train_p.add_argument("--model-dir", type=Path, default=None, help="Where to save model (default: data/models)")
    train_p.add_argument("--skip-download", action="store_true", help="Do not download from Kaggle")
    train_p.add_argument("--max-cicids", type=int, default=None, help="Max CICIDS2017 rows (default: all)")
    train_p.add_argument("--max-unsw", type=int, default=None, help="Max UNSW-NB15 rows (default: all)")

    daemon_p = sub.add_parser("daemon", help="Run as daemon: watch flows file and report anomalies")
    daemon_p.add_argument("--flows", type=Path, required=True, help="Path to flows.csv")
    daemon_p.add_argument("--model", "-m", default="isolation_forest", help="Model name")
    daemon_p.add_argument(
        "--interval",
        "-i",
        type=float,
        default=5.0,
        help="Poll interval in seconds (default: 5)",
    )
    daemon_p.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directory to save/load model (persistence across restarts)",
    )
    daemon_p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file for anomalies (default: stdout)",
    )
    daemon_p.add_argument(
        "--min-fit-flows",
        type=int,
        default=10,
        help="Min flows to collect before fitting (default: 10)",
    )
    daemon_p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)",
    )

    sub.add_parser("list-models", help="List available models")

    args = parser.parse_args()

    if args.cmd == "track":
        return cmd_track(
            args.flows,
            getattr(args, "model", "baseline"),
            getattr(args, "follow", False),
            getattr(args, "model_dir", None),
        )
    if args.cmd == "train":
        return cmd_train(
            getattr(args, "data_dir", Path("data")),
            getattr(args, "model", "isolation_forest"),
            getattr(args, "model_dir", None),
            getattr(args, "skip_download", False),
            getattr(args, "max_cicids", None),
            getattr(args, "max_unsw", None),
        )
    elif args.cmd == "daemon":
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return cmd_daemon(
            args.flows,
            args.model,
            args.interval,
            args.model_dir,
            args.output,
            args.min_fit_flows,
        )
    elif args.cmd == "list-models":
        return cmd_list_models()

    return 0


if __name__ == "__main__":
    sys.exit(main())
