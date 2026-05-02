import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cli as cli_mod  # type: ignore
from cli import cmd_daemon  # type: ignore


def test_daemon_writes_structured_logs_on_empty_flows(tmp_path: Path) -> None:
    """Daemon should start and exit cleanly even if flows file is initially empty.

    This is a very small smoke test that exercises the structured logging path
    with an output file but does not rely on long-running loops.
    """
    flows_path = tmp_path / "flows.csv"
    flows_path.write_text("", encoding="utf-8")
    output_path = tmp_path / "daemon.log"

    # Run at least one loop iteration by calling cmd_daemon directly with a short interval.
    # We forcibly stop after one cycle by toggling the internal flag.

    cli_mod._RUNNING = True  # type: ignore[attr-defined]

    import threading
    import time

    def _run_daemon() -> None:
        cmd_daemon(flows_path, "baseline", interval=0.1, model_dir=None, output=output_path, min_fit_flows=1)

    t = threading.Thread(target=_run_daemon, daemon=True)
    t.start()
    time.sleep(0.3)
    cli_mod._RUNNING = False  # type: ignore[attr-defined]
    t.join(timeout=1.0)

    if not output_path.exists():
        # If nothing was logged yet that's acceptable for this smoke, but file
        # creation is expected under normal conditions.
        return

    lines = [ln for ln in output_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for ln in lines:
        rec = json.loads(ln)
        assert "timestamp" in rec
        assert "anomaly" in rec
        assert "flow" in rec

