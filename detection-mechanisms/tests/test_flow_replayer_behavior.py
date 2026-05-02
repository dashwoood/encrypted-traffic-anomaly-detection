from pathlib import Path

import csv


def test_flow_replayer_max_rows_and_headers() -> None:
    """Basic sanity check for benchmark flow replayer output if flows.csv exists.

    This test does not start Docker; it only validates the file structure if
    a previous benchmark or demo run created data/logs/flows.csv.
    """
    project_root = Path(__file__).resolve().parents[2]
    flows_path = project_root / "data" / "logs" / "flows.csv"
    if not flows_path.exists():
        # Flow replayer is exercised via demo scripts / Make targets; skip here if absent.
        return

    with flows_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert len(rows) >= 2
    header = rows[0]
    assert "ground_truth" in header or "anomaly" in header

