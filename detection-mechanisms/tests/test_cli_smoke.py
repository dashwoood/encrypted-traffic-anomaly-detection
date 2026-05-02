import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cli  # type: ignore


def test_list_models_exits_success(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [sys.argv[0], "list-models"],
    )
    exit_code = cli.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Available models" in captured.out

