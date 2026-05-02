#!/usr/bin/env python3
"""
Write ~/.kaggle/kaggle.json from environment variables.

Usage:
  export KAGGLE_USERNAME=your_username
  export KAGGLE_KEY=your_key_from_kaggle_settings
  python scripts/setup_kaggle.py

Or from project root:
  KAGGLE_USERNAME=... KAGGLE_KEY=... python detection-mechanisms/scripts/setup_kaggle.py
"""
import os
from pathlib import Path

def main() -> int:
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not username or not key:
        print("Set KAGGLE_USERNAME and KAGGLE_KEY environment variables.", file=__import__("sys").stderr)
        print("Get your key from https://www.kaggle.com/settings -> Create New Token.", file=__import__("sys").stderr)
        return 1
    import json
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(mode=0o700, exist_ok=True)
    path = kaggle_dir / "kaggle.json"
    content = json.dumps({"username": username, "key": key})
    path.write_text(content)
    path.chmod(0o600)
    print(f"Wrote {path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
