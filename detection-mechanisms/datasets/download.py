"""
Download benchmark datasets from Kaggle (not stored in git).

Setup:
  1. pip install -e ".[benchmark]"   # or: pip install kaggle
  2. Create ~/.kaggle/kaggle.json with your API credentials:
     From https://www.kaggle.com/settings -> Create New Token you get a JSON
     with "username" and "key". Save it as ~/.kaggle/kaggle.json (chmod 600).
     Or use: KAGGLE_USERNAME=... KAGGLE_KEY=... python scripts/setup_kaggle.py
"""
import logging
import subprocess
import zipfile
from pathlib import Path

LOG = logging.getLogger(__name__)

# Primary Kaggle dataset slugs; alternates tried if primary fails (different uploads, same data)
CICIDS2017_SLUGS = ["dhoogla/cicids2017", "sweety18/cicids2017-full-dataset"]
UNSW_NB15_SLUGS = ["mrwellsdavid/unsw-nb15", "ucimachinelearning/unsw-nb15-dataset"]


def _run_kaggle_download(slug: str, dest_dir: Path) -> bool:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-p", str(dest_dir), slug],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        LOG.warning("Kaggle download failed for %s: %s", slug, e)
        return False


def _unzip_first(directory: Path) -> None:
    for z in directory.glob("*.zip"):
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(directory)
        LOG.info("Extracted %s", z.name)


def download_cicids2017(dest_dir: Path) -> Path:
    """Download CICIDS2017 to dest_dir/cicids2017/. Tries alternate slugs if first fails."""
    folder = Path(dest_dir) / "cicids2017"
    folder.mkdir(parents=True, exist_ok=True)
    for slug in CICIDS2017_SLUGS:
        if _run_kaggle_download(slug, folder):
            _unzip_first(folder)
            return folder
    LOG.warning("All CICIDS2017 slugs failed; folder %s may be empty", folder)
    return folder


def download_unsw_nb15(dest_dir: Path) -> Path:
    """Download UNSW-NB15 to dest_dir/unsw_nb15/. Tries alternate slugs if first fails."""
    folder = Path(dest_dir) / "unsw_nb15"
    folder.mkdir(parents=True, exist_ok=True)
    for slug in UNSW_NB15_SLUGS:
        if _run_kaggle_download(slug, folder):
            _unzip_first(folder)
            return folder
    LOG.warning("All UNSW-NB15 slugs failed; folder %s may be empty", folder)
    return folder


def download_all(dest_dir: Path) -> dict[str, Path]:
    """Download both benchmarks. Returns dict dataset_name -> folder path."""
    dest_dir = Path(dest_dir)
    out = {}
    out["cicids2017"] = download_cicids2017(dest_dir)
    out["unsw_nb15"] = download_unsw_nb15(dest_dir)
    return out
