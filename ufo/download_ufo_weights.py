"""Download UFO model weights from Google Drive and copy into installed ufo package weights/.

Usage: run this script from your environment where the `ufo` package is installed.
It will:
 - download the file from Google Drive (supports large files via confirmation token)
 - place the downloaded file into the local repo at UFO/weights/ (for repo users)
 - copy the file into the site-packages `ufo/weights/` directory so installed package can access it

This script avoids using gdown as an external dependency and uses requests + streaming.
"""
import os
import sys
import shutil
import requests
from pathlib import Path
import importlib
import subprocess

GDRIVE_FILE_ID = "1eIAoCy-sV_9ueC9-KmQKDyc8nex2yWxL"
FILENAME = "ufo_weights.pth"
CHUNK_SIZE = 32768


def download_from_google_drive(file_id: str, destination: Path):
    """Download the Google Drive file using gdown only (installing gdown if necessary)."""
    _try_gdown(file_id, destination)


def _try_gdown(file_id: str, destination: Path):
    """Download using gdown. Installs gdown if missing."""
    try:
        import gdown
    except Exception:
        print('gdown not installed; installing via pip...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gdown'])
        import gdown

    url = f'https://drive.google.com/uc?id={file_id}'
    out = str(destination)
    print(f'Running gdown to fetch {file_id} -> {out}')
    # Use gdown to download and overwrite if necessary
    gdown.download(url, out, quiet=False, fuzzy=True)


def get_installed_ufo_site_packages_path():
    """Return path to installed ufo package directory in site-packages."""
    try:
        import ufo
        pkg_dir = Path(ufo.__file__).parent
        return pkg_dir
    except Exception as e:
        print("Error: could not import installed 'ufo' package:", e)
        return None


def main():
    # Prefer downloading directly into the installed package weights folder (site-packages/ufo/weights)
    site_pkg = get_installed_ufo_site_packages_path()
    if site_pkg is not None:
        site_weights_dir = site_pkg / "weights"
        site_weights_dir.mkdir(parents=True, exist_ok=True)
        site_dest = site_weights_dir / FILENAME

        if site_dest.exists():
            print(f"Weights already present in installed package at {site_dest}")
            print("All done. Installed package weights path:", site_dest)
            return site_dest

        # Download directly to the installed package weights folder
        print(f"Downloading weights directly to installed package at {site_dest} ...")
        download_from_google_drive(GDRIVE_FILE_ID, site_dest)
        print("Download complete.")
        print("All done. Installed package weights path:", site_dest)
        return site_dest

    # Fallback: download to repo-level weights folder if installed package not available
    repo_weights_dir = Path(__file__).resolve().parents[1] / "weights"
    repo_weights_dir.mkdir(parents=True, exist_ok=True)
    local_dest = repo_weights_dir / FILENAME

    if local_dest.exists():
        print(f"Weights already downloaded in repo at {local_dest}")
        print("All done. Repo weights path:", local_dest)
        return local_dest
    else:
        print(f"Installed package not found; downloading weights to repo path {local_dest} ...")
        download_from_google_drive(GDRIVE_FILE_ID, local_dest)
        print("Download complete (repo copy).")
        print("All done. Repo weights path:", local_dest)
        return local_dest


if __name__ == '__main__':
    main()
