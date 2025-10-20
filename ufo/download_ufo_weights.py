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

GDRIVE_FILE_ID = "1eIAoCy-sV_9ueC9-KmQKDyc8nex2yWxL"
FILENAME = "ufo_weights.pth"
CHUNK_SIZE = 32768


def download_from_google_drive(file_id: str, destination: Path):
    """Download a large file from Google Drive handling confirmation token for virus scan."""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    _save_response_content(response, destination)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def _save_response_content(response, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive chunks
                f.write(chunk)


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
    repo_weights_dir = Path(__file__).resolve().parents[1] / "weights"
    repo_weights_dir.mkdir(parents=True, exist_ok=True)
    local_dest = repo_weights_dir / FILENAME

    if local_dest.exists():
        print(f"Weights already downloaded in repo at {local_dest}")
    else:
        print(f"Downloading weights to {local_dest} ...")
        download_from_google_drive(GDRIVE_FILE_ID, local_dest)
        print("Download complete.")

    site_pkg = get_installed_ufo_site_packages_path()
    if site_pkg is None:
        print("Installed ufo package not found; leaving weights in the repo.")
        return

    site_weights_dir = site_pkg / "weights"
    site_weights_dir.mkdir(parents=True, exist_ok=True)
    site_dest = site_weights_dir / FILENAME

    if site_dest.exists():
        print(f"Weights already present in installed package at {site_dest}")
    else:
        print(f"Copying weights to installed package at {site_dest} ...")
        shutil.copy2(local_dest, site_dest)
        print("Copy complete.")

    print("All done. Installed package weights path:", site_dest)


if __name__ == '__main__':
    main()
