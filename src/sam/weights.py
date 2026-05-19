"""Management of weights including downloads."""
import zipfile
import os
import shutil
import requests
import logging
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

DEFAULT_DOWNLOAD_PATH = "~/.sam2"

DOWNLOAD_PATH_ENV_VAR_NAME = "SAM_WEIGHTS_PATH"

GITHUB_RELEASES_URL = "https://github.com/giacomo-janson/sam2/releases/download/data-1.0"

WEIGHTS_FLAVORS = frozenset({
    "atlas_1.0",
    "mdcath_1.0",
})

FLAVOR_OPTIONS = ["all"] + list(sorted(WEIGHTS_FLAVORS))

DEFAULT_FLAVOR = "atlas_1.0"

WEIGHTS_SUBDIR_NAME = "weights"

def resolve_weights_url(flavor: str) -> (str, str):

    if flavor not in WEIGHTS_FLAVORS:
        raise ValueError(f"Invalid weights flavor '{flavor}', choose from {WEIGHTS_FLAVORS}")

    filename = f"{flavor}.zip"
    url = GITHUB_RELEASES_URL + "/" + filename

    return url, filename

def download_weights(
        target_path: Path,
        flavor: str,
        force: bool = False,
) -> Path:

    _target_dir = target_path.expanduser().resolve()

    weights_dir = _target_dir / WEIGHTS_SUBDIR_NAME

    flavor_dir = weights_dir / flavor
    if flavor_dir.exists():
        _LOGGER.info(f"The weights flavor directory already exists: {flavor_dir}")

        if not force:
            _LOGGER.warn("force writing is not enabled, skipping download")
            return flavor_dir
        else:
            _LOGGER.warn("Forcing redownload of weights after deletion of files.")
            shutil.rmtree(flavor_dir)

    else:
        _LOGGER.info("No weights flavor directory found, creating.")

    flavor_dir.mkdir(parents=True, exist_ok=True)

    url, filename = resolve_weights_url(flavor)

    _LOGGER.info(f"Initiating download of weights.")

    # download
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(
            f"Unable to download file (status code {response.status_code})."
        )

    archive_path = flavor_dir / filename
    _LOGGER.info(f"HTTP request successful, saving archive file: {archive_path}")
    with open(archive_path, 'wb') as wf:
        wf.write(response.content)

    _LOGGER.info("Saved archive file")


    _LOGGER.info(f"Unpacking archive file to: {flavor_dir}")
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(weights_dir)

    _LOGGER.info("Unpacked archive, removing archive file.")

    os.remove(archive_path)
    
    return flavor_dir


def download_weights_main(directory: Path, flavor: str, force: bool = False) -> Path:

    if (env_value := os.getenv(DOWNLOAD_PATH_ENV_VAR_NAME, None)) is not None:
        _LOGGER.info(f"Value of environment variable '{DOWNLOAD_PATH_ENV_VAR_NAME}' set and is being used for download directory.")
        target_directory = Path(env_value).expanduser().resolve()

    else:
        _LOGGER.info("Using directory from CLI argument for download directory.")
        target_directory = Path(directory).expanduser().resolve()

    _LOGGER.info(f"Full download path resolved to: {target_directory}")

    if flavor == "all":
        _LOGGER.info("Requested download of all weights flavors")
        flavors = WEIGHTS_FLAVORS
    else:
        flavors = {flavor}

    for flavor in flavors:
        _LOGGER.info(f"Downloading weights for flavor: {flavor}")
        download_weights(target_directory, flavor, force)
        _LOGGER.info(f"Downloaded weights for flavor: {flavor}")

    return target_directory
