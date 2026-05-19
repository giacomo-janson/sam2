import os
from pathlib import Path
import json
import requests
import yaml
import zipfile


def read_cfg_file(cfg_fp: str):
    if cfg_fp.endswith(".json"):
        with open(cfg_fp, "r") as i_fh:
            model_cfg = json.load(i_fh)
    elif cfg_fp.endswith(".yaml"):
        with open(cfg_fp, 'r') as i_fh:
            model_cfg = yaml.safe_load(i_fh)
    else:
        raise TypeError(
            f"Invalid extension for configuration file: {cfg_fp}. Must be a"
            " json or yaml file.")

    # expand path and resolve if there is a path in the weights
    if "weights" in model_cfg and "path" in model_cfg["weights"] and model_cfg["weights"]["path"] is not None:
        model_cfg["weights"]["path"] = str(Path(model_cfg["weights"]["path"]).expanduser().resolve())
    
    return model_cfg

def print_msg(msg, verbose=True, tag="verbose"):
    if verbose:
        print(f"[{tag}]:", msg)

def to_json_compatible(report):
    _report = {}
    for k in report:
        if isinstance(report[k], (str, int, float)):
            _report[k] = report[k]
        else:
            try:
                _report[k] = float(report[k])
            except TypeError:
                _report[k] = "ERROR"
    return _report


github_releases_url = "https://github.com/giacomo-janson/sam2/releases/download/data-1.0"

def check_sam_weights(cfg_path: str, verbose: bool = True):
    model_cfg = read_cfg_file(cfg_path)
    if model_cfg["weights"]["path"] is None:
        download_path = download_sam_weights(model_cfg["weights"]["version"])
        model_cfg["weights"]["path"] = download_path
        with open(cfg_path, "w") as o_fh:
            yaml.dump(model_cfg, o_fh)
    else:
        if not os.path.isdir(str(Path(model_cfg["weights"]["path"]).expanduser())):
            raise FileNotFoundError(
                "Weights directory not found at: {}".format(
                    model_cfg["weights"]["path"]
                )
            )

def download_sam_weights(version: str, verbose: bool = True):
    filename = f"{version}.zip"
    url = github_releases_url + "/" + filename

    if os.getenv("SAM_WEIGHTS_PATH") is None:
        download_path = os.path.expanduser("~/.sam2/weights")
    else:
        download_path = os.getenv("SAM_WEIGHTS_PATH")
    os.makedirs(download_path, exist_ok=True)

    print_msg(
        f"# No aSAM weights were detected, beginning download now.",
        verbose=verbose, tag="download"
    )
    print_msg(
        f"- Downloading aSAM weights from: {url}",
        verbose=verbose, tag="download"
    )
    print_msg(
        f"- Weights will be saved at: {download_path}",
        verbose=verbose, tag="download"
    )
    res = requests.get(url)
    if res.status_code != 200:
        raise OSError(
            f"unable to download file (status code {res.status_code})."
        )
    save_path = os.path.join(download_path, filename)
    with open(save_path, "wb") as f:
        f.write(res.content)
    print_msg("- Download completed.", verbose=verbose, tag="download")
    print_msg("- Unzipping weight files.", verbose=verbose, tag="download")
    with zipfile.ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall(download_path)
    os.remove(save_path)
    print_msg(f"- Weights are now ready.", verbose=verbose, tag="download")
    print_msg(
        "- Will update the input .yaml configuration file.",
        verbose=verbose, tag="download"
    )
    return os.path.join(download_path, version)
