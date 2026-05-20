"""
Generate with SAM conformational ensemble for an input PDB file.
Notes:
    On the first time you use this script, weights for the aSAM models will be
    automatically downloaded to ~/.sam2/weights. Change the $SAM_WEIGHTS_PATH
    environmemtal variable to change the download path.
"""
import logging
from pathlib import Path
import os
from typing import Literal
import importlib.resources
import textwrap
import sys
import argparse
import time
import numpy as np
import mdtraj
from sam.model import AllAtomSAM
from sam.utils import read_cfg_file, print_msg, check_sam_weights
from sam.data.topology import get_seq_from_top
from sam.minimizer.runner import Minimizer
from sam.weights import has_flavor, download_weights_main, WEIGHTS_FLAVORS, DEFAULT_FLAVOR, DEFAULT_DOWNLOAD_PATH, DOWNLOAD_PATH_ENV_VAR_NAME, WEIGHTS_SUBDIR_NAME
from sam.config import MODEL_CONFIGS, DEFAULT_MODEL_CONFIG, MODEL_FILENAMES

_LOGGER = logging.getLogger(__name__)

help_string = textwrap.dedent(
    f"""Generate SAM conformational ensemble.

    The `--data-dir` flag can be overridden with the
    `{DOWNLOAD_PATH_ENV_VAR_NAME}` environment variable.

    For model configurations you can choose from the built-in defaults
    or provide your own .json or .yaml file with the configuration
    parameters, see the `--config` argument. For built-in model
    configs choose from {MODEL_CONFIGS}. Default is
    {DEFAULT_MODEL_CONFIG}.
    
    """
)

def generate_ensemble(
        init: Path,
        out_path: Path,
        *,
        data_dir: Path | None = None,
        config_path: Path | None = None,
        config: str | None = "atlas",
        no_download: bool = False,
        out_fmt: Literal["xtc", "dcd"] = "dcd",
        n_samples: int = 250,
        n_steps: int = 100,
        batch_size: int = 8,
        device: Literal["cuda", "cpu"] = "cuda",
        temperature: float | None = None,
        quiet: bool = False,
        no_minimize: bool = False,
        keep_no_min: bool = False,
        ca: bool = False,
        track_time: bool = True,
) -> None:

    #---------------
    # Check input. -
    #---------------
    init = init.expanduser().resolve()
    out_path = out_path.expanduser().resolve()

    if config_path is not None and config is not None:
        raise ValueError("Cannot specify both config_path and config")

    elif config_path is not None:
        model_config_path = config_path.expanduser().resolve()

    elif config is not None and config not in MODEL_CONFIGS:
        raise ValueError(
            f"Non-path config requested '{config}' but is not a valid built-in. Choose one of: {MODEL_CONFIGS}"
        )
    elif config is not None:
        # resolve built-in config
        configs_dir = importlib.resources.files("sam.config")
        model_config_path = configs_dir / MODEL_FILENAMES[config]

    model_cfg = read_cfg_file(str(model_config_path))
    
    timing = {"all": time.time(), "sample": None}

    if not os.path.isfile(init):
        raise FileNotFoundError(init)


    # check if the weights for the flavor are available and download
    # if requested

    if model_cfg["weights"]["version"] not in WEIGHTS_FLAVORS:
        raise ValueError(
            f"Invalid 'weights.version' field in model config '{model_cfg["weights"]["version"]}', choose from {WEIGHTS_FLAVORS}")

    else:
        flavor = model_cfg["weights"]["version"]

    if data_dir is not None:
        _data_dir = data_dir.expanduser().resolve()

    else:
        # resolve the directory to use, either the environment
        # variable or the default
        if env_val := os.getenv(DOWNLOAD_PATH_ENV_VAR_NAME, None) is not None:
            _data_dir = Path(env_val).expanduser().resolve()
        else:
            _data_dir = Path(DEFAULT_DOWNLOAD_PATH).expanduser().resolve()

    if has_flavor(_data_dir, flavor):
        _LOGGER.info(f"Weights for flavor {flavor} already present in data dir.")
        weights_dir = _data_dir / WEIGHTS_SUBDIR_NAME / flavor
    elif no_download:
        raise RuntimeError(
            "Weights could not be found, and the --no_download flag was given."
        )
    else:
        _LOGGER.warning(f"Weights for flavor {flavor} not present, downloading.")
        weights_dir = download_weights_main(
            _data_dir,
            flavor,
        )

    _LOGGER.info("Loading initial conformation.")
    tem_traj = mdtraj.load(str(init))
    tbm_data = {"xyz": tem_traj.xyz}
    if model_cfg["generative_stack"]["data_type"] == "aa_protein":
        tbm_data["topology"] = tem_traj.topology
    seq = get_seq_from_top(tbm_data["topology"])

    #-----------
    # Run SAM. -
    #-----------

    # Initialize the SAM model.
    if model_cfg["generative_stack"]["data_type"] == "cg_protein":
        raise NotImplementedError()
    elif model_cfg["generative_stack"]["data_type"] == "aa_protein":
        model_cls = AllAtomSAM
    else:
        raise KeyError(model_cfg["generative_stack"]["data_type"])

    model = model_cls(
        config_fp=str(model_config_path),
        weights_dir=weights_dir,
        device=device,
        verbose=not quiet
    )

    conditions = {}
    if temperature is not None:
        conditions["temperature"] = temperature
    sample_args = {}
    
    # Generate ensemble.
    _LOGGER.info("Sampling model")
    timing["sample"] = time.time()
    out = model.sample(
        seq=seq,
        n_samples=n_samples,
        n_steps=n_steps,
        batch_size_eps=batch_size,
        batch_size_dec=batch_size,
        tbm_data=tbm_data,
        return_enc=False,
        sample_args=sample_args,
        conditions=conditions,
        use_cache=True
    )
    timing["sample"] = time.time() - timing["sample"]
    _LOGGER.info(f"Finished sampling model. Took: {timing['sample']} s")

    # Save the output data.
    _LOGGER.info("Saving output data.")
    save = model.save(
        out=out,
        out_path=out_path,
        out_fmt=out_fmt,
        save_ca=ca
    )
    # tem_traj.save(f"{out_path}.template.pdb")
    _LOGGER.info("Finished saving output data.")

    #-------------------------------------
    # Energy minimize the conformations. -
    #-------------------------------------

    if no_minimize or model_cfg["minimization"]["protocol"] is None:
        _LOGGER.warning("Skipping minimization of output structures.")
    else:
        _LOGGER.warning("Running minimization of output structures.")
        timing["min"] = time.time()
        min_obj = Minimizer(
            name="sam_ensemble",
            top_fp=save["aa_top"],
            ens_fp=save["aa_traj"],
            protocol=model_cfg["minimization"]["protocol"]
        )
        min_traj = min_obj.run(device=device, verbose=not quiet)
        _LOGGER.info("Finished minimization of output structures.")
        if keep_no_min:
            min_out_str = ".min"
        else:
            min_out_str = ""
        min_traj_path = f"{out_path}{min_out_str}.traj.{out_fmt}"
        _LOGGER.info(
            f"- Saving a trajectory file to: {min_traj_path}",
        )
        min_traj.save(min_traj_path)
        min_top_path = f"{out_path}{min_out_str}.top.pdb"
        _LOGGER.infox(
            f"- Saving a topology PDB file to: {min_top_path}",
        )
        min_traj[0].save(min_top_path)
        timing["min"] = time.time() - timing["min"]

    #------------
    # Complete. -
    #------------

    timing["all"] = time.time() - timing["all"]
    if track_time:
        _LOGGER.info("Saving timings.")
        with open(f"{out_path}.time.txt", "w") as o_fh:
            for stage in timing:
                o_fh.write(f"{stage}: {timing[stage]}\n")

    _LOGGER.info("Finished with structure generation.")

def main():
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        description=help_string,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '-i',
        '--init',
        type=str,
        required=True,
        help='Input PDB file with the initial structure.',
    )
    parser.add_argument(
        '-o',
        '--out_path',
        type=str,
        required=True,
        help='Output path. File extensions for different file types will be'
        ' automatically added.'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=False,
        default=None,
        help=(
            f"Model config. String for a built-in config."
            f"Choose from {MODEL_CONFIGS} (Default={DEFAULT_MODEL_CONFIG})"
            "Incompatible with `--config_path`."
        )
    )
    parser.add_argument(
        '-c', '--config_path',
        type=str,
        required=False,
        default=None,
        help="Path to a config to override using built-in default. Incompatible with `--config`."
    )
    parser.add_argument(
        '-u',
        '--out_fmt',
        type=str,
        default='dcd',
        choices=['dcd', 'xtc'],
        help='Output format for the file storing xyz coordinates.'
             ' (default: dcd)'
    )
    parser.add_argument(
        '-n',
        '--n_samples',
        type=int,
        default=250,
        help='Number of samples to generate. (default: 250)',
    )
    parser.add_argument(
        '-t',
        '--n_steps',
        type=int,
        default=100,
        help='Number of diffusion steps. (min=1, max=1000) (default: 100)'
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for sampling. (default: 8)',
    )
    parser.add_argument('-T', '--temperature', type=float,
        help='temperature (optional, only for temperature-based models)')


    parser.add_argument(
        '--data_dir',
        default=None,
        help=(
            "Directory with model data, i.e. model weights."
            "See description for default resolution process if not provided."
        ),
    )

    parser.add_argument(
        '--no_download',
        action='store_true',
        default=False,
        help=f"If given will not attempt to download the weights files, if not present."
    )
    parser.add_argument('-d', '--device', type=str, default='cuda',
        choices=['cuda', 'cpu'], help='PyTorch device. (default: cuda)')
    parser.add_argument('-q', '--quiet', action='store_true',
        help='Quiet mode, will not print any output.')
    parser.add_argument('--no_minimize', action='store_true',
        help='Do not perform energy minimization.')
    parser.add_argument('--keep_no_min', action='store_true',
        help='If performing energy minimization, save also a trajectory file'
             ' for the non-minimized ensemble.')
    parser.add_argument('--ca', action='store_true',
        help='Save an additional Ca-only trajectory.')
    parser.add_argument('--time', action='store_true',
        help='Save an output file with the wall clock time of sampling.')
    args = parser.parse_args()

    generate_ensemble(
        config=args.config,
        config_path=args.config_path,
        init=Path(args.init).expanduser().resolve(),
        out_path=Path(args.out_path).expanduser().resolve(),
        data_dir=Path(args.data_dir).expanduser().resolve(),
        no_download=args.no_download,
        out_fmt=args.out_fmt,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        device=args.device,
        temperature=args.temperature,
        quiet=args.quiet,
        no_minimize=args.no_minimize,
        keep_no_min=args.keep_no_min,
        ca=args.ca,
        track_time=args.time,
    )



if __name__ == "__main__":
    main()
