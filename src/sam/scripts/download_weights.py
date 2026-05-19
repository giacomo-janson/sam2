import os
import textwrap
import sys
import logging
import argparse
from pathlib import Path

from sam.weights import download_weights_main, DEFAULT_DOWNLOAD_PATH, DEFAULT_FLAVOR, DOWNLOAD_PATH_ENV_VAR_NAME, WEIGHTS_FLAVORS, FLAVOR_OPTIONS

_LOGGER = logging.getLogger(__name__)

help_string = textwrap.dedent(
f"""Download weights for sam2.

Download the weights files to a folder of your choosing, otherwise
will fall back to the defaults.


Order of precedence for choosing folder:


1. Environment variable: {DOWNLOAD_PATH_ENV_VAR_NAME}
2. `--directory` CLI argument
3. Default path: {DEFAULT_DOWNLOAD_PATH}


Each flavor will go under a `weights/<flavor>` directory


Choose from flavors: {FLAVOR_OPTIONS}


If you specify 'all' all the flavors will be downloaded.


By default download will be skipped if the flavor directory is present
already.

"""
)

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
        "--flavor",
        type=str,
        default=DEFAULT_FLAVOR,
        help=f"Flavor of weights to download. (Default='{DEFAULT_FLAVOR}')"
    )
    parser.add_argument(
        '--directory',
        type=str,
        default=DEFAULT_DOWNLOAD_PATH,
        help=f"Alternative directory to download the weights to. (Default='{DEFAULT_DOWNLOAD_PATH}')"
    )

    parser.add_argument(
        '-f',
        '--force',
        action='store_true',
        default=False,
        help="If True will force redownload of weights to the folder."
    )
    
    args = parser.parse_args()

    download_weights_main(args.directory, args.flavor, force=args.force)
    
if __name__ == "__main__":

    main()
