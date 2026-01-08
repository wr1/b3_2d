"""ANBA single command for CLI."""

import os
import shutil
import subprocess
from treeparse import command, option


def anba_single_command(
    json_file: str,
    anba_env: str = "anba4-env",
    verbose: bool = False,
) -> None:
    """Run ANBA4 on a single anba.json file."""
    import logging

    logger = logging.getLogger(__name__)
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    conda_path = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda_path:
        logger.error("Conda not found")
        return
    result = subprocess.run([conda_path, "env", "list"], capture_output=True, text=True)
    if anba_env not in result.stdout:
        logger.error(f"Conda env {anba_env} not found")
        return
    conda_command = [conda_path, "run", "-n", anba_env, "anba4-run", "-i", json_file]
    env_vars = {
        **os.environ.copy(),
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "-1",
    }
    result = subprocess.run(
        conda_command,
        capture_output=True,
        text=True,
        env=env_vars,
    )
    if result.stdout:
        logger.info(result.stdout.strip())
    if result.returncode != 0:
        logger.error(f"ANBA4 failed for {json_file}: {result.stderr}")
    else:
        logger.info(f"ANBA4 completed for {json_file}")


anba_single_cmd = command(
    name="single",
    help="Run ANBA4 on a single anba.json file.",
    callback=anba_single_command,
    options=[
        option(
            flags=["--json-file", "-j"],
            arg_type=str,
            required=True,
            help="Input anba.json file",
        ),
        option(
            flags=["--anba-env", "-e"],
            arg_type=str,
            default="anba4-env",
            help="Conda environment for ANBA4",
        ),
        option(
            flags=["--verbose", "-V"],
            arg_type=bool,
            default=False,
            help="Verbose output",
        ),
    ],
)
