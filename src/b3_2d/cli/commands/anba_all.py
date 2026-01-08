"""ANBA all command for CLI."""

import os
import shutil
import subprocess
from pathlib import Path
from treeparse import command, option


def anba_all_command(
    output_dir: str,
    anba_env: str = "anba4-env",
    verbose: bool = False,
) -> None:
    """Run ANBA4 on all anba.json files in output_dir."""
    import logging

    logger = logging.getLogger(__name__)
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    output_path = Path(output_dir)
    anba_files = list(output_path.glob("section_*/anba.json"))
    if not anba_files:
        logger.warning("No anba.json files found")
        return
    conda_path = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda_path:
        logger.error("Conda not found")
        return
    result = subprocess.run([conda_path, "env", "list"], capture_output=True, text=True)
    if anba_env not in result.stdout:
        logger.error(f"Conda env {anba_env} not found")
        return
    conda_command = [conda_path, "run", "-n", anba_env, "anba4-run", "-i"]
    for anba_file in anba_files:
        conda_command.append(str(anba_file))
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
    if result.returncode != 0:
        logger.error(f"ANBA4 failed: {result.stderr}")
    else:
        logger.info("ANBA4 completed for all sections")


anba_all_cmd = command(
    name="all",
    help="Run ANBA4 on all section anba.json files.",
    callback=anba_all_command,
    options=[
        option(
            flags=["--output-dir", "-o"],
            arg_type=str,
            required=True,
            help="Output directory containing section_*/anba.json",
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
