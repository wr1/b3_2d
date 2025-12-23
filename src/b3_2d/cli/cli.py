"""CLI entry point using treeparse."""

import logging
from rich.logging import RichHandler
from treeparse import cli, command, option, group

import pyvista as pv
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(show_time=False)],
)

logger = logging.getLogger(__name__)


def mesh_command(
    vtp_file: str,
    output_dir: str,
    num_processes: int = None,
    verbose: bool = False,
) -> None:
    """Mesh sections from VTP file."""
    from ..core.mesh import process_vtp_multi_section

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    process_vtp_multi_section(vtp_file, output_dir, num_processes)


def plot_command(
    mesh_file: str,
    output_file: str,
    scalar: str = "material_id",
    verbose: bool = False,
) -> None:
    """Plot a mesh."""
    from ..core.plotting import plot_mesh

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    mesh = pv.read(mesh_file)
    plot_mesh(mesh, scalar=scalar, output_file=output_file)


def anba_all_command(
    output_dir: str,
    anba_env: str = "anba4-env",
    verbose: bool = False,
) -> None:
    """Run ANBA4 on all anba.json files in output_dir."""
    import os
    import shutil
    import subprocess
    from pathlib import Path

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
        logger.info(f"ANBA4 completed for all sections")


def anba_single_command(
    json_file: str,
    anba_env: str = "anba4-env",
    verbose: bool = False,
) -> None:
    """Run ANBA4 on a single anba.json file."""
    import os
    import shutil
    import subprocess

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


def anba_plot_command(
    json_file: str,
    output_file: str,
    verbose: bool = False,
) -> None:
    """Plot ANBA4 results for a single section."""
    from ..core.plotting import plot_anba_results

    json_path = Path(json_file)
    section_dir = json_path.parent
    vtk_file = section_dir / "output.vtk"
    if not vtk_file.exists():
        logger.error(f"VTK file not found: {vtk_file}")
        return
    mesh = pv.read(str(vtk_file))
    with open(json_file, "r") as f:
        data = json.load(f)
    plot_anba_results(mesh, data, output_file)


mesh_cmd = command(
    name="mesh",
    help="Process VTP file for multi-section meshing.",
    callback=mesh_command,
    options=[
        option(
            flags=["--vtp-file", "-v"],
            arg_type=str,
            required=True,
            help="Input VTP file",
        ),
        option(
            flags=["--output-dir", "-o"],
            arg_type=str,
            required=True,
            help="Output directory",
        ),
        option(
            flags=["--num-processes", "-n"],
            arg_type=int,
            help="Number of processes",
        ),
        option(
            flags=["--verbose", "-V"],
            arg_type=bool,
            default=False,
            help="Verbose output",
        ),
    ],
)

plot_cmd = command(
    name="plot",
    help="Plot a mesh.",
    callback=plot_command,
    options=[
        option(
            flags=["--mesh-file", "-m"],
            arg_type=str,
            required=True,
            help="Input mesh file",
        ),
        option(
            flags=["--output-file", "-o"],
            arg_type=str,
            required=True,
            help="Output plot file",
        ),
        option(
            flags=["--scalar", "-s"],
            arg_type=str,
            default="material_id",
            help="Scalar field to plot",
        ),
        option(
            flags=["--verbose", "-V"],
            arg_type=bool,
            default=False,
            help="Verbose output",
        ),
    ],
)

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

anba_plot_cmd = command(
    name="plot",
    help="Plot ANBA4 results for a section.",
    callback=anba_plot_command,
    options=[
        option(
            flags=["--json-file", "-j"],
            arg_type=str,
            required=True,
            help="Input anba_out.json file with results",
        ),
        option(
            flags=["--output-file", "-o"],
            arg_type=str,
            required=True,
            help="Output plot file",
        ),
        option(
            flags=["--verbose", "-V"],
            arg_type=bool,
            default=False,
            help="Verbose output",
        ),
    ],
)

anba_group = group(
    name="anba",
    help="Run ANBA4 on meshes.",
    commands=[anba_all_cmd, anba_single_cmd, anba_plot_cmd],
)

app = cli(
    name="b3_2d",
    help="2D meshing for b3m using cgfoil.",
    commands=[mesh_cmd, plot_cmd],
    subgroups=[anba_group],
    show_types=True,
    show_defaults=True,
    line_connect=True,
    theme="monochrome",
    max_width=120,
)


def main():
    app.run()


if __name__ == "__main__":
    main()
