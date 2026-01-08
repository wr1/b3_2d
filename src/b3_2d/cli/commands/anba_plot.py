"""ANBA plot command for CLI."""

import json
from pathlib import Path
import pyvista as pv
from treeparse import command, option
from ...core.plotting import plot_section_anba


def anba_plot_command(
    json_file: str,
    output_file: str,
    verbose: bool = False,
) -> None:
    """Plot ANBA4 results for a single section."""
    import logging

    logger = logging.getLogger(__name__)
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    json_path = Path(json_file)
    section_dir = json_path.parent
    vtk_file = section_dir / "output.vtk"
    if not vtk_file.exists():
        logger.error(f"VTK file not found: {vtk_file}")
        return
    mesh = pv.read(str(vtk_file))
    with open(json_file, "r") as f:
        data = json.load(f)
    plot_section_anba(mesh, data, output_file)


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
