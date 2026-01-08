"""Plot command for CLI."""

import pyvista as pv
from treeparse import command, option
from ...core.plotting import plot_mesh


def plot_command(
    mesh_file: str,
    output_file: str,
    scalar: str = "material_id",
    verbose: bool = False,
) -> None:
    """Plot a mesh."""
    import logging

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    mesh = pv.read(mesh_file)
    plot_mesh(mesh, scalar=scalar, output_file=output_file)


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
