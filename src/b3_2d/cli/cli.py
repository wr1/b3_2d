"""CLI entry point using treeparse."""

import logging
from rich.logging import RichHandler
from treeparse import cli, command, option

import pyvista as pv

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(show_time=False)],
)


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

app = cli(
    name="b3_2d",
    help="2D meshing for b3m using cgfoil.",
    commands=[mesh_cmd, plot_cmd],
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
