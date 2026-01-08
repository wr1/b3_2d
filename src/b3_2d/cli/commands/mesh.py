"""Mesh command for CLI."""

from treeparse import command, option
from ...core.mesh import process_vtp_multi_section


def mesh_command(
    vtp_file: str,
    output_dir: str,
    num_processes: int = None,
    verbose: bool = False,
) -> None:
    """Mesh sections from VTP file."""
    import logging

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    process_vtp_multi_section(vtp_file, output_dir, num_processes)


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
