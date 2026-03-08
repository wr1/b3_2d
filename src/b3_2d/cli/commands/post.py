"""Post command for CLI."""

import json
from treeparse import command, option
from ...core.bom_plotting import plot_bom_spanwise
from ...core.span_plotting import plot_span_anba


def post_command(
    output_dir: str,
    matdb_file: str = None,
    verbose: bool = False,
) -> None:
    """Run postprocessing plots for BOM and ANBA."""
    import logging

    logging.getLogger(__name__)
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    matdb = {}
    if matdb_file:
        with open(matdb_file, "r") as f:
            matdb = json.load(f)
    plot_bom_spanwise(output_dir, f"{output_dir}/bom_spanwise.png", matdb)
    plot_span_anba(output_dir, f"{output_dir}/anba_spanwise.png")


post_cmd = command(
    name="post",
    help="Run postprocessing plots for BOM and ANBA.",
    callback=post_command,
    options=[
        option(
            flags=["--output-dir", "-o"],
            arg_type=str,
            required=True,
            help="Output directory containing section_*/",
        ),
        option(
            flags=["--matdb-file", "-m"],
            arg_type=str,
            help="Material database JSON file",
        ),
        option(
            flags=["--verbose", "-V"],
            arg_type=bool,
            default=False,
            help="Verbose output",
        ),
    ],
)
