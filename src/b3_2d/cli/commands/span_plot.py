"""Span plot command for CLI."""

from treeparse import command, option
from ...core.span_plotting import plot_span_anba


def span_plot_command(
    output_dir: str,
    output_file: str,
    verbose: bool = False,
) -> None:
    """Plot ANBA stiffnesses and masses along blade span."""
    import logging

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    plot_span_anba(output_dir, output_file)


span_cmd = command(
    name="span",
    help="Plot ANBA stiffnesses and masses along blade span.",
    callback=span_plot_command,
    options=[
        option(
            flags=["--output-dir", "-o"],
            arg_type=str,
            required=True,
            help="Output directory containing section_*/",
        ),
        option(
            flags=["--output-file", "-f"],
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
