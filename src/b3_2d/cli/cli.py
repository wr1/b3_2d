"""CLI entry point using treeparse."""

import logging
from rich.logging import RichHandler
from treeparse import cli, group

import pyvista as pv
import json
from pathlib import Path

from .commands.mesh import mesh_cmd
from .commands.plot import plot_cmd
from .commands.anba_all import anba_all_cmd
from .commands.anba_single import anba_single_cmd
from .commands.anba_plot import anba_plot_cmd
from .commands.span_plot import span_cmd
from .commands.post import post_cmd

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(show_time=False)],
)

logger = logging.getLogger(__name__)

anba_group = group(
    name="anba",
    help="Run ANBA4 on meshes.",
    commands=[anba_all_cmd, anba_single_cmd, anba_plot_cmd],
)

app = cli(
    name="b3_2d",
    help="2D meshing for b3m using cgfoil.",
    commands=[mesh_cmd, plot_cmd, span_cmd, post_cmd],
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
