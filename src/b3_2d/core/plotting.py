"""Plotting utilities for b3_2d."""

import pyvista as pv
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def plot_mesh(
    mesh,
    scalar: Optional[str] = None,
    output_file: str = "plot.png",
) -> None:
    """Plot the mesh with scalar coloring and save screenshot."""
    plotter = pv.Plotter(off_screen=True)
    if scalar and scalar in mesh.cell_data:
        plotter.add_mesh(mesh, scalars=scalar)
    else:
        plotter.add_mesh(mesh)
    plotter.view_xy()
    plotter.screenshot(output_file)
    logger.info(f"Plot saved to {output_file}")
