"""Plotting utilities for b3_2d."""

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
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


def plot_anba_results(
    mesh,
    data: dict,
    output_file: str = "anba_plot.png",
) -> None:
    """Plot ANBA4 results on the mesh using matplotlib."""
    # Extract centers and angle
    centers = {
        "mass_center": np.array(data["mass_center"]),
        "shear_center": np.array(data["shear_center"]),
        "elastic_center": np.array(data["tension_center"]),
    }
    principal_angle = data["principal_angle"]
    principal_angle_deg = np.degrees(principal_angle)

    # Plot mesh
    fig, ax = plt.subplots()
    x = mesh.points[:, 0]
    y = mesh.points[:, 1]
    triangles = mesh.cells.reshape(-1, 4)[:, 1:4]  # Assuming triangular cells
    if "material_id" in mesh.cell_data:
        scalars = mesh.cell_data["material_id"]
        ax.tripcolor(x, y, triangles, scalars, cmap="viridis")
    else:
        ax.triplot(x, y, triangles, color="black", linewidth=0.5)

    # Add centers
    for name, point in centers.items():
        ax.plot(point[0], point[1], "ro", markersize=10, label=name.replace("_", " ").title())
        ax.text(point[0] + 0.01, point[1] + 0.01, name.replace("_", " ").title(), fontsize=12)

    # Add principal angle line
    if "elastic_center" in centers:
        ec = centers["elastic_center"]
        length = 0.1  # Arbitrary length
        dx = length * np.cos(principal_angle)
        dy = length * np.sin(principal_angle)
        ax.plot([ec[0] - dx, ec[0] + dx], [ec[1] - dy, ec[1] + dy], "b-", label="Principal Angle")
        ax.text(ec[0] + 0.02, ec[1] + 0.02, f"Principal Angle: {principal_angle_deg:.2f}°", fontsize=12)

    ax.legend()
    ax.set_aspect("equal")
    plt.savefig(output_file)
    plt.close()
    logger.info(f"ANBA plot saved to {output_file}")
