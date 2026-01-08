"""Plotting utilities for b3_2d."""

import os
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def bb_size(mesh: pv.PolyData) -> float:
    """Compute bounding box size."""
    bounds = mesh.bounds
    return ((bounds[1] - bounds[0]) ** 2 + (bounds[3] - bounds[2]) ** 2) ** 0.5


def plot_mesh(
    mesh: pv.PolyData,
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


def plot_section_anba(
    mesh: pv.PolyData,
    data: dict,
    output_file: str = "anba_plot.png",
    log_file: Optional[str] = None,
    lock=None,
) -> None:
    """Plot ANBA4 results on the mesh using matplotlib."""
    required_keys = ["mass_center", "shear_center", "tension_center", "principal_angle"]
    if not all(k in data for k in required_keys):
        if log_file and lock:
            with lock:
                with open(log_file, "a") as f:
                    f.write(
                        f"ANBA results not found, skipping plot for {output_file}\n"
                    )
        return
    centers = {
        "mass_center": np.array(data["mass_center"]),
        "shear_center": np.array(data["shear_center"]),
        "elastic_center": np.array(data["tension_center"]),
    }
    principal_angle = data["principal_angle"]
    principal_angle_deg = np.degrees(principal_angle)
    fig, ax = plt.subplots(figsize=(12.8, 6.6))
    x = mesh.points[:, 0]
    y = mesh.points[:, 1]
    triangles = mesh.cells.reshape(-1, 4)[:, 1:4]
    if "material_id" in mesh.cell_data:
        scalars = mesh.cell_data["material_id"]
        ax.tripcolor(x, y, triangles, scalars, cmap="viridis")
    else:
        ax.triplot(x, y, triangles, color="black", linewidth=0.5)
    markers = ["ro", "bs", "g^"]
    marker_iter = iter(markers)
    for name, point in centers.items():
        marker = next(marker_iter)
        ax.plot(
            point[0],
            point[1],
            marker,
            markersize=10,
            label=name.replace("_", " ").title(),
        )
    if "elastic_center" in centers:
        ec = centers["elastic_center"]
        bounds = mesh.bounds
        x_min, x_max = bounds[0], bounds[1]
        y_min, y_max = bounds[2], bounds[3]
        length = max(x_max - x_min, y_max - y_min) * 0.4
        dx = length * np.cos(principal_angle)
        dy = length * np.sin(principal_angle)
        ax.plot(
            [ec[0] - dx, ec[0] + dx],
            [ec[1] - dy, ec[1] + dy],
            "b-",
            label="Principal Angle",
        )
        ax.text(
            ec[0] + 0.02,
            ec[1] + 0.02,
            f"Principal Angle: {principal_angle_deg:.2f}°",
            fontsize=12,
        )
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=400)
    plt.close()
    if log_file and lock:
        with lock:
            with open(log_file, "a") as f:
                f.write(f"ANBA plot saved to {output_file}\n")


def plot_section_debug(
    section_mesh: pv.PolyData, output_dir: str, section_id: int
) -> None:
    """Plot original and transformed section for debugging."""
    # Plot original section
    plot_mesh(
        section_mesh,
        scalar="panel_id",
        output_file=os.path.join(output_dir, f"section_{section_id}_original.png"),
    )
    # Apply transformations
    min_panel_id = section_mesh.cell_data["panel_id"].min()
    section = section_mesh.threshold(
        value=(0, section_mesh.cell_data["panel_id"].max()), scalars="panel_id"
    )
    twist = np.unique(section.cell_data["twist"])[0]
    dx = np.unique(section.cell_data["dx"])[0]
    dy = np.unique(section.cell_data["dy"])[0]
    section.points[:, 0] -= dx
    section.points[:, 1] -= dy
    section.rotate_z(-twist)
    te = section_mesh.threshold(value=(min_panel_id, min_panel_id), scalars="panel_id")
    section_bb = bb_size(section)
    te_bb = bb_size(te)
    if te_bb > 0.03 * section_bb:
        transformed = pv.merge([section, te])
    else:
        transformed = section
    # Plot transformed section
    plot_mesh(
        transformed,
        scalar="panel_id",
        output_file=os.path.join(output_dir, f"section_{section_id}_transformed.png"),
    )
