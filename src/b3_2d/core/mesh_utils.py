"""Mesh utility functions for b3_2d."""

import pyvista as pv


def sort_points_by_y(mesh: pv.PolyData) -> pv.PolyData:
    """Sort points by y-coordinate and update connectivity."""
    mesh = mesh.copy()
    n = mesh.n_points
    sorted_indices = sorted(range(n), key=lambda i: mesh.points[i, 1])
    mesh.points = mesh.points[sorted_indices]
    cells = mesh.cells.copy()
    old_to_new = {old: new for new, old in enumerate(sorted_indices)}
    cells[1::4] = [old_to_new[cells[i]] for i in range(1, len(cells), 4)]
    cells[2::4] = [old_to_new[cells[i]] for i in range(2, len(cells), 4)]
    cells[3::4] = [old_to_new[cells[i]] for i in range(3, len(cells), 4)]
    mesh.cells = cells
    for key in mesh.point_data.keys():
        mesh.point_data[key] = mesh.point_data[key][sorted_indices]
    return mesh


def validate_points(points_2d: list) -> bool:
    """Validate points_2d is a list of 2D points."""
    if not isinstance(points_2d, list):
        return False
    for p in points_2d:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            return False
        if not all(isinstance(c, (int, float)) for c in p):
            return False
    return True


def bb_size(mesh: pv.PolyData) -> float:
    """Compute bounding box size."""
    bounds = mesh.bounds
    return ((bounds[1] - bounds[0]) ** 2 + (bounds[3] - bounds[2]) ** 2) ** 0.5
