"""Pytest configuration and fixtures for b3_2d."""

import pytest
import numpy as np
import pyvista as pv


@pytest.fixture
def simple_mock_mesh():
    """Create a simple valid PyVista mesh fixture."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.5, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.5, 1.0, 0.0],
            [2.0, 2.0, 0.0],
        ]
    )

    # Connectivity in VTK legacy format: npts followed by point ids per cell
    cells = np.array([3, 0, 1, 2, 3, 1, 3, 4, 3, 3, 2, 4, 3, 2, 5, 4], dtype=pv.ID_TYPE)

    celltypes = np.full(4, pv.CellType.TRIANGLE.value, dtype=np.uint8)
    # Order: cells, celltypes, points
    mesh = pv.UnstructuredGrid(cells, celltypes, points)
    return mesh


@pytest.fixture
def mock_mesh_with_data(simple_mock_mesh):
    """Mock mesh with all required cell data."""
    mesh = simple_mock_mesh.copy(deep=True)
    n_cells = mesh.n_cells

    # Add cell data
    mesh.cell_data["section_id"] = np.full(n_cells, 1, dtype=float)
    mesh.cell_data["panel_id"] = np.array([-1.0, 0.0, 0.0, -2.0])
    mesh.cell_data["twist"] = np.full(n_cells, 5.0)
    mesh.cell_data["dx"] = np.full(n_cells, 0.1)
    mesh.cell_data["dy"] = np.full(n_cells, 0.05)

    # Add ply data: thicknesses as point_data, materials as cell_data
    for i in range(3):
        mesh.point_data[f"ply_{i}_thickness"] = np.full(
            mesh.n_points, 0.002 + i * 0.001, dtype=float
        )
        mesh.cell_data[f"ply_{i}_material"] = np.full(n_cells, i + 1, dtype=int)

    return mesh


@pytest.fixture
def mock_vtp_mesh():
    """Mock VTP mesh with multiple sections."""
    sections = []
    for sec_id in [1, 2]:
        points = np.random.rand(12, 3)
        cells = np.tile([3, 0, 1, 2], 4)  # 4 triangles
        sec_mesh = pv.PolyData(points, cells)
        sec_mesh.cell_data["section_id"] = np.full(4, sec_id)
        sections.append(sec_mesh)

    mesh = pv.merge(sections)
    mesh.rotate_z(-90)
    mesh.rotate_x(180)
    return mesh
