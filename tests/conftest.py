"""Pytest configuration and fixtures for b3_2d."""

import pytest
import numpy as np
import pyvista as pv


@pytest.fixture
def simple_mock_mesh():
    """Create a simple valid PyVista mesh fixture."""
    # Create points
    n_points = 9
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.5, 1.0, 0.0],
        [1.0, 2.0, 0.0],
        [3.0, 0.0, 0.0],
        [2.5, 1.0, 0.0],
        [2.0, 2.0, 0.0]
    ])
    
    # Create triangle connectivity (point ids only, no npts prefix)
    cells = np.array([0,1,2, 1,3,4, 3,2,4, 2,5,4])
    
    n_cells = 4
    celltypes = np.full(n_cells, 5, dtype=np.uint8)  # VTK_TRIANGLE = 5
    mesh = pv.UnstructuredGrid(celltypes, cells, points)
    return mesh


@pytest.fixture
def mock_mesh_with_data(simple_mock_mesh):
    """Mock mesh with all required cell data."""
    mesh = simple_mock_mesh.copy()
    n_cells = mesh.n_cells
    
    # Add cell data
    mesh.cell_data['section_id'] = np.full(n_cells, 1, dtype=float)
    mesh.cell_data['panel_id'] = np.array([-1.0, 0.0, 0.0, -2.0])
    mesh.cell_data['twist'] = np.full(n_cells, 5.0)
    mesh.cell_data['dx'] = np.full(n_cells, 0.1)
    mesh.cell_data['dy'] = np.full(n_cells, 0.05)
    
    # Add ply data - use point data for cell_data_to_point_data
    point_data = mesh.cell_data_to_point_data()
    for i in range(3):
        point_data[f'ply_{i}_thickness'] = np.full(mesh.n_points, 0.002 + i*0.001, dtype=float)
        mesh.cell_data[f'ply_{i}_material'] = np.full(n_cells, i + 1, dtype=int)
    
    return mesh


@pytest.fixture
def mock_vtp_mesh():
    """Mock VTP mesh with multiple sections."""
    sections = []
    for sec_id in [1, 2]:
        points = np.random.rand(12, 3)
        cells = np.tile([3, 0, 1, 2], 4)  # 4 triangles
        sec_mesh = pv.PolyData(points, cells)
        sec_mesh.cell_data['section_id'] = np.full(4, sec_id)
        sections.append(sec_mesh)
    
    mesh = pv.merge(sections)
    mesh.rotate_z(-90)
    mesh.rotate_x(180)
    return mesh
