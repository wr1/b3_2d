"""Tests for meshing functionality."""

import pytest
from unittest.mock import patch, MagicMock

try:
    from b3_2d.core.mesh import process_vtp_multi_section
except ImportError:
    process_vtp_multi_section = None


@patch("pyvista.read")
@patch("b3_2d.core.mesh.Progress")
@patch("b3_2d.core.mesh.multiprocessing.Pool")
def test_process_vtp_multi_section(mock_pool, mock_progress, mock_pv_read):
    """Test process_vtp_multi_section with mocked dependencies."""
    if process_vtp_multi_section is None:
        pytest.skip("cgfoil not available")
    # Mock the VTP mesh
    mock_mesh = MagicMock()
    mock_mesh.cell_data = {"section_id": [1, 1, 2, 2]}
    mock_mesh.rotate_z.return_value = mock_mesh
    mock_pv_read.return_value = mock_mesh

    # Mock progress
    mock_progress_instance = MagicMock()
    mock_progress_instance.add_task.return_value = "task"
    mock_progress.return_value.__enter__.return_value = mock_progress_instance
    mock_progress.return_value.__exit__ = MagicMock()

    # Mock pool
    mock_pool_instance = mock_pool.return_value

    # Call the function
    process_vtp_multi_section("dummy.vtp", "/tmp/output", num_processes=1)

    # Assertions
    mock_pv_read.assert_called_with("dummy.vtp")
    mock_pool.assert_called_with(processes=1)
