"""Tests for B32dStep and BOM functions."""

from unittest.mock import patch, MagicMock
import numpy as np

try:
    from b3_2d.state.b3_2d_mesh import B32dStep
    from b3_2d.core.bom import compute_bom
except ImportError:
    B32dStep = None
    compute_bom = None


def test_b32d_step():
    """Test B32dStep execution."""
    if B32dStep is None:
        import pytest

        pytest.skip("Statesman not available")
    with patch(
        "b3_2d.state.b3_2d_mesh.B32dStep.load_config",
        return_value={"workdir": "work", "num_processes": 4},
    ), patch("pyvista.read") as mock_pv_read, patch(
        "b3_2d.state.b3_2d_mesh.process_vtp_multi_section"
    ) as mock_process, patch("pathlib.Path") as mock_path_class:
        mock_pv_read.return_value = MagicMock()
        mock_pv_read.return_value.cell_data = {"section_id": [1, 2]}
        mock_config_path = MagicMock()
        mock_config_dir = MagicMock()
        mock_workdir = MagicMock()
        mock_vtp_dir = MagicMock()
        mock_vtp_file = MagicMock()
        mock_output_dir = MagicMock()
        mock_path_class.return_value = mock_config_path
        mock_config_path.parent = mock_config_dir
        mock_config_dir.__truediv__ = MagicMock(return_value=mock_workdir)
        mock_workdir.__truediv__ = MagicMock(
            side_effect=[mock_vtp_dir, mock_output_dir]
        )
        mock_vtp_dir.__truediv__ = MagicMock(return_value=mock_vtp_file)
        mock_output_dir.mkdir = MagicMock()
        mock_vtp_file.__str__ = MagicMock(return_value="vtp_path")
        mock_output_dir.__str__ = MagicMock(return_value="output_path")

        step = B32dStep("config.yaml")
        step.logger = MagicMock()

        step._execute()

        assert mock_process.called
        step.logger.info.assert_called()


def test_compute_bom():
    """Test BOM computation."""
    if compute_bom is None:
        import pytest

        pytest.skip("Function not available")
    mock_mesh = MagicMock()
    mock_mesh.cell_data = {
        "Area": np.array([1.0, 2.0, 3.0]),
        "material_id": np.array([1, 1, 2]),
    }
    result = compute_bom(mock_mesh)
    expected = {"total_area": 6.0, "areas_per_material": {1: 3.0, 2: 3.0}}
    assert result == expected


def test_compute_bom_missing_data():
    """Test BOM computation with missing data."""
    if compute_bom is None:
        import pytest

        pytest.skip("Function not available")
    mock_mesh = MagicMock()
    mock_mesh.cell_data = {}
    result = compute_bom(mock_mesh)
    assert result is None


def test_compute_bom_with_mass():
    """Test BOM computation with mass using matdb."""
    if compute_bom is None:
        import pytest

        pytest.skip("Function not available")
    mock_mesh = MagicMock()
    mock_mesh.cell_data = {
        "Area": np.array([1.0, 2.0, 3.0]),
        "material_id": np.array([1, 1, 2]),
    }
    matdb = {"carbon": {"id": 1, "rho": 1600.0}, "glass": {"id": 2, "rho": 2000.0}}
    result = compute_bom(mock_mesh, matdb)
    expected = {
        "total_area": 6.0,
        "areas_per_material": {1: 3.0, 2: 3.0},
        "total_mass": 3.0 * 1600.0 + 3.0 * 2000.0,  # 4800.0 + 6000.0 = 10800.0
        "masses_per_material": {1: 3.0 * 1600.0, 2: 3.0 * 2000.0},  # {4800.0, 6000.0}
    }
    assert result == expected


def test_compute_bom_with_partial_matdb():
    """Test BOM computation with partial matdb (missing density for some materials)."""
    if compute_bom is None:
        import pytest

        pytest.skip("Function not available")
    mock_mesh = MagicMock()
    mock_mesh.cell_data = {
        "Area": np.array([1.0, 2.0, 3.0]),
        "material_id": np.array([1, 1, 2]),
    }
    matdb = {"carbon": {"id": 1, "rho": 1600.0}}  # Missing entry for id 2
    with patch("b3_2d.core.bom.logger") as mock_logger:
        result = compute_bom(mock_mesh, matdb)
        expected = {
            "total_area": 6.0,
            "areas_per_material": {1: 3.0, 2: 3.0},
            "total_mass": 3.0 * 1600.0,  # Only for material 1
            "masses_per_material": {1: 3.0 * 1600.0},  # Only material 1
        }
        assert result == expected
        mock_logger.warning.assert_called_once_with(
            "Material ID 2 not found in matdb. Available material IDs: [1]. Skipping mass calculation."
        )


def test_compute_bom_with_wrong_key_matdb():
    """Test BOM computation with matdb having wrong key type."""
    if compute_bom is None:
        import pytest

        pytest.skip("Function not available")
    mock_mesh = MagicMock()
    mock_mesh.cell_data = {
        "Area": np.array([1.0, 2.0, 3.0]),
        "material_id": np.array([1, 1, 2]),
    }
    matdb = {"carbon": {"id": "1", "rho": 1600.0}}  # String id instead of int
    with patch("b3_2d.core.bom.logger") as mock_logger:
        result = compute_bom(mock_mesh, matdb)
        expected = {
            "total_area": 6.0,
            "areas_per_material": {1: 3.0, 2: 3.0},
            "total_mass": 0.0,  # No masses calculated
            "masses_per_material": {},
        }
        assert result == expected
        # Should call warning for both 1 and 2
        assert mock_logger.warning.call_count == 2
        mock_logger.warning.assert_any_call(
            "Material ID 1 not found in matdb. Available material IDs: ['1']. Skipping mass calculation."
        )
        mock_logger.warning.assert_any_call(
            "Material ID 2 not found in matdb. Available material IDs: ['1']. Skipping mass calculation."
        )
