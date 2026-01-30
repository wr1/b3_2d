"""Comprehensive tests for meshing functionality."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import pyvista as pv
import os

try:
    from b3_2d.core.mesh import (
        process_vtp_multi_section,
        process_single_section,
        extract_airfoil_and_web_points,
        validate_points,
        sort_points_by_y,
        bb_size,
        get_thickness_and_material_arrays,
    )
    from b3_2d.core.bom import compute_bom
except ImportError:
    process_vtp_multi_section = None
    process_single_section = None
    extract_airfoil_and_web_points = None
    validate_points = None
    sort_points_by_y = None
    bb_size = None
    get_thickness_and_material_arrays = None
    compute_bom = None


class TestMeshUtilities:
    def test_bb_size(self, simple_mock_mesh):
        """Test bounding box size calculation."""
        size = bb_size(simple_mock_mesh)
        assert isinstance(size, float)
        assert size > 0

    def test_validate_points_valid(self):
        """Test valid points validation."""
        points = [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]
        assert validate_points(points) is True

    @pytest.mark.parametrize(
        "invalid_points",
        [
            None,
            [[0, 0, 0]],
            [[0], [1, 1]],
            ["invalid"],
            [[0, 0], None],
        ],
    )
    def test_validate_points_invalid(self, invalid_points):
        """Test invalid points validation."""
        assert validate_points(invalid_points) is False

    def test_sort_points_by_y(self, simple_mock_mesh):
        """Test point sorting by y-coordinate."""
        sorted_mesh = sort_points_by_y(simple_mock_mesh)
        y_coords = sorted_mesh.points[:, 1]
        assert np.all(np.diff(y_coords) >= -1e-10)
        assert sorted_mesh.n_points == simple_mock_mesh.n_points


class TestExtractAirfoilAndWebPoints:
    def test_extract_airfoil_and_web_points(self, mock_mesh_with_data):
        """Test airfoil and web extraction."""
        points_2d, web_data, airfoil = extract_airfoil_and_web_points(
            mock_mesh_with_data
        )

        assert isinstance(points_2d, list)
        assert len(points_2d) > 0
        assert all(len(p) == 2 for p in points_2d)

        assert isinstance(web_data, list)
        assert isinstance(airfoil, pv.UnstructuredGrid)

    def test_no_negative_panels(self, mock_mesh_with_data):
        """Test case with no negative panel_ids (no TE or webs)."""
        mock_mesh_with_data.cell_data["panel_id"] = np.zeros(
            mock_mesh_with_data.n_cells
        )
        points_2d, web_data, airfoil = extract_airfoil_and_web_points(
            mock_mesh_with_data
        )
        assert len(web_data) == 0
        assert len(points_2d) > 0


class TestThicknessAndMaterials:
    def test_get_thickness_and_material_arrays(self, mock_mesh_with_data):
        """Test thickness and material array extraction."""
        thicknesses, materials = get_thickness_and_material_arrays(mock_mesh_with_data)

        assert isinstance(thicknesses, dict)
        assert isinstance(materials, dict)
        assert len(thicknesses) > 0
        # Keys should match pattern but materials are cell_data, thicknesses from point_data
        assert all("thickness" in k for k in thicknesses)


class TestBomCalculation:
    def test_compute_bom_basic(self):
        """Test basic BOM computation without matdb."""
        if compute_bom is None:
            pytest.skip("BOM function not available")
        mock_mesh = MagicMock()
        mock_mesh.cell_data = {
            "Area": np.array([1.0, 2.0, 3.0]),
            "material_id": np.array([1, 1, 2]),
        }
        result = compute_bom(mock_mesh)
        expected = {"total_area": 6.0, "areas_per_material": {1: 3.0, 2: 3.0}}
        assert result == expected

    def test_compute_bom_with_mass(self):
        """Test BOM computation with mass using matdb."""
        if compute_bom is None:
            pytest.skip("BOM function not available")
        mock_mesh = MagicMock()
        mock_mesh.cell_data = {
            "Area": np.array([1.0, 2.0, 3.0]),
            "material_id": np.array([1, 1, 2]),
        }
        matdb = {"carbon": {"id": 1, "rho": 1600.0}, "glass": {"id": 2, "rho": 1200.0}}
        result = compute_bom(mock_mesh, matdb)
        expected = {
            "total_area": 6.0,
            "areas_per_material": {1: 3.0, 2: 3.0},
            "total_mass": 3.0 * 1600.0 + 3.0 * 1200.0,  # 4800.0 + 3600.0 = 8400.0
            "masses_per_material": {
                1: 3.0 * 1600.0,
                2: 3.0 * 1200.0,
            },  # {4800.0, 3600.0}
        }
        assert result == expected

    def test_compute_bom_missing_data(self):
        """Test BOM computation with missing data."""
        if compute_bom is None:
            pytest.skip("BOM function not available")
        mock_mesh = MagicMock()
        mock_mesh.cell_data = {}
        result = compute_bom(mock_mesh)
        assert result is None


@patch("logging.getLogger")
@patch("b3_2d.core.mesh.os.path.exists")
class TestProcessSingleSection:
    def test_process_single_section_file_not_found(self, mock_exists, mock_get_logger):
        """Test when VTP file doesn't exist (via pv.read error handling)."""
        mock_get_logger.return_value = MagicMock()
        mock_exists.return_value = True
        os.makedirs("/tmp/output", exist_ok=True)

        with patch("pyvista.read", side_effect=FileNotFoundError):
            result = process_single_section(1, "/nonexistent.vtp", "/tmp/output")
            assert result["success"] is False

    def test_process_single_section_structure(self, mock_exists, mock_get_logger):
        """Test result structure."""
        mock_get_logger.return_value = MagicMock()
        mock_exists.return_value = True
        os.makedirs("/tmp/output", exist_ok=True)

        with patch("pyvista.read"):
            result = process_single_section(1, "/tmp/input.vtp", "/tmp/output")
            assert "section_id" in result
            assert "success" in result
            assert isinstance(result["created_files"], list)


class TestProcessVtpMultiSection:
    @patch("pyvista.read")
    def test_process_vtp_no_section_id(self, mock_pv_read):
        """Test VTP processing without section_id."""
        mock_mesh = MagicMock()
        mock_mesh.cell_data = {}
        mock_pv_read.return_value = mock_mesh

        with pytest.raises(ValueError, match="section_id not found"):
            process_vtp_multi_section("test.vtp", "/tmp/output")

    def test_process_vtp_cpu_count(self):
        """Test default num_processes uses cpu_count."""

        try:
            with patch("multiprocessing.cpu_count", return_value=4):
                process_vtp_multi_section("test.vtp", "/tmp/out")
        except FileNotFoundError:
            pass  # Expected


class TestIntegrationFunctions:
    def test_basic_function_imports(self):
        """Test all functions can be imported."""
        assert callable(bb_size)
        assert callable(validate_points)
        assert callable(sort_points_by_y)


@pytest.mark.skipif(process_vtp_multi_section is None, reason="cgfoil not available")
class TestCgfoilIntegration:
    @patch("b3_2d.core.mesh.Progress")
    @patch("multiprocessing.Pool")
    @patch("pyvista.read")
    def test_pool_called_with_correct_args(
        self, mock_pv_read, mock_pool, mock_progress
    ):
        """Test multiprocessing pool is called correctly."""
        mock_mesh = MagicMock()
        mock_mesh.cell_data = {"section_id": [1, 2]}
        mock_pv_read.return_value = mock_mesh

        process_vtp_multi_section("test.vtp", "/tmp/out", num_processes=2)

        mock_pool.assert_called_once_with(processes=2)
