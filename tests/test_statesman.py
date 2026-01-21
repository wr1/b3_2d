"""Tests for statesman functionality."""

from unittest.mock import patch, MagicMock
import numpy as np

try:
    from b3_2d.state.b3_2d_mesh import B32dStep, compute_bom
    from b3_2d.state.b3_2d_anba import B32dAnbaStep
except ImportError:
    B32dStep = None
    compute_bom = None
    B32dAnbaStep = None


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


@patch("b3_2d.state.b3_2d_anba.open")
@patch("b3_2d.state.b3_2d_anba.subprocess.run")
@patch("b3_2d.state.b3_2d_anba.shutil.which")
@patch("b3_2d.state.b3_2d_anba.Path")
@patch(
    "b3_2d.state.b3_2d_anba.B32dAnbaStep.load_config",
    return_value={"workdir": "work", "anba_env": "anba4-env"},
)
def test_b32d_anba_step(
    mock_load_config, mock_path_class, mock_which, mock_subprocess, mock_open
):
    """Test B32dAnbaStep execution."""
    if B32dAnbaStep is None:
        import pytest

        pytest.skip("Statesman not available")
    mock_subprocess.return_value = MagicMock(
        returncode=0, stdout="anba4-env", stderr=""
    )
    mock_which.return_value = "conda"
    mock_open.return_value.__enter__.return_value = MagicMock()
    mock_config_path = MagicMock()
    mock_config_path.parent = MagicMock()
    mock_config_path.parent.__truediv__ = MagicMock(return_value=MockPath())
    mock_path_class.return_value = mock_config_path

    step = B32dAnbaStep("config.yaml")
    step.logger = MagicMock()

    step._execute()

    assert mock_subprocess.call_count == 2  # env list and anba run
    step.logger.info.assert_called()


class MockPath:
    def __init__(self):
        self.path = "/work"

    def __str__(self):
        return self.path

    def __fspath__(self):
        return self.path

    def __truediv__(self, other):
        new_path = f"{self.path}/{other}"
        mock = MockPath()
        mock.path = new_path
        return mock

    def mkdir(self, **kwargs):
        pass

    def glob(self, pattern):
        # Mock glob to return some files
        mock_file = MockPath()
        mock_file.path = "/work/b3_2d/section_1/anba.json"
        return [mock_file]
