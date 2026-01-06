"""Tests for statesman functionality."""

from unittest.mock import patch, MagicMock

try:
    from b3_2d.state.b3_2d_mesh import B32dStep
except ImportError:
    B32dStep = None


def test_b32d_step():
    """Test B32dStep execution."""
    if B32dStep is None:
        import pytest
        pytest.skip("Statesman not available")
    with (
        patch("b3_2d.state.b3_2d_mesh.B32dStep.load_config", return_value={"workdir": "work", "num_processes": 4}),
        patch("pyvista.read") as mock_pv_read,
        patch("b3_2d.state.b3_2d_mesh.process_vtp_multi_section") as mock_process,
        patch("pathlib.Path") as mock_path_class,
    ):
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
