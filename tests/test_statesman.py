"""Tests for statesman functionality."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


def test_b32d_step():
    """Test B32dStep execution."""
    from b3_2d.core import mesh  # Ensure module is loaded
    with patch("b3_2d.core.mesh.process_vtp_multi_section") as mock_process, \
         patch("pathlib.Path") as mock_path_class:
        mock_config_path = MagicMock()
        mock_config_dir = MagicMock()
        mock_workdir = MagicMock()
        mock_vtp_dir = MagicMock()
        mock_vtp_file = MagicMock()
        mock_output_dir = MagicMock()
        mock_path_class.return_value = mock_config_path
        mock_config_path.parent = mock_config_dir
        mock_config_dir.__truediv__ = MagicMock(return_value=mock_workdir)
        mock_workdir.__truediv__ = MagicMock(side_effect=[mock_vtp_dir, mock_output_dir])
        mock_vtp_dir.__truediv__ = MagicMock(return_value=mock_vtp_file)
        mock_output_dir.mkdir = MagicMock()
        mock_vtp_file.__str__ = MagicMock(return_value="vtp_path")
        mock_output_dir.__str__ = MagicMock(return_value="output_path")

        from b3_2d.statesman.b3_2d_step import B32dStep
        step = B32dStep()
        step.config_path = "config.yaml"
        step.config = {"workdir": "work", "num_processes": 4}
        step.logger = MagicMock()

        step._execute()

        mock_output_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_process.assert_called_with("vtp_path", "output_path", 4)
        step.logger.info.assert_called()
