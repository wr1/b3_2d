"""Tests for CLI functionality."""

import sys
import pytest
from unittest.mock import patch
from b3_2d.cli.cli import app


def test_cli_app():
    """Test that the CLI app is properly configured."""
    assert app.name == "b3_2d"
    assert len(app.commands) == 2  # mesh and plot commands


def test_mesh_command_help(capsys):
    """Test mesh command help output."""
    sys.argv = ["b3-2d", "mesh", "--help"]
    with pytest.raises(SystemExit):
        app.run()
    captured = capsys.readouterr()
    assert "Process VTP file for multi-section meshing" in captured.out


def test_plot_command_help(capsys):
    """Test plot command help output."""
    sys.argv = ["b3-2d", "plot", "--help"]
    with pytest.raises(SystemExit):
        app.run()
    captured = capsys.readouterr()
    assert "Plot a mesh" in captured.out


@patch("b3_2d.core.mesh.process_vtp_multi_section")
def test_mesh_command(mock_process):
    """Test mesh command execution."""
    from b3_2d.core import mesh  # Ensure module is loaded
    sys.argv = ["b3-2d", "mesh", "--vtp-file", "test.vtp", "--output-dir", "out", "--num-processes", "2"]
    app.run()
    mock_process.assert_called_with("test.vtp", "out", 2)


@patch("b3_2d.core.plotting.plot_mesh")
@patch("pyvista.read")
def test_plot_command(mock_pv_read, mock_plot):
    """Test plot command execution."""
    mock_mesh = mock_pv_read.return_value
    sys.argv = ["b3-2d", "plot", "--mesh-file", "mesh.vtk", "--output-file", "plot.png", "--scalar", "material_id"]
    app.run()
    mock_pv_read.assert_called_with("mesh.vtk")
    mock_plot.assert_called_with(mock_mesh, scalar="material_id", output_file="plot.png")
