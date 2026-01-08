"""Tests for CLI functionality."""

import sys
import pytest
from unittest.mock import patch, MagicMock
from b3_2d.cli.cli import app


def test_cli_app():
    """Test that the CLI app is properly configured."""
    assert app.name == "b3_2d"
    assert len(app.commands) == 3  # mesh, plot, span commands
    assert len(app.subgroups) == 1  # anba subgroup


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


def test_anba_all_command_help(capsys):
    """Test anba all command help output."""
    sys.argv = ["b3-2d", "anba", "all", "--help"]
    with pytest.raises(SystemExit):
        app.run()
    captured = capsys.readouterr()
    assert "Run ANBA4 on all section anba.json files" in captured.out


def test_anba_single_command_help(capsys):
    """Test anba single command help output."""
    sys.argv = ["b3-2d", "anba", "single", "--help"]
    with pytest.raises(SystemExit):
        app.run()
    captured = capsys.readouterr()
    assert "Run ANBA4 on a single anba.json file" in captured.out


def test_anba_plot_command_help(capsys):
    """Test anba plot command help output."""
    sys.argv = ["b3-2d", "anba", "plot", "--help"]
    with pytest.raises(SystemExit):
        app.run()
    captured = capsys.readouterr()
    assert "Plot ANBA4 results for a section" in captured.out


@patch("b3_2d.core.mesh.process_vtp_multi_section")
def test_mesh_command(mock_process):
    """Test mesh command execution."""
    sys.argv = [
        "b3-2d",
        "mesh",
        "--vtp-file",
        "test.vtp",
        "--output-dir",
        "out",
        "--num-processes",
        "2",
    ]
    app.run()
    mock_process.assert_called_with("test.vtp", "out", 2)


@patch("b3_2d.core.plotting.plot_mesh")
@patch("pyvista.read")
def test_plot_command(mock_pv_read, mock_plot):
    """Test plot command execution."""
    mock_mesh = mock_pv_read.return_value
    sys.argv = [
        "b3-2d",
        "plot",
        "--mesh-file",
        "mesh.vtk",
        "--output-file",
        "plot.png",
        "--scalar",
        "material_id",
    ]
    app.run()
    mock_pv_read.assert_called_with("mesh.vtk")
    mock_plot.assert_called_with(
        mock_mesh, scalar="material_id", output_file="plot.png"
    )


@patch("subprocess.run")
@patch("shutil.which")
@patch("pathlib.Path")
def test_anba_all_command(mock_path_class, mock_which, mock_subprocess):
    """Test anba all command execution."""
    mock_path = MagicMock()
    mock_path.glob.return_value = ["file"]
    mock_path_class.return_value = mock_path
    mock_which.return_value = "conda"
    mock_subprocess.return_value = MagicMock(
        returncode=0, stdout="anba4-env", stderr=""
    )
    sys.argv = [
        "b3-2d",
        "anba",
        "all",
        "--output-dir",
        "out",
    ]
    app.run()
    # Check that subprocess was called
    assert mock_subprocess.called


@patch("subprocess.run")
def test_anba_single_command(mock_subprocess):
    """Test anba single command execution."""
    mock_subprocess.return_value = MagicMock(returncode=0, stdout="output", stderr="")
    sys.argv = [
        "b3-2d",
        "anba",
        "single",
        "--json-file",
        "file.json",
    ]
    app.run()
    # Check that subprocess was called
    assert mock_subprocess.called


@patch("pathlib.Path.exists", return_value=True)
@patch("builtins.open")
@patch("b3_2d.cli.cli.json.load")
@patch("pyvista.read")
@patch("b3_2d.core.plotting.plot_section_anba")
def test_anba_plot_command(mock_plot, mock_pv_read, mock_json, mock_open, mock_exists):
    """Test anba plot command execution."""
    mock_open.return_value = MagicMock()
    mock_mesh = mock_pv_read.return_value
    mock_data = {
        "mass_center": [0, 0],
        "shear_center": [0, 0],
        "tension_center": [0, 0],
        "principal_angle": 0,
    }
    mock_json.return_value = mock_data
    sys.argv = [
        "b3-2d",
        "anba",
        "plot",
        "--json-file",
        "file.json",
        "--output-file",
        "plot.png",
    ]
    app.run()
    mock_plot.assert_called_with(mock_mesh, mock_data, "plot.png")
