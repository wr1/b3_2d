"""Tests for CLI functionality."""

import sys
import pytest
from unittest.mock import patch, MagicMock
from b3_2d.cli.cli import app


def test_cli_app():
    """Test that the CLI app is properly configured."""
    assert app.name == "b3_2d"
    assert len(app.commands) == 4  # mesh, plot, span, post commands
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


def test_span_command_help(capsys):
    """Test span command help output."""
    sys.argv = ["b3-2d", "span", "--help"]
    with pytest.raises(SystemExit):
        app.run()
    captured = capsys.readouterr()
    assert "Plot ANBA stiffnesses and masses along blade span" in captured.out


def test_post_command_help(capsys):
    """Test post command help output."""
    sys.argv = ["b3-2d", "post", "--help"]
    with pytest.raises(SystemExit):
        app.run()
    captured = capsys.readouterr()
    assert "Run postprocessing plots for BOM and ANBA" in captured.out


@patch("b3_2d.cli.commands.mesh.process_vtp_multi_section")
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


@patch("b3_2d.cli.commands.plot.plot_mesh")
@patch("pyvista.read")
def test_plot_command(mock_pv_read, mock_plot):
    """Test plot command execution."""
    import pyvista as pv

    mock_pv_read.return_value = pv.PolyData(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]]
    )
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


@patch("b3_2d.cli.commands.anba_all.subprocess.run")
@patch("shutil.which")
@patch("b3_2d.cli.commands.anba_all.Path")
def test_anba_all_command(mock_path_class, mock_which, mock_subprocess):
    """Test anba all command execution."""
    mock_path = MagicMock()
    mock_path.glob.return_value = ["section_1/anba.json"]
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


@patch("b3_2d.cli.commands.anba_single.subprocess.run")
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


@patch("b3_2d.cli.commands.anba_plot.Path")
@patch("builtins.open")
@patch("b3_2d.cli.cli.json.load")
@patch("pyvista.read")
@patch("b3_2d.cli.commands.anba_plot.plot_section_anba")
def test_anba_plot_command(
    mock_plot, mock_pv_read, mock_json, mock_open, mock_path_class
):
    """Test anba plot command execution."""
    import numpy as np

    mock_open.return_value = MagicMock()
    mock_mesh = mock_pv_read.return_value
    mock_mesh.points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    mock_mesh.cells = np.array([3, 0, 1, 2])
    mock_mesh.cell_data = {"material_id": [1]}
    mock_mesh.bounds = [0, 1, 0, 1, 0, 0]
    mock_data = {
        "mass_center": [0, 0],
        "shear_center": [0, 0],
        "tension_center": [0, 0],
        "principal_angle": 0,
    }
    mock_json.return_value = mock_data
    mock_path = MagicMock()
    mock_path.exists.return_value = True
    mock_path_class.return_value = mock_path
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


@patch("b3_2d.cli.commands.span_plot.plot_span_anba")
def test_span_command(mock_plot_span):
    """Test span command execution."""
    sys.argv = [
        "b3-2d",
        "span",
        "--output-dir",
        "out",
        "--output-file",
        "span.png",
    ]
    app.run()
    mock_plot_span.assert_called_with("out", "span.png")


@patch("b3_2d.cli.commands.post.plot_bom_spanwise")
@patch("b3_2d.cli.commands.post.plot_span_anba")
@patch("builtins.open")
@patch("b3_2d.cli.cli.json.load")
def test_post_command(mock_json_load, mock_open, mock_plot_anba, mock_plot_bom):
    """Test post command execution."""
    mock_json_load.return_value = {}
    sys.argv = [
        "b3-2d",
        "post",
        "--output-dir",
        "out",
        "--matdb-file",
        "matdb.json",
    ]
    app.run()
    mock_plot_bom.assert_called_with("out", "out/bom_spanwise.png", {})
    mock_plot_anba.assert_called_with("out", "out/anba_spanwise.png")
