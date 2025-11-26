"""Tests for CLI functionality."""

import pytest
from b3_2d.cli.cli import app


def test_cli_app():
    """Test that the CLI app is properly configured."""
    assert app.name == "b3_2d"
    assert len(app.commands) == 2  # mesh and plot commands


def test_mesh_command_help(capsys):
    """Test mesh command help output."""
    with pytest.raises(SystemExit):
        app.run(["mesh", "--help"])
    captured = capsys.readouterr()
    assert "Process VTP file for multi-section meshing" in captured.out


def test_plot_command_help(capsys):
    """Test plot command help output."""
    with pytest.raises(SystemExit):
        app.run(["plot", "--help"])
    captured = capsys.readouterr()
    assert "Plot a mesh" in captured.out
