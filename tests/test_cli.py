"""Tests for CLI functionality."""

import sys
import pytest
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
