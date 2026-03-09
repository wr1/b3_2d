"""Tests for CLI functionality."""

from b3_2d.cli.cli import main


def test_main_callable():
    """Test that main is callable."""
    assert callable(main)
