"""Tests for plotting functionality."""

from unittest.mock import patch, MagicMock


def test_plot_mesh():
    """Test plot_mesh function."""
    with patch("b3_2d.core.plotting.pv.Plotter") as mock_plotter_class:
        mock_plotter = MagicMock()
        mock_plotter_class.return_value = mock_plotter
        mock_mesh = MagicMock()
        mock_mesh.cell_data = {"test_scalar": []}

        from b3_2d.core.plotting import plot_mesh

        plot_mesh(mock_mesh, scalar="test_scalar", output_file="test.png")

        mock_plotter_class.assert_called_once()
        mock_plotter.add_mesh.assert_called_once_with(mock_mesh, scalars="test_scalar")
        mock_plotter.view_xy.assert_called_once()
        mock_plotter.screenshot.assert_called_once_with("test.png")
