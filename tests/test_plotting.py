"""Tests for plotting functionality."""

from unittest.mock import patch, MagicMock
import numpy as np


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


def test_plot_anba_results():
    """Test plot_anba_results function."""
    with patch("b3_2d.core.plotting.plt") as mock_plt:
        mock_plt.subplots.return_value = (MagicMock(), MagicMock())
        mock_mesh = MagicMock()
        mock_mesh.points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        mock_mesh.cells = np.array([3, 0, 1, 2])
        mock_mesh.cell_data = {"material_id": [1]}
        mock_mesh.bounds = [0, 1, 0, 1, 0, 0]
        mock_data = {
            "mass_center": [0.5, 0.5],
            "shear_center": [0.4, 0.4],
            "tension_center": [0.3, 0.3],
            "principal_angle": 0.0,
        }

        from b3_2d.core.plotting import plot_anba_results

        plot_anba_results(mock_mesh, mock_data, "test.png")

        mock_plt.subplots.assert_called_once_with(figsize=(12.8, 9.6))
        # Further assertions can be added if needed
