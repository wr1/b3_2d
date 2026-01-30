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


def test_plot_section_anba():
    """Test plot_section_anba function."""
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

        from b3_2d.core.plotting import plot_section_anba

        plot_section_anba(mock_mesh, mock_data, "test.png")

        mock_plt.subplots.assert_called_once_with(figsize=(12.8, 6.6))
        # Further assertions can be added if needed


@patch("b3_2d.core.span_plotting.plt")
def test_plot_span_centers(mock_plt):
    """Test plot_span_centers function."""
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())
    from b3_2d.core.span_plotting import plot_span_centers

    data_list = [
        (
            1,
            {
                "mass_center": [0, 0],
                "shear_center": [1, 1],
                "tension_center": [2, 2],
                "principal_angle": 0.0,
            },
        ),
        (
            2,
            {
                "mass_center": [0.1, 0.1],
                "shear_center": [1.1, 1.1],
                "tension_center": [2.1, 2.1],
                "principal_angle": 0.1,
            },
        ),
    ]
    plot_span_centers(data_list, "test.png")

    mock_plt.subplots.assert_called_once_with(2, 2, figsize=(12, 8))
    mock_plt.savefig.assert_called_once_with("test.png", dpi=400)


@patch("b3_2d.core.span_plotting.plt")
def test_plot_span_stiff(mock_plt):
    """Test plot_span_stiff function."""
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())
    from b3_2d.core.span_plotting import plot_span_stiff

    data_list = [
        (1, {"stiffness": [[1, 2], [3, 4]], "mass": [[0.1, 0.2], [0.3, 0.4]]}),
        (
            2,
            {
                "stiffness": [[1.1, 2.1], [3.1, 4.1]],
                "mass": [[0.11, 0.21], [0.31, 0.41]],
            },
        ),
    ]
    plot_span_stiff(data_list, "test.png")

    mock_plt.subplots.assert_called_once_with(2, 2, figsize=(18, 12))
    mock_plt.savefig.assert_called_once_with("test.png", dpi=400)


@patch("b3_2d.core.span_plotting.plt")
@patch("pathlib.Path.glob")
@patch("builtins.open")
@patch("json.load")
def test_plot_span_anba(mock_json_load, mock_open, mock_glob, mock_plt):
    """Test plot_span_anba function."""
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())
    from b3_2d.core.span_plotting import plot_span_anba

    mock_file = MagicMock()
    mock_file.parent.name = "section_1"
    mock_glob.return_value = [mock_file]
    mock_json_load.return_value = {
        "mass_center": [0, 0],
        "shear_center": [1, 1],
        "tension_center": [2, 2],
        "principal_angle": 0.0,
        "stiffness": [[1, 2], [3, 4]],
        "mass": [[0.1, 0.2], [0.3, 0.4]],
    }

    plot_span_anba("output_dir", "test.png")

    assert mock_plt.subplots.call_count >= 2  # Called for centers and stiff
    mock_plt.savefig.assert_called()


@patch("b3_2d.core.bom_plotting.plt")
@patch("pathlib.Path.glob")
@patch("builtins.open")
@patch("json.load")
def test_plot_bom_spanwise(mock_json_load, mock_open, mock_glob, mock_plt):
    """Test plot_bom_spanwise function."""
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())
    from b3_2d.core.bom_plotting import plot_bom_spanwise

    mock_file = MagicMock()
    mock_file.parent.name = "section_1"
    mock_glob.return_value = [mock_file]
    mock_json_load.return_value = {
        "total_area": 10.0,
        "areas_per_material": {1: 5.0, 2: 5.0},
        "total_mass": 100.0,
        "masses_per_material": {1: 50.0, 2: 50.0},
    }
    matdb = {"carbon": {"id": 1}, "glass": {"id": 2}}

    plot_bom_spanwise("output_dir", "test.png", matdb)

    mock_plt.subplots.assert_called_once_with(2, 1, figsize=(12, 8))
    mock_plt.savefig.assert_called_once_with("test.png", dpi=400)
