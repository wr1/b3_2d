"""Tests for B32dPostStep."""

from unittest.mock import patch, MagicMock

try:
    from b3_2d.state.b3_2d_post import B32dPostStep
except ImportError:
    B32dPostStep = None


def test_b32d_post_step():
    """Test B32dPostStep execution."""
    if B32dPostStep is None:
        import pytest
        pytest.skip("B32dPostStep not available")
    
    with patch("b3_2d.state.b3_2d_post.B32dPostStep.load_config", return_value={"workdir": "work", "matdb": {}}), \
         patch("pathlib.Path") as mock_path_class, \
         patch("b3_2d.state.b3_2d_post.plot_bom_spanwise") as mock_plot_bom, \
         patch("b3_2d.state.b3_2d_post.plot_span_anba") as mock_plot_anba:
        
        mock_config_path = MagicMock()
        mock_config_path.parent = MagicMock()
        mock_workdir = MagicMock()
        mock_output_dir = MagicMock()
        mock_bom_plot = MagicMock()
        mock_anba_plot = MagicMock()
        
        mock_path_class.return_value = mock_config_path
        mock_config_path.parent.__truediv__ = MagicMock(return_value=mock_workdir)
        mock_workdir.__truediv__ = MagicMock(return_value=mock_output_dir)
        mock_output_dir.__truediv__.side_effect = [mock_bom_plot, mock_anba_plot]
        mock_bom_plot.exists.return_value = True
        mock_anba_plot.exists.return_value = True
        
        step = B32dPostStep("config.yaml")
        step.logger = MagicMock()
        
        step._execute()
        
        mock_plot_bom.assert_called_once()
        mock_plot_anba.assert_called_once()
        step.logger.info.assert_called()
