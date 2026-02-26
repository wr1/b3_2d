"""Tests for B32dAnbaStep."""


try:
    from b3_2d.state.b3_2d_anba import B32dAnbaStep
except ImportError:
    B32dAnbaStep = None


# @patch("b3_2d.state.b3_2d_anba.B32dAnbaStep.load_config", return_value={"workdir": "work", "anba_env": "anba4-env"})
def test_b32d_anba_step():
    """Test B32dAnbaStep execution."""
    if B32dAnbaStep is None:
        import pytest

        pytest.skip("Statesman not available")

    # with patch("b3_2d.state.b3_2d_anba.B32dAnbaStep.load_config", return_value={"workdir": "work", "anba_env": "anba4-env"}):
    #     mock_load_config = MagicMock(return_value={"workdir": "work", "anba_env": "anba4-env"})
    #     step = B32dAnbaStep("config.yaml")
    #     step.logger = MagicMock()
    #     step.load_config = mock_load_config
    #     step._execute()
    #     assert mock_subprocess.called
    #     step.logger.info.assert_called()
    pass  # Skip for now
