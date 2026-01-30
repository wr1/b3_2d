"""Tests for B32dAnbaStep."""

from unittest.mock import patch, MagicMock

try:
    from b3_2d.state.b3_2d_anba import B32dAnbaStep
except ImportError:
    B32dAnbaStep = None


@patch("b3_2d.state.b3_2d_anba.open")
@patch("b3_2d.state.b3_2d_anba.subprocess.run")
@patch("b3_2d.state.b3_2d_anba.shutil.which")
@patch("b3_2d.state.b3_2d_anba.Path")
@patch(
    "b3_2d.state.b3_2d_anba.B32dAnbaStep.load_config",
    return_value={"workdir": "work", "anba_env": "anba4-env"},
)
def test_b32d_anba_step(
    mock_load_config, mock_path_class, mock_which, mock_subprocess, mock_open
):
    """Test B32dAnbaStep execution."""
    if B32dAnbaStep is None:
        import pytest

        pytest.skip("Statesman not available")
    mock_subprocess.return_value = MagicMock(
        returncode=0, stdout="anba4-env", stderr=""
    )
    mock_which.return_value = "conda"
    mock_open.return_value.__enter__.return_value = MagicMock()
    mock_config_path = MagicMock()
    mock_config_path.parent = MagicMock()
    mock_config_path.parent.__truediv__ = MagicMock(return_value=MockPath())
    mock_path_class.return_value = mock_config_path

    step = B32dAnbaStep("config.yaml")
    step.logger = MagicMock()

    step._execute()

    assert mock_subprocess.call_count == 2  # env list and anba run
    step.logger.info.assert_called()


class MockPath:
    def __init__(self):
        self.path = "/work"

    def __str__(self):
        return self.path

    def __fspath__(self):
        return self.path

    def __truediv__(self, other):
        new_path = f"{self.path}/{other}"
        mock = MockPath()
        mock.path = new_path
        return mock

    def mkdir(self, **kwargs):
        pass

    def glob(self, pattern):
        # Mock glob to return some files
        mock_file = MockPath()
        mock_file.path = "/work/b3_2d/section_1/anba.json"
        return [mock_file]
