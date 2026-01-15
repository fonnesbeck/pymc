#   Copyright 2025 - present The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from unittest.mock import MagicMock, patch

import pymc as pm

from pymc.progress_bar import (
    MarimoProgressBackend,
    detect_environment,
    in_marimo_notebook,
)


class TestEnvironmentDetection:
    """Tests for environment detection functions."""

    def test_in_marimo_notebook_not_installed(self):
        """Test that in_marimo_notebook returns False when marimo is not installed."""
        with patch.dict("sys.modules", {"marimo": None}):
            # Force reimport by clearing any cached result
            assert in_marimo_notebook() is False

    def test_in_marimo_notebook_not_running(self):
        """Test that in_marimo_notebook returns False when not running in notebook."""
        mock_marimo = MagicMock()
        mock_marimo.running_in_notebook.return_value = False
        with patch.dict("sys.modules", {"marimo": mock_marimo}):
            result = in_marimo_notebook()
            # The function imports marimo fresh, so we need to check its behavior
            assert result is False

    def test_detect_environment_default(self):
        """Test that detect_environment returns 'rich' by default."""
        with patch("pymc.progress_bar.in_marimo_notebook", return_value=False):
            assert detect_environment() == "rich"

    def test_detect_environment_marimo(self):
        """Test that detect_environment returns 'marimo' when in marimo."""
        with patch("pymc.progress_bar.in_marimo_notebook", return_value=True):
            assert detect_environment() == "marimo"


class TestMarimoProgressBackend:
    """Tests for the MarimoProgressBackend class."""

    def test_format_stats_inline(self):
        """Test stats inline formatting."""
        # Create a mock step method
        mock_step = MagicMock()
        mock_step._progressbar_config.return_value = (
            [],
            {"divergences": [0, 0], "step_size": [0.5, 0.5]},
        )
        mock_step._make_progressbar_update_functions.return_value = []

        backend = MarimoProgressBackend(
            step_method=mock_step,
            chains=2,
            draws=100,
            tune=50,
            progressbar=True,
        )

        # Test formatting with abbreviated names
        stats = {"divergences": 5, "step_size": 0.123}
        result = backend._format_stats_inline(stats)
        assert "Div: 5" in result
        assert "Step: 0.123" in result

    def test_format_timing(self):
        """Test timing formatting."""
        mock_step = MagicMock()
        mock_step._progressbar_config.return_value = ([], {"divergences": [0, 0]})
        mock_step._make_progressbar_update_functions.return_value = []

        backend = MarimoProgressBackend(
            step_method=mock_step,
            chains=2,
            draws=100,
            tune=50,
            progressbar=True,
        )

        # Fast rate
        result = backend._format_timing(100.0, 1.5)
        assert "100 it/s" in result
        assert "1.50s" in result

        # Slow rate
        result = backend._format_timing(0.5, 10.0)
        assert "s/it" in result
        assert "10.00s" in result

    def test_disabled_backend(self):
        """Test that disabled backend doesn't create progress bars."""
        mock_step = MagicMock()
        mock_step._progressbar_config.return_value = ([], {"divergences": [0, 0]})
        mock_step._make_progressbar_update_functions.return_value = []

        backend = MarimoProgressBackend(
            step_method=mock_step,
            chains=2,
            draws=100,
            tune=50,
            progressbar=False,  # Disabled
        )

        assert backend._disabled is True
        assert backend._started is False

        # Enter context should not fail
        backend.__enter__()
        assert backend._started is False  # Still False when disabled

        # Update should be a no-op
        backend.update(chain_idx=0, is_last=False, draw=0, tuning=True, stats=[])

        # Exit should not fail
        backend.__exit__(None, None, None)

    def test_combined_mode(self):
        """Test that combined mode is detected correctly."""
        mock_step = MagicMock()
        mock_step._progressbar_config.return_value = ([], {"divergences": [0, 0]})
        mock_step._make_progressbar_update_functions.return_value = []

        backend = MarimoProgressBackend(
            step_method=mock_step,
            chains=2,
            draws=100,
            tune=50,
            progressbar="combined+stats",
        )

        assert backend._combined is True
        assert backend._show_full_stats is True

    def test_split_mode(self):
        """Test that split mode is detected correctly."""
        mock_step = MagicMock()
        mock_step._progressbar_config.return_value = ([], {"divergences": [0, 0]})
        mock_step._make_progressbar_update_functions.return_value = []

        backend = MarimoProgressBackend(
            step_method=mock_step,
            chains=2,
            draws=100,
            tune=50,
            progressbar="split+stats",
        )

        assert backend._combined is False
        assert backend._show_full_stats is True


def test_progressbar_nested_compound():
    # Regression test for https://github.com/pymc-devs/pymc/issues/7721

    with pm.Model():
        a = pm.Poisson("a", mu=10)
        b = pm.Binomial("b", n=a, p=0.8)
        c = pm.Poisson("c", mu=11)
        d = pm.Dirichlet("d", a=[c, b])

        step = pm.CompoundStep(
            [
                pm.CompoundStep([pm.Metropolis(a), pm.Metropolis(b), pm.Metropolis(c)]),
                pm.NUTS([d]),
            ]
        )

        kwargs = {
            "draws": 10,
            "tune": 10,
            "chains": 2,
            "compute_convergence_checks": False,
            "step": step,
        }

        # We don't parametrize to avoid recompiling the model functions
        for cores in (1, 2):
            pm.sample(**kwargs, cores=cores, progressbar=True)  # default is split+stats
            pm.sample(**kwargs, cores=cores, progressbar="combined")
            pm.sample(**kwargs, cores=cores, progressbar="split")
            pm.sample(**kwargs, cores=cores, progressbar=False)
