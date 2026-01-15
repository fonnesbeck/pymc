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
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

from rich.box import SIMPLE_HEAD
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    Task,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.style import Style
from rich.table import Column, Table
from rich.theme import Theme

if TYPE_CHECKING:
    from pymc.step_methods.compound import BlockedStep, CompoundStep

ProgressBarType = Literal[
    "combined",
    "split",
    "combined+stats",
    "stats+combined",
    "split+stats",
    "stats+split",
]
default_progress_theme = Theme(
    {
        "bar.complete": "#1764f4",
        "bar.finished": "green",
        "progress.remaining": "none",
        "progress.elapsed": "none",
    }
)


def in_marimo_notebook() -> bool:
    """Check if running inside a marimo notebook.

    Returns
    -------
    bool
        True if running in a marimo notebook, False otherwise.
    """
    try:
        import marimo as mo

        return mo.running_in_notebook()
    except (ImportError, AttributeError):
        return False


def detect_environment() -> Literal["marimo", "rich"]:
    """Detect the runtime environment for progress bar display.

    Returns
    -------
    str
        "marimo" if running in a marimo notebook, "rich" otherwise.
        Rich is used for both terminal and Jupyter environments.
    """
    if in_marimo_notebook():
        return "marimo"
    return "rich"


class MarimoProgressBackend:
    """Progress bar backend using custom compact HTML display for marimo.

    This backend provides space-efficient progress bars for marimo notebooks,
    using custom HTML/CSS with `mo.output.replace()` for dynamic updates.
    Each chain is displayed on a single line with inline stats.

    Parameters
    ----------
    step_method : BlockedStep or CompoundStep
        The step method being used to sample
    chains : int
        Number of chains being sampled
    draws : int
        Number of draws per chain
    tune : int
        Number of tuning steps per chain
    progressbar : bool or ProgressBarType
        How and whether to display the progress bar
    show_stats : list of str, optional
        Which statistics to display. If None, all available stats are shown.
    """

    # CSS styles for compact progress display
    _CSS = """
    <style>
    .pymc-progress-container {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 13px;
        line-height: 1.4;
        padding: 8px 0;
    }
    .pymc-progress-row {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 2px 0;
        flex-wrap: nowrap;
    }
    .pymc-chain-label {
        min-width: 70px;
        font-weight: 500;
        white-space: nowrap;
    }
    .pymc-chain-label.failing {
        color: #dc2626;
    }
    .pymc-bar-container {
        width: 150px;
        min-width: 150px;
        height: 14px;
        background: #e5e7eb;
        border-radius: 3px;
        overflow: hidden;
    }
    .pymc-bar-fill {
        height: 100%;
        background: #1764f4;
        transition: width 0.1s ease-out;
    }
    .pymc-bar-fill.failing {
        background: #dc2626;
    }
    .pymc-bar-fill.complete {
        background: #16a34a;
    }
    .pymc-progress-text {
        min-width: 110px;
        white-space: nowrap;
        color: #374151;
    }
    .pymc-stats {
        color: #6b7280;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .pymc-timing {
        color: #9ca3af;
        white-space: nowrap;
        margin-left: auto;
    }
    </style>
    """

    def __init__(
        self,
        step_method: "BlockedStep | CompoundStep",
        chains: int,
        draws: int,
        tune: int,
        progressbar: bool | ProgressBarType,
        show_stats: list[str] | None = None,
    ):
        import time

        self._disabled = progressbar is False
        self._combined = "combined" in str(progressbar) if isinstance(progressbar, str) else False
        self._show_full_stats = (
            "stats" in str(progressbar) if isinstance(progressbar, str) else True
        )

        # If progressbar=True, default to split+stats
        if progressbar is True:
            self._combined = False
            self._show_full_stats = True

        self.chains = chains
        self.total_draws = draws + tune
        self.completed_draws = 0

        # Get stats config from step method
        _, stats_config = step_method._progressbar_config(chains)
        self._stat_keys = show_stats if show_stats is not None else list(stats_config.keys())
        self._update_stats_functions = step_method._make_progressbar_update_functions()

        # Track per-chain state
        self._chain_draws: dict[int, int] = dict.fromkeys(range(chains), 0)
        self._chain_stats: dict[int, dict[str, Any]] = {i: {} for i in range(chains)}
        self._chain_failing: dict[int, bool] = dict.fromkeys(range(chains), False)
        self._chain_start_times: dict[int, float] = {}
        self._chain_elapsed: dict[int, float] = dict.fromkeys(range(chains), 0.0)

        # For combined mode
        self._combined_start_time: float | None = None
        self._combined_failing: bool = False
        self._combined_stats: dict[str, Any] = {}

        # Track if we've started
        self._started: bool = False
        self._time = time

    def __enter__(self):
        """Enter context manager and initialize display."""
        if self._disabled:
            return self

        import marimo as mo

        # Initialize timing
        self._combined_start_time = self._time.time()
        for i in range(self.chains):
            self._chain_start_times[i] = self._time.time()

        # Create initial display
        self._started = True
        mo.output.replace(mo.Html(self._render_html()))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - display remains visible."""
        if self._disabled or not self._started:
            return

        import marimo as mo

        # Final update to show completion
        mo.output.replace(mo.Html(self._render_html(complete=True)))

    def update(
        self, chain_idx: int, is_last: bool, draw: int, tuning: bool, stats: list[dict[str, Any]]
    ):
        """Update progress bar with new draw information.

        Parameters
        ----------
        chain_idx : int
            Index of the chain being updated
        is_last : bool
            Whether this is the last draw for the chain
        draw : int
            Current draw number
        tuning : bool
            Whether we are still in tuning phase
        stats : list of dict
            Statistics from each step method
        """
        if self._disabled or not self._started:
            return

        import marimo as mo

        self.completed_draws += 1
        self._chain_draws[chain_idx] = draw + 1

        # Calculate elapsed time
        current_time = self._time.time()
        self._chain_elapsed[chain_idx] = current_time - self._chain_start_times[chain_idx]

        # Process stats through update functions
        failing = False
        all_step_stats: dict[str, Any] = {}

        chain_progress_stats = [
            update_stats_fn(step_stats)
            for update_stats_fn, step_stats in zip(self._update_stats_functions, stats, strict=True)
        ]
        for step_stats in chain_progress_stats:
            for key, val in step_stats.items():
                if key == "failing":
                    failing |= val
                    continue
                if not self._show_full_stats:
                    continue
                if key not in all_step_stats:
                    all_step_stats[key] = val

        # Update chain state
        self._chain_failing[chain_idx] = failing
        self._chain_stats[chain_idx] = all_step_stats

        # For combined mode, aggregate
        if self._combined:
            self._combined_failing = any(self._chain_failing.values())
            self._combined_stats = all_step_stats

        # Update display
        mo.output.replace(mo.Html(self._render_html()))

    def _render_html(self, complete: bool = False) -> str:
        """Render the complete HTML for all progress bars."""
        if self._combined:
            rows = [self._render_combined_row(complete)]
        else:
            rows = [self._render_chain_row(i, complete) for i in range(self.chains)]

        return f'{self._CSS}<div class="pymc-progress-container">{"".join(rows)}</div>'

    def _render_combined_row(self, complete: bool = False) -> str:
        """Render a single row for combined mode."""
        current = self.completed_draws
        total = self.total_draws * self.chains
        pct = (current / total * 100) if total > 0 else 0
        failing = self._combined_failing

        elapsed = self._time.time() - self._combined_start_time if self._combined_start_time else 0
        rate = current / max(elapsed, 1e-6)

        label_class = "pymc-chain-label failing" if failing else "pymc-chain-label"
        bar_class = "pymc-bar-fill"
        if failing:
            bar_class += " failing"
        elif complete and pct >= 100:
            bar_class += " complete"

        label = "Sampling"
        stats_str = self._format_stats_inline(self._combined_stats)
        timing = self._format_timing(rate, elapsed)

        return f"""
        <div class="pymc-progress-row">
            <span class="{label_class}">{label}</span>
            <div class="pymc-bar-container">
                <div class="{bar_class}" style="width: {pct:.1f}%"></div>
            </div>
            <span class="pymc-progress-text">{current}/{total} ({pct:.0f}%)</span>
            <span class="pymc-stats">{stats_str}</span>
            <span class="pymc-timing">{timing}</span>
        </div>
        """

    def _render_chain_row(self, chain_idx: int, complete: bool = False) -> str:
        """Render a single row for a chain."""
        current = self._chain_draws[chain_idx]
        total = self.total_draws
        pct = (current / total * 100) if total > 0 else 0
        failing = self._chain_failing[chain_idx]
        stats = self._chain_stats[chain_idx]
        elapsed = self._chain_elapsed[chain_idx]
        rate = current / max(elapsed, 1e-6)

        label_class = "pymc-chain-label failing" if failing else "pymc-chain-label"
        bar_class = "pymc-bar-fill"
        if failing:
            bar_class += " failing"
        elif complete and pct >= 100:
            bar_class += " complete"

        label = f"Chain {chain_idx}"
        stats_str = self._format_stats_inline(stats)
        timing = self._format_timing(rate, elapsed)

        return f"""
        <div class="pymc-progress-row">
            <span class="{label_class}">{label}</span>
            <div class="pymc-bar-container">
                <div class="{bar_class}" style="width: {pct:.1f}%"></div>
            </div>
            <span class="pymc-progress-text">{current}/{total} ({pct:.0f}%)</span>
            <span class="pymc-stats">{stats_str}</span>
            <span class="pymc-timing">{timing}</span>
        </div>
        """

    def _format_stats_inline(self, stats: dict[str, Any]) -> str:
        """Format stats as a compact inline string with abbreviated names."""
        if not self._show_full_stats or not stats:
            return ""

        # Abbreviated display names for compactness
        abbrev_names = {
            "divergences": "Div",
            "step_size": "Step",
            "tree_size": "Grad",
            "n_steps": "Grad",
            "tune": "Tune",
            "scaling": "Scale",
            "accept_rate": "Acc",
            "accept": "Acc",
            "n_evals": "Evals",
        }

        parts = []
        for key in self._stat_keys:
            if key not in stats:
                continue
            val = stats[key]
            name = abbrev_names.get(key, key[:4].title())

            if isinstance(val, bool):
                formatted = "Y" if val else "N"
            elif isinstance(val, float):
                formatted = f"{val:.3f}"
            else:
                formatted = str(val)

            parts.append(f"{name}: {formatted}")

        return " | ".join(parts)

    def _format_timing(self, rate: float, elapsed: float) -> str:
        """Format timing information compactly."""
        if rate > 1:
            rate_str = f"{rate:.0f} it/s"
        else:
            rate_str = f"{1 / rate:.2f} s/it" if rate > 0 else "-- s/it"

        return f"{rate_str} | {elapsed:.2f}s"


class CustomProgress(Progress):
    """A child of Progress that allows to disable progress bars and its container.

    The implementation simply checks an `is_enabled` flag and generates the progress bar only if
    it's `True`.
    """

    def __init__(self, *args, disable=False, include_headers=False, **kwargs):
        self.is_enabled = not disable
        self.include_headers = include_headers

        if self.is_enabled:
            super().__init__(*args, **kwargs)

    def __enter__(self):
        """Enter the context manager."""
        if self.is_enabled:
            self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        if self.is_enabled:
            super().__exit__(exc_type, exc_val, exc_tb)

    def add_task(self, *args, **kwargs):
        if self.is_enabled:
            return super().add_task(*args, **kwargs)
        return None

    def advance(self, task_id, advance=1) -> None:
        if self.is_enabled:
            super().advance(task_id, advance)
        return None

    def update(
        self,
        task_id,
        *,
        total=None,
        completed=None,
        advance=None,
        description=None,
        visible=None,
        refresh=False,
        **fields,
    ):
        if self.is_enabled:
            super().update(
                task_id,
                total=total,
                completed=completed,
                advance=advance,
                description=description,
                visible=visible,
                refresh=refresh,
                **fields,
            )
        return None

    def make_tasks_table(self, tasks: Iterable[Task]) -> Table:
        """Get a table to render the Progress display.

        Unlike the parent method, this one returns a full table (not a grid), allowing for column headings.

        Parameters
        ----------
        tasks: Iterable[Task]
            An iterable of Task instances, one per row of the table.

        Returns
        -------
        table: Table
            A table instance.
        """

        def call_column(column, task):
            # Subclass rich.BarColumn and add a callback method to dynamically update the display
            if hasattr(column, "callbacks"):
                column.callbacks(task)

            return column(task)

        table_columns = (
            (
                Column(no_wrap=True)
                if isinstance(_column, str)
                else _column.get_table_column().copy()
            )
            for _column in self.columns
        )
        if self.include_headers:
            table = Table(
                *table_columns,
                padding=(0, 1),
                expand=self.expand,
                show_header=True,
                show_edge=True,
                box=SIMPLE_HEAD,
            )
        else:
            table = Table.grid(*table_columns, padding=(0, 1), expand=self.expand)

        for task in tasks:
            if task.visible:
                table.add_row(
                    *(
                        (
                            column.format(task=task)
                            if isinstance(column, str)
                            else call_column(column, task)
                        )
                        for column in self.columns
                    )
                )

        return table


class RecolorOnFailureBarColumn(BarColumn):
    """Rich colorbar that changes color when a chain has detected a failure."""

    def __init__(self, *args, failing_color="red", **kwargs):
        from matplotlib.colors import to_rgb

        self.failing_color = failing_color
        self.failing_rgb = [int(x * 255) for x in to_rgb(self.failing_color)]

        super().__init__(*args, **kwargs)

        self.default_complete_style = self.complete_style
        self.default_finished_style = self.finished_style

    def callbacks(self, task: "Task"):
        if task.fields["failing"]:
            self.complete_style = Style.parse("rgb({},{},{})".format(*self.failing_rgb))
            self.finished_style = Style.parse("rgb({},{},{})".format(*self.failing_rgb))
        else:
            # Recovered from failing yay
            self.complete_style = self.default_complete_style
            self.finished_style = self.default_finished_style


class ProgressBarManager:
    """Manage progress bars displayed during sampling."""

    def __init__(
        self,
        step_method: "BlockedStep | CompoundStep",
        chains: int,
        draws: int,
        tune: int,
        progressbar: bool | ProgressBarType = True,
        progressbar_theme: Theme | None = None,
        backend: Literal["auto", "rich", "marimo"] | None = None,
        show_stats: list[str] | None = None,
    ):
        """
        Manage progress bars displayed during sampling.

        When sampling, Step classes are responsible for computing and exposing statistics that can be reported on
        progress bars. Each Step implements two class methods: :meth:`pymc.step_methods.BlockedStep._progressbar_config`
        and :meth:`pymc.step_methods.BlockedStep._make_progressbar_update_functions`. `_progressbar_config` reports which
        columns should be displayed on the progress bar, and `_make_progressbar_update_functions` computes the statistics
        that will be displayed on the progress bar.

        Parameters
        ----------
        step_method: BlockedStep or CompoundStep
            The step method being used to sample
        chains: int
            Number of chains being sampled
        draws: int
            Number of draws per chain
        tune: int
            Number of tuning steps per chain
        progressbar: bool or ProgressType, optional
            How and whether to display the progress bar. If False, no progress bar is displayed. Otherwise, you can ask
            for one of the following:
            - "combined": A single progress bar that displays the total progress across all chains. Only timing
                information is shown.
            - "split": A separate progress bar for each chain. Only timing information is shown.
            - "combined+stats" or "stats+combined": A single progress bar displaying the total progress across all
                chains. Aggregate sample statistics are also displayed.
            - "split+stats" or "stats+split": A separate progress bar for each chain. Sample statistics for each chain
                are also displayed.

            If True, the default is "split+stats" is used.

        progressbar_theme: Theme, optional
            The theme to use for the progress bar. Defaults to the default theme.
        backend: {"auto", "rich", "marimo"}, optional
            Which backend to use for progress display. If "auto" (default), automatically
            detects the environment: uses marimo's native progress bars when running in
            a marimo notebook, otherwise uses Rich for terminal/Jupyter display.
        show_stats: list of str, optional
            Which statistics to display in the progress bar. If None, all available
            statistics from the step method are shown. Only used with the marimo backend.
        """
        # Auto-detect environment if not specified
        if backend is None or backend == "auto":
            backend = detect_environment()

        self._backend_type = backend

        # Use marimo backend if in marimo environment
        if backend == "marimo":
            self._marimo_backend: MarimoProgressBackend | None = MarimoProgressBackend(
                step_method=step_method,
                chains=chains,
                draws=draws,
                tune=tune,
                progressbar=progressbar,
                show_stats=show_stats,
            )
            # Set attributes for compatibility (used by sampling code)
            self.combined_progress = self._marimo_backend._combined
            self.chains = chains
            return

        # Rich backend (existing implementation)
        self._marimo_backend = None

        if progressbar_theme is None:
            progressbar_theme = default_progress_theme

        match progressbar:
            case True:
                self.combined_progress = False
                self.full_stats = True
                show_progress = True
            case False:
                self.combined_progress = False
                self.full_stats = True
                show_progress = False
            case "combined":
                self.combined_progress = True
                self.full_stats = False
                show_progress = True
            case "split":
                self.combined_progress = False
                self.full_stats = False
                show_progress = True
            case "combined+stats" | "stats+combined":
                self.combined_progress = True
                self.full_stats = True
                show_progress = True
            case "split+stats" | "stats+split":
                self.combined_progress = False
                self.full_stats = True
                show_progress = True
            case _:
                raise ValueError(
                    "Invalid value for `progressbar`. Valid values are True (default), False (no progress bar), "
                    "one of 'combined', 'split', 'split+stats', or 'combined+stats."
                )

        progress_columns, progress_stats = step_method._progressbar_config(chains)

        self._progress = self.create_progress_bar(
            progress_columns,
            progressbar=progressbar,
            progressbar_theme=progressbar_theme,
        )
        self.progress_stats = progress_stats
        self.update_stats_functions = step_method._make_progressbar_update_functions()

        self._show_progress = show_progress
        self.completed_draws = 0
        self.total_draws = draws + tune
        self.desc = "Sampling chain"
        self.chains = chains

        self._tasks: list[Task] | None = None  # type: ignore[annotation-unchecked]

    def __enter__(self):
        if self._marimo_backend is not None:
            self._marimo_backend.__enter__()
            return self

        self._initialize_tasks()
        return self._progress.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._marimo_backend is not None:
            return self._marimo_backend.__exit__(exc_type, exc_val, exc_tb)

        return self._progress.__exit__(exc_type, exc_val, exc_tb)

    def _initialize_tasks(self):
        if self.combined_progress:
            self.tasks = [
                self._progress.add_task(
                    self.desc.format(self),
                    completed=0,
                    draws=0,
                    total=self.total_draws * self.chains - 1,
                    chain_idx=0,
                    sampling_speed=0,
                    speed_unit="draws/s",
                    failing=False,
                    **{stat: value[0] for stat, value in self.progress_stats.items()},
                )
            ]

        else:
            self.tasks = [
                self._progress.add_task(
                    self.desc.format(self),
                    completed=0,
                    draws=0,
                    total=self.total_draws - 1,
                    chain_idx=chain_idx,
                    sampling_speed=0,
                    speed_unit="draws/s",
                    failing=False,
                    **{stat: value[chain_idx] for stat, value in self.progress_stats.items()},
                )
                for chain_idx in range(self.chains)
            ]

    @staticmethod
    def compute_draw_speed(elapsed, draws):
        speed = draws / max(elapsed, 1e-6)

        if speed > 1 or speed == 0:
            unit = "draws/s"
        else:
            unit = "s/draw"
            speed = 1 / speed

        return speed, unit

    def update(self, chain_idx, is_last, draw, tuning, stats):
        if self._marimo_backend is not None:
            self._marimo_backend.update(chain_idx, is_last, draw, tuning, stats)
            return

        if not self._show_progress:
            return

        self.completed_draws += 1
        if self.combined_progress:
            draw = self.completed_draws
            chain_idx = 0

        elapsed = self._progress.tasks[chain_idx].elapsed
        speed, unit = self.compute_draw_speed(elapsed, draw)

        failing = False
        all_step_stats = {}

        chain_progress_stats = [
            update_stats_fn(step_stats)
            for update_stats_fn, step_stats in zip(self.update_stats_functions, stats, strict=True)
        ]
        for step_stats in chain_progress_stats:
            for key, val in step_stats.items():
                if key == "failing":
                    failing |= val
                    continue
                if not self.full_stats:
                    # Only care about the "failing" flag
                    continue

                if key in all_step_stats:
                    # TODO: Figure out how to integrate duplicate / non-scalar keys, ignoring them for now
                    continue
                else:
                    all_step_stats[key] = val

        self._progress.update(
            self.tasks[chain_idx],
            completed=draw,
            draws=draw,
            sampling_speed=speed,
            speed_unit=unit,
            failing=failing,
            **all_step_stats,
        )

        if is_last:
            self._progress.update(
                self.tasks[chain_idx],
                draws=draw + 1 if not self.combined_progress else draw,
                failing=failing,
                **all_step_stats,
                refresh=True,
            )

    def create_progress_bar(self, step_columns, progressbar, progressbar_theme):
        columns = [TextColumn("{task.fields[draws]}", table_column=Column("Draws", ratio=1))]

        if self.full_stats:
            columns += step_columns

        columns += [
            TextColumn(
                "{task.fields[sampling_speed]:0.2f} {task.fields[speed_unit]}",
                table_column=Column("Sampling Speed", ratio=1),
            ),
            TimeElapsedColumn(table_column=Column("Elapsed", ratio=1)),
            TimeRemainingColumn(table_column=Column("Remaining", ratio=1)),
        ]

        return CustomProgress(
            RecolorOnFailureBarColumn(
                table_column=Column("Progress", ratio=2),
                failing_color="tab:red",
                complete_style=Style.parse("rgb(31,119,180)"),  # tab:blue
                finished_style=Style.parse("rgb(31,119,180)"),  # tab:blue
            ),
            *columns,
            console=Console(theme=progressbar_theme),
            disable=not progressbar,
            include_headers=True,
        )
