"""Rich progress rendering for the per-file / per-detector processing loops.

The pipeline reports progress through a single chokepoint
(``tools.messaging.message_calibration``); this module turns those calls into a
live rich progress bar. It is presentation only and never affects processing.

Behaviour:
- The bar fills the line (``bar_width=None`` + ``expand``) so the count and the
  time column stay pinned to the right edge regardless of description length.
- Completed bars persist (``transient=False``); the time column shows elapsed
  while running and the wall-clock finish time once a loop completes.
- When a loop finishes, the finished bar stays and an animated "finalizing"
  spinner is shown for any post-loop work, until the next stage output
  finalizes the display (see :func:`stop_progress`, called by the messaging
  banner/footer helpers and the worker's top-level handler).

Safety rules:
- The live bar is driven ONLY from the main thread on a TTY. Calls from joblib
  worker threads, or when stdout is not a TTY, fall back to a DEBUG log line
  (handlers are thread-safe; rich Live is not safe to drive from many threads).
- The progress shares the one console singleton with the logging RichHandler,
  so log records render cleanly above the live bar.
"""

import datetime
import logging
import threading

from vircampype.pipeline.logsetup import get_console

__all__ = ["report_progress", "stop_progress"]

log = logging.getLogger(__name__)


class _ProgressDriver:
    """Owns a single rich Progress, updated from the main thread only."""

    def __init__(self):
        self._progress = None
        self._outer = None
        self._inner = None
        self._working = None  # animated spinner shown during post-loop work
        self._outer_total = None
        self._inner_total = None

    def _build(self):
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            ProgressColumn,
            SpinnerColumn,
            TextColumn,
        )
        from rich.text import Text

        class _ElapsedOrClockColumn(ProgressColumn):
            """Elapsed time while running; wall-clock finish time once done."""

            def render(self, task):
                finished_at = task.fields.get("finished_at")
                if finished_at:
                    return Text(finished_at, style="progress.elapsed")
                elapsed = task.elapsed
                if elapsed is None:
                    return Text("--:--", style="progress.elapsed")
                minutes, seconds = divmod(int(elapsed), 60)
                hours, minutes = divmod(minutes, 60)
                return Text(
                    f"{hours:d}:{minutes:02d}:{seconds:02d}",
                    style="progress.elapsed",
                )

        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            _ElapsedOrClockColumn(),
            console=get_console(),
            expand=True,
        )

    def _ensure_started(self):
        if self._progress is None:
            self._progress = self._build()
            self._progress.start()

    def _clear_inner(self):
        if self._inner is not None:
            self._progress.remove_task(self._inner)
        self._inner = None
        self._inner_total = None

    def _clear_working(self):
        if self._working is not None and self._progress is not None:
            self._progress.remove_task(self._working)
        self._working = None

    def update(self, n_current, n_total, name, d_current, d_total):
        self._ensure_started()
        # Fresh loop activity: drop any leftover "finalizing" spinner.
        self._clear_working()

        # Outer (per-file / per-group) task. A changed total starts a fresh one.
        if self._outer is None or self._outer_total != n_total:
            if self._outer is not None:
                self._progress.remove_task(self._outer)
            self._outer = self._progress.add_task(name, total=n_total)
            self._outer_total = n_total
            self._clear_inner()
        self._progress.update(self._outer, completed=n_current, description=name)

        # Optional inner (per-detector) task.
        if d_total is not None:
            if self._inner is None or self._inner_total != d_total:
                self._clear_inner()
                self._inner = self._progress.add_task("  detectors", total=d_total)
                self._inner_total = d_total
            self._progress.update(self._inner, completed=d_current)

        # Loop finished: stamp the finish clock, keep the bar, and show a
        # "finalizing" spinner for any post-loop work. The bar is finalized
        # (persisted) by the next stage output via stop_progress().
        if n_current >= n_total and (d_total is None or d_current >= d_total):
            clock = datetime.datetime.now().strftime("%H:%M:%S")
            self._progress.update(self._outer, finished_at=clock)
            if self._inner is not None:
                self._progress.update(self._inner, finished_at=clock)
            self._working = self._progress.add_task("finalizing", total=None)
            # Forget the finished tasks so the next loop adds fresh ones while
            # the completed bars stay rendered above the spinner.
            self._outer = None
            self._inner = None
            self._outer_total = None
            self._inner_total = None

    def finalize(self):
        """Persist completed bars, drop the spinner, and stop the live display."""
        if self._progress is not None:
            self._clear_working()
            self._progress.stop()
        self._progress = None
        self._outer = None
        self._inner = None
        self._working = None
        self._outer_total = None
        self._inner_total = None


_driver = _ProgressDriver()


def report_progress(n_current, n_total, name, d_current=None, d_total=None):
    """Report per-file/per-detector progress (file DEBUG always; bar on a TTY)."""
    detail = f" det {d_current}/{d_total}" if d_total is not None else ""
    log.debug(f"processing {n_current}/{n_total} {name}{detail}")

    if (
        threading.current_thread() is threading.main_thread()
        and get_console().is_terminal
    ):
        _driver.update(n_current, n_total, name, d_current, d_total)


def stop_progress():
    """Finalize the live display: persist completed bars, clear the spinner.

    Called at stage boundaries (the messaging banner/footer helpers) and by the
    worker's top-level handler, so the live region is stopped before any raw
    print() and completed bars are committed to the scrollback.
    """
    _driver.finalize()
