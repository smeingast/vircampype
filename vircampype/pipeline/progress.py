"""Rich progress rendering for the per-file / per-detector processing loops.

The pipeline reports progress through a single chokepoint
(``tools.messaging.message_calibration``); this module turns those calls into a
live rich progress bar. It is presentation only and never affects processing.

Safety rules:
- The live bar is driven ONLY from the main thread on a TTY. Calls from joblib
  worker threads, or when stdout is not a TTY, fall back to a DEBUG log line
  (handlers are thread-safe; rich Live is not safe to drive from many threads).
- The progress shares the one console singleton with the logging RichHandler,
  so log records render cleanly above the live bar.
- ``n_current``/``n_total`` (and ``d_current``/``d_total``) drive the task
  lifecycle: a new total starts a fresh task; reaching the total stops the bar.
"""

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
        self._outer_total = None
        self._inner_total = None

    def _ensure_started(self):
        if self._progress is None:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
            )

            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                # Expand the bar to fill the line so the count + elapsed time
                # stay pinned to the right edge regardless of description length.
                BarColumn(bar_width=None),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=get_console(),
                transient=True,
                expand=True,
            )
            self._progress.start()

    def _clear_inner(self):
        if self._inner is not None:
            self._progress.remove_task(self._inner)
        self._inner = None
        self._inner_total = None

    def update(self, n_current, n_total, name, d_current, d_total):
        self._ensure_started()

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

        # Stop when the loop has finished.
        if n_current >= n_total and (d_total is None or d_current >= d_total):
            self.stop()

    def stop(self):
        if self._progress is not None:
            self._progress.stop()
        self._progress = None
        self._outer = None
        self._inner = None
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
    """Stop and clear any active progress bar (e.g. on completion or abort)."""
    _driver.stop()
