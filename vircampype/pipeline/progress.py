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

import contextlib
import datetime
import logging
import threading

from vircampype.pipeline.logsetup import get_console

__all__ = ["monitor", "report_progress", "spinner", "stop_progress", "track"]

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
        )
        from rich.spinner import Spinner
        from rich.table import Column
        from rich.text import Text

        # The spinner rides inside the description cell, right after the label,
        # so it sits next to the text on every row - short label or long filename
        # - while the text stays flush left. Finished tasks drop the spinner (but
        # keep its width, so the columns to the right do not jump). no_wrap keeps
        # a long filename + spinner on one line (the flex bar shrinks instead).
        class _LabelColumn(ProgressColumn):
            def __init__(self):
                super().__init__(table_column=Column(no_wrap=True))
                self._spinner = Spinner("dots")

            def render(self, task):
                label = Text(task.description, style="progress.description")
                if task.finished:
                    return Text.assemble(label, "  ")
                frame = self._spinner.render(task.get_time())
                return Text.assemble(label, " ", frame)

        # Indeterminate tasks (total=None) carry no count, so the bar and the
        # M/N are always blanked. The "finalizing" spinner is icon + label only;
        # a named spinner (show_elapsed=True, e.g. SCAMP / the tile coadd) also
        # keeps the elapsed timer.
        # ratio=1 keeps the (flex) bar column reserving width even when it is
        # blank for every row (a spinner-only stage), so the timer still pins to
        # the right edge instead of floating after the label.
        class _BarColumn(BarColumn):
            def render(self, task):
                if task.total is None:
                    return Text("")
                return super().render(task)

        class _MofNColumn(MofNCompleteColumn):
            def render(self, task):
                if task.total is None:
                    return Text("")
                # Byte-monitor tasks show a percentage, not raw counts.
                if task.fields.get("percent"):
                    return Text(
                        f"{int(task.completed):d}%", style="progress.percentage"
                    )
                return super().render(task)

        class _ElapsedOrClockColumn(ProgressColumn):
            """Elapsed time while running; wall-clock finish time once done."""

            def render(self, task):
                # Finalizing spinner (indeterminate, no show_elapsed): no timer.
                if task.total is None and not task.fields.get("show_elapsed"):
                    return Text("")
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
            _LabelColumn(),
            _BarColumn(bar_width=None, table_column=Column(ratio=1)),
            _MofNColumn(),
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

    def _prepare(self):
        """Start the live display and drop any leftover "finalizing" spinner.

        Run before adding a fresh task (a loop bar, a batch bar, or a spinner)
        so new activity supersedes the post-loop spinner of the previous stage.
        """
        self._ensure_started()
        self._clear_working()

    def update(self, n_current, n_total, name, d_current, d_total):
        self._prepare()

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
            clock = self._timestamp_now()
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

    def _task(self, task_id):
        """Look up a live task by id (rich offers no public getter)."""
        if self._progress is None:
            return None
        for task in self._progress.tasks:
            if task.id == task_id:
                return task
        return None

    @staticmethod
    def _timestamp_now() -> str:
        """Current wall-clock time as HH:MM:SS (the finish stamp on a done bar)."""
        return datetime.datetime.now().strftime("%H:%M:%S")

    def start_task(self, description, total, **fields):
        """Add a standalone task (determinate bar, or a 1-step spinner) and
        return its id. Used by the shell-command batch stages, which run between
        the ``message_calibration`` loops, so they share the one live display.
        """
        self._prepare()
        return self._progress.add_task(description, total=total, **fields)

    def start_spinner(self, description):
        """Add an indeterminate spinner (icon + label + elapsed; no bar/count).

        Used for a single long-running command (SCAMP, the tile coadd), where a
        determinate bar and an "N/1" count would be meaningless.
        """
        self._prepare()
        return self._progress.add_task(description, total=None, show_elapsed=True)

    def advance_task(self, task_id):
        """Advance a batch task by one finished command."""
        if self._progress is not None and task_id is not None:
            self._progress.advance(task_id, 1)

    def set_task(self, task_id, completed):
        """Set a task's absolute completion (the byte-monitor path)."""
        if self._progress is not None and task_id is not None:
            self._progress.update(task_id, completed=completed)

    def finish_task(self, task_id):
        """Stamp the finish clock and keep the bar rendered.

        On the success path advance() already drove the bar to its total, so we
        deliberately do NOT snap completed=total here: a bar left short by a
        mid-batch failure then stays truthful instead of jumping to full. The
        completed bar persists (like the loop bars) until the next stage output
        calls :func:`stop_progress`, which commits it to the scrollback.
        """
        if self._progress is None or task_id is None:
            return
        self._progress.update(task_id, finished_at=self._timestamp_now())


_driver = _ProgressDriver()


def _in_main_process() -> bool:
    """True only in the genuine main process (never a loky/mp worker).

    A ``prefer="processes"`` joblib worker has its own main thread, so the
    main-thread check below is not enough on its own: a worker process would
    pass it and try to start a second live display. Both signals must agree.
    """
    import multiprocessing

    return (
        multiprocessing.parent_process() is None
        and multiprocessing.current_process().name == "MainProcess"
    )


def _can_drive_live() -> bool:
    """Whether it is safe to render a live bar from here.

    Only from the real main thread of the main process, on a TTY. The rich Live
    is not safe to drive from many threads, and a worker process shares neither
    the parent's display nor its handlers.
    """
    return (
        threading.current_thread() is threading.main_thread()
        and _in_main_process()
        and get_console().is_terminal
    )


def report_progress(
    n_current, n_total, name, d_current=None, d_total=None, display=True
):
    """Report per-file/per-detector progress (file DEBUG always; bar on a TTY).

    ``display=False`` suppresses only the live bar (quiet mode); the DEBUG file
    record is written regardless.
    """
    detail = f" det {d_current}/{d_total}" if d_total is not None else ""
    log.debug(f"processing {n_current}/{n_total} {name}{detail}")

    if display and _can_drive_live():
        _driver.update(n_current, n_total, name, d_current, d_total)


def stop_progress():
    """Finalize the live display: persist completed bars, clear the spinner.

    Called at stage boundaries (the messaging banner/footer helpers) and by the
    worker's top-level handler, so the live region is stopped before any raw
    print() and completed bars are committed to the scrollback.
    """
    _driver.finalize()


@contextlib.contextmanager
def track(label, total):
    """Bar for a main-thread batch of shell commands (e.g. source detection).

    Yields a no-arg ``advance`` callable; call it once per finished command. For
    a single long command pass ``total=1`` (the spinner column animates while it
    runs, e.g. SCAMP / the tile coadd).

    The bar is a no-op that renders nothing - while still yielding a working
    ``advance`` - when ``label`` is None, the batch is empty, off the main
    thread, in a worker process, or off a TTY, so callers can wrap
    unconditionally. The file-side DEBUG record of each command lives in
    ``systemtools`` regardless.
    """
    if label is None or not total or not _can_drive_live():
        yield lambda: None
        return
    task = _driver.start_task(label, total)
    try:
        yield lambda: _driver.advance_task(task)
    finally:
        _driver.finish_task(task)


@contextlib.contextmanager
def monitor(label, total):
    """Percentage bar driven by absolute ``completed`` values (e.g. bytes
    written to an output file while an external command runs).

    Yields a one-arg ``set_completed(value)`` callable taking values in the
    same units as ``total``; the display is scaled to percent. Like
    :func:`track`, it is a no-op that renders nothing - while still yielding a
    working callable - when ``label`` is None, ``total`` is falsy, off the
    main thread, in a worker process, or off a TTY.
    """
    if label is None or not total or not _can_drive_live():
        yield lambda completed: None
        return
    task = _driver.start_task(label, total=100, percent=True)
    try:
        yield lambda completed: _driver.set_task(
            task, min(100.0 * completed / total, 100.0)
        )
    finally:
        _driver.finish_task(task)


@contextlib.contextmanager
def spinner(label):
    """Indeterminate spinner for a single long-running main-thread command.

    Shows the animated icon + label + elapsed timer, with NO bar and NO count
    (an "N/1" would be meaningless), e.g. SCAMP or the tile coadd. A no-op that
    renders nothing when ``label`` is None, off the main thread, in a worker
    process, or off a TTY, so callers can wrap unconditionally.
    """
    if label is None or not _can_drive_live():
        yield
        return
    task = _driver.start_spinner(label)
    try:
        yield
    finally:
        _driver.finish_task(task)
