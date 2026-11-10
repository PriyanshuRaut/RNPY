"""Exception-safe observation with execution-local state."""

from contextlib import contextmanager
import math
from numbers import Real

from . import state
from .registry import reset_registry
from .session import ObservationSession


@contextmanager
def memory_watcher(tag="block", silent=False, warn_threshold_mb=10):
    """Observe a block and yield a handle for explicit worker registration.

    Traced memory is process-wide, not isolated NumPy usage. Registrations belong
    only to the innermost session by default.
    """
    if not isinstance(tag, str):
        raise TypeError("tag must be a string")
    if (
        not isinstance(warn_threshold_mb, Real)
        or not math.isfinite(warn_threshold_mb)
        or warn_threshold_mb < 0
    ):
        raise ValueError("warn_threshold_mb must be a finite non-negative number")
    session = ObservationSession(tag)
    token = state.activate(session)
    exception_type = None
    try:
        yield session
    except BaseException as error:
        exception_type = type(error).__name__
        raise
    finally:
        try:
            session.finish(exception_type)
            observation = session.observation
            state.save_observation(observation)
        finally:
            state.restore(token)
        peak = observation["peak_mb"]
        if not silent and exception_type is None and peak is not None and peak > warn_threshold_mb:
            print("[npguard] Process-wide traced memory exceeded the warning threshold")


def last_observation():
    return state.last_observation()


def reset():
    """Clear this context's last observation and standalone registrations."""
    if state.sessions():
        raise RuntimeError("Cannot reset during an active monitoring session")
    state.clear_observation()
    reset_registry()
