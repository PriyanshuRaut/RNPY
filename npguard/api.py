"""Public API; memory units use MiB (bytes / 1024**2)."""

from contextlib import contextmanager
from functools import wraps
import inspect
import sys
from typing import Any, Callable, Dict, Optional

from . import core
from .core import memory_watcher, last_observation, reset
from .registry import register_array as _register_array
from .reporting import report
from .suggestions import suggest
from .log import log


__all__ = [
    "memory_watcher", "register_array", "suggest", "report", "watch", "capture",
    "profile", "last_observation", "reset", "last", "log",
]


def register_array(arr, label="array", *, source=None, session=None):
    """Register ndarray metadata, returning arr unchanged.

    source is an optional caller-asserted dtype provenance relationship. session
    explicitly joins a handle yielded by memory_watcher, including from workers.
    Re-registering a live object in a session is ignored (first metadata wins).
    """
    frame = sys._getframe(1)
    location = (frame.f_code.co_filename, frame.f_lineno)
    del frame
    return _register_array(arr, label, source=source, session=session, _location=location)


def watch(tag: Optional[str] = None, **watcher_kwargs):
    """Observe a synchronous or async function, preserving its metadata/result."""
    def decorator(fn: Callable):
        name = tag if tag is not None else getattr(fn, "__name__", type(fn).__name__)
        if inspect.iscoroutinefunction(fn):
            @wraps(fn)
            async def async_wrapper(*args, **kwargs):
                with memory_watcher(name, **watcher_kwargs):
                    return await fn(*args, **kwargs)
            return async_wrapper

        @wraps(fn)
        def wrapper(*args, **kwargs):
            with memory_watcher(name, **watcher_kwargs):
                return fn(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def capture(tag: str = "block", **watcher_kwargs):
    """Yield a dict populated on exit, including when the observed block raises."""
    box: Dict[str, Any] = {}
    watcher_kwargs.pop("silent", None)  # Capture is always silent.
    session = None
    try:
        with memory_watcher(tag, silent=True, **watcher_kwargs) as session:
            yield box
    finally:
        if session is not None:
            box.update(session.observation)


def profile(fn: Callable, *args, **kwargs):
    """Observe a callable and preserve its return value (await async results)."""
    return watch()(fn)(*args, **kwargs)


def last(key=None):
    """Read a defensive observation snapshot or a dotted dictionary path."""
    observation = core.last_observation()
    if key is None:
        return observation
    current = observation
    for part in key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current
