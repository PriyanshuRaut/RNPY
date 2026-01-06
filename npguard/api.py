"""
npguard.api

Public API façade for npguard.
"""

from contextlib import contextmanager
from typing import Callable, Any, Dict

# ---- Core execution ----
from .core import memory_watcher as _memory_watcher
from .core import last_observation as _last_observation

# ---- Registry / state ----
from .registry import register_array as _register_array
from .registry import reset_registry as _reset_registry

# ---- Explanation / reporting ----
from .suggestions import suggest as _suggest
from .reporting import report as _report
from npguard import core

# --- Custom Log ---
from .log import log


# ==========================================================
# Core v0.2 APIs (guaranteed stable)
# ==========================================================

def memory_watcher(tag: str = "block", **kwargs):
    """
    Context manager to observe NumPy memory behavior.

    Compatible with v0.2:
    - tag
    - warn_threshold_mb
    - silent
    """
    return _memory_watcher(tag=tag, **kwargs)


def register_array(arr, label: str = "array"):
    """
    Register a NumPy array for observability.

    This is explicit and opt-in by design.
    """
    return _register_array(arr, label)


def suggest(*, temp_threshold_mb: float = 5):
    """
    Print explanation-first suggestions based on the last observation.
    """
    return _suggest(temp_threshold_mb=temp_threshold_mb)


def report():
    """
    Print cumulative allocation summary.
    """
    return _report()


# ==========================================================
# State / Control APIs
# ==========================================================

def last_observation() -> Dict[str, Any]:
    """
    Return a read-only copy of the most recent observation.
    """
    obs = _last_observation()
    return dict(obs) if obs else {}


def reset():
    """
    Reset live observability state.

    Does NOT clear cumulative allocation history unless documented.
    """
    core.reset()


# ==========================================================
# Ergonomic APIs (v0.2-compatible)
# ==========================================================

def watch(tag: str | None = None, **watcher_kwargs):
    """
    Decorator to observe a function as a single memory block.

    Example:
        @npguard.watch("step1", warn_threshold_mb=10)
        def fn(): ...
    """
    def decorator(fn: Callable):
        def wrapper(*args, **kwargs):
            name = tag or fn.__name__
            with _memory_watcher(tag=name, **watcher_kwargs):
                return fn(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def capture(tag: str = "block", **watcher_kwargs):
    """
    Silent context manager that captures observation data.

    Example:
        with npguard.capture("step") as obs:
            ...
        print(obs)
    """
    box: Dict[str, Any] = {}
    with _memory_watcher(tag=tag, silent=True, **watcher_kwargs):
        yield box
    box.update(last_observation())


def profile(fn: Callable, *args, **kwargs):
    """
    Profile a callable using memory_watcher.

    Returns the callable's original return value.
    """
    with _memory_watcher(tag=fn.__name__):
        return fn(*args, **kwargs)

def last(key=None):
    obs = last_observation()
    if key is None:
        return obs

    cur = obs
    for part in key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur
