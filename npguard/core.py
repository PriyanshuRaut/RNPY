"""
npguard.py

NumPy Memory Guard (v0.2)

A NumPy memory observability and explanation tool.

Features:
1. Watch NumPy memory behavior
2. Notify users about memory pressure & temporaries
3. Suggest opt-in ways to reduce memory pressure

This module does NOT modify NumPy internals.
It only observes and explains memory behavior.
"""

"""
Structure of npguard
npguard.py
│
├─ State
│   ├─ ArrayRegistry
│   ├─ Alloc_Tracker
│   └─ _last_observation
│
├─ Collection layer
│   ├─ register_array()
│   ├─ _cleanup_dead_arrays()
│
├─ Analysis layer (signals)
│   ├─ _estimate_temporaries()
│   ├─ _detect_repeated_allocations()
│   ├─ _detect_broadcasting()
│
├─ Presentation layer
│   ├─ memory_watcher()
│   ├─ _emit_warning()
│   └─ suggest()
│
└─ Reporting
    └─ report()

"""

import numpy as np
import tracemalloc
import gc
import inspect
from collections import defaultdict
from contextlib import contextmanager

# ===========================
# Global State
# ===========================

ArrayRegistry = {}
Alloc_Tracker = defaultdict(int)
_last_observation = {}
_temp_arrays_created = 0


# ===========================
# Collection Layer
# ===========================

def register_array(arr: np.ndarray, label: str = "array"):
    global _temp_arrays_created

    frame = inspect.stack()[1]
    callsite = f"{frame.filename}:{frame.lineno}"

    ArrayRegistry[id(arr)] = {
        "label": label,
        "size": arr.nbytes,
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "owndata": arr.flags["OWNDATA"],
        "contiguous": arr.flags["C_CONTIGUOUS"],
        "callsite": callsite,
    }

    Alloc_Tracker[label] += arr.nbytes
    _temp_arrays_created += 1


def _cleanup_dead_arrays():
    live_ids = {id(o) for o in gc.get_objects() if isinstance(o, np.ndarray)}
    for k in set(ArrayRegistry) - live_ids:
        del ArrayRegistry[k]


# ===========================
# Analysis Layer (Signals)
# ===========================

def _detect_repeated_allocations():
    groups = defaultdict(list)
    for info in ArrayRegistry.values():
        key = (info["shape"], info["dtype"], info["callsite"])
        groups[key].append(info)

    return {k: v for k, v in groups.items() if len(v) >= 2}


def _estimate_temporaries(python_peak_mb, numpy_live_mb):
    return max(0.0, python_peak_mb - numpy_live_mb)


def _find_user_frame():
    for frame in inspect.stack():
        if "npguard" not in frame.filename and "contextlib" not in frame.filename:
            return frame
    return None


# ===========================
# API
# ===========================

#watcher
@contextmanager
def memory_watcher(tag="block", warn_threshold_mb=10, silent=False):
    global _temp_arrays_created

    tracemalloc.start()
    start_current, _ = tracemalloc.get_traced_memory()
    start_temp_arrays = _temp_arrays_created

    yield

    _cleanup_dead_arrays()

    arrays_created = _temp_arrays_created - start_temp_arrays
    arrays_alive = len(ArrayRegistry)
    temp_array_count = max(0, arrays_created - arrays_alive)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    python_peak_mb = (peak - start_current) / 1024 / 1024
    python_cur_mb = (current - start_current) / 1024 / 1024
    numpy_live_mb = sum(i["size"] for i in ArrayRegistry.values()) / 1024 / 1024
    temp_estimate_mb = _estimate_temporaries(python_peak_mb, numpy_live_mb)

    _last_observation.update({
        "tag": tag,
        "python_peak_mb": python_peak_mb,
        "python_current_mb": python_cur_mb,
        "numpy_live_mb": numpy_live_mb,
        "temp_estimate_mb": temp_estimate_mb,
        "temp_array_count": temp_array_count,
    })
    if not silent:
        print(f"\n[npguard] Memory Watch: {tag}")
        print(f"  Python peak diff:    {python_peak_mb:.2f} MB")
        print(f"  Python current diff: {python_cur_mb:.2f} MB")
        print(f"  NumPy live arrays:   {numpy_live_mb:.2f} MB")
        print(f"  Estimated temporaries: {temp_estimate_mb:.2f} MB")
        print(f"  Estimated temporary arrays: {temp_array_count}")

    if python_peak_mb > warn_threshold_mb and not silent:
        _emit_warning()

#last observation
def last_observation():
    """
    Return the most recent memory observation (read-only).
    """
    return dict(_last_observation)

#reset
def reset():
    """
    Reset live state (does not clear cumulative allocation stats).
    """
    ArrayRegistry.clear()
    _last_observation.clear()


#watch
def watch(tag=None, **watcher_kwargs):
    """
    Decorator to observe a function as a single block.
    """
    def decorator(fn):
        def wrapper(*args, **kwargs):
            name = tag or fn.__name__
            with memory_watcher(name, **watcher_kwargs):
                return fn(*args, **kwargs)
        return wrapper
    return decorator

#capture
@contextmanager
def capture(tag="block", **kwargs):
    container = {}

    with memory_watcher(tag, silent=True, **kwargs):
        yield container

    container.update(_last_observation)


#profile
def profile(fn, *args, **kwargs):
    """
    Profile a callable using memory_watcher.
    """
    with memory_watcher(fn.__name__):
        return fn(*args, **kwargs)


# ===========================
# Explanation & Suggestions
# ===========================

def _emit_warning():
    obs = _last_observation
    frame = _find_user_frame()
    location = f"{frame.filename}:{frame.lineno}" if frame else "<unknown>"

    print("\n⚠️  Memory pressure detected")
    print(f"  Location: {location}")
    print(f"  Peak memory increase: {obs['python_peak_mb']:.2f} MB")

    if obs["temp_estimate_mb"] > 5:
        print(f"  Likely temporary allocations: ~{obs['temp_estimate_mb']:.2f} MB")
        print("  Cause: chained NumPy operations or broadcasting")


def suggest(temp_threshold_mb=5):
    print("\n[npguard] Suggestions:")

    obs = _last_observation
    if not obs:
        print("  No observation data available.")
        return

    if obs["temp_array_count"] >= 3:
        print(
            f"  This block created at least {obs['temp_array_count']} temporary arrays,"
        )

    if obs["temp_estimate_mb"] < temp_threshold_mb:
        print("  No significant temporary allocations detected.")
        return

    print(f"  allocating ~{obs['temp_estimate_mb']:.2f} MB in total.")

    if obs["temp_estimate_mb"] > obs["numpy_live_mb"] * 2:
        print(
            "  • Likely cause: chained NumPy expressions creating intermediate arrays."
        )
        print(
            "    Suggestion: split expressions or use ufuncs with `out=`."
        )

    repeats = _detect_repeated_allocations()
    for (shape, dtype, site), items in repeats.items():
        print(
            f"  • Detected {len(items)} repeated allocations of shape {shape} "
            f"and dtype {dtype} at {site}."
        )
        print(
            "    Suggestion: reuse a preallocated buffer inside loops."
        )
        break


# ===========================
# Reporting
# ===========================

def report():
    print("\n[npguard] Allocation Summary (cumulative)")
    for label, size in Alloc_Tracker.items():
        print(f"  {label:<12}: {size/1024/1024:.2f} MB")
