"""
npguard.py

NumPy Memory Guard (v1)

A NumPy memory observability and explanation tool.

Features:
1. Watch NumPy memory behavior
2. Notify users about memory pressure & temporaries
3. Suggest opt-in ways to reduce memory pressure

This module does NOT modify NumPy internals.
It only observes and explains memory behavior.
"""

import numpy as np
import tracemalloc
import gc
import inspect
from collections import defaultdict
from contextlib import contextmanager

# ---------------------------
# Global State (v1)
# ---------------------------

ArrayRegistry = {}
Alloc_Tracker = defaultdict(int)
_last_observation = {}

# ---------------------------
# Helpers
# ---------------------------

def array_size(arr: np.ndarray) -> int:
    return arr.nbytes


def register_array(arr: np.ndarray, label: str = "array"):
    key = id(arr)
    ArrayRegistry[key] = {
        "label": label,
        "size": arr.nbytes,
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "owndata": arr.flags["OWNDATA"],
        "contiguous": arr.flags["C_CONTIGUOUS"],
    }
    Alloc_Tracker[label] += arr.nbytes


def _cleanup_dead_arrays():
    live_ids = {id(obj) for obj in gc.get_objects() if isinstance(obj, np.ndarray)}
    dead = set(ArrayRegistry) - live_ids
    for k in dead:
        del ArrayRegistry[k]


def _find_user_frame():
    for frame_info in inspect.stack():
        fname = frame_info.filename
        if "npguard" not in fname and "contextlib" not in fname:
            return frame_info
    return None


# ---------------------------
# Watcher (core feature)
# ---------------------------

@contextmanager
def memory_watcher(tag="block", warn_threshold_mb=10):
    tracemalloc.start()
    start_current, _ = tracemalloc.get_traced_memory()
    start_snapshot = set(ArrayRegistry.keys())

    yield

    _cleanup_dead_arrays()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    python_peak_mb = (peak - start_current) / 1024 / 1024
    python_cur_mb = (current - start_current) / 1024 / 1024

    new_arrays = len(set(ArrayRegistry.keys()) - start_snapshot)

    numpy_live_mb = sum(info["size"] for info in ArrayRegistry.values()) / 1024 / 1024
    temp_estimate_mb = max(0.0, python_peak_mb - numpy_live_mb)

    _last_observation.update({
        "tag": tag,
        "python_peak_mb": python_peak_mb,
        "python_current_mb": python_cur_mb,
        "numpy_live_mb": numpy_live_mb,
        "temp_estimate_mb": temp_estimate_mb,
        "new_arrays": new_arrays,
    })

    print(f"\n[npguard] Memory Watch: {tag}")
    print(f"  Python peak diff:    {python_peak_mb:.2f} MB")
    print(f"  Python current diff: {python_cur_mb:.2f} MB")
    print(f"  NumPy live arrays:   {numpy_live_mb:.2f} MB")
    print(f"  Estimated temporaries: {temp_estimate_mb:.2f} MB")

    if python_peak_mb > warn_threshold_mb:
        _emit_warning()


# ---------------------------
# Warning & Explanation
# ---------------------------

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


# ---------------------------
# Suggestions (educational)
# ---------------------------

def _same_shape_groups():
    groups = defaultdict(list)
    for info in ArrayRegistry.values():
        groups[(info["shape"], info["dtype"])].append(info)
    return groups


def _forced_copy_detected():
    for info in ArrayRegistry.values():
        if info["owndata"] and not info["contiguous"]:
            return True
    return False

def suggest(temp_threshold_mb=5):
    print("\n[npguard] Suggestions:")

    obs = _last_observation
    if not obs:
        print("  No observation data available.")
        return

    temp_mb = obs["temp_estimate_mb"]

    if temp_mb < temp_threshold_mb:
        print("  No significant temporary allocations detected.")
        return

    print(f"  This block allocated ~{temp_mb:.2f} MB in temporary arrays.")

    # 1 Chained expression detection
    if temp_mb > obs["numpy_live_mb"] * 2:
        print(
            "  • Likely cause: chained NumPy expressions creating intermediate arrays."
        )
        print(
            "    Suggestion: split expressions into steps or use ufuncs with `out=`."
        )

    # 2️ Same-shape allocation reuse hint
    shape_groups = _same_shape_groups()
    for (shape, dtype), items in shape_groups.items():
        if len(items) >= 2:
            print(
                f"  • Detected {len(items)} arrays with same shape {shape} and dtype {dtype}."
            )
            print(
                "    Suggestion: consider reusing a preallocated buffer for these arrays."
            )
            break  # avoid spam

    # 3️ Forced copy hint
    for info in ArrayRegistry.values():
        if info["owndata"] and info["contiguous"]:
            continue
        if info["owndata"]:
            print(
                f"  • Array '{info['label']}' appears to be a forced copy."
            )
            print(
                "    Suggestion: avoid operations that require contiguity (e.g. ascontiguousarray) if possible."
            )
            break

# ---------------------------
# Reporting
# ---------------------------

def report():
    print("\n[npguard] Allocation Summary (cumulative)")
    for label, size in Alloc_Tracker.items():
        print(f"  {label:<12}: {size/1024/1024:.2f} MB")
