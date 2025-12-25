"""
npguard.py

NumPy Memory Guard (MVP)
A NumPy memory observability and explanation tool

Features:
1. Watch NumPy memory behavior
2. Notify users about allocations & temporaries
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

# Global registry to track NumPy array allocations
ArrayRegistry = {}
# Memory allocation tracker
Alloc_Tracker = defaultdict(int)

#To track array size
def array_size(arr: np.ndarray) ->int:
    return arr.nbytes

#Track allocations
def register_array(arr: np.ndarray,label:str = "Unknown"):
    key = id(arr)
    ArrayRegistry[key] = {
        "label": label,
        "size": array_size(arr),
        "shape": arr.shape,
        "dtype": arr.dtype,
        "owndata": arr.flags["OWNDATA"],
        "contiguous": arr.flags["C_CONTIGUOUS"],
    }
    Alloc_Tracker[label] += array_size(arr)
    # NOTE: Alloc_Tracker tracks cumulative allocations, not live memory.

#Cleaner
def cleaner():
    live_ids = {id(obj) for obj in gc.get_objects() if isinstance(obj, np.ndarray)}
    deadKeys = set(ArrayRegistry) - live_ids
    for k in deadKeys:
        del ArrayRegistry[k]

#Watcher
@contextmanager
def memory_watcher(tag = "block",warn_threshold_mb=10):
    tracemalloc.start()
    start_current, start_peak = tracemalloc.get_traced_memory()
    start_snapshot = set(ArrayRegistry.keys())

    yield

    cleaner()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    python_peak_mb = (peak - start_current) / 1024 / 1024
    python_cur_mb = (current - start_current) / 1024 / 1024
    new_arrays = len(set(ArrayRegistry.keys()) - start_snapshot)

    print(f"\n[npguard] Memory Watch: {tag}")
    print(f"  Python current diff: {python_cur_mb:.2f} MB")
    print(f"  Python peak diff:    {python_peak_mb:.2f} MB")
    print(f"  New NumPy arrays:    {new_arrays}")
    if python_peak_mb > warn_threshold_mb:
        emit_warning(tag, python_peak_mb, new_arrays)

#User_Frame
def find_user_frame():
    for frame_info in inspect.stack():
        fname = frame_info.filename
        if "npguard" not in fname and "contextlib" not in fname:
            return frame_info
    return None


#Warning emitter
def emit_warning(tag, peak_mb, array_count):
    frame = find_user_frame()
    location = f"{frame.filename}:{frame.lineno}" if frame else "<unknown>"

    print("\nPotential memory pressure detected")
    print(f"  Location: {location}")
    print(f"  Peak increase: {peak_mb:.2f} MB")
    print(f"  Arrays created: {array_count}")

#suggester
def suggest():
    print("\n[npguard] Suggestions:")

    for info in ArrayRegistry.values():
        if info["owndata"] and info["contiguous"]:
            print(
                f"  • Array '{info['label']}' ({info['size']/1024/1024:.2f} MB): "
                "Consider reusing via `out=` or a scratch buffer."
            )

#Reporter
def report():
    print("\n[npguard] Allocation Summary")
    for label, size in Alloc_Tracker.items():
        print(f"  {label:<12}: {size/1024/1024:.2f} MB")
