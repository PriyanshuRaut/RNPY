"""
np_memory_guard.py

A learning-oriented module that:
- Shows where NumPy creates hidden memory overhead
- Tracks NumPy array allocations
- Demonstrates safer, memory-aware alternatives

This file does NOT modify NumPy internals.
It only observes and optimizes usage patterns.
"""

import numpy as np
import tracemalloc
import weakref
import gc
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
        "dtype": arr.dtype
    }
    Alloc_Tracker[label] += array_size(arr)

#Watcher
@contextmanager
def memory_watcher(tag = "block"):
    tracemalloc.start()
    start_current, start_peak = tracemalloc.get_traced_memory()
    yield
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\n[Memory Watcher: {tag}]")
    print(f"  Current diff: {(current - start_current)/1024:.2f} KB")
    print(f"  Peak diff:    {(peak - start_peak)/1024:.2f} KB")
    print(f"  Live arrays:  {len(ArrayRegistry)}")

#Producing Errors
def error_numpy_operation():
    a = np.random.rand(10_000, 100)
    register_array(a,"array a")
    b = a * 2 + a.mean(axis = 0) - 1
    register_array(b,"array b")
    c = np.ascontiguousarray(b.T)
    register_array(c,"array c")
    return c

#Fixing Errors
def safe_numpy_operation(buffer = None):
    a = np.random.rand(10_000, 100)
    register_array(a,"array a")
    mean = a.mean(axis = 0)
    if buffer is None:
        buffer = np.empty_like(a)
        register_array(buffer,"buffer")
    np.multiply(a,2,out=buffer)
    np.add(buffer,mean,out=buffer)
    np.subtract(buffer,1,out=buffer)
    return buffer

#Report
def memory_report():
    print("\nMemory Allocation Report:\n")
    for label,size in Alloc_Tracker.items():
        print(f"{label:<10}: {size/1024:.2f} KB")

#Execution
if __name__ == "__main__":
    print("\nDemonstrating NumPy Memory Management\n")
    with memory_watcher("Error Operation"):
        out1 = error_numpy_operation()
    gc.collect()
    print("\nCorrect Operation\n")
    with memory_watcher("Safe Operation"):
        out2 = safe_numpy_operation()
    memory_report()