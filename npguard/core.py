"""
npguard.py

NumPy Memory Guard (v0.3)

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
import tracemalloc
import time
from contextlib import contextmanager

from .registry import ArrayRegistry, AllocationTimeline, reset_registry
from .signals.threading import detect_parallel_spikes
from .signals.dtype import detect_dtype_promotion
from .signals.repetition import detect_repeated_allocations
from .signals.temporaries import detect_temporaries

_last = {}

@contextmanager
def memory_watcher(tag="block", silent=False, warn_threshold_mb=10):
    tracemalloc.start()
    start_mem, _ = tracemalloc.get_traced_memory()
    start_time = time.perf_counter()

    yield

    _, peak = tracemalloc.get_traced_memory()
    end_time = time.perf_counter()
    tracemalloc.stop()

    peak_mb = (peak - start_mem) / 1024 / 1024

    temporaries = detect_temporaries(
        AllocationTimeline,
        start_time,
        end_time
    )

    temp_mb = sum(t["size"] for t in temporaries) / 1024 / 1024

    signals = {
        "parallel": detect_parallel_spikes(AllocationTimeline),
        "dtype_promotions": detect_dtype_promotion(ArrayRegistry),
        "repeated": detect_repeated_allocations(ArrayRegistry),
        "temporaries": {
            "count": len(temporaries),
            "mb": temp_mb,
        }
    }

    _last.clear()
    _last.update({
        "tag": tag,
        "peak_mb": peak_mb,
        "signals": signals,
    })

    if not silent and peak_mb > warn_threshold_mb:
        print("[npguard] Memory spike detected")


def last_observation():
    return dict(_last)


def reset():
    _last.clear()
    reset_registry()
