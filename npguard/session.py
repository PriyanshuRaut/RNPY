"""Independent observation state and conservative detector orchestration."""

from copy import deepcopy
import time

from . import state
from .registry import ArrayCollection
from .signals.dtype import detect_dtype_promotion
from .signals.repetition import detect_repeated_allocations
from .signals.temporaries import detect_temporaries
from .signals.threading import detect_parallel_spikes
from .tracing import coordinator


class ObservationSession:
    """Handle yielded by memory_watcher for deliberate worker registration."""

    def __init__(self, tag):
        self.tag = tag
        self.owner = state.execution_owner()
        self._collection = ArrayCollection()
        self._observation = {}
        self._started_at = time.perf_counter()
        self._interval = coordinator.acquire()

    @property
    def observation(self):
        """Defensive snapshot; empty until the session finishes."""
        return deepcopy(self._observation)

    def finish(self, exception_type):
        measurement = coordinator.release(self._interval)
        entries = self._collection.snapshot(close=True)
        end = time.perf_counter()
        errors = []

        def analyze(name, fn, fallback):
            try:
                return fn()
            except Exception as error:
                errors.append({"detector": name, "error_type": type(error).__name__})
                return fallback

        signals = {
            "temporaries": analyze(
                "temporaries",
                lambda: detect_temporaries(entries, self._started_at, end),
                {"count": None, "mb": None, "status": "analysis_failed"},
            ),
            "dtype_promotions": analyze("dtype", lambda: detect_dtype_promotion(entries), []),
            "repeated": analyze("repetition", lambda: detect_repeated_allocations(entries), {}),
            "parallel": analyze("threading", lambda: detect_parallel_spikes(entries), None),
        }
        peak = measurement["session_peak_bytes"]
        self._observation = {
            "tag": self.tag,
            "status": "error" if exception_type else "completed",
            "exception_type": exception_type,
            "duration_seconds": end - self._started_at,
            "peak_mb": peak / 1024**2 if peak is not None else None,
            "measurements": {
                "python_traced": measurement,
                "numpy_data_buffers": {"status": "not_measured"},
                "process_memory": {"status": "not_measured"},
                "registered_arrays": {
                    "count": len(entries),
                    "logical_bytes": sum(e["size"] for e in entries),
                    "unique_buffer_bytes": None,
                },
            },
            "registrations": entries,
            "signals": signals,
            "analysis_errors": errors,
        }
