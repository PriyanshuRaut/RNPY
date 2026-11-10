"""Manage process-global tracemalloc ownership without resetting caller peaks."""

from dataclasses import dataclass
import threading
import tracemalloc


@dataclass(eq=False)
class TraceInterval:
    baseline_bytes: int
    owns_interval: bool
    overlapping: bool = False


class TraceCoordinator:
    def __init__(self):
        self._lock = threading.RLock()
        self._active = set()
        self._owns_tracer = False

    def acquire(self):
        with self._lock:
            if not self._active:
                self._owns_tracer = not tracemalloc.is_tracing()
                if self._owns_tracer:
                    tracemalloc.start()
            for active in self._active:
                active.overlapping = True
            interval = TraceInterval(
                baseline_bytes=tracemalloc.get_traced_memory()[0],
                owns_interval=self._owns_tracer and not self._active,
                overlapping=bool(self._active),
            )
            self._active.add(interval)
            return interval

    def release(self, interval):
        with self._lock:
            tracing = tracemalloc.is_tracing()
            current, peak = tracemalloc.get_traced_memory()
            attributable = tracing and interval.owns_interval and not interval.overlapping
            result = {
                "available": tracing,
                "scope": "process",
                "baseline_bytes": interval.baseline_bytes,
                "current_bytes": current if tracing else None,
                "net_bytes": current - interval.baseline_bytes if tracing else None,
                "process_peak_bytes": peak if tracing else None,
                "session_peak_bytes": max(0, peak - interval.baseline_bytes) if attributable else None,
                "overlapping_sessions": interval.overlapping,
                "tracer_owned_by_npguard": self._owns_tracer,
                "session_peak_unavailable_reason": (
                    None if attributable else
                    "tracing_stopped_externally" if not tracing else
                    "overlapping_sessions" if interval.overlapping else
                    "preexisting_tracer"
                ),
            }
            self._active.remove(interval)
            if not self._active:
                if self._owns_tracer and tracing:
                    tracemalloc.stop()
                self._owns_tracer = False
            return result


coordinator = TraceCoordinator()
