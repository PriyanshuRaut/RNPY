"""Registration thread diversity does not establish simultaneous allocation."""

from collections import defaultdict


def detect_parallel_spikes(timeline):
    per_thread = defaultdict(int)
    for entry in timeline:
        per_thread[entry["thread"]] += entry["size"]
    if len(per_thread) <= 1:
        return None
    return {
        "threads": len(per_thread),
        "total_bytes": sum(per_thread.values()),
        "per_thread": dict(per_thread),
        "kind": "multi_thread_registrations",
        "concurrent_allocation_confirmed": False,
        "size_basis": "registered_logical_bytes_not_unique_allocated_bytes",
    }
