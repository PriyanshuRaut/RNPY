"""Lifecycle evidence cannot establish temporary buffer allocation lifetimes."""


def detect_temporaries(timeline, start_time, end_time):
    released = [
        entry for entry in timeline
        if entry.get("released_at") is not None
        and start_time <= entry["registered_at"] <= entry["released_at"] <= end_time
    ]
    return {
        "count": None,
        "mb": None,
        "status": "unmeasured",
        "released_array_objects": len(released),
        "released_logical_mb": sum(e["size"] for e in released) / 1024**2,
        "buffer_release_confirmed": False,
        "explanation": (
            "Weak references observe registered array-object release, not buffer "
            "allocation or release. Temporary-buffer counts and sizes are unknown."
        ),
    }
