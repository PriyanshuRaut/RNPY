"""Human-readable registered logical sizes; no allocation-size claims."""

from . import state
from .registry import current_collection


def report():
    """Summarize the active/last session, or standalone registrations initially."""
    if state.sessions():
        entries = current_collection().snapshot()
    else:
        observation = state.last_observation()
        entries = observation["registrations"] if observation else current_collection().snapshot()
    print("\n[npguard] Registered array sizes (logical; shared buffers may be counted twice)")
    if not entries:
        print("  No arrays registered.")
        return
    totals = {}
    for info in entries:
        totals[info["label"]] = totals.get(info["label"], 0) + info["size"]
    for label, size in totals.items():
        print(f"  {label:<12}: {size / 1024**2:.2f} MiB")
