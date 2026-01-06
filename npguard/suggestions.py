from .core import last_observation
from .log import log


def suggest(temp_threshold_mb: float = 5):
    """
    Explanation-first suggestions based on the last observation.
    - Noise-compressed via logging
    """
    obs = last_observation()

    if not obs:
        log.info("npguard", "No observation data available.")
        return

    peak_mb = obs.get("peak_mb", 0.0)
    signals = obs.get("signals", {}) or {}

    log.info("npguard", f"Memory analysis completed (peak ~{peak_mb:.2f} MB)")

    if peak_mb < temp_threshold_mb:
        log.debug(
            "analysis",
            "Peak memory below threshold; no significant temporary pressure detected"
        )
        return

    # --------------------------------------------------
    # Signal-based explanations (compressed)
    # --------------------------------------------------

    emitted = False

    if signals.get("parallel"):
        log.warn(
            "signals.parallel",
            "Parallel temporary allocations detected across threads"
        )
        log.info(
            "suggestion",
            "Consider thread-local buffers or avoiding shared temporaries"
        )
        emitted = True

    if signals.get("dtype_promotions"):
        log.warn(
            "signals.dtype",
            "Dtype promotion caused full-size array copies"
        )
        log.info(
            "suggestion",
            "Make dtype conversions explicit to avoid hidden copies"
        )
        emitted = True

    if signals.get("repeated"):
        count = len(signals["repeated"])
        log.warn(
            "signals.repetition",
            f"Repeated allocations detected at {count} site(s)"
        )
        log.info(
            "suggestion",
            "Reuse preallocated buffers inside loops"
        )
        emitted = True

    # --------------------------------------------------
    # Fallback heuristic (v0.2 behavior)
    # --------------------------------------------------

    if not emitted:
        log.warn(
            "signals.heuristic",
            "Memory spike likely caused by chained NumPy expressions"
        )
        log.info(
            "suggestion",
            "Split expressions or use ufuncs with `out=`"
        )
