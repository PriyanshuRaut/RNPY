"""Conditional, evidence-based advice; never modify arrays or application code."""

import math
from numbers import Real

from .core import last_observation
from .log import log


def suggest(temp_threshold_mb: float = 5):
    """Explain registered evidence; legacy threshold controls the traced summary.

    High traced memory alone does not identify NumPy temporaries or their cause.
    """
    if (
        not isinstance(temp_threshold_mb, Real)
        or not math.isfinite(temp_threshold_mb)
        or temp_threshold_mb < 0
    ):
        raise ValueError("temp_threshold_mb must be finite and non-negative")
    observation = last_observation()
    if not observation:
        log.info("npguard", "No observation data available.")
        return
    peak = observation["peak_mb"]
    if peak is None:
        log.info("npguard", "Session traced peak unavailable; see measurement scope and reason.")
    elif peak >= temp_threshold_mb:
        log.info("npguard", f"Process-wide traced peak increase: {peak:.2f} MiB (includes instrumentation)")
    else:
        log.debug("analysis", "Process-wide traced peak increase is below the summary threshold.")
    for error in observation["analysis_errors"]:
        log.warn("analysis", f"{error['detector']} analysis unavailable ({error['error_type']})")
    signals = observation["signals"]
    if signals["parallel"]:
        log.info(
            "signals.parallel",
            "Arrays were registered from multiple threads; concurrent allocation is unproven.",
        )
    for finding in signals["dtype_promotions"]:
        log.info(
            "signals.dtype",
            f"Related arrays differ in dtype: {finding['from']} -> {finding['to']} "
            f"at {finding['callsite']}",
        )
        log.info("suggestion", "Review whether the result dtype is required; preserve numeric range and precision.")
    if signals["repeated"]:
        log.info(
            "signals.repetition",
            f"Distinct owning arrays registered at {len(signals['repeated'])} repeated site(s).",
        )
        log.info(
            "suggestion",
            "If results need not coexist, consider preallocated buffers or out=; "
            "check aliasing and correctness first.",
        )
    if not signals["dtype_promotions"] and not signals["repeated"] and not signals["parallel"]:
        log.info("npguard", "No supported cause identified from registered evidence.")
