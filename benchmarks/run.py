"""Reproducible raw workload measurements; no pass/fail performance claims.

Install the project first: python -m pip install -e '.[dev]'.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import platform
import sys
import time
from importlib.metadata import version

import numpy as np
import npguard as ng


def retained(size, register):
    arr = np.ones(size, dtype=np.float64)
    register(arr, "retained")
    return arr


def shared_view(size, register):
    arr = retained(size, register)
    view = arr[::2]
    register(view, "view")
    return arr, view


def copied_buffer(size, register):
    arr = retained(size, register)
    copied = arr.copy()
    register(copied, "copy", source=arr)
    return arr, copied


def chained_expression(size, register):
    arr = retained(size, register)
    result = arr * 2 + arr.mean() - 1
    register(result, "result")
    return arr, result


def broadcasting_output(size, register):
    column = np.ones((size, 1))
    row = np.arange(8)
    result = column + row
    for arr in (column, row, result):
        register(arr)
    return column, row, result


def dtype_conversion(size, register):
    source = np.ones(size, dtype=np.float32)
    result = source.astype(np.float64)
    register(source, "source")
    register(result, "result", source=source)
    return source, result


def repeated_distinct(size, register):
    arrays = []
    for _ in range(5):
        arr = np.ones(size)
        register(arr)
        arrays.append(arr)
    return arrays


def reused_buffer(size, register):
    arr = np.zeros(size)
    for _ in range(5):
        np.add(arr, 1, out=arr)
        register(arr)
    return arr


def short_lived(size, register):
    for _ in range(5):
        arr = np.ones(size)
        register(arr)
        del arr


def nested(size, register):
    outer = retained(size, register)
    context = ng.capture("nested-inner") if register is ng.register_array else nullcontext()
    with context:
        inner = retained(size, register)
    return outer, inner


def threads(size, register):
    def worker(_):
        context = ng.capture("worker") if register is ng.register_array else nullcontext()
        with context:
            return retained(size, register)

    with ThreadPoolExecutor(max_workers=2) as pool:
        return list(pool.map(worker, range(2)))


SCENARIOS = {
    "retained_owning": retained,
    "shared_view": shared_view,
    "copied_buffer": copied_buffer,
    "chained_expression": chained_expression,
    "broadcasting_output": broadcasting_output,
    "dtype_conversion": dtype_conversion,
    "repeated_distinct": repeated_distinct,
    "reused_buffer": reused_buffer,
    "short_lived": short_lived,
    "nested_sessions": nested,
    "threaded_sessions": threads,
}


def no_registration(arr, *args, **kwargs):
    return arr


def run(size, repeats):
    results = []
    for name, workload in SCENARIOS.items():
        samples = []
        workload(min(size, 100), no_registration)  # Untimed warmup.
        for _ in range(repeats):
            gc.collect()
            started = time.perf_counter()
            result = workload(size, no_registration)
            baseline_seconds = time.perf_counter() - started
            del result
            gc.collect()
            started = time.perf_counter()
            with ng.capture(name) as observation:
                result = workload(size, ng.register_array)
            observed_seconds = time.perf_counter() - started
            del result
            samples.append({
                "baseline_seconds": baseline_seconds,
                "observed_seconds_including_instrumentation": observed_seconds,
                "python_traced": observation["measurements"]["python_traced"],
                "registered_arrays": observation["measurements"]["registered_arrays"],
                "released_registered_objects": observation["signals"]["temporaries"]["released_array_objects"],
                "repeated_registration_sites": len(observation["signals"]["repeated"]),
                "dtype_findings": len(observation["signals"]["dtype_promotions"]),
                "analysis_errors": observation["analysis_errors"],
            })
            ng.reset()
        results.append({"name": name, "samples": samples})
    return {
        "schema": "npguard-benchmark-v0.4",
        "python": sys.version,
        "numpy": np.__version__,
        "npguard": version("npguard"),
        "platform": platform.platform(),
        "size": size,
        "repeats": repeats,
        "scenarios": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=20_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.size < 1 or args.repeats < 1:
        parser.error("size and repeats must be positive")
    results = run(args.size, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Measured {len(SCENARIOS)} scenarios x {args.repeats} samples -> {args.output}")


if __name__ == "__main__":
    main()
