"""Executable v0.4 examples; all arrays and diagnostic metadata stay local."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import npguard as ng


with ng.memory_watcher("basic", warn_threshold_mb=1):
    source = np.ones(100_000, dtype=np.float32)
    result = source.astype(np.float64)
    ng.register_array(source, "source")
    ng.register_array(result, "result", source=source)
ng.report()
ng.suggest()

with ng.capture("capture") as observation:
    arrays = []
    for _ in range(3):
        arr = np.ones(1000)
        ng.register_array(arr, "repeated")
        arrays.append(arr)
print("Repeated registration sites:", len(observation["signals"]["repeated"]))
print("Temporary-buffer count:", observation["signals"]["temporaries"]["count"])
ng.suggest()


@ng.watch("decorator", silent=True)
def compute():
    return ng.register_array(np.arange(1000), "decorated")


computed = compute()
ng.profile(compute)
print("Last tag:", ng.last("tag"))

with ng.capture("outer") as outer:
    with ng.capture("inner"):
        ng.register_array(np.ones(10))
print("Nested session peak (unavailable):", outer["peak_mb"])

with ng.memory_watcher("workers", silent=True) as session:
    with ThreadPoolExecutor(max_workers=2) as pool:
        worker_arrays = list(pool.map(
            lambda _: ng.register_array(np.ones(1000), "worker", session=session),
            range(2),
        ))
print("Explicit worker registrations:", len(session.observation["registrations"]))
ng.suggest()

try:
    with ng.capture("failure") as failed:
        raise ValueError("example application failure")
except ValueError:
    print("Failure finalized:", failed["status"], failed["exception_type"])

ng.reset()
print("After reset:", ng.last())
print("[npguard] v0.4 demo complete")
