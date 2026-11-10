# npguard

NPGuard helps explain NumPy memory use through process-traced measurements and
optional array metadata. It observes, analyzes, and suggests; it never rewrites
code, monkey-patches NumPy, or changes array behavior. Application data stays
local. This checkout is **v0.4.0.dev0**, a reliability development version, not a
published stable release or a claim of production readiness.

## Installation

Python **3.10+**, NumPy **1.23.5–2.x** (subject to each NumPy release's Python
support). From this checkout:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e '.[dev]'
```

`pip install npguard` installs the version available on PyPI, which may differ
from this development checkout. See the [compatibility matrix](docs/development.md).

## Observe a NumPy workload

```python
import numpy as np
import npguard as ng

with ng.capture("step") as observation:
    source = np.ones(100_000, dtype=np.float32)
    result = source.astype(np.float64)
    ng.register_array(source, "source")
    ng.register_array(result, "result", source=source)

print(observation["measurements"]["registered_arrays"]["logical_bytes"])
print(observation["signals"]["dtype_promotions"])
ng.report()
ng.suggest()
```

`source=` is optional and asserts a relationship supplied by the caller. Dtype
findings include the two dtypes, shape/size evidence, storage-overlap bounds, and
safe-cast information. They do not prove an implicit NumPy promotion or an
unnecessary copy. High confidence applies to the dtype difference, not its cause.

## What the numbers mean

| Category | v0.4 behavior |
| --- | --- |
| Python-traced memory | Process-wide tracemalloc bytes; includes Python objects, supported NumPy buffers, other threads, and instrumentation |
| NumPy data-buffer allocations | Not measured separately; NumPy-domain snapshots are planned for v0.5 |
| Process memory / RSS | Not measured |
| Registered array sizes | Logical `nbytes`; views/shared buffers can be counted more than once |
| Temporary/intermediate allocations | Unknown; registration does not establish allocation time or buffer lifetime |
| Object lifecycle | Weak references observe whether a registered array object was released during observation; this does not prove its buffer was freed |

`peak_mb` is a compatibility alias for the process-wide traced peak increase
above the session's starting current traced bytes, in **MiB**. It is available
only when NPGuard owns an exclusive, uninterrupted tracing interval. It is
`None` for nested/overlapping sessions or an already-running external tracer.
The process's raw peak remains available under
`measurements.python_traced.process_peak_bytes`, with its scope and attribution
reason. No individual session peak is invented from a shared high-water mark.

## Public APIs

```python
with ng.memory_watcher("block", warn_threshold_mb=10) as session:
    array = np.ones(100)
    ng.register_array(array, "array")

@ng.watch("function", silent=True)
def compute():
    return np.ones(100)

result = ng.profile(compute)
print(ng.last("signals.temporaries.status"))
ng.reset()
```

- `memory_watcher(tag="block", silent=False, warn_threshold_mb=10)` yields a
  session handle. Warnings describe process-traced memory, not proven NumPy
  temporary pressure. Async functions can use `@watch` or await `profile(fn)`.
- `capture(tag="block", **kwargs)` yields a dict filled on exit, including on
  application exceptions. Capture is always silent and propagates exceptions.
- `register_array(arr, label="array", *, source=None, session=None)` returns
  `arr` unchanged. Metadata only; no strong array references. The first
  registration of a live object in a session wins. Register a derived array
  separately. Registration timestamps are not allocation timestamps.
- `last_observation()` and `last(path=None)` return defensive snapshots in the
  current execution context. Unknown dotted paths return `None`.
- `report()` prints active/last-session logical sizes; with no completed session,
  it prints standalone registrations. `suggest(temp_threshold_mb=5)` prints
  conditional evidence-based advice; its legacy threshold controls the traced
  memory summary, not proof of temporary allocations.
- `reset()` clears this context's last result and standalone metadata; it is
  rejected while a session is active. `log` exposes debug/info/warn/error output.

See [API and measurement contracts](docs/measurement-and-api.md) for nesting,
threads, tasks, exceptions, generator limitations, and tracing ownership.

## Threads and nesting

New threads and new asyncio tasks have independent sessions/results. To register
worker arrays in an existing session, share its handle explicitly:

```python
from concurrent.futures import ThreadPoolExecutor

with ng.memory_watcher("workers", silent=True) as session:
    with ThreadPoolExecutor(max_workers=2) as pool:
        arrays = list(pool.map(
            lambda _: ng.register_array(np.ones(100), session=session), range(2)
        ))
```

Join workers before closing the session. Thread diversity alone does not prove
simultaneous allocation. Nested sessions collect their own registrations;
parents' process measurements necessarily include nested work. No NumPy memory
is automatically attributed to a line of user code in v0.4.

## Development and roadmap

```bash
python -m pytest --cov=npguard --cov-report=term-missing
python -m flake8 npguard tests benchmarks --select=E9,F63,F7,F82
python main.py
python benchmarks/run.py --size 20000 --repeats 5 --output /tmp/npguard-benchmarks.json
python -m build
```

- [v0.3.1 audit and architecture decisions](docs/audit-v0.3.1.md)
- [Milestones and acceptance criteria through v1.0](docs/roadmap.md)
- [Migration from v0.3](docs/migration-v0.4.md)
- [Development guide and compatibility matrix](docs/development.md)
- [Benchmark methodology](benchmarks/README.md)
- [Contributing](CONTRIBUTING.md), [changelog](CHANGELOG.md), [security](SECURITY.md)

Automatic allocation tracking, AST-backed diagnostics, a profiling CLI,
JSON/HTML profiling reports, optimization benchmarking, regression budgets, and
native-profiler integrations are later-phase work. The internal AST helper is a
syntactic hint only: it cannot establish that a binary expression uses NumPy.
NPGuard does not identify all native/short-lived allocations or memory leaks.

MIT licensed. Contributions to detectors, testing, documentation, and benchmarks
are welcome. Optional project support: https://patreon.com/PriyanshuRauth.
