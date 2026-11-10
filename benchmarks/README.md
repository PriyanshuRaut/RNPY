# Reproducible workloads

Install `python -m pip install -e '.[dev]'`, then from the checkout:

```bash
python benchmarks/run.py --size 20000 --repeats 5 --output /tmp/npguard-benchmarks.json
```

Eleven scenarios cover retained owning arrays, shared views, explicit copies,
chained expressions, broadcasting outputs, dtype changes, repeated distinct
arrays, reused buffers, short-lived array objects, nesting, and worker sessions.
Each sample executes the same workload without registration and with capture.
An untimed small warmup precedes each scenario, and GC runs before each timed
sample. Measured timings include the instrumentation, tracing startup, metadata
collection, and final analysis. The baseline never runs a watcher, including in
nested/worker scenarios. Run multiple fresh processes, alternate run order for
serious comparisons, and retain raw data; these short ordered samples are a
starting point, not statistically controlled optimization results.

JSON records runtime/platform versions and raw time, process-traced memory,
registered logical sizes, lifecycle evidence, and diagnostic counts. A null
session peak for nested/worker sessions is intentional. Worker registrations
belong to worker sessions rather than the parent. Logical sizes double-count
shared views; object release does not prove buffer release. No RSS or
NumPy-domain-specific bytes are measured. This is a benchmark harness, not the
future public profiling-report schema, accuracy certification, or regression CLI.
Do not compare different metric types or use noisy wall times as hard CI budgets.

The tests validate evidence and controlled NumPy buffers separately. Later phases
will add domain-level ground truth, confidence calibration, and statistical
regression policy. No benchmark result is committed as a universal performance
claim.
