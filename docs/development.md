# Development guide

## Setup and validation

Use the existing checkout in isolated cloud tasks; do not create a worktree
unless requested. Create a virtual environment and install `python -m pip
install -e '.[dev]'`. NumPy is the only runtime dependency; test/build tools are
optional extras. Run from the repository root:

```bash
python -m pytest --cov=npguard --cov-report=term-missing
python -m flake8 npguard tests benchmarks --select=E9,F63,F7,F82 --count
python main.py
python benchmarks/run.py --size 20000 --repeats 3 --output /tmp/npguard-benchmarks.json
python -m build
```

Coverage includes branches and has an 85% minimum. Prefer tests with positive
AND negative evidence, observable state restoration, and controlled workloads
over tests that mirror implementation details. Memory tests use controlled
NumPy arrays; timing benchmarks have no brittle pass/fail speed thresholds.
Benchmarks write raw samples to the requested local path. No external service,
account, GPU, API credential, or network destination is needed to run the code.
Dependency installation uses the normal Python package index.

## Compatibility policy and configured CI matrix

Runtime requirement: CPython >=3.10 and NumPy >=1.23.5,<3. NumPy's own Python
compatibility still applies. CI targets:

| OS | Python | NumPy |
| --- | --- | --- |
| Linux | 3.10 | 1.23.5 (minimum) |
| Linux | 3.11 | 1.26.4 |
| Linux | 3.12 | 1.26.4 and latest compatible 2.x |
| Linux | 3.13, 3.14 | latest compatible 2.x |
| Windows, macOS | 3.12 | latest compatible 2.x |

The workflows are configuration, not evidence that hosted jobs ran. Local
validation is recorded separately in the phase report. Other interpreters,
free-threaded builds, and Python versions outside this tested matrix are not
certified. Future dependency-major versions need an explicit compatibility
review. During 0.x, preserve public calls where feasible and document necessary
correctness changes; v1.0 will freeze the stable API/serialization policy.

## Architecture and extension points

- `api.py`: public ergonomics and caller locations; `core.py`: lifecycle.
- `state.py`: context ownership and defensive snapshots.
- `tracing.py`: global tracer coordination; never reset external peaks.
- `registry.py`: weakref lifecycle metadata and explicit source relationships.
- `session.py`: observation assembly, measurement categories, detector errors.
- `signals/`: independently testable evidence analyzers.
- `reporting.py`, `suggestions.py`: qualified presentation, never auto-fixes.
- `static.py`: legacy internal syntax hints, not NumPy execution evidence.

A new detector should consume metadata, state its evidence and uncertainty, and
have both confirming and counterexample tests. Do not retain application arrays
or inspect their values. Add dedicated metric names for new measurement sources;
never silently replace Python-traced bytes with RSS or NumPy-domain bytes.
Consult the [audit/decisions](audit-v0.3.1.md), [API contracts](measurement-and-api.md),
and [roadmap](roadmap.md) before expanding scope.
