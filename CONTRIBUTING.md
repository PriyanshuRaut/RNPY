# Contributing to NPGuard

Contributions to diagnostics, reports, tests, examples, benchmarks, and docs are
welcome. Follow the [Code of Conduct](CODE_OF_CONDUCT.md). Start with the
[development guide](docs/development.md), [measurement contracts](docs/measurement-and-api.md),
and [roadmap](docs/roadmap.md).

## Local workflow

1. Fork/clone the project and create a feature branch.
2. Use Python 3.10+ with a virtual environment, then install:
   `python -m pip install -e '.[dev]'`.
3. Add a failing test for the problem. Include negative cases for detector rules
   and lifecycle/concurrency/error tests for state changes.
4. Make a focused change. Keep NumPy as the only core runtime dependency unless
   a new dependency is justified. Never monkey-patch NumPy, rewrite user code,
   or apply optimization advice automatically.
5. Run `python -m pytest --cov=npguard --cov-report=term-missing` and
   `python -m flake8 npguard tests benchmarks --select=E9,F63,F7,F82`.
   CI enforces 85% branch-aware package coverage; useful tests matter more than
   chasing a number. Run `python main.py`, relevant benchmarks, and
   `python -m build` for packaging changes.
6. Document public behavior, evidence limits, compatibility changes, and a
   changelog entry. Open a PR describing the problem, resulting behavior,
   validation, and limitations.

Explain every memory metric: Python-traced bytes, NumPy-domain bytes, RSS,
registered logical sizes, and estimates are distinct. Do not turn registration
into proof of allocation/freeing or infer causal optimizations from a high peak.
Never fabricate accuracy or benchmark results; record raw samples and runtime
metadata using the [benchmark harness](benchmarks/README.md).

## Issues and review

Bug reports should include a minimal reproducer, Python/NumPy/OS versions,
expected/actual behavior, and whether another tracer is running. Redact private
paths, labels, application data, and secrets. Discuss large API/architecture
changes before implementing unrelated roadmap phases. Keep work small enough to
review, with one clear evidence contract per detector.

Use the [security policy](SECURITY.md) for sensitive reports. Maintainers review
compatibility and evidence quality before release. Publishing packages, creating
public releases, or merging requires explicit maintainer approval.
