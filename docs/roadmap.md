# NPGuard implementation roadmap

The project observes, explains, and advises. It never rewrites user code,
monkey-patches NumPy, or changes array behavior. Each phase is a separate review
and validation boundary; package publication, public releases, and merges need
the maintainer's explicit approval. A completed local check is not evidence that
the remote compatibility matrix passed.

## v0.4 — reliability foundation (current implementation)

1. **Audit and contracts.** Record baseline behavior, known defects, memory
   vocabulary, compatibility policy, and intentional corrections. Acceptance:
   every existing detector has an evidence requirement and documented limits.
2. **Session and tracing ownership.** Replace shared registries/results; support
   nested sessions, independent threads/tasks, explicit worker registration,
   and cleanup on exceptions. Acceptance: regression tests demonstrate no
   cross-session contamination and preserve external tracing/peak state.
3. **Evidence and presentation.** Use weakref lifecycle evidence, true caller
   locations, deduplicated registrations, explicit dtype sources, and qualified
   recommendations. Acceptance: live/pre-existing arrays are not reported as
   destroyed temporary buffers; unsupported diagnoses stay unknown.
4. **Release-candidate validation.** Cover public APIs, errors, detectors,
   reporting, concurrency, NumPy workloads, installation, and wheel contents.
   Acceptance: existing regression tests pass; important core modules reach
   >=85% branch-aware coverage; demo/benchmarks run; CI is configured for the
   documented Python/NumPy/OS matrix. Document unexecuted matrix jobs honestly.

## v0.5 — diagnostics engine

1. Prototype tracemalloc-domain snapshots using NumPy's published domain ID;
   test known retained buffers, views, external native buffers, and short-lived
   intermediates. Acceptance: ground-truth byte comparisons and explicit
   observability limits, including allocations freed between snapshots.
2. Define stage snapshots/diffs and structured diagnostics (severity,
   confidence, explanation, evidence, source). Acceptance: stable serialization
   schema, deterministic diffs, and measured versus inferred fields.
3. Add snapshot hotspots and conservative AST rules for copies, broadcasting,
   conversions, and repeated allocation. Acceptance: positive AND negative
   fixtures, reliable locations only, no claims of universal native tracking.
4. Measure overhead with/without manual registration and tune collection.
   Acceptance: reproducible accuracy/overhead results, not unmeasured claims.

## v0.6 — CLI and reports

1. Add `npguard run script.py` with configurable collection and preserved target
   exit behavior. Acceptance: script arguments/errors tested in subprocesses.
2. Versioned JSON save/load/compare and offline standalone HTML. Acceptance:
   schema round trips, malicious text escaped, no external assets/accounts.
3. Show hotspots, trends, evidence, confidence, locations, and recommendations.
   Acceptance: reports expose unknown metrics and profiling limitations; clear
   CLI help, examples, and migration documentation.

## v0.7 — optimization advisor

1. Build a rule interface for copies, `out=`, reuse, broadcasting, and dtype
   changes. Acceptance: every rule states preconditions, aliasing/numeric
   caveats, evidence, confidence, and correctness constraints.
2. Add optional before/after experiments on isolated, user-selected functions.
   Acceptance: output equality/tolerances validated before suggesting gains;
   repeated memory/timing measurements; no automatic application.
3. Publish examples and negative cases. Acceptance: unsupported or ambiguous
   situations produce conditional advice or no recommendation.

## v0.8 — memory regression testing

1. Define budgets and baseline metadata (metric, runtime, NumPy, platform,
   workload, sampling). Acceptance: reject incompatible/missing measurements.
2. Add repeated-sample comparisons with absolute/relative tolerances and a
   documented statistical method. Acceptance: controlled increases fail;
   known noisy/no-change cases do not systematically fail.
3. Integrate pytest, JSON results, and `npguard compare`. Acceptance: meaningful
   exit codes, CI examples, and platform/variance limitations documented.

## v0.9 — scale and integrations

1. Profile overhead and retention, add justified filters/sampling, expand
   scientific workloads and concurrency scenarios. Acceptance: at least ten
   reproducible benchmark scenarios with explicit metric provenance, raw data,
   and methodology; do not turn noisy timing into brittle unit-test budgets.
2. Investigate optional Memray/native integration separately from core.
   Acceptance: dependency remains optional, unsupported platforms degrade
   clearly, native versus traced measurements are not conflated.
3. Improve short-lived allocation evidence where observable. Acceptance:
   controlled lifecycle experiments and accuracy/coverage tradeoffs published.

## v1.0 — stability gate

1. Freeze documented public APIs, report schema, semantic versioning, and
   compatibility/migration policy. Acceptance: public examples and upgrade tests.
2. Resolve critical correctness/security issues; audit dependencies and local
   data handling. Acceptance: supported matrix passes on CI and a fresh wheel
   install works independently of the source checkout.
3. Publish reproducible accuracy/performance methodology, full API guide,
   tutorials, installation/troubleshooting, and contributor instructions.
   Acceptance: independent developer can install, diagnose a known problem,
   interpret uncertainty, and run tests/benchmarks from the documentation.

## Review discipline

Keep changes grouped by contract, collection, detectors, presentation, and
validation. Each meaningful behavior change needs a reproducer and a regression
test. After each phase record commands/results, diagnosed failures, limitations,
and the next acceptance gate. Never mark a later phase complete merely because
an earlier phase adds a supporting data field.
