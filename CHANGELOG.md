## v0.4.1 — Prism

This release improves session isolation, tracing ownership, and the evidence
behind memory diagnostics. It distinguishes process-traced memory from registered
array sizes and reports unavailable measurements explicitly.

### Added
- Independent nested/thread/async sessions and explicit worker session handles.
- Weakref object-lifecycle evidence without retaining arrays or their contents.
- Optional `source=` metadata for evidenced dtype differences and safe widening.
- Separate measurement categories and reasons for unavailable session peaks.
- Defensive result snapshots, exception status, and explicit detector failures.
- Pytest reliability/integration coverage, compatibility CI, and eleven raw-data
  benchmark scenarios. Audit, roadmap, architecture, and migration guides.

### Fixed
- Live registrations no longer masquerade as destroyed temporary buffers.
- Repeated registrations, object-ID reuse, views, and unrelated sessions no
  longer generate misleading repeated-allocation reports.
- Call sites resolve to user code; context/tracing restore on exceptions.
- External tracemalloc ownership and peak/depth settings are preserved.
- Decorators preserve metadata and support async execution.
- Removed unsupported chained-expression and concurrent-spike claims in advice.

### Compatibility
- Python >=3.10; NumPy >=1.23.5,<3, subject to NumPy's Python compatibility.
- `peak_mb` and temporary count/size can be `None` when unmeasured/unattributable.
- Reports use active/last-session logical sizes; reset during observation raises.
- Worker registration requires explicit sharing. See `docs/migration-v0.4.md`.

### Measurement limits
- Process-traced memory is process-wide and can include unrelated work and
  instrumentation. Individual session peaks require an exclusive tracing interval.
- Registered sizes are logical `nbytes`; views and shared buffers can be counted
  more than once. Object release does not establish that its buffer was freed.
- NumPy data-buffer allocations, process RSS, and temporary-allocation lifetimes
  are not measured separately in v0.4.1.

## v0.3.0 — Structured Signals & Ergonomics

### Added
- Structured memory signals:
  - repeated allocation detection
  - parallel/threaded allocation detection
  - dtype promotion signals
- Estimated temporary memory usage and array counts
- Programmatic signal access via `ng.last(...)`
- Logging-style output (info / warn / debug levels)
- Decorator API: `@ng.watch`
- Silent capture API: `ng.capture`
- One-shot profiling: `ng.profile`
- Proper reset semantics

### Improved
- Explanation clarity over raw memory dumps
- Signal aggregation across blocks and functions
- Noise reduction in repeated warnings

### Preserved
- All v0.2 APIs remain compatible
- Explanation-first, non-invasive philosophy
- No NumPy monkey-patching or auto-fixing

This release is focused on **debugging and understanding memory pressure**, not enforcing behavior.
