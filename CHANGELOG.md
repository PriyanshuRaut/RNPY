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
