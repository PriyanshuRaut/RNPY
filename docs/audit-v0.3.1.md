# v0.3.1 audit and v0.4 decisions

Baseline: commit `0ae844a8bea540d3b175727c2e484decef24b758`.
The existing import test passes on Python 3.12.14 / NumPy 2.5.4; fatal-error
flake8 passes. This establishes importability, not detector accuracy.

| Finding | Consequence | v0.4 acceptance criterion |
| --- | --- | --- |
| Every registration in the time window is called a destroyed temporary | Live arrays and pre-existing arrays produce false findings | Registration and object release are separate events; unknown buffer lifetimes stay unknown |
| `parent_dtype` is read but never populated | Dtype detector cannot produce evidence | Accept an explicit `source=` array; report evidenced conversions/widening without claiming implicit promotion |
| `inspect.stack()[1]` points at the API wrapper | Unrelated registrations collapse to one apparent hotspot | Capture the external caller's filename and line |
| Registry keyed only by object ID, timeline never cleared | Duplicate registrations, ID reuse, and history contaminate findings | Session-local, lifetime-aware identity and first-registration deduplication |
| Module-global registry and last result | Nested, threaded, and async users overwrite one another | Context-local state with execution-owner checks; explicit sharing for worker registration |
| Context manager has no `finally` | Exceptions leave tracing/state active | Finalize on all exits and preserve the original exception |
| Every watcher starts/stops process-global tracemalloc | Nested watchers disable parents; external tracers are stopped | Locked reference-counted ownership, no peak reset, stop only NPGuard-owned tracing |
| Process peak is presented as a block's NumPy usage | Old peaks and concurrent work are attributed to the wrong session | Separate process-traced metrics, registered sizes, and unavailable per-session peaks |
| Shallow copies expose nested internal dictionaries | Callers can mutate future results | Defensive snapshots for all public observations |
| Thread diversity is called a parallel spike | Sequential activity implies concurrent allocation | Report registration thread diversity only; no concurrency/spike claim |
| Suggestions infer chained expressions from any high peak | Advice has no supporting evidence | Remove speculative fallback and qualify recommendations |
| Python >=3.8 but evaluated `str | None` annotations | Import fails on advertised older Python | Declare Python >=3.10 and a CI compatibility matrix |
| Only one import test | Lifecycle/concurrency/errors are untested | Meaningful regression tests and >=85% branch-aware core coverage |

## Architectural decisions

1. A session owns metadata, weak references, and its final observation. It never
   retains registered arrays or their contents. A child session collects its own
   registrations; parent process measurements necessarily include nested work.
2. Context variables isolate results. Inherited contexts in new asyncio tasks do
   not inherit a mutable monitoring session. New threads do not join sessions
   implicitly. `memory_watcher()` yields a session handle that can be passed as
   `register_array(..., session=handle)` for deliberate worker registration.
3. A locked coordinator manages process-global tracemalloc ownership. It never
   resets a caller's peak. A session peak is available only for an uninterrupted,
   exclusive NPGuard-owned tracing interval. Overlap or pre-existing tracing
   makes that metric unavailable. These measurements still include other
   threads, instrumentation, Python allocations, and NumPy buffers traced by
   NumPy; they are neither RSS nor isolated NumPy-domain measurements.
4. Weakref callbacks prove array-object release after registration, not buffer
   allocation time or buffer release. v0.4 leaves temporary-buffer counts/bytes
   unknown. Logical array sizes may double-count views and are not allocation
   measurements. This conservative alternative avoids pretending that a
   registration timestamp is an allocation timestamp.
5. `source=` records a caller-asserted relationship plus dtype, shape, storage
   overlap, and logical-size evidence. It proves a dtype difference between
   related registrations, not why NumPy chose that dtype or when storage was
   allocated. Recommendations remain conditional and never modify code.
6. Failed detector analysis is recorded explicitly; cleanup still runs. An
   observed application's exception is propagated, not replaced with a detector
   error. Reports retain the exception type, not its potentially sensitive text.

## Scope and compatibility corrections

Keep public function names and common calling forms. Add `source=` and
`session=` to registration and preserve decorator metadata. `peak_mb` remains a
key but can be `None` when attribution is unavailable. Temporary `count`/`mb`
can be `None`: unknown is not zero. Reports describe the current/last session
instead of accumulating unrelated sessions. Reset during an active session is
rejected. These correctness changes require migration guidance.

Automatic NumPy-domain snapshots, full allocation provenance, AST-backed
diagnostics, CLI, HTML, optimization validation, statistical regressions, and
native-profiler integrations remain later-phase work. The existing AST helper
is a syntactic hint only, not proof of NumPy execution.
