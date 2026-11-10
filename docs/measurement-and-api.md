# Measurement and API contracts (v0.4)

## Collection and evidence

A session owns an array collection and a final observation. `ArrayCollection`
(`registry.py`) stores immutable-by-convention metadata snapshots and weak
references, never array values or strong references to arrays/sources. IDs are
paired with registration lifetimes so ID reuse cannot overwrite dead records.
Multiple registrations of one live object count once; the first label, location,
and source metadata win. Object metadata is sampled at registration; resizing
or mutating an array later does not retroactively change it.

A record's size is logical `nbytes`. It is not unique allocated storage, retained
process memory, or a measurement of Python referents inside object arrays.
Registration may happen long after allocation. Weakref release records prove
array-object release after registration only. A view, external buffer, allocator
cache, or other owner may retain the data. `signals.temporaries.count` and `.mb`
are therefore `None`, with `status="unmeasured"`; released object counts and
logical sizes are separate evidence, never a lower bound on freed buffer bytes.

Repetition groups distinct registered owning arrays by user file/line, shape,
and dtype. Re-registering an object or registering views does not count. A group
can include pre-existing arrays or arrays intentionally kept alive: it is a
repeated registration pattern, not proof of repeated allocation or wasted work.
Advice to reuse buffers is conditional on results not needing to coexist and on
preserving aliasing and numerical correctness.

Dtype findings require `register_array(result, source=source)`. This relationship
is asserted by the caller. Metadata records dtype, shape, logical bytes,
`np.can_cast(..., casting="safe")`, and `np.may_share_memory`. A false overlap
result establishes disjoint bounds; true establishes only possible sharing.
Neither says when an allocation occurred. A wider safely castable numeric dtype
is classified as `dtype_widening`; other differences as `dtype_difference`.
Views can reinterpret dtypes without a copy. No rule claims implicit promotion,
value conversion, unnecessary allocation, or proof of NumPy's decision process.
`confidence="high"` refers only to the observed dtype difference under the
caller-supplied relationship. Structured and nonnumeric dtype metadata are
supported without reconstructing a dtype from its display string.

## Tracing ownership

`tracing.py` coordinates process-global tracemalloc under a lock. The first
watcher starts the tracer only if it was off. The last watcher stops it only if
NPGuard started it. Existing traceback depth and high-water marks are preserved;
NPGuard never calls `reset_peak()`. All overlapping intervals are marked.

The `python_traced` object records starting current bytes, ending current bytes,
signed net change, the process's high-water mark, overlap, availability, and
whether NPGuard owned tracing. `session_peak_bytes` is peak minus starting current
bytes only for an exclusive NPGuard-owned interval. It is otherwise `None`, with
an explicit reason. Even an exclusive interval is **process-wide**, including
unrelated threads, supported NumPy allocations, and NPGuard itself. It is not
isolated source-level attribution. Tracemalloc does not cover every native
allocator; an array allocated before tracing started may not be represented.

Existing external tracers are preserved, but callers must not stop, restart, or
reset the tracer while a watcher runs. An observed external stop yields an
unavailable measurement. A stop/restart or peak reset between observations
cannot always be detected without intercepting global functions; such
interference invalidates measurements and is unsupported. NPGuard does not
monkey-patch tracing functions to discover it.

Do not equate traced bytes with NumPy-domain-only bytes, RSS, logical array sizes,
or estimated intermediate buffers. NumPy-domain snapshot collection belongs to
v0.5; native/RSS integrations are separate metrics in later phases.

## Execution contexts

`state.py` uses ContextVars and checks the owning thread/task so copied task
contexts do not share mutable sessions, standalone registries, or results.
Independent threads/tasks have independent last results. Same-owner explicitly
copied contexts follow normal ContextVar propagation semantics. Async tasks use
separate contexts but the process tracer is still shared. Sessions that overlap
lose attributable peaks even if they never register the same arrays.

By default a registration belongs only to the innermost active session. An outer
session resumes collection after its child exits. The last completed session
becomes `last()` until another finishes, so a child may be inspected inside its
parent. A session handle's `.observation` is empty while active and returns a
defensive snapshot afterward. Workers can pass `session=handle`; collection is
locked. Join workers before the session exits; late registration raises
`RuntimeError`. No implicit cross-thread aggregation or native allocation
concurrency is inferred from thread IDs.

## Exceptions and public behavior

Watchers finalize and restore context/tracer ownership on normal return,
`Exception`, `KeyboardInterrupt`, `SystemExit`, and async cancellation. Capture's
dict is filled before the original exception propagates. Results retain only the
exception's type name, not its value, traceback, or potentially sensitive text.
Detectors run independently: an analysis error is recorded in `analysis_errors`
without replacing application exceptions or pretending the detector succeeded.
An empty detector result must be interpreted alongside these errors.

`watch` preserves function names, docstrings, signatures, results, and exceptions.
For `async def`, observation spans the await. Synchronous functions returning a
lazy generator or arbitrary awaitable are observed only until they return;
wrap actual iteration/await in a watcher to monitor lazy execution. `profile`
passes all supplied arguments to the callable and uses the same rules.

`report` prints registered logical sizes for the active or last completed
session. Before any session completes, standalone metadata is available for
manual reports; it does not leak into subsequent sessions. `reset` clears
standalone metadata and the last result in the calling context, never another
thread/task or an active session. Public snapshots are deep copies, not a
read-only view into shared nested dictionaries. These dictionaries are not yet
a versioned JSON report format (repetition keys include tuples).

Metadata memory grows with unique registered lifetimes within a session and
persists in the last result until reset/replacement. Long-running unbounded
sessions are inappropriate for continuous production monitoring. Standard
CPython frame metadata supplies callsites; no source text or data values are
collected. Treat local file paths and labels as potentially sensitive metadata.
