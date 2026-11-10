# Migrating from v0.3.x

The public names remain: `memory_watcher`, `register_array`, `suggest`, `report`,
`watch`, `capture`, `profile`, `last_observation`, `reset`, `last`, and `log`.
Most calls continue unchanged, but results intentionally correct false claims:

1. Use Python >=3.10. v0.3.1 advertised >=3.8 but evaluated annotations that
   require 3.10. NumPy support is now >=1.23.5,<3, subject to Python compatibility.
2. `peak_mb` is in MiB and can be `None` for overlapping/nested sessions or
   pre-existing tracing. Check `measurements.python_traced` for scope/reason.
   Do not substitute a historical process peak for a missing session peak.
3. `signals.temporaries.count` and `.mb` are `None`, not guessed from
   registrations. Inspect `.released_array_objects` for weakref lifecycle
   evidence, and never equate it with freed buffer bytes.
4. Dtype differences require an explicit `source=` array. No relationship is
   inferred between arbitrary arrays. Findings carry evidence, not claims that
   NumPy implicitly promoted or copied data.
5. Repetition groups distinct owning arrays at real user callsites. A duplicate
   registration, view, or unrelated old session no longer generates a finding.
6. Reports show active/last-session logical sizes rather than global cumulative
   allocations. Standalone registrations are used only before a session result
   exists. Session state/results are local to the thread/async task.
7. To include worker registrations, use `with memory_watcher() as handle` and
   pass `session=handle` to `register_array`. Join workers before the watcher
   closes. New threads/tasks no longer silently contaminate other sessions.
8. `reset()` during monitoring raises `RuntimeError`. Reset outside a watcher.
9. `capture` populates its dict on errors and still propagates the original
   exception. `watch` now preserves metadata and supports async functions.
10. Advice wording is conditional. A high traced peak alone no longer produces
    an unsupported assertion about chained expressions or temporary pressure.

Internal global `ArrayRegistry` and `AllocationTimeline` are removed; they were
not public exports. Use public observations/registrations instead. The old
observation dict shape is extended but is not yet a stable serialized schema.
No optimization is ever applied automatically.
