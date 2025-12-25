# npguard

**npguard** is a NumPy memory observability and explanation tool.

It helps developers understand why NumPy memory usage spikes by detecting hidden temporary allocations and explaining their causes, with safe, opt-in suggestions to reduce memory pressure.

npguard focuses on explanation, not automatic optimization.

---

## Installation

Install from PyPI:

```bash
pip install npguard
````

PyPI project page: [https://pypi.org/project/npguard/](https://pypi.org/project/npguard/)

---

## Motivation

NumPy can silently allocate large temporary arrays during chained expressions, broadcasting, or forced copies.

For example:

```python
b = a * 2 + a.mean(axis=0) - 1
```

This single line can create multiple full-sized temporary arrays, leading to sudden memory spikes that are not obvious from the code and are often poorly explained by traditional profilers.

**npguard** exists to answer the question:

> Why did memory spike here?

---

## Features

* Watch NumPy-heavy code blocks
* Detect memory pressure and hidden temporary allocations
* Explain likely causes (chained ops, broadcasting, forced copies)
* Provide safe, opt-in optimization suggestions

**No monkey-patching, no unsafe automation.**

---

## What npguard does not do

* Does not modify NumPy internals
* Does not automatically reuse buffers
* Does not rewrite user code
* Does not detect memory leaks
* Does not act as a production monitoring tool

---

## Example

```python
import numpy as np
import npguard as ng

with ng.memory_watcher("matrix_pipeline"):
    a = np.random.rand(10_000, 100)
    ng.register_array(a, "a")

    b = a * 2 + a.mean(axis=0) - 1
    ng.register_array(b, "b")

    c = np.ascontiguousarray(b.T)
    ng.register_array(c, "c")

ng.report()
ng.suggest()
```

---

## Example output

```
[npguard] Memory Watch: matrix_pipeline
  Python peak diff:    23.95 MB
  Estimated temporaries: 23.95 MB

⚠️  Memory pressure detected
  Likely cause: chained NumPy operations

[npguard] Suggestions:
  • Split expressions or use ufuncs with `out=`
```

---

## When to use npguard

* Debugging unexpected NumPy memory spikes
* Understanding temporary array creation
* Learning memory-aware NumPy patterns
* Investigating performance regressions during development

**npguard** is intended for development and debugging, not production monitoring.

---

## Target audience

* NumPy users working with medium to large arrays
* Developers debugging memory pressure (not leaks)
* People who want explanations rather than automatic optimization

---

## Comparison with existing tools

* Traditional profilers show how much memory is used, but not why
* Leak detectors focus on long-lived leaks, not short-lived spikes
* NumPy itself does not expose temporary allocation behavior at a high level

**npguard** complements these tools by explaining temporary allocation pressure at the code-block level.

---

## Project status

* Version: 0.1.0
* Status: early but stable
* API: intentionally small and conservative

Future versions may add:

* decorator-based APIs
* better loop-level signals
* improved attribution of temporaries

---

## License

MIT License
