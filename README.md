# npguard

**npguard** is a NumPy memory observability and explanation tool.

It helps developers understand **why** NumPy memory usage spikes by detecting
temporary allocations and explaining their causes, with safe, opt-in suggestions
to reduce memory pressure.

npguard focuses on *explanation*, not automatic optimization.

---

## Installation

```bash
pip install npguard
````

PyPI: [https://pypi.org/project/npguard/](https://pypi.org/project/npguard/)

---

## Motivation

NumPy can silently allocate large temporary arrays during chained expressions,
broadcasting, or forced copies.

For example:

```python
b = a * 2 + a.mean(axis=0) - 1
```

This single line can create multiple full-sized temporary arrays, leading to
sudden memory spikes that are not obvious from the code and are often poorly
explained by traditional profilers.

npguard exists to answer the question:

**“Why did memory spike here?”**

---

## Features

* Watch NumPy-heavy code blocks
* Detect memory pressure and hidden temporary allocations
* Estimate temporary memory usage and temporary array counts
* Explain likely causes (chained ops, broadcasting, repeated allocations)
* Provide safe, opt-in optimization suggestions
* Multiple ergonomics:

  * context manager
  * decorator
  * capture API
  * programmatic access

---

## What npguard does NOT do

* Does not modify NumPy behavior
* Does not monkey-patch NumPy
* Does not automatically reuse buffers
* Does not rewrite user code
* Does not detect memory leaks
* Is not a production monitoring tool

npguard is intended for **development and debugging**, not runtime enforcement.

---

## Example Usage

```python
import numpy as np
import npguard as ng


# -----------------------------------
# 1. Basic block observation
# -----------------------------------

with ng.memory_watcher("basic_block"):
    a = np.random.rand(10_000, 100)
    ng.register_array(a, "a")

    b = a * 2 + a.mean(axis=0) - 1
    ng.register_array(b, "b")

    c = np.ascontiguousarray(b.T)
    ng.register_array(c, "c")

ng.report()
ng.suggest()


# -----------------------------------
# 2. Silent + capture API
# -----------------------------------

with ng.capture("captured_block") as obs:
    x = np.random.rand(10_000, 100)
    ng.register_array(x, "x")

    y = x * 3
    ng.register_array(y, "y")

print("\nCaptured observation:")
print(obs)


# -----------------------------------
# 3. Decorator API (@watch)
# -----------------------------------

@ng.watch("decorated_function", warn_threshold_mb=5)
def compute_step():
    a = np.random.rand(10_000, 100)
    ng.register_array(a, "a")

    return a * 2 + a.mean(axis=0)

compute_step()
ng.suggest()


# -----------------------------------
# 4. profile() helper
# -----------------------------------

def pipeline():
    a = np.random.rand(10_000, 100)
    ng.register_array(a, "a")

    return np.ascontiguousarray(a.T)

ng.profile(pipeline)
ng.suggest()


# -----------------------------------
# 5. last_observation() + reset()
# -----------------------------------

print("\nLast observation dict:")
print(ng.last_observation())

ng.reset()
print("\nAfter reset():")
print(ng.last_observation())
```

---

## Example Output (abridged)

```
[npguard] Memory Watch: basic_block
  Python peak diff:    24.02 MB
  NumPy live arrays:   0.00 MB
  Estimated temporaries: 24.02 MB
  Estimated temporary arrays: 3

⚠️  Memory pressure detected
  Likely temporary allocations caused by chained operations or broadcasting
```

---

## When to Use npguard

* Debugging unexpected NumPy memory spikes
* Understanding temporary array creation
* Learning memory-aware NumPy patterns
* Investigating OOMs during pipeline scaling
* Second-pass performance and memory analysis

---

## Target Audience

* NumPy users working with medium to large arrays
* Developers debugging memory pressure (not leaks)
* Engineers who want explanations rather than automatic optimization

---

## Comparison with Existing Tools

* Traditional profilers show *how much* memory is used, not *why*
* Leak detectors focus on long-lived leaks, not short-lived spikes
* NumPy itself does not expose temporary allocation behavior at a high level

npguard complements these tools by explaining **temporary allocation pressure**
at the code-block or function level.

---

## Project Status

* Version: 0.2.0
* Status: early but stable
* API: intentionally small and conservative

Future versions may improve explanation quality and aggregation,
but will preserve the explanation-first philosophy.

---

## License

MIT License
