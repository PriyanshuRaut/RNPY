# npguard

**npguard** is a NumPy memory observability and explanation tool.

It helps developers understand **why** NumPy memory usage spikes by detecting
temporary allocations and explaining their causes, with safe, opt-in suggestions
to reduce memory pressure.

npguard focuses on **explanation**, not automatic optimization.

---

## Installation

```bash
pip install npguard
````

PyPI: [https://pypi.org/project/npguard/](https://pypi.org/project/npguard/)

---

## Motivation

NumPy can silently allocate large temporary arrays during:

* chained expressions
* broadcasting
* repeated allocations inside loops
* parallel execution

For example:

```python
b = a * 2 + a.mean(axis=0) - 1
```

This single line can create multiple full-sized temporary arrays, leading to
sudden memory spikes that are not obvious from the code and are often poorly
explained by traditional profilers.

**npguard exists to answer one question:**

> *“Why did memory spike here?”*

---

## Features

* Watch NumPy-heavy code blocks
* Detect memory pressure and hidden temporary allocations
* Estimate temporary memory usage and array counts
* Detect repeated allocations
* Detect parallel/threaded allocation spikes
* Detect dtype promotion signals
* Explain causes in human-readable terms
* Provide safe, opt-in optimization suggestions

### Multiple ergonomics

* Context manager
* Decorator API
* Silent capture API
* Programmatic signal access

---

## What npguard does NOT do

* Does **not** modify NumPy behavior
* Does **not** monkey-patch NumPy
* Does **not** automatically reuse buffers
* Does **not** rewrite user code
* Does **not** detect memory leaks
* Is **not** a production monitoring tool

npguard is intended for **development and debugging**, not runtime enforcement.

---

## Example Usage

### 1. Basic block observation

```python
import numpy as np
import npguard as ng

with ng.memory_watcher("basic_block"):
    a = np.random.rand(10_000, 100)
    ng.register_array(a, "a")

    b = a * 2 + a.mean(axis=0) - 1
    ng.register_array(b, "b")

    c = np.ascontiguousarray(b.T)
    ng.register_array(c, "c")

ng.report()
ng.suggest()
```

**Output:**

```
[npguard] Memory spike detected

[npguard] Allocation Summary (cumulative)
  a           : 7.63 MB
  b           : 7.63 MB
  c           : 7.63 MB
[INFO][npguard] Memory analysis completed (peak ~23.77 MB)
[WARN][signals.repetition] Repeated allocations detected at 1 site(s)
[INFO][suggestion] Reuse preallocated buffers inside loops
```

---

## 2. Silent Capture API

```python
with ng.capture("captured_block") as obs:
    x = np.random.rand(10_000, 100)
    ng.register_array(x, "x")

    y = x * 3
    ng.register_array(y, "y")

print(obs)
```

**Output (excerpt):**

```
{
  'tag': 'captured_block',
  'peak_mb': 15.26,
  'signals': {
    'parallel': None,
    'dtype_promotions': [],
    'repeated': {...},
    'temporaries': {'count': 2, 'mb': 15.26}
  }
}
```

---

## 3. Decorator API

```python
@ng.watch("decorated_function", warn_threshold_mb=5)
def compute_step():
    a = np.random.rand(10_000, 100)
    ng.register_array(a, "a")
    return a * 2 + a.mean(axis=0)

compute_step()
ng.suggest()
```

**Output:**

```
[npguard] Memory spike detected
[INFO][npguard] Memory analysis completed (peak ~15.26 MB)
[WARN][signals.repetition] Repeated allocations detected at 1 site(s)
[INFO][suggestion] Reuse preallocated buffers inside loops
```

---

## 4. profile() Helper

```python
def pipeline():
    a = np.random.rand(10_000, 100)
    ng.register_array(a, "a")
    return np.ascontiguousarray(a.T)

ng.profile(pipeline)
ng.suggest()
```

---

## 5. NEW in v0.3 — Structured Signal Access

```python
print("Peak MB:", ng.last("peak_mb"))
print("Repeated allocations:", bool(ng.last("signals.repeated")))
print("Dtype promotions:", bool(ng.last("signals.dtype_promotions")))
print("Parallel allocations:", bool(ng.last("signals.parallel")))
```

**Output:**

```
Peak MB: 15.26
Repeated allocations: True
Dtype promotions: False
Parallel allocations: False
```

---

## 6. NEW in v0.3 — Parallel Allocation Detection

```python
import threading

def threaded_alloc():
    a = np.random.rand(5_000, 100)
    ng.register_array(a, "threaded")

with ng.memory_watcher("thread_test"):
    t1 = threading.Thread(target=threaded_alloc)
    t2 = threading.Thread(target=threaded_alloc)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

ng.suggest()
```

**Output:**

```
[WARN][signals.parallel] Parallel temporary allocations detected across threads
[INFO][suggestion] Consider thread-local buffers or avoiding shared temporaries
```

---

## 7. Reset State (v0.3 fix)

```python
print("Before reset:", ng.last())
ng.reset()
print("After reset:", ng.last())
```

**Output:**

```
Before reset: {...}
After reset: {}
```

---


## 8. Logging

npguard exposes a lightweight logger for structured diagnostic output:

```python
ng.log.info("tag", "message")
ng.log.warn("tag", "message")
ng.log.debug("tag", "message")
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
* Engineers who want **explanations rather than auto-fixes**

---

## Project Status

* **Version:** 0.3.0
* **Status:** early but stable
* **API:** conservative and explanation-first

Future versions may improve signal accuracy and aggregation,
while preserving the non-invasive philosophy.

---

## License

MIT License