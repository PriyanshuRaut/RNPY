# npguard

**npguard** is a NumPy memory observability and explanation tool.

It helps developers understand *why* NumPy memory usage spikes by detecting
temporary allocations and explaining their causes, with safe, opt-in suggestions
to reduce memory pressure.

## Features

- Watch NumPy-heavy code blocks
- Detect memory pressure and hidden temporaries
- Explain causes (chained ops, broadcasting)
- Provide safe optimization suggestions
- No monkey-patching, no unsafe automation

## What npguard does not do
- Does not modify NumPy behavior
- Does not automatically reuse buffers
- Does not rewrite code

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
