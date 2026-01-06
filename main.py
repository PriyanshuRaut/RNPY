import numpy as np
import npguard as ng
import threading


ng.log.info("demo", "Starting npguard demo")


# =========================================================
# 1. Basic block observation (v0.2 baseline)
# =========================================================

with ng.memory_watcher("basic_block"):
    a = np.random.rand(10_000, 100)
    ng.register_array(a, "a")

    b = a * 2 + a.mean(axis=0) - 1
    ng.register_array(b, "b")

    c = np.ascontiguousarray(b.T)
    ng.register_array(c, "c")

ng.report()
ng.suggest()


# =========================================================
# 2. Silent + capture API (v0.2)
# =========================================================

with ng.capture("captured_block") as obs:
    x = np.random.rand(10_000, 100)
    ng.register_array(x, "x")

    y = x * 3
    ng.register_array(y, "y")

ng.log.debug("capture", "Captured observation snapshot")
print(obs)


# =========================================================
# 3. Decorator API (@watch) (v0.2)
# =========================================================

@ng.watch("decorated_function", warn_threshold_mb=5)
def compute_step():
    a = np.random.rand(10_000, 100)
    ng.register_array(a, "a")
    return a * 2 + a.mean(axis=0)

compute_step()
ng.suggest()


# =========================================================
# 4. profile() helper (v0.2)
# =========================================================

def pipeline():
    a = np.random.rand(10_000, 100)
    ng.register_array(a, "a")
    return np.ascontiguousarray(a.T)

ng.profile(pipeline)
ng.suggest()


# =========================================================
# 5. NEW v0.3 — structured signal access
# =========================================================

ng.log.info("v0.3", "Structured signal access")

print("Peak MB:", ng.last("peak_mb"))
print("Repeated allocations:", bool(ng.last("signals.repeated")))
print("Dtype promotions:", bool(ng.last("signals.dtype_promotions")))
print("Parallel allocations:", bool(ng.last("signals.parallel")))


# =========================================================
# 6. NEW v0.3 — parallel allocations (thread signal)
# =========================================================

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


# =========================================================
# 7. Reset correctness (v0.3 fixed)
# =========================================================

ng.log.info("reset", "Resetting npguard state")

print("Before reset:", ng.last())
ng.reset()
print("After reset:", ng.last())

ng.log.info("demo", "npguard demo complete")
