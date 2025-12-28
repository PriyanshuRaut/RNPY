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
