import inspect
import threading
import time
import numpy as np

ArrayRegistry = {}
AllocationTimeline = []

def register_array(arr: np.ndarray, label="array"):
    frame = inspect.stack()[1]
    callsite = f"{frame.filename}:{frame.lineno}"

    info = {
        "id": id(arr),
        "label": label,
        "size": arr.nbytes,
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "owndata": arr.flags["OWNDATA"],
        "contiguous": arr.flags["C_CONTIGUOUS"],
        "callsite": callsite,
        "thread": threading.get_ident(),
        "time": time.perf_counter(),
    }

    ArrayRegistry[id(arr)] = info
    AllocationTimeline.append(info)

    return arr


def reset_registry():
    """
    Reset live registry and allocation timeline.

    """
    ArrayRegistry.clear()
    AllocationTimeline.clear()
