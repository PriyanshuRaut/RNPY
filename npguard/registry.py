"""Session-local metadata collection; never retain array objects or values."""

from contextvars import ContextVar
from copy import deepcopy
import sys
import threading
import time
import weakref

import numpy as np

from . import state


_standalone = ContextVar("npguard_standalone_registry", default=None)


class ArrayCollection:
    def __init__(self):
        self._lock = threading.RLock()
        self._records = []
        self._live = {}
        self._closed = False

    def register(self, arr, label, source, location):
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot register an array in a closed session")
            previous = self._live.get(id(arr))
            if previous is not None and previous[1]() is arr:
                return arr  # First registration of this object's lifetime wins.
            now = time.perf_counter()
            record = {
                "registration_index": len(self._records),
                "id": id(arr),
                "label": label,
                "size": arr.nbytes,
                "shape": arr.shape,
                "dtype": str(arr.dtype),
                "dtype_kind": arr.dtype.kind,
                "itemsize": arr.itemsize,
                "owndata": bool(arr.flags["OWNDATA"]),
                "contiguous": bool(arr.flags["C_CONTIGUOUS"]),
                "filename": location[0],
                "lineno": location[1],
                "callsite": f"{location[0]}:{location[1]}",
                "thread": threading.get_ident(),
                "time": now,  # Compatibility alias: registration, NOT allocation time.
                "registered_at": now,
                "released_at": None,
                "allocation_time_known": False,
                "source": None,
            }
            if source is not None:
                record["source"] = {
                    "dtype": str(source.dtype),
                    "dtype_kind": source.dtype.kind,
                    "itemsize": source.itemsize,
                    "shape": source.shape,
                    "size": source.nbytes,
                    "relationship": "user_supplied_source",
                    "may_share_memory": bool(np.may_share_memory(arr, source)),
                    "safe_cast_to_result": bool(np.can_cast(source.dtype, arr.dtype, casting="safe")),
                }
            collection_ref = weakref.ref(self)
            array_id = id(arr)

            def released(ref):
                collection = collection_ref()
                if collection is not None:
                    with collection._lock:
                        if not collection._closed:
                            record["released_at"] = time.perf_counter()
                        live = collection._live.get(array_id)
                        if live is not None and live[1] is ref:
                            collection._live.pop(array_id)

            ref = weakref.ref(arr, released)
            self._records.append((record, ref))
            self._live[array_id] = (record, ref)
            return arr

    def snapshot(self, *, close=False):
        with self._lock:
            result = []
            for record, ref in self._records:
                item = deepcopy(record)
                item["alive_at_end"] = ref() is not None
                result.append(item)
            if close:
                self._closed = True
            return result


def current_collection():
    active = state.sessions()
    if active:
        return active[-1]._collection
    saved = _standalone.get()
    owner = state.execution_owner()
    if saved is None or saved[0] != owner:
        saved = (owner, ArrayCollection())
        _standalone.set(saved)
    return saved[1]


def register_array(arr, label="array", *, source=None, session=None, _location=None):
    if not isinstance(arr, np.ndarray):
        raise TypeError("arr must be a NumPy ndarray")
    if source is not None and not isinstance(source, np.ndarray):
        raise TypeError("source must be a NumPy ndarray or None")
    if not isinstance(label, str):
        raise TypeError("label must be a string")
    if session is None:
        collection = current_collection()
    else:
        from .session import ObservationSession

        if not isinstance(session, ObservationSession):
            raise TypeError("session must be a handle yielded by memory_watcher")
        collection = session._collection
    if _location is None:
        frame = sys._getframe(1)
        _location = (frame.f_code.co_filename, frame.f_lineno)
        del frame
    return collection.register(arr, label, source, _location)


def reset_registry():
    _standalone.set(None)
