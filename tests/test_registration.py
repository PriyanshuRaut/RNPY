import gc
import inspect
import weakref

import numpy as np
import pytest

import npguard as ng
from npguard.registry import register_array as direct_register
from npguard.signals.repetition import detect_repeated_allocations


def test_live_and_preexisting_arrays_are_not_destroyed_temporaries():
    before = np.ones(12)
    with ng.capture() as observation:
        ng.register_array(before)
        current = ng.register_array(np.zeros(12))
    assert all(e["alive_at_end"] for e in observation["registrations"])
    assert all(e["released_at"] is None for e in observation["registrations"])
    assert all(not e["allocation_time_known"] for e in observation["registrations"])
    assert observation["signals"]["temporaries"]["count"] is None
    assert observation["signals"]["temporaries"]["mb"] is None
    assert observation["signals"]["temporaries"]["released_array_objects"] == 0
    assert current.shape == before.shape


def test_weakref_release_is_recorded_without_retaining_array():
    with ng.capture() as observation:
        arr = np.zeros(10)
        ref = weakref.ref(arr)
        ng.register_array(arr)
        del arr
        gc.collect()
        assert ref() is None
    entry = observation["registrations"][0]
    assert not entry["alive_at_end"]
    assert entry["released_at"] >= entry["registered_at"]
    evidence = observation["signals"]["temporaries"]
    assert evidence["released_array_objects"] == 1
    assert evidence["released_logical_mb"] == 80 / 1024**2
    assert evidence["buffer_release_confirmed"] is False
    assert evidence["count"] is None


def test_view_release_is_not_buffer_release_and_sizes_remain_logical():
    owner = np.ones(16)
    with ng.capture() as observation:
        ng.register_array(owner, "owner")
        view = owner[:]
        ng.register_array(view, "view")
        del view
        gc.collect()
    assert observation["measurements"]["registered_arrays"]["logical_bytes"] == owner.nbytes * 2
    assert observation["signals"]["temporaries"]["released_array_objects"] == 1
    assert observation["signals"]["repeated"] == {}
    assert owner.sum() == 16


def test_same_array_registration_is_idempotent_and_returns_identity():
    arr = np.ones(4)
    with ng.capture() as observation:
        assert ng.register_array(arr, "first") is arr
        for _ in range(20):
            ng.register_array(arr, "alias")
    assert len(observation["registrations"]) == 1
    assert observation["registrations"][0]["label"] == "first"
    assert observation["signals"]["repeated"] == {}


def test_many_released_objects_survive_id_reuse_in_history():
    with ng.capture() as observation:
        for _ in range(150):
            arr = np.zeros(3)
            ng.register_array(arr)
            del arr
    assert len(observation["registrations"]) == 150
    assert observation["signals"]["temporaries"]["released_array_objects"] == 150
    groups = list(observation["signals"]["repeated"].values())
    assert len(groups) == 1 and len(groups[0]) == 150


def test_repetition_has_actual_user_location_and_no_duplicate_entries():
    arrays = [np.ones(4) for _ in range(3)]
    with ng.capture() as observation:
        line = inspect.currentframe().f_lineno + 2
        for arr in arrays:
            ng.register_array(arr)
    entries = observation["registrations"]
    assert {e["filename"] for e in entries} == {__file__}
    assert {e["lineno"] for e in entries} == {line}
    assert len(observation["signals"]["repeated"]) == 1
    groups = detect_repeated_allocations({0: entries[0], 1: entries[0]})
    assert groups == {}


def test_different_call_sites_do_not_form_a_repeated_pattern():
    with ng.capture() as observation:
        a = ng.register_array(np.ones(4))
        b = ng.register_array(np.ones(4))
    assert a.shape == b.shape
    assert observation["signals"]["repeated"] == {}


def test_direct_registry_entry_point_has_external_callsite():
    with ng.capture() as observation:
        direct_register(np.ones(2))
    assert observation["registrations"][0]["filename"] == __file__


def test_results_are_defensive_deep_snapshots():
    with ng.memory_watcher(silent=True) as session:
        ng.register_array(np.ones(2))
    snapshot = ng.last()
    snapshot["signals"]["temporaries"]["count"] = 999
    snapshot["registrations"][0]["label"] = "tampered"
    session.observation["registrations"].clear()
    assert ng.last("signals.temporaries.count") is None
    assert ng.last("registrations")[0]["label"] == "array"
    assert session.observation["registrations"][0]["label"] == "array"
    assert ng.last("signals.temporaries.missing") is None
    assert ng.last("tag.not.a.dictionary") is None


@pytest.mark.parametrize("arr,kwargs", [
    ([1, 2], {}),
    (np.ones(2), {"source": [1, 2]}),
    (np.ones(2), {"label": 4}),
    (np.ones(2), {"session": object()}),
])
def test_registration_rejects_invalid_input(arr, kwargs):
    with pytest.raises(TypeError):
        ng.register_array(arr, **kwargs)


def test_source_metadata_does_not_retain_source():
    with ng.capture() as observation:
        source = np.ones(4, dtype=np.float32)
        ref = weakref.ref(source)
        target = source.astype(np.float64)
        ng.register_array(target, source=source)
        del source
        gc.collect()
        assert ref() is None
    assert observation["signals"]["dtype_promotions"]
