import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import tracemalloc

import numpy as np
import pytest

import npguard as ng


def test_measurement_categories_and_real_numpy_buffer():
    with ng.capture("array") as observation:
        arr = np.ones(250_000, dtype=np.float64)
        ng.register_array(arr, "retained")
    measurement = observation["measurements"]
    assert measurement["python_traced"]["scope"] == "process"
    assert measurement["python_traced"]["session_peak_bytes"] >= arr.nbytes
    assert observation["peak_mb"] == measurement["python_traced"]["session_peak_bytes"] / 1024**2
    assert measurement["numpy_data_buffers"]["status"] == "not_measured"
    assert measurement["process_memory"]["status"] == "not_measured"
    assert measurement["registered_arrays"]["logical_bytes"] == arr.nbytes
    assert measurement["registered_arrays"]["unique_buffer_bytes"] is None


def test_nested_sessions_restore_parent_and_do_not_stop_tracing():
    with ng.memory_watcher("outer", silent=True) as outer:
        a = ng.register_array(np.ones(16), "outer-a")
        with ng.capture("inner") as inner:
            b = ng.register_array(np.zeros(32), "inner-b")
        assert tracemalloc.is_tracing()
        c = ng.register_array(np.ones(8), "outer-c")
    assert {e["label"] for e in outer.observation["registrations"]} == {"outer-a", "outer-c"}
    assert [e["label"] for e in inner["registrations"]] == ["inner-b"]
    assert ng.last("tag") == "outer"
    assert a.size + b.size + c.size == 56
    for result in (outer.observation, inner):
        assert result["peak_mb"] is None
        assert result["measurements"]["python_traced"]["session_peak_unavailable_reason"] == "overlapping_sessions"


@pytest.mark.parametrize("error", [ValueError("private payload"), KeyboardInterrupt(), SystemExit(4)])
def test_capture_preserves_exception_and_finalizes(error):
    with pytest.raises(type(error)) as caught:
        with ng.capture("failed") as observation:
            ng.register_array(np.ones(10))
            raise error
    assert caught.value is error
    assert observation["status"] == "error"
    assert observation["exception_type"] == type(error).__name__
    assert "private payload" not in repr(observation)
    assert ng.last("status") == "error"
    with ng.capture("next") as next_observation:
        pass
    assert next_observation["status"] == "completed"


def test_nested_exception_restores_registration_target():
    with ng.capture("outer") as outer:
        with pytest.raises(RuntimeError):
            with ng.capture("inner") as inner:
                raise RuntimeError("application")
        ng.register_array(np.zeros(4), "after-error")
    assert inner["status"] == "error"
    assert outer["registrations"][0]["label"] == "after-error"


def test_external_tracer_and_old_peak_are_preserved():
    tracemalloc.start(3)
    try:
        previous = bytearray(2_000_000)
        old_peak = tracemalloc.get_traced_memory()[1]
        del previous
        with ng.capture("external") as observation:
            ng.register_array(np.ones(8))
        assert tracemalloc.is_tracing()
        assert tracemalloc.get_traceback_limit() == 3
        assert tracemalloc.get_traced_memory()[1] >= old_peak
        assert observation["peak_mb"] is None
        assert observation["measurements"]["python_traced"]["session_peak_unavailable_reason"] == "preexisting_tracer"
    finally:
        tracemalloc.stop()


def test_external_stop_makes_measurement_unavailable():
    with ng.capture() as observation:
        tracemalloc.stop()
    assert observation["peak_mb"] is None
    assert observation["measurements"]["python_traced"]["available"] is False
    assert observation["measurements"]["python_traced"]["session_peak_unavailable_reason"] == "tracing_stopped_externally"


def test_independent_overlapping_threads():
    barrier = threading.Barrier(2, timeout=10)

    def worker(label, size):
        assert ng.last() == {}
        with ng.capture(label) as observation:
            arr = ng.register_array(np.ones(size), label)
            barrier.wait()
            barrier.wait()
        assert arr.size == size
        assert ng.last("tag") == label
        return observation

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(worker, "first", 12), pool.submit(worker, "second", 24)]
        results = [job.result(timeout=15) for job in jobs]
    for result, size in zip(results, (12, 24)):
        assert len(result["registrations"]) == 1
        assert result["registrations"][0]["size"] == size * 8
        assert result["peak_mb"] is None
    assert ng.last() == {}


def test_worker_registration_requires_explicit_session():
    with ng.memory_watcher(silent=True) as handle:
        a = ng.register_array(np.ones(2), "main")
        with ThreadPoolExecutor(max_workers=1) as pool:
            b = pool.submit(ng.register_array, np.ones(3), "joined", session=handle).result()
            pool.submit(ng.register_array, np.ones(4), "unjoined").result()
    result = handle.observation
    assert [r["label"] for r in result["registrations"]] == ["main", "joined"]
    assert a.size + b.size == 5
    assert result["signals"]["parallel"]["threads"] == 2
    assert result["signals"]["parallel"]["concurrent_allocation_confirmed"] is False
    with pytest.raises(RuntimeError, match="closed"):
        ng.register_array(a, session=handle)


def test_async_tasks_do_not_inherit_mutable_parent_state():
    async def scenario():
        async def child(label):
            assert ng.last() == {}
            ng.register_array(np.zeros(2), "outside-child-session")
            with ng.capture(label) as result:
                arr = ng.register_array(np.ones(5), label)
                await asyncio.sleep(0)
            assert arr.size == 5
            assert ng.last("tag") == label
            return result

        with ng.capture("parent") as parent:
            results = await asyncio.gather(child("a"), child("b"))
        return parent, results

    parent, results = asyncio.run(scenario())
    assert parent["registrations"] == []
    assert [r["registrations"][0]["label"] for r in results] == ["a", "b"]
    assert all(r["peak_mb"] is None for r in results)
    assert ng.last() == {}


def test_failed_detector_is_explicit_and_does_not_mask_application(monkeypatch):
    def broken(entries):
        raise LookupError("private detector payload")

    monkeypatch.setattr("npguard.session.detect_dtype_promotion", broken)
    with pytest.raises(ValueError, match="original"):
        with ng.capture() as observation:
            raise ValueError("original")
    assert observation["analysis_errors"] == [{"detector": "dtype", "error_type": "LookupError"}]
    assert observation["signals"]["dtype_promotions"] == []
    assert "private detector payload" not in repr(observation)


def test_reset_is_local_and_rejects_active_session():
    with ng.capture():
        with pytest.raises(RuntimeError, match="active"):
            ng.reset()
    assert ng.last()
    ng.reset()
    assert ng.last() == {}


@pytest.mark.parametrize("threshold", [-1, float("nan"), float("inf"), None, "large"])
def test_invalid_threshold_does_not_start_tracing(threshold):
    with pytest.raises(ValueError):
        with ng.memory_watcher(warn_threshold_mb=threshold):
            pytest.fail("Should not enter")


def test_invalid_tag_does_not_start_tracing():
    with pytest.raises(TypeError):
        with ng.memory_watcher(tag=123):
            pytest.fail("Should not enter")


def test_async_cancellation_finalizes_without_swallowing_cancel():
    captured = {}

    async def scenario():
        entered = asyncio.Event()

        async def worker():
            with ng.capture("cancelled") as observation:
                captured["observation"] = observation
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(worker())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert captured["observation"]["exception_type"] == "CancelledError"
    assert captured["observation"]["status"] == "error"


def test_net_traced_change_preserves_negative_sign():
    tracemalloc.start()
    try:
        arr = np.ones(250_000)
        with ng.capture() as observation:
            del arr
        assert observation["measurements"]["python_traced"]["net_bytes"] < -1_000_000
        assert observation["peak_mb"] is None
    finally:
        tracemalloc.stop()
