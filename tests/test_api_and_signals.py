import asyncio
import inspect

import numpy as np
import pytest

import npguard as ng
from npguard.signals.dtype import detect_dtype_promotion
from npguard.static import analyze_source


@pytest.mark.parametrize("before,after,kind", [
    ("float32", "float64", "dtype_widening"),
    ("float64", "float32", "dtype_difference"),
    ("int64", "float64", "dtype_difference"),
    ("int16", "int64", "dtype_widening"),
])
def test_dtype_findings_include_relationship_and_storage_evidence(before, after, kind):
    source = np.ones(20, dtype=before)
    result = source.astype(after)
    with ng.capture() as observation:
        ng.register_array(result, "converted", source=source)
    finding, = observation["signals"]["dtype_promotions"]
    assert finding["from"] == before
    assert finding["to"] == after
    assert finding["kind"] == kind
    assert finding["evidence"]["relationship"] == "user_supplied_source"
    assert finding["evidence"]["may_share_memory"] is False
    assert finding["evidence"]["logical_size_difference_bytes"] == result.nbytes - source.nbytes
    assert not finding["implicit_promotion_confirmed"]
    assert not finding["allocation_confirmed"]
    assert finding["confidence"] == "high"


def test_dtype_difference_is_not_guessed_between_unrelated_arrays():
    with ng.capture() as observation:
        a = ng.register_array(np.zeros(20, dtype="float32"))
        b = ng.register_array(np.zeros(20, dtype="float64"))
    assert a.dtype != b.dtype
    assert observation["signals"]["dtype_promotions"] == []


def test_same_dtype_and_reshape_do_not_imply_conversion():
    source = np.ones(20, dtype="float32")
    with ng.capture() as observation:
        ng.register_array(source.reshape(4, 5), source=source)
    assert observation["signals"]["dtype_promotions"] == []
    entries = {e["registration_index"]: e for e in observation["registrations"]}
    assert detect_dtype_promotion(entries) == []


def test_dtype_reinterpretation_is_not_claimed_as_allocated_copy():
    source = np.ones(20, dtype="int32")
    with ng.capture() as observation:
        ng.register_array(source.view("int16"), source=source)
    finding, = observation["signals"]["dtype_promotions"]
    assert finding["evidence"]["may_share_memory"] is True
    assert finding["evidence"]["logical_size_difference_bytes"] is None
    assert not finding["allocation_confirmed"]


def test_watch_preserves_metadata_arguments_and_return_identity():
    @ng.watch("decorated", silent=True)
    def calculate(value, *, scale=2):
        """Original docstring."""
        ng.register_array(value)
        return value * scale

    assert calculate.__name__ == "calculate"
    assert calculate.__doc__ == "Original docstring."
    assert inspect.signature(calculate) == inspect.signature(calculate.__wrapped__)
    np.testing.assert_array_equal(calculate(np.arange(3), scale=3), [0, 3, 6])
    assert ng.last("tag") == "decorated"
    marker = object()
    assert ng.profile(lambda: marker) is marker


def test_profile_accepts_callable_instances():
    class Callable:
        def __call__(self, value):
            return value

    assert ng.profile(Callable(), 12) == 12
    assert ng.last("tag") == "Callable"


def test_async_decorator_and_profile_observe_awaited_execution():
    @ng.watch(silent=True)
    async def compute():
        await asyncio.sleep(0)
        return ng.register_array(np.ones(4))

    async def scenario():
        result = await compute()
        assert ng.last("tag") == "compute"
        assert ng.last("registrations")[0]["size"] == result.nbytes
        result = await ng.profile(compute.__wrapped__)
        assert ng.last("registrations")[0]["size"] == result.nbytes

    asyncio.run(scenario())


def test_decorator_propagates_error():
    @ng.watch(silent=True)
    def fail():
        raise ArithmeticError("original")

    with pytest.raises(ArithmeticError, match="original"):
        fail()
    assert ng.last("exception_type") == "ArithmeticError"


def test_capture_is_silent_and_warn_threshold_has_honest_scope(capsys):
    with ng.capture(silent=False, warn_threshold_mb=0):
        ng.register_array(np.ones(100))
    assert capsys.readouterr().out == ""
    with ng.memory_watcher(warn_threshold_mb=0):
        ng.register_array(np.ones(100))
    assert "Process-wide traced memory" in capsys.readouterr().out


def test_reporting_empty_standalone_active_and_last_session(capsys):
    ng.report()
    assert "No arrays registered" in capsys.readouterr().out
    arr = ng.register_array(np.ones(4), "standalone")
    ng.report()
    assert "standalone" in capsys.readouterr().out
    with ng.capture():
        ng.register_array(arr, "active")
        ng.report()
        output = capsys.readouterr().out
        assert "logical" in output and "active" in output and "standalone" not in output
    ng.report()
    output = capsys.readouterr().out
    assert "active" in output and "standalone" not in output
    ng.reset()
    ng.report()
    assert "No arrays registered" in capsys.readouterr().out


def test_suggestions_do_not_invent_causes_from_high_peak(capsys):
    ng.suggest()
    assert "No observation" in capsys.readouterr().out
    with ng.capture():
        arr = np.ones(250_000)
    assert arr.size == 250_000
    ng.suggest(temp_threshold_mb=0)
    text = capsys.readouterr().out
    assert "No supported cause identified" in text
    assert "chained" not in text and "temporary allocations detected" not in text


def test_suggestions_qualify_repetition_and_dtype_advice(capsys):
    source = np.ones(10, dtype="float32")
    with ng.capture():
        arrays = [source.astype("float64") for _ in range(3)]
        for arr in arrays:
            ng.register_array(arr, source=source)
    ng.suggest(temp_threshold_mb=0)
    text = capsys.readouterr().out
    assert "dtype" in text and "precision" in text
    assert "If results need not coexist" in text
    assert "aliasing and correctness" in text
    assert "caused full-size" not in text


def test_suggestions_handle_unknown_peak_and_analysis_failure(capsys, monkeypatch):
    def broken(entries):
        raise RuntimeError("detector")

    monkeypatch.setattr("npguard.session.detect_dtype_promotion", broken)
    with ng.capture():
        with ng.capture():
            pass
    ng.suggest()
    text = capsys.readouterr().out
    assert "unavailable" in text and "dtype analysis unavailable" in text


@pytest.mark.parametrize("threshold", [-1, float("inf"), "invalid"])
def test_suggestion_threshold_validation(threshold):
    with pytest.raises(ValueError):
        ng.suggest(temp_threshold_mb=threshold)


def test_static_hints_are_syntactic_only():
    assert analyze_source("x = 1") == []
    hints = analyze_source("x = a * 2 + b")
    assert len(hints) == 2
    assert all("may" in h["reason"] for h in hints)
    with pytest.raises(SyntaxError):
        analyze_source("def invalid(")


def test_log_facade(capsys):
    for method in (ng.log.debug, ng.log.info, ng.log.warn, ng.log.error):
        method("tag", "message")
    assert capsys.readouterr().out.count("[tag] message") == 4


def test_structured_dtype_does_not_require_parsing_display_string():
    source = np.zeros(4, dtype=[("field", "i4")])
    result = source.astype([("field", "i8")])
    with ng.capture() as observation:
        ng.register_array(result, source=source)
    assert observation["analysis_errors"] == []
    finding, = observation["signals"]["dtype_promotions"]
    assert finding["kind"] == "dtype_difference"
    assert finding["evidence"]["logical_size_difference_bytes"] == 16
