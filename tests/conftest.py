import tracemalloc

import pytest

import npguard as ng
from npguard import state
from npguard.tracing import coordinator


@pytest.fixture(autouse=True)
def isolated_state():
    tracing_before = tracemalloc.is_tracing()
    ng.reset()
    yield
    assert not state.sessions(), "Monitoring context leaked"
    assert not coordinator._active, "Tracer reference leaked"
    assert tracemalloc.is_tracing() == tracing_before, "Tracer ownership leaked"
    ng.reset()
