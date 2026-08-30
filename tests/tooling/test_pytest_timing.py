"""Regression tests for pytest timing evidence collection."""

from __future__ import annotations

from types import SimpleNamespace

from tools import pytest_timing


def test_worker_failure_without_output_is_not_masked_by_timing_hook() -> None:
    state = pytest_timing._TimingState()
    node = SimpleNamespace(
        config=SimpleNamespace(stash={pytest_timing._TIMING_STATE: state})
    )

    pytest_timing.pytest_testnodedown(node, RuntimeError("worker terminated"))

    assert state.workers == {}
