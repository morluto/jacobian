"""Assertions shared by composition-local optional-provider spike tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def assert_unavailable_spike_preserves_catalog(
    runtime: Any,
    run_spike: Callable[[], Mapping[str, object]],
) -> None:
    """Prove that an unavailable spike neither registers nor removes operations."""

    before = runtime.core.operations.snapshot().model_dump(mode="json")
    report = run_spike()
    after = runtime.core.operations.snapshot().model_dump(mode="json")

    assert report["status"] == "UNAVAILABLE"
    assert report["operation_ids_registered"] == []
    assert after == before
