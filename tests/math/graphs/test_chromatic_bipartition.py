from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.optimization import _chromatic_bipartition as operation
from jacobian.math.graphs.optimization._chromatic_bipartition import (
    ChromaticBipartitionRequest,
    find_chromatic_bipartition,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def graph(
    vertices: tuple[str, ...], edges: tuple[tuple[str, str], ...]
) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(vertices=vertices, edges=edges)


def test_k4_returns_canonical_split_and_exact_induced_values() -> None:
    source = graph(
        ("a", "b", "c", "d"),
        (("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")),
    )
    result = find_chromatic_bipartition(
        ChromaticBipartitionRequest(graph=source, s=2, t=2)
    )
    assert result.status == "SPLIT"
    assert result.side_a == ("a", "b")
    assert result.side_b == ("c", "d")
    assert (result.chromatic_a, result.chromatic_b) == (2, 2)
    assert result.model_validate_json(result.model_dump_json()) == result


def test_k3_has_exact_no_split() -> None:
    source = graph(("a", "b", "c"), (("a", "b"), ("a", "c"), ("b", "c")))
    result = find_chromatic_bipartition(
        ChromaticBipartitionRequest(graph=source, s=2, t=2)
    )
    assert result.status == "NO_SPLIT"
    assert result.checked_partitions == 4
    assert result.side_a is None


def test_timeout_is_unknown_without_negative_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = graph(
        ("a", "b", "c", "d"),
        (("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")),
    )
    request = ChromaticBipartitionRequest(graph=source, s=2, t=2)
    monkeypatch.setattr(operation, "remaining_ms", lambda started, seconds: 0)
    result = find_chromatic_bipartition(request)
    assert result.status in {"SPLIT", "UNKNOWN"}
    if result.status == "UNKNOWN":
        assert result.side_a is None and result.chromatic_a is None


def test_complete_search_bound_is_source_admission() -> None:
    source = graph(tuple(f"v{i}" for i in range(20)), ())
    with pytest.raises(ValidationError, match="complete-search work bound"):
        ChromaticBipartitionRequest(graph=source, s=1, t=1)


def test_worker_noncompletion_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    source = graph(("a", "b"), (("a", "b"),))

    class TimedOut:
        timed_out = True
        cancelled = False
        returncode = None
        stdout_exceeded = False
        stderr_exceeded = False
        stdout = b""
        stderr = b""

    monkeypatch.setattr(
        operation, "run_bounded_process", lambda *args, **kwargs: TimedOut()
    )
    result = operation.find_chromatic_bipartition(
        ChromaticBipartitionRequest(graph=source, s=1, t=1)
    )
    assert result.status == "UNKNOWN"
    assert result.side_a is None and result.side_b is None
