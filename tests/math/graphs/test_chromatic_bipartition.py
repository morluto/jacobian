from __future__ import annotations

import json
import time

import pytest
from pydantic import ValidationError

from jacobian._execution import bind_request_deadline, request_execution
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.graphs.optimization import _chromatic_bipartition as operation
from jacobian.math.graphs.optimization import (
    _chromatic_bipartition_process as process_owner,
)
from jacobian.math.graphs.optimization._chromatic_bipartition import (
    ChromaticBipartitionRequest,
    ChromaticBipartitionResult,
    find_chromatic_bipartition,
)
from jacobian.math.graphs.optimization._coloring_models import ChromaticNumberBudget
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.process import BoundedProcessResult


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
    assert result.checked_partitions == 3
    assert result.model_validate_json(result.model_dump_json()) == result


def test_k3_unit_thresholds_report_the_induced_k2_chromatic_number() -> None:
    source = graph(("a", "b", "c"), (("a", "b"), ("a", "c"), ("b", "c")))
    result = find_chromatic_bipartition(
        ChromaticBipartitionRequest(graph=source, s=1, t=1)
    )
    assert result.status == "SPLIT"
    assert result.side_a == ("a",)
    assert result.side_b == ("b", "c")
    assert (result.chromatic_a, result.chromatic_b) == (1, 2)
    assert result.model_validate_json(result.model_dump_json()) == result


def test_k4_unit_thresholds_report_the_induced_k3_chromatic_number() -> None:
    source = graph(
        ("a", "b", "c", "d"),
        (("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")),
    )
    result = find_chromatic_bipartition(
        ChromaticBipartitionRequest(graph=source, s=1, t=1)
    )
    assert result.status == "SPLIT"
    assert result.side_a == ("a",)
    assert result.side_b == ("b", "c", "d")
    assert (result.chromatic_a, result.chromatic_b) == (1, 3)


def test_k3_has_exact_no_split() -> None:
    source = graph(("a", "b", "c"), (("a", "b"), ("a", "c"), ("b", "c")))
    result = find_chromatic_bipartition(
        ChromaticBipartitionRequest(graph=source, s=2, t=2)
    )
    assert result.status == "NO_SPLIT"
    assert result.side_a is None
    assert result.checked_partitions == 0


def test_unequal_thresholds_accept_the_opposite_orientation() -> None:
    source = graph(
        ("a", "b", "c", "d", "e"),
        (("a", "b"), ("c", "d"), ("c", "e"), ("d", "e")),
    )
    result = find_chromatic_bipartition(
        ChromaticBipartitionRequest(graph=source, s=3, t=2)
    )
    assert result.status == "SPLIT"
    assert set(result.side_a) == {"c", "d", "e"}
    assert set(result.side_b) == {"a", "b"}
    assert (result.chromatic_a, result.chromatic_b) == (3, 2)


def test_kernel_timeout_is_unknown_without_a_negative_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = graph(
        ("a", "b", "c", "d"),
        (("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")),
    )
    request = ChromaticBipartitionRequest(graph=source, s=2, t=2)
    monkeypatch.setattr(operation, "remaining_ms", lambda *_args: 0)
    result = operation._find_chromatic_bipartition_kernel(request)
    assert result.status == "UNKNOWN"
    assert result.side_a is None and result.chromatic_a is None
    assert result.checked_partitions == 1


def test_worker_deadline_times_out_across_the_process_boundary() -> None:
    source = graph(
        ("a", "b", "c", "d"),
        (("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")),
    )
    request = ChromaticBipartitionRequest(
        graph=source,
        s=2,
        t=2,
        resource_budget=ChromaticNumberBudget(wall_seconds=5),
    )
    with request_execution(time.monotonic()):
        bind_request_deadline(time.monotonic() + 0.02)
        result = process_owner.find_chromatic_bipartition(request)
    assert result.status == "UNKNOWN"
    assert result.side_a is None and result.chromatic_a is None


def test_edgeless_twenty_vertex_request_is_exactly_decidable() -> None:
    source = graph(tuple(f"v{i}" for i in range(20)), ())
    result = find_chromatic_bipartition(
        ChromaticBipartitionRequest(graph=source, s=1, t=1)
    )
    assert result.status == "SPLIT"
    assert result.side_a == ("v0",)
    assert result.side_b == tuple(f"v{i}" for i in range(1, 20))
    assert (result.chromatic_a, result.chromatic_b) == (1, 1)
    assert result.model_validate_json(result.model_dump_json()) == result


def test_edgeless_graph_has_no_split_above_chromatic_one() -> None:
    source = graph(tuple(f"v{i}" for i in range(20)), ())
    result = find_chromatic_bipartition(
        ChromaticBipartitionRequest(graph=source, s=2, t=1)
    )
    assert result.status == "NO_SPLIT"
    assert result.side_a is None


def test_eight_vertex_path_fits_the_unordered_partition_bound() -> None:
    vertices = tuple(f"v{i}" for i in range(8))
    edges = tuple((vertices[index], vertices[index + 1]) for index in range(7))
    source = graph(vertices, edges)
    result = find_chromatic_bipartition(
        ChromaticBipartitionRequest(graph=source, s=2, t=2)
    )
    assert result.status == "SPLIT"
    assert result.chromatic_a == 2 and result.chromatic_b == 2
    assert set(result.side_a or ()) | set(result.side_b or ()) == set(vertices)
    assert not set(result.side_a or ()) & set(result.side_b or ())


def test_impossible_threshold_sum_is_exact_no_split_before_search() -> None:
    vertices = tuple(f"v{i}" for i in range(20))
    source = graph(vertices, (("v0", "v1"),))
    result = find_chromatic_bipartition(
        ChromaticBipartitionRequest(graph=source, s=11, t=10)
    )
    assert result.status == "NO_SPLIT"
    assert result.side_a is None and result.chromatic_a is None
    assert result.checked_partitions == 0
    assert result.model_validate_json(result.model_dump_json()) == result


def test_complete_search_bound_still_rejects_a_dense_twenty_vertex_graph() -> None:
    vertices = tuple(f"v{i}" for i in range(20))
    source = graph(vertices, (("v0", "v1"),))
    request = ChromaticBipartitionRequest(graph=source, s=2, t=2)
    with pytest.raises(OperationResourceAdmissionError, match="complete-search work"):
        find_chromatic_bipartition(request)


def test_unit_threshold_dense_thirty_two_vertex_graph_is_admitted_as_work() -> None:
    vertices = tuple(f"v{i:02d}" for i in range(32))
    edges = tuple(
        (vertices[left], vertices[right])
        for left in range(32)
        for right in range(left + 1, 32)
    )
    request = ChromaticBipartitionRequest(graph=graph(vertices, edges), s=1, t=1)
    with pytest.raises(OperationResourceAdmissionError, match="complete-search work"):
        find_chromatic_bipartition(request)


def test_unit_threshold_triangle_plus_isolates_is_exact_without_backend_overflow() -> (
    None
):
    isolates = tuple(f"u{i:02d}" for i in range(31))
    source = graph((*isolates, "x", "y", "z"), (("x", "y"), ("x", "z"), ("y", "z")))
    result = find_chromatic_bipartition(
        ChromaticBipartitionRequest(graph=source, s=1, t=1)
    )
    assert result.status == "SPLIT"
    assert result.side_a == ("u00",)
    assert result.chromatic_a == 1
    assert result.chromatic_b == 3
    assert result.model_validate_json(result.model_dump_json()) == result


def test_unit_threshold_tries_another_singleton_before_backend_overflow() -> None:
    isolated = "iso"
    cycle = tuple(f"c{i:02d}" for i in range(33))
    edges = (
        *((cycle[index], cycle[index + 1]) for index in range(32)),
        (cycle[0], cycle[32]),
    )
    result = find_chromatic_bipartition(
        ChromaticBipartitionRequest(graph=graph((isolated, *cycle), edges), s=1, t=1)
    )
    assert result.status == "SPLIT"
    assert result.chromatic_a == 1
    assert result.chromatic_b == 2
    assert len(result.side_a) == 1
    assert result.model_validate_json(result.model_dump_json()) == result


def test_unit_threshold_nonbipartite_core_above_backend_order_is_refused() -> None:
    vertices = tuple(f"v{i:02d}" for i in range(34))
    edges = tuple(
        (vertices[left], vertices[right])
        for left in range(34)
        for right in range(left + 1, 34)
    )
    request = ChromaticBipartitionRequest(graph=graph(vertices, edges), s=1, t=1)
    with pytest.raises(OperationResourceAdmissionError, match="complete-search work"):
        find_chromatic_bipartition(request)


def _completed(
    *,
    returncode: int | None = 0,
    stdout: bytes = b"",
    timed_out: bool = False,
    stdout_exceeded: bool = False,
    stderr_exceeded: bool = False,
) -> BoundedProcessResult:
    return BoundedProcessResult(
        returncode=returncode,
        stdout=stdout,
        stderr=b"",
        stdout_exceeded=stdout_exceeded,
        stderr_exceeded=stderr_exceeded,
        timed_out=timed_out,
    )


def test_worker_timeout_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    source = graph(("a", "b"), (("a", "b"),))
    monkeypatch.setattr(
        process_owner,
        "run_bounded_process",
        lambda *_args, **_kwargs: _completed(returncode=None, timed_out=True),
    )
    result = operation.find_chromatic_bipartition(
        ChromaticBipartitionRequest(graph=source, s=1, t=1)
    )
    assert result.status == "UNKNOWN"
    assert result.side_a is None and result.side_b is None


@pytest.mark.parametrize(
    ("completed", "message"),
    [
        (_completed(returncode=1), "did not establish an outcome"),
        (_completed(stdout_exceeded=True), "exceeded an output cap"),
        (_completed(stderr_exceeded=True), "exceeded an output cap"),
        (_completed(stdout=b"not-json"), "malformed output"),
    ],
)
def test_worker_failures_are_operational_errors(
    monkeypatch: pytest.MonkeyPatch,
    completed: BoundedProcessResult,
    message: str,
) -> None:
    source = graph(("a", "b"), (("a", "b"),))
    monkeypatch.setattr(
        process_owner,
        "run_bounded_process",
        lambda *_args, **_kwargs: completed,
    )
    with pytest.raises(RuntimeError, match=message):
        operation.find_chromatic_bipartition(
            ChromaticBipartitionRequest(graph=source, s=1, t=1)
        )


def test_worker_result_must_echo_the_submitted_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = graph(("a", "b", "c"), (("a", "b"), ("b", "c")))
    echoed = ChromaticBipartitionResult(
        graph=graph(("x", "y"), (("x", "y"),)),
        s=1,
        t=1,
        status="SPLIT",
        side_a=("x",),
        side_b=("y",),
        chromatic_a=1,
        chromatic_b=2,
        checked_partitions=1,
    )
    monkeypatch.setattr(
        process_owner,
        "run_bounded_process",
        lambda *_args, **_kwargs: _completed(
            stdout=json.dumps(echoed.model_dump(mode="json")).encode("utf-8")
        ),
    )
    with pytest.raises(RuntimeError, match="not bound to the submitted request"):
        operation.find_chromatic_bipartition(
            ChromaticBipartitionRequest(graph=source, s=2, t=2)
        )


def test_split_below_submitted_thresholds_cannot_bind() -> None:
    source = graph(("a", "b", "c"), (("a", "b"), ("b", "c")))
    with pytest.raises(ValidationError):
        ChromaticBipartitionResult(
            graph=source,
            s=2,
            t=2,
            status="SPLIT",
            side_a=("a", "b"),
            side_b=("c",),
            chromatic_a=2,
            chromatic_b=1,
            checked_partitions=1,
        )


def test_long_nfc_labels_on_k2_return_split_through_the_worker() -> None:
    left = "a" * 30_000
    right = "b" * 30_000
    source = graph((left, right), ((left, right),))
    request = ChromaticBipartitionRequest(graph=source, s=1, t=1)
    limit = process_owner._chromatic_bipartition_worker_stdout_limit(request)
    assert limit > 128 * 1024
    result = find_chromatic_bipartition(request)
    assert result.status == "SPLIT"
    assert result.side_a == (left,)
    assert result.side_b == (right,)
    assert (result.chromatic_a, result.chromatic_b) == (1, 1)
    dumped = json.dumps(
        result.model_dump(mode="json"),
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert len(dumped) <= limit
    assert result.model_validate_json(result.model_dump_json()) == result
