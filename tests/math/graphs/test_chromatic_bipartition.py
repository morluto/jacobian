from __future__ import annotations

import json
import time
from itertools import combinations, product

import pytest
from pydantic import ValidationError

from jacobian._execution import (
    BackendFailureReason,
    OperationBackendError,
    OperationExecutionTimeoutError,
    OperationResourceExhaustedError,
    bind_request_deadline,
    request_execution,
)
from jacobian._worker_protocol import encode_worker_result_frame
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


def _brute_chromatic_number(
    vertices: tuple[str, ...], edges: tuple[tuple[str, str], ...]
) -> int:
    if not vertices:
        return 0
    for colors in range(1, len(vertices) + 1):
        assignments = range(colors)
        for coloring in product(assignments, repeat=len(vertices)):
            if all(
                coloring[vertices.index(left)] != coloring[vertices.index(right)]
                for left, right in edges
            ):
                return colors
    raise AssertionError("every finite graph has a finite coloring")


def _brute_splittable(source: SimpleUndirectedGraph, s: int, t: int) -> bool:
    vertices = source.vertices
    for size in range(1, len(vertices)):
        for selected in combinations(vertices, size):
            side_a = tuple(selected)
            side_b = tuple(vertex for vertex in vertices if vertex not in side_a)
            edges_a = tuple(
                edge for edge in source.edges if edge[0] in side_a and edge[1] in side_a
            )
            edges_b = tuple(
                edge for edge in source.edges if edge[0] in side_b and edge[1] in side_b
            )
            if (
                _brute_chromatic_number(side_a, edges_a) >= s
                and _brute_chromatic_number(side_b, edges_b) >= t
            ):
                return True
    return False


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


def test_kernel_matches_independent_exhaustive_oracle_on_all_four_vertex_graphs() -> (
    None
):
    vertices = ("a", "b", "c", "d")
    possible_edges = tuple((left, right) for left, right in combinations(vertices, 2))
    for edge_count in range(len(possible_edges) + 1):
        for selected_edges in combinations(possible_edges, edge_count):
            source = graph(vertices, selected_edges)
            for s, t in ((1, 1), (2, 1), (1, 2), (2, 2)):
                request = ChromaticBipartitionRequest(graph=source, s=s, t=t)
                result = operation._find_chromatic_bipartition_kernel(request)
                assert (result.status == "SPLIT") == _brute_splittable(source, s, t)
                if result.status == "SPLIT":
                    assert result.side_a is not None and result.side_b is not None
                    assert result.chromatic_a == _brute_chromatic_number(
                        result.side_a,
                        tuple(
                            edge
                            for edge in source.edges
                            if edge[0] in result.side_a and edge[1] in result.side_a
                        ),
                    )
                    assert result.chromatic_b == _brute_chromatic_number(
                        result.side_b,
                        tuple(
                            edge
                            for edge in source.edges
                            if edge[0] in result.side_b and edge[1] in result.side_b
                        ),
                    )


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
    assert result.side_a is not None
    assert result.side_b is not None
    assert set(result.side_a) == {"c", "d", "e"}
    assert set(result.side_b) == {"a", "b"}
    assert (result.chromatic_a, result.chromatic_b) == (3, 2)


def test_kernel_timeout_raises_an_execution_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = graph(
        ("a", "b", "c", "d"),
        (("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")),
    )
    request = ChromaticBipartitionRequest(graph=source, s=2, t=2)
    monkeypatch.setattr(operation, "remaining_ms", lambda *_args: 0)
    with pytest.raises(OperationExecutionTimeoutError):
        operation._find_chromatic_bipartition_kernel(request)


def test_worker_deadline_returns_source_bound_unknown_across_process_boundary() -> None:
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
        with pytest.raises(OperationExecutionTimeoutError):
            process_owner.find_chromatic_bipartition(request)


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
    assert result.side_a is not None
    assert result.chromatic_a == 1
    assert result.chromatic_b == 2
    assert len(result.side_a) == 1
    assert result.checked_partitions == 1
    assert result.model_validate_json(result.model_dump_json()) == result


def test_unit_threshold_charges_only_the_first_usable_remainder() -> None:
    isolates = tuple(f"u{i:03d}" for i in range(248))
    clique = tuple(f"k{i}" for i in range(8))
    edges = tuple(
        (clique[left], clique[right])
        for left in range(8)
        for right in range(left + 1, 8)
    )
    request = ChromaticBipartitionRequest(
        graph=graph((*isolates, *clique), edges), s=1, t=1
    )
    result = find_chromatic_bipartition(request)
    assert result.status == "SPLIT"
    assert result.side_a == ("u000",)
    assert result.chromatic_b == 8


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


def test_worker_timeout_is_an_execution_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = graph(("a", "b"), (("a", "b"),))
    monkeypatch.setattr(
        process_owner,
        "run_bounded_process",
        lambda *_args, **_kwargs: _completed(returncode=None, timed_out=True),
    )
    with pytest.raises(OperationExecutionTimeoutError) as caught:
        operation.find_chromatic_bipartition(
            ChromaticBipartitionRequest(graph=source, s=1, t=1)
        )
    assert caught.value.configured_seconds == 5
    assert caught.value.adjustable_field_path == ("resource_budget", "wall_seconds")


def test_worker_timeout_retains_recovery_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = graph(("a", "b"), (("a", "b"),))
    monkeypatch.setattr(
        process_owner,
        "run_bounded_process",
        lambda *_args, **_kwargs: _completed(returncode=None, timed_out=True),
    )
    request = ChromaticBipartitionRequest(
        graph=source,
        s=1,
        t=1,
        resource_budget=ChromaticNumberBudget(wall_seconds=5),
    )
    with pytest.raises(OperationExecutionTimeoutError) as error:
        find_chromatic_bipartition(request)
    assert error.value.configured_seconds == 5
    assert error.value.adjustable_field_path == ("resource_budget", "wall_seconds")


@pytest.mark.parametrize(
    ("completed", "error_type", "reason"),
    [
        (
            _completed(returncode=1),
            OperationBackendError,
            BackendFailureReason.ABNORMAL_EXIT,
        ),
        (_completed(stdout_exceeded=True), OperationResourceExhaustedError, None),
        (_completed(stderr_exceeded=True), OperationResourceExhaustedError, None),
        (
            _completed(stdout=b"not-json"),
            OperationBackendError,
            BackendFailureReason.MALFORMED_RESPONSE,
        ),
    ],
)
def test_worker_failures_are_operational_errors(
    monkeypatch: pytest.MonkeyPatch,
    completed: BoundedProcessResult,
    error_type: type[Exception],
    reason: BackendFailureReason | None,
) -> None:
    source = graph(("a", "b"), (("a", "b"),))
    monkeypatch.setattr(
        process_owner,
        "run_bounded_process",
        lambda *_args, **_kwargs: completed,
    )
    with pytest.raises(error_type) as caught:
        operation.find_chromatic_bipartition(
            ChromaticBipartitionRequest(graph=source, s=1, t=1)
        )
    if reason is not None:
        assert isinstance(caught.value, OperationBackendError)
        assert caught.value.reason == reason


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
            stdout=encode_worker_result_frame(echoed.model_dump(mode="json"))
        ),
    )
    with pytest.raises(OperationBackendError) as caught:
        operation.find_chromatic_bipartition(
            ChromaticBipartitionRequest(graph=source, s=2, t=2)
        )
    assert caught.value.reason == BackendFailureReason.INVALID_OUTPUT


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


def test_unknown_status_cannot_bind_as_an_exact_result() -> None:
    source = graph(("a", "b"), (("a", "b"),))
    with pytest.raises(ValidationError):
        ChromaticBipartitionResult(
            graph=source,
            s=2,
            t=2,
            status="UNKNOWN",
            checked_partitions=0,
        )


def test_unknown_worker_frame_is_an_execution_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = graph(("a", "b"), (("a", "b"),))
    frame = {
        "graph": source.model_dump(mode="json"),
        "s": 1,
        "t": 1,
        "status": "UNKNOWN",
        "side_a": None,
        "side_b": None,
        "chromatic_a": None,
        "chromatic_b": None,
        "checked_partitions": 0,
    }
    monkeypatch.setattr(
        process_owner,
        "run_bounded_process",
        lambda *_args, **_kwargs: _completed(
            stdout=encode_worker_result_frame(frame)
        ),
    )
    with pytest.raises(OperationBackendError) as caught:
        operation.find_chromatic_bipartition(
            ChromaticBipartitionRequest(graph=source, s=1, t=1)
        )
    assert caught.value.reason == BackendFailureReason.MALFORMED_RESPONSE


def test_split_sides_must_follow_the_source_vertex_axis() -> None:
    source = graph(
        ("b", "a", "d", "c"),
        (("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")),
    )
    with pytest.raises(ValidationError, match="source vertex axis"):
        ChromaticBipartitionResult(
            graph=source,
            s=2,
            t=2,
            status="SPLIT",
            side_a=("a", "b"),
            side_b=("d", "c"),
            chromatic_a=2,
            chromatic_b=2,
            checked_partitions=3,
        )


def test_operation_rejects_a_result_axis_above_its_admitted_envelope() -> None:
    vertices = tuple(f"v{i}" for i in range(257))
    request = ChromaticBipartitionRequest(graph=graph(vertices, ()), s=1, t=1)
    with pytest.raises(OperationResourceAdmissionError, match="at most 256"):
        find_chromatic_bipartition(request)


def test_edgeless_graph_above_the_witness_cap_is_exact_no_split() -> None:
    vertices = tuple(f"v{i}" for i in range(257))
    request = ChromaticBipartitionRequest(graph=graph(vertices, ()), s=2, t=1)
    limit = process_owner._chromatic_bipartition_worker_stdout_limit(request)
    result = find_chromatic_bipartition(request)
    assert result.status == "NO_SPLIT"
    assert result.side_a is None
    assert result.checked_partitions == 0
    dumped = json.dumps(
        result.model_dump(mode="json"),
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert len(dumped) <= limit


def test_one_vertex_long_label_is_exact_no_split() -> None:
    source = graph(("x" * 400_000,), ())
    result = find_chromatic_bipartition(
        ChromaticBipartitionRequest(graph=source, s=1, t=1)
    )
    assert result.status == "NO_SPLIT"
    assert result.checked_partitions == 0


def test_operation_rejects_excessive_retained_source_labels() -> None:
    vertices = tuple(f"v{i:03d}" + "x" * 60 for i in range(256))
    edges = tuple(
        (left, right)
        for index, left in enumerate(vertices)
        for right in vertices[index + 1 :]
    )
    request = ChromaticBipartitionRequest(graph=graph(vertices, edges), s=2, t=2)
    with pytest.raises(OperationResourceAdmissionError, match="label"):
        find_chromatic_bipartition(request)


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


def test_unit_threshold_counts_only_computed_remainders() -> None:
    cycle = tuple(f"c{index}" for index in range(33))
    vertices = ("iso", *cycle)
    edges = tuple(
        (min(cycle[index], cycle[(index + 1) % 33]), max(cycle[index], cycle[(index + 1) % 33]))
        for index in range(33)
    )
    result = operation._find_chromatic_bipartition_kernel(
        ChromaticBipartitionRequest(graph=graph(vertices, edges), s=1, t=1)
    )
    assert result.status == "SPLIT"
    assert result.checked_partitions == 1


def test_split_charges_partition_labels_once() -> None:
    left = "a" * 140_000
    right = "b" * 140_000
    result = find_chromatic_bipartition(
        ChromaticBipartitionRequest(
            graph=graph((left, right), ((left, right),)),
            s=1,
            t=1,
        )
    )
    assert result.status == "SPLIT"
