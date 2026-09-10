"""Operational failures do not establish an augmentation result."""

import time

import networkx as nx
import pytest

from jacobian._execution import (
    BackendFailureReason,
    ExecutionResource,
    OperationBackendError,
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    OperationResourceExhaustedError,
)
from jacobian._worker_protocol import encode_worker_result_frame
from jacobian.math.graphs.triangle_free_diameter_augmentation import (
    _augmentation_z3 as owner,
)
from jacobian.math.graphs.triangle_free_diameter_augmentation._models import (
    TriangleFreeDiameterAugmentationBudget,
)
from jacobian.math.graphs.triangle_free_diameter_augmentation.operations import (
    triangle_free_diameter_augmentation,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.process import BoundedProcessResult


def test_triangle_check_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = SimpleUndirectedGraph(vertices=("a", "b"), edges=(("a", "b"),))

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("triangle backend failure")

    monkeypatch.setattr(nx, "triangles", fail)
    with pytest.raises(RuntimeError, match="triangle backend failure"):
        triangle_free_diameter_augmentation(graph, 1)


def _nontrivial_source() -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=("0", "1", "2", "3"),
        edges=(("0", "1"), ("1", "2"), ("2", "3")),
    )


@pytest.mark.parametrize(
    ("completed", "expected"),
    [
        (
            BoundedProcessResult(0, b"", b"", False, False, True),
            OperationExecutionTimeoutError,
        ),
        (
            BoundedProcessResult(0, b"", b"", False, False, False, True),
            OperationExecutionCancelledError,
        ),
        (
            BoundedProcessResult(0, b"", b"", True, False, False),
            OperationResourceExhaustedError,
        ),
        (
            BoundedProcessResult(0, b"", b"", False, True, False),
            OperationResourceExhaustedError,
        ),
        (BoundedProcessResult(2, b"", b"", False, False, False), OperationBackendError),
        (
            BoundedProcessResult(0, b"not-json", b"", False, False, False),
            OperationBackendError,
        ),
    ],
)
def test_worker_noncompletion_is_an_execution_error(
    monkeypatch: pytest.MonkeyPatch,
    completed: BoundedProcessResult,
    expected: type[Exception],
) -> None:
    monkeypatch.setattr(
        owner,
        "run_bounded_process",
        lambda *args, **kwargs: completed,
    )
    with pytest.raises(expected):
        triangle_free_diameter_augmentation(
            _nontrivial_source(),
            2,
            resource_budget=TriangleFreeDiameterAugmentationBudget(wall_seconds=5),
        )


def test_worker_startup_failure_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise OSError

    monkeypatch.setattr(owner, "run_bounded_process", fail)
    with pytest.raises(OperationBackendError) as raised:
        triangle_free_diameter_augmentation(_nontrivial_source(), 2)
    assert raised.value.reason is BackendFailureReason.STARTUP


def test_worker_output_exhaustion_identifies_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        owner,
        "run_bounded_process",
        lambda *args, **kwargs: BoundedProcessResult(0, b"", b"", True, False, False),
    )
    with pytest.raises(OperationResourceExhaustedError) as raised:
        triangle_free_diameter_augmentation(_nontrivial_source(), 2)
    assert raised.value.resource is ExecutionResource.OUTPUT


@pytest.mark.parametrize(
    ("added_edges", "diameter"),
    [
        ([["0", "2"]], 2),
        ([["0", "3"]], 1),
    ],
)
def test_parent_rejects_forged_exact_worker_witness(
    monkeypatch: pytest.MonkeyPatch,
    added_edges: list[list[str]],
    diameter: int,
) -> None:
    projection = {
        "target_diameter": 2,
        "status": "EXACT",
        "added_edge_count": 1,
        "added_edges": added_edges,
        "augmented_diameter": diameter,
        "detail": "forged worker result",
    }
    monkeypatch.setattr(
        owner,
        "run_bounded_process",
        lambda *args, **kwargs: BoundedProcessResult(
            0,
            encode_worker_result_frame(projection),
            b"",
            False,
            False,
            False,
        ),
    )
    with pytest.raises(OperationBackendError) as raised:
        triangle_free_diameter_augmentation(_nontrivial_source(), 2)
    assert raised.value.reason is BackendFailureReason.INVALID_OUTPUT


def test_expired_startup_failure_preserves_timeout_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]

    def fail(*args: object, **kwargs: object) -> None:
        now[0] = 10.0
        raise OSError

    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    monkeypatch.setattr(owner, "run_bounded_process", fail)
    with pytest.raises(OperationExecutionTimeoutError):
        triangle_free_diameter_augmentation(_nontrivial_source(), 2)
