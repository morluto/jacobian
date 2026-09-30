"""Face closure observes late cancellation inside each admitted work phase."""

from threading import Event

import pytest

from jacobian._execution import (
    OperationExecutionCancelledError,
    request_cancellation,
    request_checkpoint,
)
from jacobian.math.topology.cubical_complexes import operations
from jacobian.math.topology.cubical_complexes._models import CubicalCell


@pytest.mark.parametrize(
    ("phase", "dimension"),
    [("expansion", 10), ("materialization", 5)],
)
def test_face_closure_cancels_after_work_has_started(
    monkeypatch: pytest.MonkeyPatch, phase: str, dimension: int
) -> None:
    cell = CubicalCell(intervals=((0, 1),) * dimension)
    cancelled = Event()
    observed: list[str] = []

    def cancel_during_work(stage: str) -> None:
        if phase in stage:
            observed.append(stage)
            if len(observed) == 2:
                cancelled.set()
        request_checkpoint(stage)

    monkeypatch.setattr(operations, "request_checkpoint", cancel_during_work)
    with (
        request_cancellation(cancelled),
        pytest.raises(OperationExecutionCancelledError, match=phase),
    ):
        operations.face_closure((cell,))
    assert len(observed) == 2
    assert cancelled.is_set()


def test_face_closure_preserves_complete_sorted_faces() -> None:
    result = operations.face_closure((CubicalCell(intervals=((0, 1),) * 4),))
    assert result.total_cells == 3**4
    intervals = tuple(cell.intervals for cell in result.complex.cells)
    assert intervals == tuple(sorted(set(intervals)))
