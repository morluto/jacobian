"""Source-authored timeout context survives a checked production worker."""

import json

import pytest
from tests.support.chromatic_timeout import chromatic_worker_timeout

from jacobian._execution import OperationExecutionTimeoutError
from jacobian.math.graphs.optimization import (
    _chromatic_bipartition_process as process_owner,
)
from jacobian.math.graphs.optimization._chromatic_bipartition import (
    ChromaticBipartitionRequest,
    find_chromatic_bipartition,
)
from jacobian.process import BoundedProcessResult


@pytest.mark.parametrize("wall_seconds", [1, 5, 120])
def test_chromatic_kernel_context_survives_checked_parent(
    monkeypatch: pytest.MonkeyPatch, wall_seconds: int
) -> None:
    request, output = chromatic_worker_timeout(monkeypatch, wall_seconds)

    def complete(*args: object, **kwargs: object) -> BoundedProcessResult:
        assert kwargs["stdout_limit"] == (
            process_owner._chromatic_bipartition_worker_stdout_limit(request)
        )
        return BoundedProcessResult(0, output, b"", False, False, False)

    monkeypatch.setattr(process_owner, "run_bounded_process", complete)
    with pytest.raises(OperationExecutionTimeoutError) as caught:
        find_chromatic_bipartition(request)
    assert caught.value.configured_seconds == wall_seconds
    assert caught.value.adjustable_field_path == ("resource_budget", "wall_seconds")
    assert caught.value.timeout_owner == "operation_wall"
    assert caught.value.elapsed_seconds is None
    assert caught.value.maximum_seconds is None

    assert json.loads(output) == {
        "kind": "execution_error",
        "stage": "operation_execution",
        "reason": "timeout",
        "configured_seconds": wall_seconds,
        "adjustable_field_path": ["resource_budget", "wall_seconds"],
    }
    # This is the largest actual source-authored frame, even at the legal wall
    # maximum. It fits the owner's smallest result envelope without more stdout.
    empty = ChromaticBipartitionRequest.model_validate(
        {"graph": {"vertices": [], "edges": []}, "s": 1, "t": 1}
    )
    assert len(output) <= 160
    assert len(output) <= process_owner._chromatic_bipartition_worker_stdout_limit(
        empty
    )
    assert len(output) <= process_owner._chromatic_bipartition_worker_stdout_limit(
        request
    )


def test_inherited_worker_checkpoint_does_not_claim_the_source_wall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, output = chromatic_worker_timeout(
        monkeypatch, wall_seconds=120, expired_parent_deadline=True
    )
    monkeypatch.setattr(
        process_owner,
        "run_bounded_process",
        lambda *args, **kwargs: BoundedProcessResult(
            0, output, b"", False, False, False
        ),
    )
    with pytest.raises(OperationExecutionTimeoutError) as caught:
        find_chromatic_bipartition(request)
    assert caught.value.configured_seconds is None
    assert caught.value.adjustable_field_path is None
    assert json.loads(output) == {
        "kind": "execution_error",
        "stage": "operation_execution",
        "reason": "timeout",
    }
