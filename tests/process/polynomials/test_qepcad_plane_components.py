"""Kill-safe QEPCAD protocol for exact plane-component profiles."""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import time
from pathlib import Path
from threading import Event, Thread
from typing import Never

import pytest

from jacobian._execution import OperationExecutionCancelledError
from jacobian.math.polynomials.real_algebra import (
    _qepcad_plane_process,
    _qepcad_plane_worker,
)
from jacobian.math.polynomials.real_algebra._plane_component_models import (
    PlaneComponentProfileRequest,
)
from jacobian.math.polynomials.real_algebra._qepcad_plane_process import (
    QepcadPlaneProcessOutcome,
    _run_worker,
    _worker_outcome,
    run_plane_sample_recognition,
    run_qepcad_plane_components,
)
from jacobian.math.polynomials.real_algebra._qepcad_plane_protocol import (
    PlaneSampleWorkerRequest,
    QepcadPlaneWorkerRejected,
    QepcadPlaneWorkerRequest,
)
from jacobian.math.polynomials.real_algebra._qepcad_plane_worker import (
    _parse_cell_indices,
    _parse_true_cells,
    _QepcadCellLimitError,
    _QepcadProtocolError,
)
from jacobian.process import (
    BoundedProcessResult,
    bounded_process_cancellation,
    run_bounded_process,
)

_NESTED_DIALOGUE_WORKER = Path(__file__).with_name(
    "_nested_dialogue_cancellation_worker.py"
)
_TRUE_CELL_PREFIX = "\nd-true-cells\n"
_TRUE_CELL_SUFFIX = "\nBefore Solution >"


def _true_cell_block(index: tuple[int, int] = (3, 3)) -> str:
    return (
        f"---------- Information about the cell ({index[0]},{index[1]}) "
        "----------\n\n"
        "Level                       : 2\n"
        "Dimension                   : 0\n"
        "Number of children          : 0\n"
        "Truth value                 : TRUE\n"
        "----------   Sample point  ---------- \n"
        "alpha = 0\n"
        "\n----------------------------------------------------\n"
    )


def _open_unit_disk_request() -> PlaneComponentProfileRequest:
    return PlaneComponentProfileRequest.model_validate(
        {
            "semialgebraic_set": {
                "axis": ["x", "y"],
                "polynomials": [
                    {
                        "variables": ["x", "y"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [2, 0],
                                },
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [0, 2],
                                },
                                {
                                    "coefficient": {"num": "-1", "den": "1"},
                                    "exponents": [0, 0],
                                },
                            ]
                        },
                    }
                ],
                "sign_conditions": [{"signs": ["NEGATIVE"]}],
            },
            "samples": [],
        }
    )


def _whole_plane_with_origin() -> PlaneComponentProfileRequest:
    coordinate = {"polynomial": ["1", "0"], "real_root_index": 0}
    interval = {
        "lower": {"num": "0", "den": "1"},
        "upper": {"num": "0", "den": "1"},
    }
    return PlaneComponentProfileRequest.model_validate(
        {
            "semialgebraic_set": {
                "axis": ["x", "y"],
                "polynomials": [],
                "sign_conditions": [{"signs": []}],
            },
            "samples": [
                {
                    "axis": ["x", "y"],
                    "coordinates": [coordinate, coordinate],
                    "isolating_box": {
                        "domain": "QQ",
                        "variables": ["x", "y"],
                        "intervals": [interval, interval],
                    },
                }
            ],
        }
    )


def test_worker_is_not_launched_when_serialization_exhausts_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = iter((10.0, 12.0))
    monkeypatch.setattr(_qepcad_plane_process, "monotonic", lambda: next(times))

    def unexpected_launch(*args: object, **kwargs: object) -> Never:
        raise AssertionError("expired worker request was launched")

    monkeypatch.setattr("jacobian.process.run_bounded_process", unexpected_launch)

    assert (
        _run_worker(
            PlaneSampleWorkerRequest(samples=()),
            deadline=11.0,
            stdout_limit=1_024,
        )
        is None
    )


def test_degenerate_sample_recognition_does_not_require_qepcad() -> None:
    outcome = run_plane_sample_recognition(
        _whole_plane_with_origin(),
        wall_seconds=30.0,
    )

    assert outcome == QepcadPlaneProcessOutcome(status="COMPUTED")


def test_degenerate_sample_recognition_deadline_is_nonconclusive() -> None:
    outcome = run_plane_sample_recognition(
        _whole_plane_with_origin(),
        wall_seconds=0.000_001,
    )

    assert outcome == QepcadPlaneProcessOutcome(
        status="TIMEOUT",
        reason="SAMPLE_RECOGNITION_DEADLINE_EXPIRED",
    )


def test_degenerate_sample_recognition_observes_cancellation() -> None:
    cancellation = Event()
    cancellation.set()

    with (
        bounded_process_cancellation(cancellation),
        pytest.raises(OperationExecutionCancelledError, match="cancelled"),
    ):
        run_plane_sample_recognition(
            _whole_plane_with_origin(),
            wall_seconds=30.0,
        )


@pytest.mark.requires_backend("qepcad")
@pytest.mark.skipif(
    shutil.which("qepcad") is None,
    reason="the exact plane-component backend is unavailable",
)
def test_qepcad_process_returns_the_compact_exact_projection() -> None:
    outcome = run_qepcad_plane_components(
        _open_unit_disk_request(),
        wall_seconds=30.0,
    )

    assert outcome.status == "COMPUTED"
    assert outcome.version == "1.74"
    assert outcome.projection is not None
    assert len(outcome.projection.representatives) == 1


@pytest.mark.requires_backend("qepcad")
@pytest.mark.skipif(
    shutil.which("qepcad") is None,
    reason="the exact plane-component backend is unavailable",
)
def test_qepcad_deadline_is_an_explicit_noncompletion() -> None:
    outcome = run_qepcad_plane_components(
        _open_unit_disk_request(),
        wall_seconds=0.000_001,
    )

    assert outcome == QepcadPlaneProcessOutcome(
        status="TIMEOUT",
        reason="QEPCAD_DEADLINE_EXPIRED",
    )


@pytest.mark.skipif(
    os.name != "posix",
    reason="process-group descendant cleanup is exercised on POSIX",
)
def test_live_nested_dialogue_child_is_killed_when_the_request_is_cancelled(
    tmp_path: Path,
) -> None:
    cancellation = Event()
    marker = tmp_path / "nested.pid"
    completed: list[BoundedProcessResult] = []
    failures: list[BaseException] = []

    def run_outer_worker() -> None:
        try:
            with bounded_process_cancellation(cancellation):
                completed.append(
                    run_bounded_process(
                        [sys.executable, str(_NESTED_DIALOGUE_WORKER), str(marker)],
                        input_bytes=b"",
                        timeout_seconds=20,
                        environment=dict(os.environ),
                        stdout_limit=4_096,
                        stderr_limit=4_096,
                    )
                )
        except BaseException as exc:  # retain failures for the asserting thread
            failures.append(exc)

    outer_worker = Thread(target=run_outer_worker)
    outer_worker.start()
    nested_pid: int | None = None
    try:
        ready_deadline = time.monotonic() + 5
        while time.monotonic() < ready_deadline:
            if marker.exists() and (encoded_pid := marker.read_text().strip()):
                nested_pid = int(encoded_pid)
                break
            time.sleep(0.01)
        if nested_pid is None:
            raise AssertionError("nested dialogue child did not publish readiness")

        cancellation.set()
        outer_worker.join(timeout=15)

        assert not outer_worker.is_alive()
        assert failures == []
        assert len(completed) == 1
        assert completed[0].cancelled
        assert not completed[0].timed_out

        exit_deadline = time.monotonic() + 15
        while time.monotonic() < exit_deadline:
            try:
                os.kill(nested_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("nested dialogue child survived request cancellation")
    finally:
        cancellation.set()
        outer_worker.join(timeout=15)
        if nested_pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(nested_pid, 9)


def test_missing_qepcad_is_explicitly_nonconclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_which = shutil.which

    def without_qepcad(name: str) -> str | None:
        if name == "qepcad":
            return None
        return original_which(name)

    monkeypatch.setattr(shutil, "which", without_qepcad)

    outcome = run_qepcad_plane_components(
        _open_unit_disk_request(),
        wall_seconds=30.0,
    )

    assert outcome == QepcadPlaneProcessOutcome(
        status="BACKEND_UNAVAILABLE",
        reason="SUPPORTED_QEPCAD_NOT_INSTALLED",
    )


def test_unsupported_qepcad_version_is_backend_unavailability() -> None:
    outcome = _worker_outcome(
        QepcadPlaneWorkerRejected(reason="UNSUPPORTED_QEPCAD_VERSION")
    )

    assert outcome == QepcadPlaneProcessOutcome(
        status="BACKEND_UNAVAILABLE",
        reason="UNSUPPORTED_QEPCAD_VERSION",
    )


def test_nested_qepcad_deadline_remains_nonconclusive() -> None:
    outcome = _worker_outcome(
        QepcadPlaneWorkerRejected(reason="QEPCAD_DEADLINE_EXPIRED")
    )

    assert outcome == QepcadPlaneProcessOutcome(
        status="TIMEOUT",
        reason="QEPCAD_DEADLINE_EXPIRED",
    )


def test_oversized_qepcad_cell_index_is_a_typed_cell_limit() -> None:
    output = (
        _TRUE_CELL_PREFIX
        + "---------- Information about the cell ("
        + "9" * 5_000
        + ",1) ----------\n\n"
        + "\n----------------------------------------------------\n"
        + _TRUE_CELL_SUFFIX
    )

    with pytest.raises(_QepcadCellLimitError, match="index exceeded"):
        _parse_cell_indices(output)


def test_exact_empty_true_cell_frame_is_the_only_empty_family() -> None:
    output = _TRUE_CELL_PREFIX + _TRUE_CELL_SUFFIX

    assert _parse_true_cells(output) == ()
    assert _parse_cell_indices(output) == ()


@pytest.mark.parametrize(
    "output",
    (
        "QEPCAD command failed\nBefore Solution >",
        _TRUE_CELL_PREFIX + "QEPCAD diagnostic" + _TRUE_CELL_SUFFIX,
        _TRUE_CELL_PREFIX
        + _true_cell_block()
        + "QEPCAD trailing diagnostic"
        + _TRUE_CELL_SUFFIX,
    ),
)
def test_true_cell_parser_rejects_diagnostics_and_unconsumed_text(
    output: str,
) -> None:
    with pytest.raises(_QepcadProtocolError):
        _parse_true_cells(output)
    with pytest.raises(_QepcadProtocolError):
        _parse_cell_indices(output)


def test_true_cell_parser_consumes_one_complete_cell_block() -> None:
    output = _TRUE_CELL_PREFIX + _true_cell_block() + _TRUE_CELL_SUFFIX

    cells = _parse_true_cells(output)

    assert len(cells) == 1
    assert cells[0].index == (3, 3)
    assert cells[0].dimension == 0
    assert cells[0].sample == "alpha = 0"
    assert _parse_cell_indices(output) == ((3, 3),)


def test_duplicate_true_cell_indices_are_rejected_by_both_consumers() -> None:
    output = (
        _TRUE_CELL_PREFIX + _true_cell_block() + _true_cell_block() + _TRUE_CELL_SUFFIX
    )

    with pytest.raises(_QepcadProtocolError, match="repeated"):
        _parse_true_cells(output)
    with pytest.raises(_QepcadProtocolError, match="repeated"):
        _parse_cell_indices(output)


def test_malformed_true_cell_frame_becomes_typed_invalid_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed = _TRUE_CELL_PREFIX + "QEPCAD diagnostic" + _TRUE_CELL_SUFFIX

    def malformed_backend(*_args: object, **_kwargs: object) -> Never:
        _parse_true_cells(malformed)
        raise AssertionError("malformed frame unexpectedly parsed")

    monkeypatch.setattr(_qepcad_plane_worker, "_run_qepcad", malformed_backend)
    response = _qepcad_plane_worker._compute(
        QepcadPlaneWorkerRequest(
            executable="unused",
            qepcad_root="unused",
            deadline_monotonic=time.monotonic() + 30,
            request=_open_unit_disk_request(),
        )
    )

    assert response == QepcadPlaneWorkerRejected(reason="QEPCAD_INVALID_OUTPUT")
