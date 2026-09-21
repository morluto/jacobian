from __future__ import annotations

import json
import sys
import threading
import time

import pytest

from jacobian._execution import (
    BackendFailureReason,
    OperationBackendError,
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    OperationResourceExhaustedError,
    ProgressSink,
    request_execution,
)
from jacobian._worker_protocol import (
    encode_worker_error_frame,
    encode_worker_progress_frame,
    encode_worker_result_frame,
)
from jacobian.process import (
    decode_checked_worker_output,
    run_checked_worker_process,
    worker_environment,
)


class _Progress(ProgressSink):
    def __init__(self) -> None:
        self.values: list[tuple[int, int | None, str | None]] = []

    def report(
        self, progress: int, *, total: int | None = None, message: str | None = None
    ) -> None:
        self.values.append((progress, total, message))


def test_checked_worker_decodes_progress_and_one_result() -> None:
    sink = _Progress()
    output = encode_worker_progress_frame(2, total=5, message="searched")
    output += encode_worker_result_frame({"answer": 7})
    with request_execution(time.monotonic(), progress_sink=sink):
        result = decode_checked_worker_output(output, decode_result=lambda value: value)
    assert result == {"answer": 7}
    assert sink.values == [(2, 5, "searched")]


def test_checked_worker_accepts_a_none_result() -> None:
    assert (
        decode_checked_worker_output(
            encode_worker_result_frame(None), decode_result=lambda value: value
        )
        is None
    )


@pytest.mark.parametrize(
    "output",
    [
        b'{"kind":"result","result":1}',
        b'{"kind":"result","result":1}\v',
        b'{"kind":"result","result":1,"result":2}\n',
    ],
)
def test_checked_worker_rejects_non_strict_framing(output: bytes) -> None:
    with pytest.raises(OperationBackendError) as caught:
        decode_checked_worker_output(output, decode_result=lambda value: value)
    assert caught.value.reason == BackendFailureReason.MALFORMED_RESPONSE


@pytest.mark.parametrize(
    "output",
    [
        b"",
        b"not-json\n",
        encode_worker_progress_frame(2) + encode_worker_progress_frame(1),
        encode_worker_result_frame(1) + encode_worker_result_frame(2),
        json.dumps({"kind": "result"}).encode() + b"\n",
    ],
)
def test_checked_worker_rejects_malformed_or_nonterminal_frames(output: bytes) -> None:
    with pytest.raises(OperationBackendError) as caught:
        decode_checked_worker_output(output, decode_result=lambda value: value)
    assert caught.value.reason == BackendFailureReason.MALFORMED_RESPONSE


def test_checked_worker_checks_deadline_after_result_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr("jacobian._execution.time.monotonic", lambda: clock[0])

    def slow_decode(value: object) -> object:
        clock[0] = 2.0
        return value

    with (
        request_execution(0.0, outer_deadline=1.0),
        pytest.raises(OperationExecutionTimeoutError),
    ):
        decode_checked_worker_output(
            encode_worker_result_frame({"answer": 7}), decode_result=slow_decode
        )


def _run_real_checked_worker(script: str, **kwargs: object) -> object:
    return run_checked_worker_process(
        [sys.executable, "-c", script],
        input_bytes=b"",
        timeout_seconds=float(kwargs.pop("timeout_seconds", 2.0)),
        environment=worker_environment(),
        stdout_limit=int(kwargs.pop("stdout_limit", 1024)),
        stderr_limit=int(kwargs.pop("stderr_limit", 1024)),
        decode_result=lambda value: value,
        **kwargs,
    )


def test_real_checked_worker_timeout_is_classified_before_decode() -> None:
    with pytest.raises(OperationExecutionTimeoutError):
        _run_real_checked_worker("import time; time.sleep(1)", timeout_seconds=0.02)


def test_real_checked_worker_cancellation_is_classified_before_decode() -> None:
    cancellation = threading.Event()
    cancellation.set()
    with pytest.raises(OperationExecutionCancelledError):
        _run_real_checked_worker("print('never')", cancellation_event=cancellation)


def test_real_checked_worker_output_overflow_is_classified_before_decode() -> None:
    script = "import sys; sys.stdout.write('x' * 4096); sys.stdout.flush()"
    with pytest.raises(OperationResourceExhaustedError):
        _run_real_checked_worker(script, stdout_limit=128)


def test_real_checked_worker_abnormal_exit_is_classified_before_decode() -> None:
    with pytest.raises(OperationBackendError) as caught:
        _run_real_checked_worker("raise SystemExit(7)")
    assert caught.value.reason is BackendFailureReason.ABNORMAL_EXIT


def test_checked_worker_accepts_owner_frame_limit_above_default() -> None:
    payload = {"answer": "x" * (16 * 1024 * 1024)}
    output = encode_worker_result_frame(payload)
    assert (
        decode_checked_worker_output(
            output,
            decode_result=lambda value: value,
            max_frame_bytes=17 * 1024 * 1024,
        )
        == payload
    )


def test_checked_worker_preserves_classified_backend_error() -> None:
    with pytest.raises(OperationBackendError) as caught:
        decode_checked_worker_output(
            encode_worker_error_frame(BackendFailureReason.INVALID_OUTPUT),
            decode_result=lambda value: value,
        )
    assert caught.value.reason == BackendFailureReason.INVALID_OUTPUT
