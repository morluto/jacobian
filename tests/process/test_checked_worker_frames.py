from __future__ import annotations

import json
import time

import pytest

from jacobian._execution import (
    BackendFailureReason,
    OperationBackendError,
    ProgressSink,
    request_execution,
)
from jacobian.process import (
    decode_checked_worker_output,
    encode_worker_error_frame,
    encode_worker_progress_frame,
    encode_worker_result_frame,
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


def test_checked_worker_preserves_classified_backend_error() -> None:
    with pytest.raises(OperationBackendError) as caught:
        decode_checked_worker_output(
            encode_worker_error_frame(BackendFailureReason.INVALID_OUTPUT),
            decode_result=lambda value: value,
        )
    assert caught.value.reason == BackendFailureReason.INVALID_OUTPUT
