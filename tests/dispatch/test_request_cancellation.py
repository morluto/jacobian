from __future__ import annotations

import threading
import time

import pytest

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionStage,
    request_cancellation,
    request_cancelled,
    request_checkpoint,
    request_execution,
    request_stage,
)


class _FalseyEvent(threading.Event):
    def __bool__(self) -> bool:
        return False


@pytest.mark.parametrize("signal_type", [threading.Event, _FalseyEvent])
@pytest.mark.parametrize("stage", list(OperationExecutionStage))
def test_request_checkpoint_honors_legacy_cancellation(
    signal_type: type[threading.Event], stage: OperationExecutionStage
) -> None:
    signal = signal_type()
    signal.set()

    with request_cancellation(signal), request_stage(stage):
        with pytest.raises(
            OperationExecutionCancelledError, match="request cancelled at checkpoint"
        ) as caught:
            request_checkpoint("at checkpoint")
        assert caught.value.stage is stage
        assert request_cancelled()


@pytest.mark.parametrize("signal_type", [threading.Event, _FalseyEvent])
def test_unset_legacy_signal_overrides_cancelled_envelope(
    signal_type: type[threading.Event],
) -> None:
    legacy = signal_type()
    envelope = threading.Event()
    envelope.set()

    with request_execution(time.monotonic(), cancellation_signal=envelope):
        with request_cancellation(legacy):
            assert not request_cancelled()
            request_checkpoint("with unset legacy signal")
        assert request_cancelled()
        with pytest.raises(OperationExecutionCancelledError):
            request_checkpoint("with envelope signal")

    assert not request_cancelled()
    request_checkpoint("without cancellation signal")
