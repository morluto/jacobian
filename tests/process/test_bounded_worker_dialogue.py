from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable
from typing import Any, Never

import pytest

from jacobian.process import (
    BoundedWorkerDialogue,
    BoundedWorkerDialogueCompleted,
    BoundedWorkerDialogueError,
    BoundedWorkerDialogueErrorReason,
    run_bounded_worker_dialogue,
)


def _run_dialogue[ValueT](
    source: str,
    callback: Callable[[BoundedWorkerDialogue], ValueT],
    *,
    timeout_seconds: float = 2,
    stdout_limit: int = 4096,
    stderr_limit: int = 4096,
) -> BoundedWorkerDialogueCompleted[ValueT]:
    return run_bounded_worker_dialogue(
        [sys.executable, "-u", "-c", source],
        callback,
        absolute_deadline=time.monotonic() + timeout_seconds,
        environment=dict(os.environ),
        stdout_limit=stdout_limit,
        stderr_limit=stderr_limit,
    )


def test_adaptive_exchange_returns_callback_value_and_cumulative_output() -> None:
    source = (
        "import sys\n"
        "out = sys.stdout.buffer\n"
        "inp = sys.stdin.buffer\n"
        "out.write(b'number?'); out.flush()\n"
        "value = int(inp.readline())\n"
        "out.write(b'offset?'); out.flush()\n"
        "offset = int(inp.readline())\n"
        "out.write(f'{value + offset}!'.encode()); out.flush()\n"
    )

    def exchange(dialogue: BoundedWorkerDialogue) -> int:
        assert dialogue.read_until(b"?", frame_limit=32) == b"number?"
        dialogue.send(b"12\n")
        assert dialogue.read_until(b"?", frame_limit=32) == b"offset?"
        dialogue.send(b"5\n")
        frame = dialogue.read_until(b"!", frame_limit=32)
        return int(frame.removesuffix(b"!"))

    completed = _run_dialogue(source, exchange)

    assert completed.value == 17
    assert completed.stdout_bytes == len(b"number?offset?17!")


def test_read_until_preserves_the_unread_suffix() -> None:
    frames: list[bytes] = []

    def exchange(dialogue: BoundedWorkerDialogue) -> None:
        frames.append(dialogue.read_until(b"<END>", frame_limit=32))
        frames.append(dialogue.read_until(b"<END>", frame_limit=32))

    completed = _run_dialogue(
        "import os\nos.write(1, b'first<END>second<END>')\n",
        exchange,
    )

    assert frames == [b"first<END>", b"second<END>"]
    assert completed.stdout_bytes == len(b"first<END>second<END>")


def test_stdout_limit_is_cumulative_across_adaptive_frames() -> None:
    source = "import os\nos.write(1, b'A!')\nos.read(0, 1)\nos.write(1, b'BBBB!')\n"

    def exchange(dialogue: BoundedWorkerDialogue) -> None:
        assert dialogue.read_until(b"!", frame_limit=8) == b"A!"
        dialogue.send(b"x")
        dialogue.read_until(b"!", frame_limit=8)

    with pytest.raises(BoundedWorkerDialogueError) as raised:
        _run_dialogue(source, exchange, stdout_limit=6)

    assert raised.value.reason is BoundedWorkerDialogueErrorReason.STDOUT_LIMIT


def test_read_until_enforces_its_frame_limit() -> None:
    with pytest.raises(BoundedWorkerDialogueError) as raised:
        _run_dialogue(
            "import os\nos.write(1, b'12345!')\n",
            lambda dialogue: dialogue.read_until(b"!", frame_limit=5),
        )

    assert raised.value.reason is BoundedWorkerDialogueErrorReason.STDOUT_LIMIT


def test_expired_absolute_deadline_does_not_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_to_launch(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError("expired dialogue must not launch a child")

    monkeypatch.setattr("jacobian.process.subprocess.Popen", fail_to_launch)

    with pytest.raises(BoundedWorkerDialogueError) as raised:
        run_bounded_worker_dialogue(
            [sys.executable, "-c", "raise SystemExit(0)"],
            lambda _dialogue: None,
            absolute_deadline=time.monotonic() - 1,
            environment=dict(os.environ),
            stdout_limit=1,
            stderr_limit=1,
        )

    assert raised.value.reason is BoundedWorkerDialogueErrorReason.DEADLINE_EXPIRED


def test_start_failure_is_typed() -> None:
    with pytest.raises(BoundedWorkerDialogueError) as raised:
        run_bounded_worker_dialogue(
            ["/definitely/not/a/worker-dialogue-executable"],
            lambda _dialogue: None,
            absolute_deadline=time.monotonic() + 1,
            environment=dict(os.environ),
            stdout_limit=1,
            stderr_limit=1,
        )

    assert raised.value.reason is BoundedWorkerDialogueErrorReason.START_FAILED
    assert raised.value.stderr == b""


def test_nonzero_exit_reports_bounded_stderr() -> None:
    with pytest.raises(BoundedWorkerDialogueError) as raised:
        _run_dialogue(
            "import os\nos.write(2, b'bad input')\nraise SystemExit(7)\n",
            lambda _dialogue: None,
            stderr_limit=4,
        )

    assert raised.value.reason is BoundedWorkerDialogueErrorReason.STDERR_LIMIT
    assert raised.value.stderr == b"bad "


def test_nonzero_exit_without_output_limit_is_distinct() -> None:
    with pytest.raises(BoundedWorkerDialogueError) as raised:
        _run_dialogue(
            "import os\nos.write(2, b'bad input')\nraise SystemExit(7)\n",
            lambda _dialogue: None,
        )

    assert raised.value.reason is BoundedWorkerDialogueErrorReason.NONZERO_EXIT
    assert raised.value.stderr == b"bad input"


def test_closed_stdout_is_not_reported_as_a_deadline() -> None:
    source = "import os, time\nos.close(1)\ntime.sleep(30)\n"

    with pytest.raises(BoundedWorkerDialogueError) as raised:
        _run_dialogue(
            source,
            lambda dialogue: dialogue.read_until(b"!", frame_limit=8),
            timeout_seconds=1,
        )

    assert raised.value.reason is BoundedWorkerDialogueErrorReason.CLOSED


@pytest.mark.parametrize(
    ("source", "callback"),
    [
        (
            "import time\ntime.sleep(30)\n",
            lambda dialogue: dialogue.read_until(b"!", frame_limit=8),
        ),
        (
            "import time\ntime.sleep(30)\n",
            lambda dialogue: dialogue.send(b"x" * (8 * 1024 * 1024)),
        ),
    ],
    ids=("blocked-read", "blocked-write"),
)
def test_blocked_pipe_operation_uses_the_shared_absolute_deadline(
    source: str,
    callback: Callable[[BoundedWorkerDialogue], Any],
) -> None:
    started = time.monotonic()

    with pytest.raises(BoundedWorkerDialogueError) as raised:
        _run_dialogue(source, callback, timeout_seconds=0.25)

    assert raised.value.reason is BoundedWorkerDialogueErrorReason.DEADLINE_EXPIRED
    assert time.monotonic() - started < 2


def test_prebuffered_frame_cannot_be_consumed_after_the_deadline() -> None:
    consumed: list[bytes] = []

    def exchange(dialogue: BoundedWorkerDialogue) -> None:
        assert dialogue.read_until(b"?", frame_limit=32) == b"ready?"
        time.sleep(0.3)
        consumed.append(dialogue.read_until(b"!", frame_limit=32))

    with pytest.raises(BoundedWorkerDialogueError) as raised:
        _run_dialogue(
            "import os, time\nos.write(1, b'ready?late!')\ntime.sleep(30)\n",
            exchange,
            timeout_seconds=0.2,
        )

    assert raised.value.reason is BoundedWorkerDialogueErrorReason.DEADLINE_EXPIRED
    assert consumed == []


@pytest.mark.parametrize("payload", (b"", b"x"), ids=("empty", "completed"))
def test_send_cannot_complete_after_the_deadline(
    payload: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[bytes] = []
    original_start = threading.Thread.start

    def start_and_complete_writer(thread: threading.Thread) -> None:
        original_start(thread)
        if thread.name == "bounded-worker-dialogue-stdin":
            thread.join(timeout=1)

    monkeypatch.setattr(threading.Thread, "start", start_and_complete_writer)

    def exchange(dialogue: BoundedWorkerDialogue) -> None:
        assert dialogue.read_until(b"?", frame_limit=32) == b"ready?"
        time.sleep(0.3)
        dialogue.send(payload)
        sent.append(payload)

    with pytest.raises(BoundedWorkerDialogueError) as raised:
        _run_dialogue(
            "import os, time\nos.write(1, b'ready?')\ntime.sleep(30)\n",
            exchange,
            timeout_seconds=0.2,
        )

    assert raised.value.reason is BoundedWorkerDialogueErrorReason.DEADLINE_EXPIRED
    assert sent == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups are required")
def test_nested_child_inherits_the_supervising_worker_process_group() -> None:
    def exchange(dialogue: BoundedWorkerDialogue) -> int:
        frame = dialogue.read_until(b"!", frame_limit=32)
        return int(frame.removesuffix(b"!"))

    completed = _run_dialogue(
        "import os\nos.write(1, f'{os.getpgrp()}!'.encode())\n",
        exchange,
    )

    assert completed.value == os.getpgrp()


def test_dialogue_cannot_be_reused_after_its_callback() -> None:
    retained: list[BoundedWorkerDialogue] = []

    completed = _run_dialogue(
        "raise SystemExit(0)\n",
        lambda dialogue: retained.append(dialogue),
    )

    assert completed.value is None
    with pytest.raises(RuntimeError, match="scope has ended"):
        retained[0].send(b"late")
