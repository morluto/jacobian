from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from jacobian.contracts.lean import LeanEnvironment
from jacobian.lean_frontend.repl import (
    LeanExplorationReplRuntime,
    LeanReplPolicy,
    PersistentLeanRepl,
)

_FAKE_REPL = r"""
import json
import pathlib
import sys

starts = pathlib.Path(sys.argv[1])
starts.write_text(starts.read_text() + "x" if starts.exists() else "x")
proof_state = 0
env = 0

while True:
    lines = []
    for line in sys.stdin:
        if not line.strip():
            break
        lines.append(line)
    if not lines:
        break
    request = json.loads("".join(lines))
    if request.get("cmd") == "import Mathlib":
        response = {"env": env}
        env += 1
    elif "cmd" in request:
        assert request.get("env") == 0
        response = {
            "env": env,
            "sorries": [{"proofState": proof_state}],
        }
        env += 1
        proof_state += 1
    else:
        assert request["proofState"] == proof_state - 1
        response = {
            "proofState": proof_state,
            "proofStatus": "Completed",
            "goals": [],
        }
        proof_state += 1
    print(json.dumps(response), end="\n\n", flush=True)
"""


def test_persistent_repl_reuses_import_then_restarts_at_request_limit(
    tmp_path: Path,
) -> None:
    starts = tmp_path / "starts"
    repl = PersistentLeanRepl(
        command=(sys.executable, "-u", "-c", _FAKE_REPL, str(starts)),
        cwd=tmp_path,
        base_command="import Mathlib",
        policy=LeanReplPolicy(max_requests=2, max_age_seconds=60, max_rss_kb=0),
    )

    first = repl.execute(command="example : True := by sorry", tactic="trivial")
    second = repl.execute(command="example : True := by sorry", tactic="trivial")
    third = repl.execute(command="example : True := by sorry", tactic="trivial")
    repl.close()

    assert all(
        response[1]["proofStatus"] == "Completed" for response in (first, second, third)
    )
    assert starts.read_text() == "xx"


def test_validated_execution_inspects_state_before_requested_tactic(
    tmp_path: Path,
) -> None:
    starts = tmp_path / "starts"
    repl = PersistentLeanRepl(
        command=(sys.executable, "-u", "-c", _FAKE_REPL, str(starts)),
        cwd=tmp_path,
        base_command="import Mathlib",
        policy=LeanReplPolicy(max_requests=1, max_age_seconds=60, max_rss_kb=0),
    )

    command, validation, transition = repl.execute_validated(
        command="example : True := by sorry",
        tactic="trivial",
    )
    repl.close()

    assert command["sorries"]
    assert validation["proofState"] == 1
    assert transition["proofState"] == 2
    assert starts.read_text() == "x"


def test_persistent_repl_kills_a_timed_out_process(tmp_path: Path) -> None:
    repl = PersistentLeanRepl(
        command=(sys.executable, "-u", "-c", "import time; time.sleep(10)"),
        cwd=tmp_path,
        base_command=None,
        policy=LeanReplPolicy(
            max_requests=2,
            max_age_seconds=60,
            max_rss_kb=0,
            timeout_seconds=0.05,
        ),
    )

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        repl.execute(command="example : True := by sorry", tactic="trivial")

    assert time.monotonic() - started < 2


def test_persistent_repl_kills_a_process_that_exceeds_rss_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jacobian.bounded_process._linux_process_tree_memory_bytes",
        lambda _pid: 2 * 1024,
    )
    repl = PersistentLeanRepl(
        command=(sys.executable, "-u", "-c", "import time; time.sleep(10)"),
        cwd=tmp_path,
        base_command=None,
        policy=LeanReplPolicy(
            max_requests=2,
            max_age_seconds=60,
            max_rss_kb=1,
            timeout_seconds=5,
        ),
    )

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="exceeded its memory limit"):
        repl.execute(command="example : True := by sorry", tactic="trivial")

    assert time.monotonic() - started < 2


def test_persistent_repl_enforces_rss_during_exchange_not_only_before(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RSS is checked inside the exchange poll loop, not only before it.

    A process that is under the limit at startup but whose tree RSS later
    exceeds the bound must be killed during the exchange wait, not after the
    full read timeout elapses.
    """

    call_count = 0

    def fake_rss(_pid: int) -> int | None:
        nonlocal call_count
        call_count += 1
        # Under the limit on the first few polls (startup / pre-exchange),
        # then over the limit once the exchange loop is actively waiting.
        return 1024 if call_count <= 2 else 4 * 1024

    monkeypatch.setattr(
        "jacobian.bounded_process._linux_process_tree_memory_bytes",
        fake_rss,
    )
    repl = PersistentLeanRepl(
        command=(sys.executable, "-u", "-c", "import time; time.sleep(10)"),
        cwd=tmp_path,
        base_command=None,
        policy=LeanReplPolicy(
            max_requests=2,
            max_age_seconds=60,
            max_rss_kb=2,
            timeout_seconds=5,
        ),
    )

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="exceeded its memory limit"):
        repl.execute(command="example : True := by sorry", tactic="trivial")

    # Must be caught during the exchange poll, well before the 5s timeout.
    assert time.monotonic() - started < 2
    # The RSS probe was called more than twice (exchange-loop polling).
    assert call_count > 2
    repl.close()


def test_clean_execution_discards_every_repl_instance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = LeanExplorationReplRuntime(tmp_path, {})

    class FakeSession:
        closed = False

        def execute_validated(
            self,
            *,
            command: str,
            tactic: str,
        ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
            assert command
            assert tactic
            return ({}, {}, {})

        def close(self) -> None:
            self.closed = True

    sessions: list[FakeSession] = []

    def create(_environment: LeanEnvironment) -> FakeSession:
        session = FakeSession()
        sessions.append(session)
        return session

    monkeypatch.setattr(runtime, "_create_session", create)

    runtime.execute_clean(
        command="example : True := by sorry",
        tactic="trivial",
        environment=LeanEnvironment.CORE,
    )
    runtime.execute_clean(
        command="example : True := by sorry",
        tactic="trivial",
        environment=LeanEnvironment.CORE,
    )

    assert len(sessions) == 2
    assert all(session.closed for session in sessions)


def test_runtime_close_releases_retained_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = LeanExplorationReplRuntime(tmp_path, {})

    class FakeSession:
        closed = False

        def execute(
            self,
            *,
            command: str,
            tactic: str,
            pickle_path: Path | None = None,
        ) -> tuple[dict[str, object], dict[str, object]]:
            assert command and tactic
            assert pickle_path is None
            return {}, {}

        def close(self) -> None:
            self.closed = True

    session = FakeSession()
    monkeypatch.setattr(runtime, "_create_session", lambda _environment: session)

    runtime.execute(
        command="example : True := by sorry",
        tactic="trivial",
        environment=LeanEnvironment.CORE,
    )
    runtime.close()
    runtime.close()

    assert session.closed


def test_runtime_close_retries_failed_sessions_and_rejects_new_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = LeanExplorationReplRuntime(tmp_path, {})

    class FakeSession:
        def __init__(self, *, fail_close: bool) -> None:
            self.fail_close = fail_close
            self.close_calls = 0

        def execute(
            self,
            *,
            command: str,
            tactic: str,
            pickle_path: Path | None = None,
        ) -> tuple[dict[str, object], dict[str, object]]:
            return {}, {}

        def close(self) -> None:
            self.close_calls += 1
            if self.fail_close:
                raise RuntimeError("injected session close failure")

    sessions = [FakeSession(fail_close=True), FakeSession(fail_close=False)]
    monkeypatch.setattr(
        runtime, "_create_session", lambda _environment: sessions.pop(0)
    )
    runtime.execute(command="first", tactic="skip", environment=LeanEnvironment.CORE)
    runtime.execute(
        command="second", tactic="skip", environment=LeanEnvironment.MATHLIB
    )
    failed, succeeded = tuple(
        runtime._sessions[environment]
        for environment in (LeanEnvironment.CORE, LeanEnvironment.MATHLIB)
    )

    with pytest.raises(ExceptionGroup, match="sessions failed to close"):
        runtime.close()
    assert failed.close_calls == 1
    assert succeeded.close_calls == 1
    with pytest.raises(RuntimeError, match="runtime is closing"):
        runtime.execute(
            command="third", tactic="skip", environment=LeanEnvironment.CORE
        )

    failed.fail_close = False
    runtime.close()
    assert failed.close_calls == 2
    assert succeeded.close_calls == 1
    with pytest.raises(RuntimeError, match="runtime is closing"):
        runtime.execute_clean(
            command="third",
            tactic="skip",
            environment=LeanEnvironment.CORE,
        )


def test_runtime_close_continues_after_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = LeanExplorationReplRuntime(tmp_path, {})

    class FakeSession:
        def __init__(self, *, interrupt: bool) -> None:
            self.interrupt = interrupt
            self.close_calls = 0

        def execute(
            self,
            *,
            command: str,
            tactic: str,
            pickle_path: Path | None = None,
        ) -> tuple[dict[str, object], dict[str, object]]:
            return {}, {}

        def close(self) -> None:
            self.close_calls += 1
            if self.interrupt:
                raise KeyboardInterrupt("injected session close interrupt")

    sessions = [FakeSession(interrupt=True), FakeSession(interrupt=False)]
    monkeypatch.setattr(
        runtime, "_create_session", lambda _environment: sessions.pop(0)
    )
    runtime.execute(command="first", tactic="skip", environment=LeanEnvironment.CORE)
    runtime.execute(
        command="second", tactic="skip", environment=LeanEnvironment.MATHLIB
    )
    interrupted, closed = tuple(
        runtime._sessions[environment]
        for environment in (LeanEnvironment.CORE, LeanEnvironment.MATHLIB)
    )

    with pytest.raises(
        BaseExceptionGroup, match="Lean exploration sessions failed to close"
    ) as exc:
        runtime.close()

    assert interrupted.close_calls == 1
    assert closed.close_calls == 1
    assert [str(failure) for failure in exc.value.exceptions] == [
        "injected session close interrupt",
    ]

    interrupted.interrupt = False
    runtime.close()
