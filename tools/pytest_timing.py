"""Emit compact xdist worker timing evidence for one pytest invocation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest


@dataclass(slots=True)
class _TimingState:
    started: float = field(default_factory=time.monotonic)
    call_seconds: float = 0.0
    call_count: int = 0
    workers: dict[str, tuple[float, int]] = field(default_factory=dict)


_TIMING_STATE = pytest.StashKey[_TimingState]()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--jacobian-timing-json",
        metavar="PATH",
        help="write wall and per-worker call timing evidence to PATH",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.stash[_TIMING_STATE] = _TimingState()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[object]) -> Any:
    outcome = yield
    report: pytest.TestReport = outcome.get_result()
    if report.when != "call":
        return
    state = item.config.stash[_TIMING_STATE]
    state.call_seconds += report.duration
    state.call_count += 1


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    del exitstatus
    config = session.config
    state = config.stash[_TIMING_STATE]
    workerinput: dict[str, Any] | None = getattr(config, "workerinput", None)
    if workerinput is not None:
        worker_config = cast(Any, config)
        worker_config.workeroutput["jacobian_timing"] = {
            "id": workerinput["workerid"],
            "call_seconds": state.call_seconds,
            "call_count": state.call_count,
        }
        return
    output = config.getoption("jacobian_timing_json")
    if output is None:
        return
    _write_timing(Path(output), state, wall_seconds=time.monotonic() - state.started)


def pytest_testnodedown(node: Any, error: BaseException | None) -> None:
    del error
    config = node.config
    state = config.stash[_TIMING_STATE]
    workeroutput = getattr(node, "workeroutput", None)
    if not isinstance(workeroutput, dict):
        # xdist does not attach worker output when a worker terminates before
        # its session-finish hook. Preserve xdist's original worker failure
        # instead of replacing it with a timing-plugin AttributeError.
        return
    timing = workeroutput.get("jacobian_timing")
    if not isinstance(timing, dict):
        return
    worker_id = timing.get("id")
    call_seconds = timing.get("call_seconds")
    call_count = timing.get("call_count")
    if (
        not isinstance(worker_id, str)
        or not isinstance(call_seconds, float)
        or not isinstance(call_count, int)
    ):
        return
    state.workers[worker_id] = (call_seconds, call_count)


def _write_timing(path: Path, state: _TimingState, *, wall_seconds: float) -> None:
    workers = (
        tuple(
            {
                "id": worker_id,
                "call_seconds": call_seconds,
                "call_count": call_count,
            }
            for worker_id, (call_seconds, call_count) in sorted(state.workers.items())
        )
        if state.workers
        else (
            {
                "id": "local",
                "call_seconds": state.call_seconds,
                "call_count": state.call_count,
            },
        )
    )
    path.write_text(
        json.dumps(
            {"version": 1, "wall_seconds": wall_seconds, "workers": workers},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
