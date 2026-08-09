"""Mechanical worker lifecycle primitives shared by durable services."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from enum import StrEnum


class LifecycleTimeoutError(TimeoutError):
    """Runtime-owned workers did not quiesce before a close deadline."""


class ServiceLifecycleState(StrEnum):
    """One-way lifecycle states for services that own background workers."""

    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"


class WorkerLaunchStatus(StrEnum):
    """Outcome of attempting to start one runtime-owned worker."""

    STARTED = "STARTED"
    ALREADY_RUNNING = "ALREADY_RUNNING"
    SERVICE_CLOSING = "SERVICE_CLOSING"


def wait_for_worker_quiescence(
    *,
    lock: threading.Lock,
    workers: Mapping[str, threading.Thread],
    starts_in_flight: Callable[[], int],
    timeout_seconds: float,
) -> None:
    """Wait for workers and launch reservations to drain.

    Search and enumeration retain their own state machines and errors. This
    helper only owns the mechanical close race shared by both services.
    """

    if timeout_seconds < 0:
        raise ValueError("timeout_seconds must be non-negative")
    deadline = time.monotonic() + timeout_seconds
    while True:
        with lock:
            active = tuple(thread for thread in workers.values() if thread.is_alive())
            if not active and starts_in_flight() == 0:
                return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise LifecycleTimeoutError("workers did not quiesce before shutdown")
        if not active:
            time.sleep(min(remaining, 0.01))
            continue
        for thread in active:
            thread.join(timeout=min(remaining, 0.05))


def launch_worker(
    *,
    lock: threading.Lock,
    lifecycle_state: Callable[[], ServiceLifecycleState],
    workers: dict[str, threading.Thread],
    worker_id: str,
    target: Callable[[str], None],
    name: str,
    lifecycle_reserved: bool,
) -> WorkerLaunchStatus:
    """Start one worker unless its service is closing or it is already active."""

    with lock:
        state = lifecycle_state()
        if state is ServiceLifecycleState.CLOSED or (
            state is ServiceLifecycleState.CLOSING and not lifecycle_reserved
        ):
            return WorkerLaunchStatus.SERVICE_CLOSING
        current = workers.get(worker_id)
        if current is not None and current.is_alive():
            return WorkerLaunchStatus.ALREADY_RUNNING
        worker = threading.Thread(
            target=target,
            args=(worker_id,),
            name=name,
            daemon=True,
        )
        workers[worker_id] = worker
        worker.start()
    return WorkerLaunchStatus.STARTED


def wait_for_worker_settlement[SnapshotT](
    *,
    lock: threading.Lock,
    workers: Mapping[str, threading.Thread],
    worker_id: str,
    inspect: Callable[[str], SnapshotT],
    is_settled: Callable[[SnapshotT], bool],
    timeout_seconds: float,
    timeout_message: str,
) -> SnapshotT:
    """Poll one durable snapshot while joining its local worker when present."""

    deadline = time.monotonic() + timeout_seconds
    while True:
        snapshot = inspect(worker_id)
        if is_settled(snapshot):
            return snapshot
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(timeout_message)
        with lock:
            worker = workers.get(worker_id)
        if worker is not None:
            worker.join(timeout=min(remaining, 0.05))
        else:
            time.sleep(min(remaining, 0.05))


__all__ = [
    "LifecycleTimeoutError",
    "ServiceLifecycleState",
    "WorkerLaunchStatus",
    "launch_worker",
    "wait_for_worker_quiescence",
    "wait_for_worker_settlement",
]
