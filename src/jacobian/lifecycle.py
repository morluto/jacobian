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


__all__ = [
    "LifecycleTimeoutError",
    "ServiceLifecycleState",
    "wait_for_worker_quiescence",
]
