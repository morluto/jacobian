"""Stable deployment-smoke exit semantics shared by smoke entry points."""

from __future__ import annotations

import sys
from collections.abc import Iterable

import httpx2

TRANSIENT_SMOKE_EXIT = 75
_TRANSIENT_HTTP_STATUSES = frozenset({502, 503, 504})


class TransientSmokeError(RuntimeError):
    """A bounded operational failure that may clear without changing inputs."""


def _leaves(exc: BaseException) -> Iterable[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        for nested in exc.exceptions:
            yield from _leaves(nested)
        return
    yield exc


def is_transient_transport_failure(exc: BaseException) -> bool:
    """Return true only when every failure is a retryable transport problem."""

    leaves = tuple(_leaves(exc))
    return bool(leaves) and all(_is_transient_leaf(leaf) for leaf in leaves)


def _is_transient_leaf(exc: BaseException) -> bool:
    if isinstance(exc, httpx2.HTTPStatusError):
        return exc.response.status_code in _TRANSIENT_HTTP_STATUSES
    return isinstance(
        exc,
        (
            TransientSmokeError,
            httpx2.TransportError,
            TimeoutError,
            ConnectionError,
        ),
    )


async def raise_for_http_error(response: httpx2.Response) -> None:
    """Preserve HTTP status identity before the MCP transport projects errors."""

    response.raise_for_status()


def exit_for_smoke_failure(label: str, exc: BaseException) -> None:
    """Print one bounded diagnostic and exit with stable retry semantics."""

    leaves = tuple(_leaves(exc))
    detail = "; ".join(
        dict.fromkeys(str(leaf) or type(leaf).__name__ for leaf in leaves)
    )
    print(f"{label} failed: {detail}", file=sys.stderr)
    raise SystemExit(TRANSIENT_SMOKE_EXIT if is_transient_transport_failure(exc) else 1)


__all__ = [
    "TRANSIENT_SMOKE_EXIT",
    "TransientSmokeError",
    "exit_for_smoke_failure",
    "is_transient_transport_failure",
    "raise_for_http_error",
]
