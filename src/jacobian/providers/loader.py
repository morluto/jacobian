"""Thread-safe lazy ownership of heavy provider implementations.

A :class:`LazyLoader` defers importing or constructing a heavy optional
implementation until the first call to :meth:`LazyLoader.get`. The loader is
thread-safe, deterministic, and owns the lifecycle of the implementation it
constructs: implementations that expose a ``close()`` method are released
exactly once when the loader is closed or reset.

The loader caches both successful and failed loads so that operation
discovery and operation execution do not hammer the import system on every
call. A failed load is always cached; to re-attempt after an operator has
installed a missing dependency in a long-lived process, call
:meth:`reset` explicitly and then :meth:`get` again. This keeps the policy
deterministic and the surface small: there is no retry flag, no mode soup, and
no implicit re-attempt that could mask a real outage.

The module deliberately avoids registries, package discovery, import-time
registration, and compatibility shims. Each loader is an independent, owned
object.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from enum import StrEnum


class LazyLoadError(RuntimeError):
    """A provider implementation could not be loaded for execution."""


class LoaderState(StrEnum):
    """Observable lifecycle state of a :class:`LazyLoader`."""

    UNLOADED = "UNLOADED"
    LOADING = "LOADING"
    LOADED = "LOADED"
    FAILED = "FAILED"
    CLOSED = "CLOSED"


# Exceptions that signal genuine provider unavailability (the optional
# dependency or its native backing is absent or unreadable). These are cached
# as a typed load failure. ``LazyLoadError`` is handled separately so the
# re-entrant-load guard diagnostic is preserved rather than double-wrapped.
#
# Arbitrary ``RuntimeError`` or ``ValueError`` from a loader callable is a
# programming defect, not unavailability: such errors propagate and reset the
# loader to ``UNLOADED`` so the defect stays visible instead of being masked
# as a cached "provider missing" failure.
_UNAVAILABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ImportError,
    ModuleNotFoundError,
    OSError,
)


class LazyLoader[ImplementationT]:
    """Load one implementation at most once and own its optional cleanup.

    The loader is thread-safe and deterministic. A successful load is cached
    for the life of the loader; a failed load is cached as a typed
    :class:`LazyLoadError` and re-raised on every subsequent :meth:`get` until
    :meth:`reset` discards the cached failure. Owned implementations that
    expose a ``close()`` method are released by :meth:`close` and
    :meth:`reset`.

    The loader never imports a package at construction time. Importing the
    provider implementation is the responsibility of the ``load`` callable
    supplied by the caller, and is deferred until the first :meth:`get` call.
    """

    __slots__ = (
        "_component_id",
        "_error",
        "_implementation",
        "_load",
        "_lock",
        "_state",
    )

    def __init__(
        self,
        load: Callable[[], ImplementationT],
        *,
        component_id: str,
    ) -> None:
        self._load = load
        self._component_id = component_id
        self._lock = threading.RLock()
        self._implementation: ImplementationT | None = None
        self._error: LazyLoadError | None = None
        self._state: LoaderState = LoaderState.UNLOADED

    @property
    def component_id(self) -> str:
        """The stable identifier used in diagnostics for this loader."""

        return self._component_id

    @property
    def state(self) -> LoaderState:
        """The observable lifecycle state of this loader."""

        with self._lock:
            return self._state

    @property
    def loaded(self) -> bool:
        """Whether a usable implementation is currently cached."""

        with self._lock:
            return self._state is LoaderState.LOADED

    @property
    def error(self) -> LazyLoadError | None:
        """The cached load failure, if any."""

        with self._lock:
            return self._error

    def get(self) -> ImplementationT:
        """Return the implementation, loading it on first use if needed.

        Raises :class:`LazyLoadError` if the loader is closed, if a previous
        load failed (call :meth:`reset` to discard the cached failure and
        re-attempt), or if the load callable raises a recognized
        unavailable-provider exception.
        """

        with self._lock:
            state = self._state
            if state is LoaderState.CLOSED:
                raise LazyLoadError(f"{self._component_id} loader is closed")
            if state is LoaderState.LOADED:
                # state is LOADED only when _implementation was assigned.
                if self._implementation is None:
                    raise RuntimeError("provider implementation is unexpectedly None")
                return self._implementation
            if state is LoaderState.LOADING:
                raise LazyLoadError(
                    f"{self._component_id} loader re-entered its own load callable"
                )
            if state is LoaderState.FAILED:
                # state is FAILED only when _error was assigned.
                if self._error is None:
                    raise RuntimeError("provider error is unexpectedly None")
                raise self._error
            self._state = LoaderState.LOADING
            try:
                implementation = self._load()
            except LazyLoadError as exc:
                # Preserve the original diagnostic (for example the
                # re-entrant-load guard) instead of double-wrapping it.
                self._error = exc
                self._state = LoaderState.FAILED
                raise
            except _UNAVAILABLE_EXCEPTIONS as exc:
                error = LazyLoadError(
                    f"{self._component_id} implementation could not be loaded"
                )
                error.__cause__ = exc
                self._error = error
                self._state = LoaderState.FAILED
                raise error from exc
            except BaseException:
                # Any other exception (RuntimeError, ValueError,
                # KeyboardInterrupt, ...) is a programming defect or an
                # interruption, not provider unavailability. Reset to
                # UNLOADED so the defect stays visible and a later call can
                # re-attempt without a stale cached failure masking it.
                self._state = LoaderState.UNLOADED
                raise
            self._implementation = implementation
            self._error = None
            self._state = LoaderState.LOADED
            return implementation

    def reset(self) -> None:
        """Discard any cached implementation or failure and release resources.

        Calls ``close()`` on a previously loaded implementation that exposes it.
        A closed loader cannot be reset.
        """

        with self._lock:
            if self._state is LoaderState.CLOSED:
                raise LazyLoadError(f"{self._component_id} loader is closed")
            implementation = self._implementation
            self._implementation = None
            self._error = None
            self._state = LoaderState.UNLOADED
        if implementation is not None:
            close = getattr(implementation, "close", None)
            if callable(close):
                close()

    def close(self) -> None:
        """Release the owned implementation, if any. Idempotent."""

        with self._lock:
            if self._state is LoaderState.CLOSED:
                return
            implementation = self._implementation
            self._implementation = None
            self._error = None
            self._state = LoaderState.CLOSED
        if implementation is not None:
            close = getattr(implementation, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> LazyLoader[ImplementationT]:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
