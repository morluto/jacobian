from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable

import pytest

from jacobian.providers import LazyLoader, LazyLoadError, LoaderState


class _Recording:
    """Minimal implementation recording whether its close was called."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _identity(value: object) -> Callable[[], object]:
    def load() -> object:
        return value

    return load


def test_initial_state_is_unloaded() -> None:
    loader = LazyLoader(_identity("v"), component_id="provider.alpha")

    assert loader.component_id == "provider.alpha"
    assert loader.state is LoaderState.UNLOADED
    assert not loader.loaded
    assert loader.error is None


def test_get_loads_once_and_caches_success() -> None:
    calls = 0

    def load() -> str:
        nonlocal calls
        calls += 1
        return "implementation"

    loader = LazyLoader(load, component_id="provider.beta")

    first = loader.get()
    second = loader.get()

    assert first == "implementation"
    assert second is first
    assert calls == 1
    assert loader.state is LoaderState.LOADED
    assert loader.loaded
    assert loader.error is None


def test_get_does_not_import_at_construction_time() -> None:
    constructed = False

    def load() -> str:
        nonlocal constructed
        constructed = True
        return "implementation"

    loader = LazyLoader(load, component_id="provider.construction")

    assert not constructed
    assert loader.state is LoaderState.UNLOADED

    loader.get()

    assert constructed


def test_failure_is_cached_by_default() -> None:
    calls = 0

    def load() -> str:
        nonlocal calls
        calls += 1
        raise ImportError("missing dependency")

    loader = LazyLoader(load, component_id="provider.gamma")

    with pytest.raises(LazyLoadError, match=r"provider\.gamma implementation"):
        loader.get()
    with pytest.raises(LazyLoadError, match=r"provider\.gamma implementation"):
        loader.get()

    assert calls == 1
    assert loader.state is LoaderState.FAILED
    assert not loader.loaded
    assert isinstance(loader.error, LazyLoadError)
    assert isinstance(loader.error.__cause__, ImportError)


def _raising_load(exc: BaseException, calls_box: list[int]) -> Callable[[], str]:
    def load() -> str:
        calls_box[0] += 1
        raise exc

    return load


def test_failure_caches_unavailability_exception_types() -> None:
    for exc_type in (ImportError, ModuleNotFoundError, OSError):
        calls = [0]
        loader = LazyLoader(
            _raising_load(exc_type("missing"), calls),
            component_id=f"provider.{exc_type.__name__}",
        )

        with pytest.raises(LazyLoadError):
            loader.get()
        with pytest.raises(LazyLoadError):
            loader.get()

        assert calls[0] == 1
        assert loader.state is LoaderState.FAILED


@pytest.mark.parametrize("exc_type", [RuntimeError, ValueError])
def test_programming_defects_propagate_and_reset(exc_type: type[BaseException]) -> None:
    calls = [0]
    loader = LazyLoader(
        _raising_load(exc_type("defect"), calls),
        component_id=f"provider.defect.{exc_type.__name__}",
    )

    # A programming defect propagates as-is, not wrapped as LazyLoadError.
    with pytest.raises(exc_type, match="defect"):
        loader.get()

    # The loader is reset to UNLOADED so the defect stays visible and is not
    # masked as a cached "provider missing" failure.
    assert loader.state is LoaderState.UNLOADED
    assert loader.error is None

    # A second call re-attempts (and re-raises) rather than returning a stale
    # cached failure.
    with pytest.raises(exc_type, match="defect"):
        loader.get()
    assert calls[0] == 2
    assert loader.state is LoaderState.UNLOADED


def test_reset_then_get_re_attempts_after_failure() -> None:
    calls = 0

    def load() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ImportError("not yet installed")
        return "implementation"

    loader = LazyLoader(load, component_id="provider.retry")

    with pytest.raises(LazyLoadError):
        loader.get()
    assert loader.state is LoaderState.FAILED

    # A cached failure is re-raised on every get() without an explicit reset.
    with pytest.raises(LazyLoadError):
        loader.get()
    assert calls == 1

    # reset() discards the cached failure; the next get() re-attempts.
    loader.reset()
    assert loader.state is LoaderState.UNLOADED

    result = loader.get()

    assert result == "implementation"
    assert calls == 2
    assert loader.state is LoaderState.LOADED
    assert loader.error is None


def test_success_is_cached_without_retry_flag() -> None:
    calls = 0

    def load() -> str:
        nonlocal calls
        calls += 1
        return "implementation"

    loader = LazyLoader(load, component_id="provider.cache-success")

    loader.get()
    loader.get()

    assert calls == 1


def test_unexpected_exception_is_not_cached_as_failure() -> None:
    calls = 0

    def load() -> str:
        nonlocal calls
        calls += 1
        raise KeyboardInterrupt

    loader = LazyLoader(load, component_id="provider.interrupt")

    with pytest.raises(KeyboardInterrupt):
        loader.get()

    # The loader remains usable after an unexpected interruption.
    assert loader.state is LoaderState.UNLOADED
    assert loader.error is None

    def load_ok() -> str:
        return "implementation"

    loader._load = load_ok  # type: ignore[attr-defined]
    assert loader.get() == "implementation"
    assert loader.state is LoaderState.LOADED


def test_close_releases_owned_implementation() -> None:
    impl = _Recording()
    loader = LazyLoader(lambda: impl, component_id="provider.close")

    assert loader.get() is impl

    loader.close()

    assert impl.closed
    assert loader.state is LoaderState.CLOSED
    assert not loader.loaded


def test_close_is_idempotent() -> None:
    impl = _Recording()
    loader = LazyLoader(lambda: impl, component_id="provider.idempotent")

    loader.get()  # load so the implementation is owned and will be released
    loader.close()
    loader.close()

    assert loader.state is LoaderState.CLOSED
    assert impl.closed is True  # close() called exactly once on the impl


def test_get_after_close_raises() -> None:
    loader = LazyLoader(_identity("v"), component_id="provider.closed-get")

    loader.close()

    with pytest.raises(LazyLoadError, match=r"provider\.closed-get loader is closed"):
        loader.get()
    assert loader.state is LoaderState.CLOSED


def test_reset_clears_cached_success_and_closes_implementation() -> None:
    impl = _Recording()
    loader = LazyLoader(lambda: impl, component_id="provider.reset")

    assert loader.get() is impl

    loader.reset()

    assert impl.closed
    assert loader.state is LoaderState.UNLOADED
    assert loader.error is None
    assert not loader.loaded


def test_reset_clears_cached_failure() -> None:
    calls = 0

    def load() -> str:
        nonlocal calls
        calls += 1
        raise ImportError("missing")

    loader = LazyLoader(load, component_id="provider.reset-failure")

    with pytest.raises(LazyLoadError):
        loader.get()

    assert loader.state is LoaderState.FAILED

    loader.reset()

    assert loader.state is LoaderState.UNLOADED
    assert loader.error is None

    with pytest.raises(LazyLoadError):
        loader.get()
    assert calls == 2


def test_reset_on_closed_loader_raises() -> None:
    loader = LazyLoader(_identity("v"), component_id="provider.reset-closed")

    loader.close()

    with pytest.raises(LazyLoadError, match="loader is closed"):
        loader.reset()


def test_context_manager_closes_on_exit() -> None:
    impl = _Recording()

    with LazyLoader(lambda: impl, component_id="provider.ctx") as loader:
        assert loader.get() is impl
        assert loader.state is LoaderState.LOADED

    assert impl.closed
    assert loader.state is LoaderState.CLOSED


def test_context_manager_closes_on_exception() -> None:
    impl = _Recording()
    loader = LazyLoader(lambda: impl, component_id="provider.ctx-exc")

    with pytest.raises(ValueError, match="boom"), loader:
        loader.get()  # load so the impl is owned before the body fails
        raise ValueError("boom")

    assert impl.closed
    assert loader.state is LoaderState.CLOSED


def test_reentrant_load_raises() -> None:
    loader: LazyLoader[str] = LazyLoader(
        lambda: "",  # placeholder, replaced below
        component_id="provider.reentrant",
    )

    def load() -> str:
        # Re-entering the loader from within its own load callable is a
        # programming error and must fail closed rather than recurse.
        return loader.get()

    loader._load = load  # type: ignore[attr-defined]

    with pytest.raises(LazyLoadError, match="re-entered its own load callable"):
        loader.get()
    assert loader.state is LoaderState.FAILED
    assert isinstance(loader.error, LazyLoadError)
    assert "re-entered" in loader.error.args[0]


def test_concurrent_load_calls_callable_once() -> None:
    barrier = threading.Barrier(16)
    load_started = threading.Event()
    release_load = threading.Event()
    calls = 0
    call_lock = threading.Lock()
    results: list[str] = []

    def load() -> str:
        with call_lock:
            nonlocal calls
            calls += 1
        load_started.set()
        assert release_load.wait(timeout=2)
        return "implementation"

    loader = LazyLoader(load, component_id="provider.concurrent")

    def worker() -> None:
        barrier.wait()
        results.append(loader.get())

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for thread in threads:
        thread.start()
    assert load_started.wait(timeout=1)
    release_load.set()
    for thread in threads:
        thread.join()

    assert calls == 1
    assert loader.state is LoaderState.LOADED
    assert all(result == "implementation" for result in results)
    assert len(results) == 16


def test_concurrent_failure_is_cached_once() -> None:
    barrier = threading.Barrier(8)
    calls = 0
    call_lock = threading.Lock()

    def load() -> str:
        with call_lock:
            nonlocal calls
            calls += 1
        raise ImportError("missing")

    loader = LazyLoader(load, component_id="provider.concurrent-failure")

    def worker() -> None:
        barrier.wait()
        with contextlib.suppress(LazyLoadError):
            loader.get()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls == 1
    assert loader.state is LoaderState.FAILED


def test_loader_state_is_string_enum() -> None:
    assert LoaderState.UNLOADED == "UNLOADED"
    assert LoaderState.LOADED == "LOADED"
    assert LoaderState.FAILED == "FAILED"
    assert LoaderState.CLOSED == "CLOSED"
    assert LoaderState.LOADING == "LOADING"


def test_implementation_without_close_is_released_silently() -> None:
    sentinel = object()
    loader = LazyLoader(lambda: sentinel, component_id="provider.no-close")

    assert loader.get() is sentinel
    loader.close()  # must not raise even though sentinel has no close()
    assert loader.state is LoaderState.CLOSED
