from __future__ import annotations

from threading import Event, Thread
from types import SimpleNamespace
from typing import Any, cast

import pytest

import jacobian.operation_dispatcher as dispatcher_module
from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationRequest,
)
from jacobian.operation_dispatcher import OperationDispatcher
from jacobian.operation_visibility import OperationVisibilityPolicy


def test_concurrent_first_invocation_resolves_and_registers_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = OperationDescriptor(
        operation_id="integer.gcd.compute",
        version="1",
        title="Greatest common divisor",
        description="Compute a greatest common divisor.",
        provider="built-in",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )
    adapter = SimpleNamespace(descriptor=descriptor)
    first_resolve_entered = Event()
    allow_first_resolve = Event()
    second_resolve_entered = Event()
    resolve_calls = 0

    class Registry:
        binder = SimpleNamespace(store=object())

        def resolve(self, operation_id: str) -> Any:
            nonlocal resolve_calls
            resolve_calls += 1
            if resolve_calls == 1:
                first_resolve_entered.set()
                assert allow_first_resolve.wait(timeout=2)
            else:
                second_resolve_entered.set()
            assert operation_id == descriptor.operation_id
            return adapter

        def close(self) -> None:
            pass

    catalog = SimpleNamespace(
        policy=OperationVisibilityPolicy(),
        inspect=lambda operation_id: (
            descriptor if operation_id == descriptor.operation_id else None
        ),
    )
    dispatcher = OperationDispatcher(cast(Any, catalog), cast(Any, Registry()))
    expected_result = object()
    monkeypatch.setattr(
        dispatcher_module,
        "dispatch_operation",
        lambda _dispatcher, _request: expected_result,
    )
    request = OperationRequest(operation_id=descriptor.operation_id, input={})
    results: list[object] = []
    failures: list[BaseException] = []

    def invoke() -> None:
        try:
            results.append(dispatcher.invoke(request))
        except BaseException as exc:
            failures.append(exc)

    first = Thread(target=invoke)
    second = Thread(target=invoke)
    first.start()
    assert first_resolve_entered.wait(timeout=2)
    second.start()
    assert not second_resolve_entered.wait(timeout=0.2)
    allow_first_resolve.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert failures == []
    assert results == [expected_result, expected_result]
    assert resolve_calls == 1
