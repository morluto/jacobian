from __future__ import annotations

from types import SimpleNamespace

import pytest

import jacobian.runtime.execution as execution


class _CloseableCore:
    def __init__(self) -> None:
        self.store = object()
        self.checkers = object()
        self.schemas = object()
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def test_runtime_construction_closes_bootstrap_resources_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _CloseableCore()
    monkeypatch.setattr(execution, "bootstrap_services", lambda *_args, **_kwargs: core)

    def fail_verification(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("verification setup failed")

    monkeypatch.setattr(execution, "VerificationService", fail_verification)

    with pytest.raises(RuntimeError, match="verification setup failed"):
        execution.create_execution_runtime(
            "/state",
            SimpleNamespace(),
            operation_policy=SimpleNamespace(),
        )

    assert core.closed == 1
