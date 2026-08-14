"""Behavioral coverage for the deployment smoke helper.

The smoke helper lives in ``jacobian._deployment_smoke`` and is shared by the
read-only ``deploy/smoke_remote.py`` and ``deploy/smoke_lean.py`` probes. These
tests cover its transient-failure classification, HTTP status surfacing, and
stable exit codes; they do not exercise installation, state migration, or
rollout machinery.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import httpx2
import pytest

from jacobian._deployment_smoke import (
    TRANSIENT_SMOKE_EXIT,
    TransientSmokeError,
    exit_for_smoke_failure,
    is_transient_transport_failure,
    raise_for_http_error,
)

ROOT = Path(__file__).parents[4]


def _load_lean_smoke() -> ModuleType:
    path = ROOT / "deploy" / "smoke_lean.py"
    spec = importlib.util.spec_from_file_location("smoke_lean", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_retry_classification_is_transport_only() -> None:
    transient = ExceptionGroup(
        "transport",
        [httpx2.ConnectError("refused"), httpx2.ReadTimeout("cold start")],
    )
    deterministic = ExceptionGroup(
        "contract",
        [httpx2.ConnectError("refused"), RuntimeError("version mismatch")],
    )

    assert is_transient_transport_failure(transient) is True
    assert is_transient_transport_failure(TransientSmokeError("cold worker")) is True
    assert is_transient_transport_failure(deterministic) is False
    assert is_transient_transport_failure(RuntimeError("catalog mismatch")) is False


@pytest.mark.parametrize(
    ("status_code", "expected"),
    ((401, False), (403, False), (500, False), (502, True), (503, True), (504, True)),
)
def test_smoke_retry_classification_preserves_http_status(
    status_code: int, expected: bool
) -> None:
    request = httpx2.Request("POST", "https://math.example.org/mcp")
    response = httpx2.Response(status_code, request=request)
    with pytest.raises(httpx2.HTTPStatusError) as exc_info:
        response.raise_for_status()

    assert is_transient_transport_failure(exc_info.value) is expected


@pytest.mark.anyio
async def test_smoke_response_hook_surfaces_http_status() -> None:
    request = httpx2.Request("POST", "https://math.example.org/mcp")
    response = httpx2.Response(503, request=request)

    with pytest.raises(httpx2.HTTPStatusError):
        await raise_for_http_error(response)


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    (
        (httpx2.ConnectError("refused"), TRANSIENT_SMOKE_EXIT),
        (RuntimeError("revision mismatch"), 1),
    ),
)
def test_smoke_failure_exit_codes_are_stable(
    failure: Exception,
    expected_code: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        exit_for_smoke_failure("smoke", failure)

    assert exc_info.value.code == expected_code
    assert str(failure) in capsys.readouterr().err


def test_lean_smoke_uses_one_shot_typed_outcomes() -> None:
    smoke = _load_lean_smoke()
    accepted = {
        "execution": {"status": "COMPLETED"},
        "output": {"result": {"outcome": "ELABORATED", "diagnostics": []}},
    }
    rejected = {
        "execution": {"status": "COMPLETED"},
        "output": {
            "result": {
                "outcome": "REJECTED",
                "diagnostics": [{"severity": "ERROR", "message": "invalid proof"}],
            }
        },
    }

    smoke._require_outcome(accepted, expected="ELABORATED")
    smoke._require_outcome(rejected, expected="REJECTED", require_diagnostics=True)

    with pytest.raises(RuntimeError, match="typed diagnostics"):
        smoke._require_outcome(
            {
                "execution": {"status": "COMPLETED"},
                "output": {"result": {"outcome": "REJECTED", "diagnostics": []}},
            },
            expected="REJECTED",
            require_diagnostics=True,
        )
