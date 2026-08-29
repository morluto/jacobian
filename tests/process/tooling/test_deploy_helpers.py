"""Behavioral coverage for the deployment smoke helper.

The smoke helper lives in ``deploy.smoke`` and is shared by the
read-only ``deploy/smoke_remote.py`` probe. These tests cover its
transient-failure classification, HTTP status surfacing, and stable exit codes;
they do not exercise installation, state migration, or rollout machinery.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx2
import pytest
from deploy.smoke import (
    TRANSIENT_SMOKE_EXIT,
    TransientSmokeError,
    exit_for_smoke_failure,
    is_transient_transport_failure,
    raise_for_http_error,
)
from deploy.smoke_remote import _validate_discovery_response, _validate_tool_surface


def _current_discovery_payload() -> dict[str, object]:
    return {
        "kind": "discovery",
        "query": "exact determinant",
        "namespace": None,
        "matches": [
            {
                "operation_id": "matrix.determinant.compute",
                "title": "Exact determinant",
                "description": "Compute one exact determinant.",
                "tags": ["matrix", "determinant"],
            }
        ],
        "total_matches": 1,
        "next_cursor": None,
        "catalog_resource": "operation://catalog",
    }


def test_remote_smoke_accepts_the_current_typed_discovery_response() -> None:
    discovery = _current_discovery_payload()
    failures: list[str] = []

    matches = _validate_discovery_response(
        discovery,
        json.dumps(discovery),
        failures,
    )

    assert matches == ("matrix.determinant.compute",)
    assert failures == []


def test_remote_smoke_derives_tool_surface_from_deployed_catalog() -> None:
    remote_operation = "test.remote.previous_release"
    listed = SimpleNamespace(
        tools=[
            SimpleNamespace(name="math.find"),
            SimpleNamespace(name="math.run"),
            SimpleNamespace(name=remote_operation),
        ]
    )
    failures: list[str] = []

    tool_names = _validate_tool_surface(listed, {remote_operation}, failures)

    assert tool_names == {"math.find", "math.run", remote_operation}
    assert failures == []


def test_remote_smoke_rejects_divergent_model_visible_discovery() -> None:
    discovery = _current_discovery_payload()
    failures: list[str] = []

    _validate_discovery_response(
        discovery,
        json.dumps({**discovery, "matches": []}),
        failures,
    )

    assert failures == [
        "deployed operation discovery text and structured content disagree"
    ]


def test_remote_smoke_reports_the_removed_discovery_limit_as_schema_drift() -> None:
    discovery = {
        **_current_discovery_payload(),
        "response_byte_limit": 16_384,
    }
    failures: list[str] = []

    matches = _validate_discovery_response(
        discovery,
        json.dumps(discovery),
        failures,
    )

    assert matches == ()
    assert len(failures) == 1
    assert failures[0].startswith("deployed operation discovery violates its schema:")


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
