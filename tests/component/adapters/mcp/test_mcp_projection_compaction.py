from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
from tests.support.mcp_projection_catalog import open_mcp_projection_catalog
from tests.support.services import DomainTestServices

from jacobian.runtime.model import JacobianRuntime


@pytest.fixture
def projection_catalog(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_mcp_projection_catalog(tmp_path / "state") as services:
        yield services


def _as_runtime(services: DomainTestServices) -> JacobianRuntime:
    """Projection helpers only require ``.core.operations``."""

    return cast(JacobianRuntime, services)


def test_math_find_compacts_ranked_matches_deterministically(
    projection_catalog: DomainTestServices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.adapters.mcp import projections

    arguments = {
        "query": "polynomial",
        "domain": None,
        "input_kind": None,
        "artifact_type": None,
        "limit": 5,
        "cursor": None,
    }
    baseline = projections._operation_discovery_response(
        _as_runtime(projection_catalog), **arguments
    )
    single = projections._operation_discovery_response(
        _as_runtime(projection_catalog), **{**arguments, "limit": 1}
    )
    byte_limit = len(projections._mcp_text_json_bytes(single)) + 64
    assert len(projections._mcp_text_json_bytes(baseline)) > byte_limit
    monkeypatch.setattr(
        projections,
        "OPERATION_DISCOVERY_RESPONSE_BYTE_LIMIT",
        byte_limit,
    )

    first = projections._operation_discovery_response(
        _as_runtime(projection_catalog), **arguments
    )
    second = projections._operation_discovery_response(
        _as_runtime(projection_catalog), **arguments
    )

    assert first == second
    assert len(projections._mcp_text_json_bytes(first)) <= byte_limit
    assert first["truncated"] is True
    assert first["truncation_reason"] == "BYTE_LIMIT"
    assert first["next_cursor"] == first["matches"][-1]["operation_id"]
