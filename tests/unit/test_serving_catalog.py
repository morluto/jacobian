from __future__ import annotations

from typing import cast
from unittest.mock import Mock

import pytest

from jacobian.adapters.mcp import server as mcp_server
from jacobian.adapters.mcp.server import _LazyLocalRuntime
from jacobian.contracts.operations import OperationRequest
from jacobian.operation_visibility import OperationVisibilityPolicy
from jacobian.runtime.execution import create_inline_serving_runtime
from jacobian.runtime.model import JacobianRuntime
from jacobian.serving_catalog import ServingCatalog


def test_serving_catalog_inspects_determinant_without_sqlite() -> None:
    catalog = ServingCatalog.open()

    descriptor = catalog.inspect("matrix.determinant.compute")
    assert descriptor is not None
    assert descriptor.operation_id == "matrix.determinant.compute"


def test_inline_serving_runtime_runs_determinant_without_state() -> None:
    catalog = ServingCatalog.open(policy=OperationVisibilityPolicy())
    runtime = create_inline_serving_runtime(catalog)
    try:
        result = runtime.core.operations.invoke(
            OperationRequest(
                operation_id="matrix.determinant.compute",
                input={
                    "matrix": {
                        "matrix_schema_version": "1",
                        "domain": "QQ",
                        "entries": [
                            [
                                {"num": "1", "den": "1"},
                                {"num": "2", "den": "1"},
                            ],
                            [
                                {"num": "3", "den": "1"},
                                {"num": "4", "den": "1"},
                            ],
                        ],
                    }
                },
            )
        )
    finally:
        runtime.close()

    assert result.execution.status.value == "COMPLETED"
    assert result.output is not None
    assert result.output["result"]["determinant"] == {"num": "-2", "den": "1"}


def test_inline_serving_runtime_reports_unknown_removed_family_id() -> None:
    catalog = ServingCatalog.open(policy=OperationVisibilityPolicy())
    runtime = create_inline_serving_runtime(catalog)
    try:
        result = runtime.core.operations.invoke(
            OperationRequest(
                operation_id="graph.construct.explicit",
                input={"vertices": ["a"], "edges": []},
            )
        )
    finally:
        runtime.close()

    assert result.execution.status.value == "ERROR"
    assert result.diagnostics[0].code == "UNKNOWN_OPERATION"
    assert result.diagnostics[0].stage == "operation_resolution"


def test_local_runtime_uses_only_the_inline_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = ServingCatalog.open(policy=OperationVisibilityPolicy())
    inline_runtime = cast(JacobianRuntime, Mock())
    create_inline = Mock(return_value=inline_runtime)
    monkeypatch.setattr(mcp_server, "create_inline_serving_runtime", create_inline)
    owner = _LazyLocalRuntime(catalog)

    access = owner.acquire("matrix.determinant.compute")

    assert access.runtime is inline_runtime
    create_inline.assert_called_once_with(catalog)
    owner.close()
