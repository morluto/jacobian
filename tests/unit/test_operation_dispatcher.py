from __future__ import annotations

import pytest

from jacobian.operation_dispatcher import invoke_operation
from jacobian.serving_catalog import ServingCatalog


def test_invoke_operation_runs_determinant_directly() -> None:
    catalog = ServingCatalog.open()
    result = invoke_operation(
        "matrix.determinant.compute",
        {
            "matrix": {
                "matrix_schema_version": "1",
                "domain": "QQ",
                "entries": [
                    [{"num": "1", "den": "1"}, {"num": "2", "den": "1"}],
                    [{"num": "3", "den": "1"}, {"num": "4", "den": "1"}],
                ],
            }
        },
        catalog,
    )
    assert result.runtime_ms >= 0
    assert result.output["determinant"] == {"num": "-2", "den": "1"}


def test_invoke_operation_reports_unknown_id() -> None:
    catalog = ServingCatalog.open()
    with pytest.raises(ValueError, match="unknown operation"):
        invoke_operation(
            "graph.construct.explicit",
            {"vertices": ["a"], "edges": []},
            catalog,
        )
