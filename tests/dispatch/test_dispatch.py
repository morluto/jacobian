from __future__ import annotations

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_invoke_operation_runs_determinant_directly() -> None:
    catalog = Catalog.open()
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
    assert result.output is not None
    assert set(result.output) == {"determinant", "method"}
    assert result.output["determinant"] == {"num": "-2", "den": "1"}
    assert result.output["method"] == "FRACTION_FREE_BAREISS"


def test_invoke_operation_reports_unknown_id() -> None:
    catalog = Catalog.open()
    with pytest.raises(ValueError, match="unknown operation"):
        invoke_operation(
            "graph.construct.explicit",
            {"vertices": ["a"], "edges": []},
            catalog,
        )
