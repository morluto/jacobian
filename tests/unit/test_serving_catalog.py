from __future__ import annotations

from pathlib import Path

from jacobian.contracts.operations import OperationRequest
from jacobian.operation_visibility import OperationVisibilityPolicy
from jacobian.runtime.execution import create_inline_serving_runtime
from jacobian.serving_catalog import ServingCatalog


def test_serving_catalog_inspects_determinant_without_sqlite(tmp_path: Path) -> None:
    catalog = ServingCatalog.open(tmp_path / "missing" / "metadata.sqlite3")

    descriptor = catalog.inspect("matrix.determinant.compute")
    assert descriptor is not None
    assert descriptor.operation_id == "matrix.determinant.compute"
    record = catalog.declaration_record("matrix.determinant.compute")
    assert record is not None
    assert record.declaration_digest == "package-index"
    assert catalog.overlay is None


def test_serving_catalog_hides_family_ids_without_overlay() -> None:
    catalog = ServingCatalog.open(
        None,
        OperationVisibilityPolicy(),
    )

    assert catalog.inspect("graph.construct.explicit") is None
    assert catalog.declaration_record("graph.construct.explicit") is None
    snapshot_ids = {item.operation_id for item in catalog.snapshot().operations}
    assert "matrix.determinant.compute" in snapshot_ids
    assert "graph.construct.explicit" not in snapshot_ids


def test_inline_serving_runtime_runs_determinant_without_state() -> None:
    catalog = ServingCatalog.open(None, OperationVisibilityPolicy())
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
