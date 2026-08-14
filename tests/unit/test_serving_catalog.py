from __future__ import annotations

from pathlib import Path

import pytest

from jacobian.contracts.graph_isomorphism import SimpleUndirectedGraph
from jacobian.contracts.operations import OperationInputKind, OperationRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.operation_catalog import OperationCatalogError
from jacobian.operation_visibility import OperationVisibilityPolicy
from jacobian.runtime.execution import create_inline_serving_runtime
from jacobian.schema_registry import model_schema_uri
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


def test_serving_catalog_rejects_non_file_catalog_path(tmp_path: Path) -> None:
    catalog_path = tmp_path / "metadata.sqlite3"
    catalog_path.mkdir()

    with pytest.raises(
        OperationCatalogError,
        match=r"STATE_UPDATE_REQUIRED: catalog state is unreadable",
    ):
        ServingCatalog.open(catalog_path)


def test_serving_catalog_inspects_family_ids_from_the_package_index() -> None:
    catalog = ServingCatalog.open(
        None,
        OperationVisibilityPolicy(),
    )

    descriptor = catalog.inspect("graph.construct.explicit")
    assert descriptor is not None
    assert descriptor.operation_id == "graph.construct.explicit"
    record = catalog.declaration_record("graph.construct.explicit")
    assert record is not None
    assert record.declaration_digest == "package-index"
    snapshot_ids = {item.operation_id for item in catalog.snapshot().operations}
    assert "matrix.determinant.compute" in snapshot_ids
    assert "graph.construct.explicit" in snapshot_ids
    assert "polynomial.expression.normalize" in snapshot_ids
    assert "lean.check" not in snapshot_ids
    assert "sat.cnf.materialize" not in snapshot_ids
    assert catalog.inspect("lean.check") is None
    assert catalog.declaration_record("sat.cnf.materialize") is None

    neighborhood = catalog.inspect("graph.compute.neighborhood_independence")
    assert neighborhood is not None
    assert neighborhood.accepted_input_kinds == (OperationInputKind.TYPED_ARTIFACT,)
    assert neighborhood.accepted_artifact_types == (
        model_schema_uri(
            name="jacobian.simple-undirected-graph",
            version="1",
            model=SimpleUndirectedGraph,
        ),
    )


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


def test_inline_serving_runtime_returns_state_failure_for_family_ids() -> None:
    catalog = ServingCatalog.open(None, OperationVisibilityPolicy())
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

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "STATE_INITIALIZATION_REQUIRED"
    assert result.diagnostics[0].stage == "operation_resolution"
    assert result.diagnostics[0].hint is not None
    assert "jacobian init" in result.diagnostics[0].hint
