from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian._models import StrictModel
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import OperationRequestValidationError, invoke_operation


class _Request(StrictModel):
    value: int


class _Result(StrictModel):
    value: int


class _InvalidResultOperation:
    operation_id = "test.invalid-result"
    request_type = _Request

    @staticmethod
    def run(request: StrictModel) -> StrictModel:
        del request
        return _Result.model_validate({"value": "not-an-integer"})


class _CatalogWithInvalidResult:
    @staticmethod
    def operation(operation_id: str) -> _InvalidResultOperation:
        del operation_id
        return _InvalidResultOperation()


def test_invoke_operation_runs_determinant_directly() -> None:
    catalog = Catalog.open()
    result = invoke_operation(
        "matrix.determinant.compute",
        {
            "matrix": {
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


def test_invoke_operation_wraps_den_num_orbit_result() -> None:
    """Vertex labels spelling the canonical rational keys must still wrap."""

    result = invoke_operation(
        "graph.symmetry.generator_orbits.compute",
        {
            "graph": {
                "graph": {"vertices": ["den", "num"], "edges": [["den", "num"]]},
            },
            "generators": [
                {
                    "generator_id": "identity",
                    "mapping": [["den", "den"], ["num", "num"]],
                }
            ],
        },
        Catalog.open(),
    )
    assert result.operation_id == "graph.symmetry.generator_orbits.compute"
    assert result.output["vertex_orbit_count"] == 2
    assert result.output["edge_orbit_count"] == 1


def test_invoke_operation_reports_unknown_id() -> None:
    catalog = Catalog.open()
    with pytest.raises(ValueError, match="unknown operation"):
        invoke_operation(
            "graph.construct.explicit",
            {"vertices": ["a"], "edges": []},
            catalog,
        )


def test_dispatch_distinguishes_request_and_result_validation() -> None:
    catalog = _CatalogWithInvalidResult()
    with pytest.raises(OperationRequestValidationError):
        invoke_operation("test.invalid-result", {"value": "bad"}, catalog)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        invoke_operation("test.invalid-result", {"value": 1}, catalog)  # type: ignore[arg-type]


def test_dispatch_classifies_noncanonical_json_as_request_validation() -> None:
    catalog = _CatalogWithInvalidResult()

    with pytest.raises(OperationRequestValidationError) as error:
        invoke_operation("test.invalid-result", {"value": 1.5}, catalog)  # type: ignore[arg-type]

    assert error.value.errors()[0]["type"] == "canonicalization_error"
