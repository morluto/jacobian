from __future__ import annotations

import time
from typing import cast

import pytest
from pydantic import ValidationError, field_serializer

from jacobian._models import StrictModel
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import OperationRequestValidationError, invoke_operation


class _Request(StrictModel):
    value: int


class _Result(StrictModel):
    value: int


class _TimedResult(_Result):
    @field_serializer("value")
    def serialize_value(self, value: int) -> int:
        time.monotonic()
        return value


class _InvalidResultOperation:
    operation_id = "test.invalid-result"
    request_type = _Request

    @staticmethod
    def run(request: StrictModel) -> StrictModel:
        del request
        return _Result.model_validate({"value": "not-an-integer"})


class _CatalogWithInvalidResult:
    @staticmethod
    def _binding(operation_id: str) -> _InvalidResultOperation:
        del operation_id
        return _InvalidResultOperation()


class _TimedResultOperation:
    operation_id = "test.timed-result"
    request_type = _Request

    @staticmethod
    def run(request: StrictModel) -> StrictModel:
        assert isinstance(request, _Request)
        return _TimedResult(value=request.value)


class _CatalogWithTimedResult:
    @staticmethod
    def _binding(operation_id: str) -> _TimedResultOperation:
        del operation_id
        return _TimedResultOperation()


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


def test_runtime_includes_canonical_result_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter((1.0, 1.1, 1.2))
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))

    result = invoke_operation(
        "test.timed-result", {"value": 7}, cast(Catalog, _CatalogWithTimedResult())
    )

    assert result.output == {"value": 7}
    assert result.runtime_ms == 200
