from __future__ import annotations

import time
from typing import cast

import pytest
from pydantic import ValidationError, field_serializer

from jacobian._models import StrictModel
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import OperationRequestValidationError, invoke_operation
from jacobian.math.finite_fields import (
    element,
    finite_field,
    finite_polynomial,
    finite_polynomial_map,
)
from jacobian.math.finite_fields._models import FiniteMapTableRequest
from jacobian.math.graphs.values import SimpleUndirectedGraph


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
    with pytest.raises(ValidationError) as error:
        invoke_operation("test.invalid-result", {"value": 1}, catalog)  # type: ignore[arg-type]
    assert error.value.errors()[0]["type"] == "int_parsing"
    assert "Input should be a valid integer" in error.value.errors()[0]["msg"]


def test_dispatch_projects_owner_admission_as_an_invalid_request() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        invoke_operation(
            "topology.simplicial_complex.barycentric_subdivision.compute",
            {
                "complex": {
                    "vertices": list("abcdef"),
                    "facets": [list("abcdef")],
                }
            },
            Catalog.open(),
        )

    assert error.value.errors() == (
        {
            "loc": ("complex",),
            "type": "topology.require_barycentric_work_bounds_1",
            "msg": "barycentric subdivision requires at most 31 faces; "
            "input would produce more than 128 subdivision facets",
        },
    )


def test_dispatch_projects_finite_map_work_admission_as_an_invalid_request() -> None:
    presentation = finite_field(2, (1, 1, 0, 1, 1, 0, 0, 0, 1))
    one = element(presentation, (1,) + (0,) * 7)
    request = FiniteMapTableRequest(
        polynomial_map=finite_polynomial_map(
            finite_polynomial(presentation, (one,) * 512)
        )
    )

    with pytest.raises(OperationDomainValidationError) as error:
        invoke_operation(
            "finite_field.polynomial_map.table.compute",
            request.model_dump(mode="json"),
            Catalog.open(),
        )

    assert error.value.errors() == (
        {
            "loc": ("polynomial_map",),
            "type": "finite_field.finite_map_exceeds_operation_work_budget",
            "msg": "finite map exceeds the operation work budget",
        },
    )


def test_dispatch_projects_triangle_profile_admission_as_an_invalid_request() -> None:
    vertices = tuple(f"{index:03d}" + "x" * 61 for index in range(100))
    graph = SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple(
            (vertices[left], vertices[right])
            for left in range(len(vertices))
            for right in range(left + 1, len(vertices))
        ),
    )

    with pytest.raises(OperationDomainValidationError) as error:
        invoke_operation(
            "graph.triangle_profile.compute",
            {"graph": graph.model_dump(mode="json")},
            Catalog.open(),
        )

    assert error.value.errors()[0]["loc"] == ("graph",)
    assert error.value.errors()[0]["type"] == "graph.triangle_profile.output_budget"


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
