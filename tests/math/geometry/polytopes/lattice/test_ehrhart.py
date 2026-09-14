"""Exact Ehrhart interpolation for integral polytopes (#1192)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.polytopes.lattice._models import (
    EhrhartRequest,
    EhrhartResult,
)
from jacobian.math.geometry.polytopes.lattice._tools import ehrhart_polynomial
from jacobian.math.geometry.polytopes.lattice.operations import (
    ehrhart_polynomial as native_ehrhart,
)
from jacobian.math.geometry.polytopes.values import Vertex
from jacobian.math.polynomials.values import RationalPolynomial


def _vertex(*coordinates: int) -> dict[str, list[dict[str, int]]]:
    return {"coordinates": [{"num": value, "den": 1} for value in coordinates]}


def test_unit_square_ehrhart_polynomial_and_dilation_counts() -> None:
    request = EhrhartRequest.model_validate(
        {
            "vertices": [_vertex(0, 0), _vertex(1, 0), _vertex(0, 1), _vertex(1, 1)],
            "degree_bound": 2,
            "max_dilation": 4,
        }
    )
    result = ehrhart_polynomial(request)
    assert result.counts == ((0, 1), (1, 4), (2, 9), (3, 16), (4, 25))
    assert result.polynomial.variables == ("t",)
    assert [
        (term.coefficient.num, term.coefficient.den, term.exponents)
        for term in result.polynomial.polynomial.terms
    ] == [(1, 1, (2,)), (2, 1, (1,)), (1, 1, (0,))]


def test_unit_interval_has_zero_dilate_and_exact_linear_coefficients() -> None:
    result = ehrhart_polynomial(
        EhrhartRequest.model_validate(
            {"vertices": [_vertex(0), _vertex(1)], "degree_bound": 1, "max_dilation": 3}
        )
    )
    assert result.counts == ((0, 1), (1, 2), (2, 3), (3, 4))
    assert [
        (term.coefficient.num, term.coefficient.den, term.exponents)
        for term in result.polynomial.polynomial.terms
    ] == [(1, 1, (1,)), (1, 1, (0,))]
    assert result.max_dilation == 3


def test_rational_vertices_are_rejected_until_quasipolynomial_scope_exists() -> None:
    with pytest.raises(ValidationError, match="requires integral vertices"):
        EhrhartRequest.model_validate(
            {
                "vertices": [
                    {"coordinates": [{"num": 0, "den": 1}]},
                    {"coordinates": [{"num": 1, "den": 2}]},
                ],
                "degree_bound": 1,
            }
        )


def test_degree_bound_must_cover_dimension() -> None:
    with pytest.raises(ValidationError, match="cover the polytope dimension"):
        EhrhartRequest.model_validate(
            {
                "vertices": [_vertex(0, 0), _vertex(1, 0), _vertex(0, 1)],
                "degree_bound": 1,
            }
        )


def test_non_full_dimensional_one_point_source_is_rejected_at_native_admission() -> (
    None
):
    with pytest.raises(OperationDomainValidationError, match="affinely span"):
        native_ehrhart(
            (Vertex.model_validate(_vertex(0)),), degree_bound=1, max_dilation=1
        )


def test_all_dilation_scans_are_admitted_before_any_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.geometry.polytopes.lattice import operations

    calls = 0

    def fail_if_scanned(*_args: Any, **_kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("scan started before aggregate admission")

    monkeypatch.setattr(operations, "_scan_box", fail_if_scanned)
    with pytest.raises(OperationResourceAdmissionError, match="aggregate"):
        native_ehrhart(
            tuple(
                Vertex.model_validate(_vertex(*point))
                for point in ((0, 0), (50, 0), (0, 50), (50, 50))
            ),
            degree_bound=2,
            max_dilation=32,
        )
    assert calls == 0


def test_aggregate_scan_budget_rejects_before_scaled_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.geometry.polytopes.lattice import operations

    calls = 0
    original = operations._facets_and_box

    def count_geometry(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(operations, "_facets_and_box", count_geometry)
    with pytest.raises(OperationResourceAdmissionError, match="aggregate"):
        native_ehrhart(
            tuple(
                Vertex.model_validate(_vertex(*point))
                for point in ((0, 0), (200, 0), (0, 200), (200, 200))
            ),
            degree_bound=2,
            max_dilation=32,
        )
    assert calls == 0


def test_axis_span_budget_rejects_before_scaled_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.geometry.polytopes.lattice import operations

    calls = 0
    original = operations._facets_and_box

    def count_geometry(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(operations, "_facets_and_box", count_geometry)
    with pytest.raises(OperationResourceAdmissionError, match="per-axis span"):
        native_ehrhart(
            tuple(
                Vertex.model_validate(_vertex(*point))
                for point in ((0, 0), (10_001, 0), (0, 1), (10_001, 1))
            ),
            degree_bound=2,
            max_dilation=2,
        )
    assert calls == 0


def test_scaled_coordinate_height_is_rejected_before_carrier_construction() -> None:
    huge = 4 * 10**32767
    vertices = (
        Vertex(coordinates=(CanonicalRational(num=huge, den=1),)),
        Vertex(coordinates=(CanonicalRational(num=huge + 1, den=1),)),
    )
    request = EhrhartRequest.model_validate(
        {
            "vertices": [_vertex(huge), _vertex(huge + 1)],
            "degree_bound": 1,
            "max_dilation": 3,
        }
    )
    assert request.max_dilation == 3
    with pytest.raises(OperationDomainValidationError, match="maximum-dilation"):
        native_ehrhart(vertices, degree_bound=1, max_dilation=3)


def test_result_round_trip_retains_source_and_canonical_polynomial() -> None:
    result = ehrhart_polynomial(
        EhrhartRequest.model_validate(
            {"vertices": [_vertex(0), _vertex(1)], "degree_bound": 1, "max_dilation": 2}
        )
    )
    restored = EhrhartResult.model_validate_json(result.model_dump_json())
    assert restored == result
    assert restored.vertices == result.vertices
    assert isinstance(restored.polynomial, RationalPolynomial)


def test_forged_count_axes_and_values_are_rejected() -> None:
    with pytest.raises(ValidationError, match="dilation"):
        EhrhartResult.model_validate(
            {
                "vertices": [_vertex(0), _vertex(1)],
                "dimension": 1,
                "degree_bound": 1,
                "max_dilation": 2,
                "counts": [[0, 1], [2, -7], [2, 3]],
                "polynomial": {
                    "variables": ["t"],
                    "polynomial": {
                        "terms": [
                            {
                                "coefficient": {"num": 1, "den": 1},
                                "exponents": [0],
                            }
                        ]
                    },
                },
            }
        )
    with pytest.raises(ValidationError, match="nonnegative"):
        EhrhartResult.model_validate(
            {
                "vertices": [_vertex(0), _vertex(1)],
                "dimension": 1,
                "degree_bound": 1,
                "max_dilation": 2,
                "counts": [[0, 1], [1, -7], [2, 3]],
                "polynomial": {
                    "variables": ["t"],
                    "polynomial": {
                        "terms": [
                            {
                                "coefficient": {"num": 1, "den": 1},
                                "exponents": [0],
                            }
                        ]
                    },
                },
            }
        )


@pytest.mark.parametrize("max_dilation", [0, 1])
def test_native_ehrhart_rejects_invalid_domain(max_dilation: int) -> None:
    from jacobian.catalog.models import OperationDomainValidationError
    from jacobian.math.geometry.polytopes.lattice.operations import (
        ehrhart_polynomial as native,
    )
    from jacobian.math.geometry.polytopes.values import Vertex

    vertices = (
        Vertex.model_validate(_vertex(0)),
        Vertex.model_validate({"coordinates": [{"num": 1, "den": 2}]}),
    )
    with pytest.raises(OperationDomainValidationError):
        native(vertices, degree_bound=1, max_dilation=max_dilation)
