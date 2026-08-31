"""Canonical contracts for exact plane semialgebraic component profiles."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.analysis.intervals import ClosedRationalInterval, RationalBox
from jacobian.math.number_theory.algebraic_numbers.real import RealAlgebraicValue
from jacobian.math.polynomials.real_algebra._plane_component_models import (
    MAX_PLANE_COMPONENT_POINT_DEGREE,
    MAX_PLANE_COMPONENT_POINT_ISOLATOR_DIGITS,
    MAX_PLANE_COMPONENT_POINT_TERMS,
    MAX_PLANE_COMPONENT_POLYNOMIALS,
    MAX_PLANE_COMPONENT_SIGN_CONDITIONS,
    MAX_PLANE_COMPONENTS,
    IsolatedRealPlanePoint,
    PlaneComponentProfileComputed,
    PlaneComponentProfileNoncompletion,
    PlaneComponentProfileRequest,
    PlaneComponentProfileResult,
    PlaneSampleDisposition,
    PlaneSemialgebraicComponent,
    PlaneSemialgebraicSet,
    PlaneSign,
    PlaneSignCondition,
)
from jacobian.math.polynomials.real_algebra._plane_components import _computed_result
from jacobian.math.polynomials.real_algebra._qepcad_plane_protocol import (
    MAX_QEPCAD_POINT_JSON_BYTES,
    MAX_QEPCAD_WORKER_RESPONSE_BYTES,
    QepcadPlaneWorkerComplete,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _q(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _polynomial(
    terms: tuple[tuple[int, tuple[int, int]], ...],
    *,
    axis: tuple[str, str] = ("x", "y"),
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=axis,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(coefficient=_q(coefficient), exponents=exponents)
                for coefficient, exponents in terms
            )
        ),
    )


def _coordinate_polynomial(axis_index: int, root: int) -> RationalPolynomial:
    coordinate_exponents = (1, 0) if axis_index == 0 else (0, 1)
    terms: tuple[tuple[int, tuple[int, int]], ...] = (
        ((1, coordinate_exponents),)
        if root == 0
        else ((1, coordinate_exponents), (-root, (0, 0)))
    )
    return _polynomial(terms)


def _rational_value(value: int) -> RealAlgebraicValue:
    return RealAlgebraicValue._from_admitted_polynomial(
        polynomial=("1", str(-value)),
        real_root_index=0,
    )


def _rational_point(x: int, y: int) -> IsolatedRealPlanePoint:
    return IsolatedRealPlanePoint(
        axis=("x", "y"),
        coordinates=(_rational_value(x), _rational_value(y)),
        isolating_box=RationalBox(
            domain="QQ",
            variables=("x", "y"),
            intervals=(
                ClosedRationalInterval(lower=_q(x), upper=_q(x)),
                ClosedRationalInterval(lower=_q(y), upper=_q(y)),
            ),
        ),
    )


def test_sign_table_canonicalizes_polynomial_and_row_order() -> None:
    circle = _polynomial(((1, (2, 0)), (1, (0, 2)), (-1, (0, 0))))
    vertical = _polynomial(((1, (1, 0)),))

    first = PlaneSemialgebraicSet(
        axis=("x", "y"),
        polynomials=(circle, vertical),
        sign_conditions=(
            PlaneSignCondition(signs=(PlaneSign.NEGATIVE, PlaneSign.POSITIVE)),
            PlaneSignCondition(signs=(PlaneSign.ZERO, PlaneSign.POSITIVE)),
        ),
    )
    second = PlaneSemialgebraicSet(
        axis=("x", "y"),
        polynomials=(vertical, circle),
        sign_conditions=(
            PlaneSignCondition(signs=(PlaneSign.POSITIVE, PlaneSign.ZERO)),
            PlaneSignCondition(signs=(PlaneSign.POSITIVE, PlaneSign.NEGATIVE)),
        ),
    )

    assert first == second
    assert first.model_dump_json() == second.model_dump_json()


def test_sign_table_deduplicates_rows() -> None:
    x = _polynomial(((1, (1, 0)),))
    semialgebraic_set = PlaneSemialgebraicSet(
        axis=("x", "y"),
        polynomials=(x,),
        sign_conditions=(
            PlaneSignCondition(signs=(PlaneSign.ZERO,)),
            PlaneSignCondition(signs=(PlaneSign.ZERO,)),
        ),
    )

    assert semialgebraic_set.sign_conditions == (
        PlaneSignCondition(signs=(PlaneSign.ZERO,)),
    )


def test_empty_and_whole_plane_have_unambiguous_zero_polynomial_tables() -> None:
    empty = PlaneSemialgebraicSet(axis=("x", "y"), polynomials=(), sign_conditions=())
    whole = PlaneSemialgebraicSet(
        axis=("x", "y"),
        polynomials=(),
        sign_conditions=(PlaneSignCondition(signs=()),),
    )

    assert empty != whole


def test_sign_rows_must_cover_the_complete_polynomial_axis() -> None:
    circle = _polynomial(((1, (2, 0)), (1, (0, 2)), (-1, (0, 0))))
    with pytest.raises(ValidationError, match="one sign per polynomial"):
        PlaneSemialgebraicSet(
            axis=("x", "y"),
            polynomials=(circle,),
            sign_conditions=(PlaneSignCondition(signs=()),),
        )


def test_plane_point_reuses_algebraic_coordinates_and_binds_box_to_one_axis() -> None:
    point = _rational_point(2, -3)

    assert point.axis == ("x", "y")
    assert point.coordinates == (_rational_value(2), _rational_value(-3))
    assert point.isolating_box.intervals[0].lower == _q(2)
    assert point.isolating_box.intervals[1].lower == _q(-3)

    with pytest.raises(ValidationError, match="complete ordered plane axis"):
        IsolatedRealPlanePoint(
            axis=("x", "y"),
            coordinates=point.coordinates,
            isolating_box=point.isolating_box.model_copy(
                update={"variables": ("y", "x")}
            ),
        )


def test_plane_point_rejects_coordinates_beyond_the_result_carrier_bound() -> None:
    point = _rational_point(0, 0)
    over_degree = RealAlgebraicValue.model_construct(
        polynomial=("1",) + ("0",) * MAX_PLANE_COMPONENT_POINT_DEGREE + ("-2",),
        real_root_index=0,
    )

    with pytest.raises(ValidationError, match="degree-sixteen"):
        IsolatedRealPlanePoint(
            axis=("x", "y"),
            coordinates=(over_degree, point.coordinates[1]),
            isolating_box=point.isolating_box,
        )


def test_plane_point_schema_advertises_the_degree_sixteen_coordinate_carrier() -> None:
    coordinate_schema = IsolatedRealPlanePoint.model_json_schema()["properties"][
        "coordinates"
    ]["prefixItems"][0]

    assert coordinate_schema["properties"]["polynomial"]["maxItems"] == (
        MAX_PLANE_COMPONENT_POINT_TERMS
    )


def _large_structural_point(index: int) -> IsolatedRealPlanePoint:
    axis = ("x" * 32, "y" * 32)
    coefficient_base = 5 * 10**510

    def coordinate_value(adjustment: int) -> RealAlgebraicValue:
        coefficients = []
        for degree in range(MAX_PLANE_COMPONENT_POINT_DEGREE, -1, -1):
            if degree == MAX_PLANE_COMPONENT_POINT_DEGREE:
                coefficient = 10**511 + 1
            elif degree == 0:
                coefficient = -(5 * 10**511 + 2)
            else:
                coefficient = (
                    ((-1) ** degree)
                    * 2
                    * (coefficient_base + (adjustment if degree == 15 else 0))
                )
            coefficients.append(str(coefficient))
        return RealAlgebraicValue._from_admitted_polynomial(
            polynomial=tuple(coefficients),
            real_root_index=0,
        )

    denominator = 9 * 10 ** (MAX_PLANE_COMPONENT_POINT_ISOLATOR_DIGITS - 1) + 7
    lower = CanonicalRational.from_integer_ratio(-(denominator - 1), denominator)
    upper = CanonicalRational.from_integer_ratio(-(denominator // 2), denominator)
    interval = ClosedRationalInterval(lower=lower, upper=upper)
    return IsolatedRealPlanePoint(
        axis=axis,
        coordinates=(coordinate_value(index), coordinate_value(0)),
        isolating_box=RationalBox(
            variables=axis,
            intervals=(interval, interval),
        ),
    )


def test_maximal_point_carrier_fits_the_per_point_worker_reservation() -> None:
    point = _large_structural_point(0)

    encoded_bytes = len(point.model_dump_json().encode())

    assert encoded_bytes > 64 * 1024
    assert encoded_bytes <= MAX_QEPCAD_POINT_JSON_BYTES


def test_aggregate_point_output_limit_is_an_explicit_noncompletion() -> None:
    points = tuple(
        sorted(
            (_large_structural_point(index) for index in range(MAX_PLANE_COMPONENTS)),
            key=lambda point: point.model_dump_json(),
        )
    )
    projection = QepcadPlaneWorkerComplete(
        version="1.74",
        representatives=points,
        sample_component_ids=(None,) * 8,
    )
    request = PlaneComponentProfileRequest(
        semialgebraic_set=PlaneSemialgebraicSet(
            axis=points[0].axis,
            polynomials=(),
            sign_conditions=(PlaneSignCondition(signs=()),),
        )
    )
    outcome = PlaneComponentProfileComputed(
        components=tuple(
            PlaneSemialgebraicComponent(
                component_id=index,
                representative=point,
            )
            for index, point in enumerate(points)
        ),
        sample_dispositions=(),
    )

    assert len(projection.model_dump_json().encode()) > MAX_QEPCAD_WORKER_RESPONSE_BYTES
    result = _computed_result(request, outcome)
    assert result.outcome.status == "RESOURCE_LIMIT"
    assert result.outcome.reason == "RESULT_OUTPUT_LIMIT"


def test_result_outcome_is_discriminated_and_source_bound() -> None:
    semialgebraic_set = PlaneSemialgebraicSet(
        axis=("x", "y"), polynomials=(), sign_conditions=()
    )
    point = _rational_point(0, 0)
    computed = PlaneComponentProfileResult(
        semialgebraic_set=semialgebraic_set,
        samples=(point,),
        outcome=PlaneComponentProfileComputed(
            components=(),
            sample_dispositions=(
                PlaneSampleDisposition(sample_index=0, status="OUTSIDE"),
            ),
        ),
    )
    unavailable = PlaneComponentProfileResult(
        semialgebraic_set=semialgebraic_set,
        samples=(point,),
        outcome=PlaneComponentProfileNoncompletion(
            status="BACKEND_UNAVAILABLE",
            reason="SUPPORTED_QEPCAD_NOT_INSTALLED",
        ),
    )

    schema = PlaneComponentProfileResult.model_json_schema()
    assert (
        schema["$defs"]["PlaneComponentProfileComputed"]["properties"]["status"][
            "const"
        ]
        == "COMPUTED"
    )
    assert computed.outcome.status == "COMPUTED"
    assert unavailable.outcome.status == "BACKEND_UNAVAILABLE"

    with pytest.raises(ValidationError, match="does not match"):
        PlaneComponentProfileNoncompletion(
            status="TIMEOUT",
            reason="QEPCAD_INVALID_OUTPUT",
        )


def test_computed_components_require_unique_canonical_representative_order() -> None:
    representatives = tuple(
        sorted(
            (_rational_point(-2, 0), _rational_point(3, 0)),
            key=lambda point: point.model_dump_json(),
        )
    )
    canonical = PlaneComponentProfileComputed(
        components=tuple(
            PlaneSemialgebraicComponent(
                component_id=index,
                representative=representative,
            )
            for index, representative in enumerate(representatives)
        ),
        sample_dispositions=(),
    )

    assert (
        tuple(component.representative for component in canonical.components)
        == representatives
    )
    with pytest.raises(ValidationError, match="canonically ordered"):
        PlaneComponentProfileComputed(
            components=(
                PlaneSemialgebraicComponent(
                    component_id=0,
                    representative=representatives[1],
                ),
                PlaneSemialgebraicComponent(
                    component_id=1,
                    representative=representatives[0],
                ),
            ),
            sample_dispositions=(),
        )
    with pytest.raises(ValidationError, match="unique"):
        PlaneComponentProfileComputed(
            components=(
                PlaneSemialgebraicComponent(
                    component_id=0,
                    representative=representatives[0],
                ),
                PlaneSemialgebraicComponent(
                    component_id=1,
                    representative=representatives[0],
                ),
            ),
            sample_dispositions=(),
        )


def test_component_identity_excludes_nonunique_isolating_boxes() -> None:
    positive_sqrt_two = RealAlgebraicValue._from_admitted_polynomial(
        polynomial=("1", "0", "-2"),
        real_root_index=1,
    )
    zero = _rational_value(0)

    def point(lower: Fraction, upper: Fraction) -> IsolatedRealPlanePoint:
        return IsolatedRealPlanePoint(
            axis=("x", "y"),
            coordinates=(positive_sqrt_two, zero),
            isolating_box=RationalBox(
                domain="QQ",
                variables=("x", "y"),
                intervals=(
                    ClosedRationalInterval(lower=_q(lower), upper=_q(upper)),
                    ClosedRationalInterval(lower=_q(0), upper=_q(0)),
                ),
            ),
        )

    broad = point(Fraction(1), Fraction(2))
    narrow = point(Fraction(7, 5), Fraction(3, 2))
    with pytest.raises(ValidationError, match="unique"):
        PlaneComponentProfileComputed(
            components=(
                PlaneSemialgebraicComponent(component_id=0, representative=broad),
                PlaneSemialgebraicComponent(component_id=1, representative=narrow),
            ),
            sample_dispositions=(),
        )

    with pytest.raises(ValidationError, match="unique"):
        QepcadPlaneWorkerComplete(
            version="1.74",
            representatives=(broad, narrow),
            sample_component_ids=(),
        )


def test_request_schema_exposes_operation_owned_collection_bounds() -> None:
    schema = PlaneSemialgebraicSet.model_json_schema()
    assert schema["properties"]["polynomials"]["maxItems"] == (
        MAX_PLANE_COMPONENT_POLYNOMIALS
    )
    assert schema["properties"]["sign_conditions"]["maxItems"] == (
        MAX_PLANE_COMPONENT_SIGN_CONDITIONS
    )


def test_request_raw_preflight_rejects_deep_malformed_scalar_without_recursing() -> (
    None
):
    raw = PlaneComponentProfileRequest(
        semialgebraic_set=PlaneSemialgebraicSet(
            axis=("x", "y"),
            polynomials=(_polynomial(((1, (1, 0)),)),),
            sign_conditions=(PlaneSignCondition(signs=(PlaneSign.ZERO,)),),
        )
    ).model_dump(mode="json")
    nested: object = "1"
    for _ in range(1_500):
        nested = {"nested": nested}
    raw["semialgebraic_set"]["polynomials"][0]["polynomial"]["terms"][0]["coefficient"][
        "num"
    ] = nested

    with pytest.raises(ValidationError, match="decimal strings"):
        PlaneComponentProfileRequest.model_validate(raw)


@pytest.mark.parametrize(
    ("path", "malformed", "message"),
    [
        (("semialgebraic_set", "axis"), {"x": "y"}, "JSON array"),
        (
            ("semialgebraic_set", "polynomials", 0, "polynomial", "terms"),
            {"term": 1},
            "JSON array",
        ),
        (
            (
                "semialgebraic_set",
                "polynomials",
                0,
                "polynomial",
                "terms",
                0,
                "exponents",
            ),
            {"x": 1},
            "JSON array",
        ),
    ],
)
def test_request_raw_preflight_rejects_non_array_axes_and_terms(
    path: tuple[str | int, ...], malformed: object, message: str
) -> None:
    raw = PlaneComponentProfileRequest(
        semialgebraic_set=PlaneSemialgebraicSet(
            axis=("x", "y"),
            polynomials=(_polynomial(((1, (1, 0)),)),),
            sign_conditions=(PlaneSignCondition(signs=(PlaneSign.ZERO,)),),
        )
    ).model_dump(mode="json")
    target: Any = raw
    for segment in path[:-1]:
        target = target[segment]
    target[path[-1]] = malformed

    with pytest.raises(ValidationError, match=message):
        PlaneComponentProfileRequest.model_validate(raw)


def test_request_raw_preflight_rejects_unknown_deep_fields_before_normalization() -> (
    None
):
    raw = PlaneComponentProfileRequest(
        semialgebraic_set=PlaneSemialgebraicSet(
            axis=("x", "y"), polynomials=(), sign_conditions=()
        )
    ).model_dump(mode="json")
    nested: object = None
    for _ in range(1_500):
        nested = {"nested": nested}
    raw["unknown"] = nested

    with pytest.raises(ValidationError, match="unknown fields"):
        PlaneComponentProfileRequest.model_validate(raw)
