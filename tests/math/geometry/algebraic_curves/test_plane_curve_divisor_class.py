import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.algebraic_curves.divisor_classes._models import (
    PlaneCurveStrictTransformRequest,
)
from jacobian.math.geometry.algebraic_curves.divisor_classes._tools import TOOLS
from jacobian.math.geometry.algebraic_curves.divisor_classes.operations import (
    plane_curve_strict_transform_class,
)
from jacobian.math.geometry.blowup_p2._models import (
    BlowupP2Surface,
    BlowupPoint,
)
from jacobian.math.geometry.blowup_p2.operations import (
    construct_divisor_class,
    intersect_classes,
)
from jacobian.math.geometry.projective.coordinates._models import (
    RationalProjectivePoint,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _point(*coordinates: int) -> dict[str, object]:
    return {"coordinates": [{"num": value, "den": 1} for value in coordinates]}


def _cuspidal_cubic_request() -> PlaneCurveStrictTransformRequest:
    return PlaneCurveStrictTransformRequest.model_validate(
        {
            "polynomial": {
                "domain": "QQ",
                "variables": ["u", "v", "w"],
                "polynomial": {
                    "terms": [
                        {"coefficient": {"num": -1, "den": 1}, "exponents": [3, 0, 0]},
                        {"coefficient": {"num": 1, "den": 1}, "exponents": [0, 2, 1]},
                    ]
                },
            },
            "surface": {
                "points": [
                    {"label": "cusp", "point": _point(1, 0, 0)},
                    {"label": "off_curve", "point": _point(0, 1, 0)},
                    {"label": "smooth", "point": _point(1, 1, 1)},
                ]
            },
            # Point coordinates [X0:X1:X2] mean [w:u:v].
            "projective_coordinate_variables": ["w", "u", "v"],
        }
    )


def _compute(request: PlaneCurveStrictTransformRequest):
    return plane_curve_strict_transform_class(
        request.polynomial,
        request.surface,
        request.projective_coordinate_variables,
    )


def test_curve_class_uses_exact_local_multiplicities_and_axis_transport() -> None:
    request = _cuspidal_cubic_request()
    divisor = _compute(request)
    assert divisor.degree == 3
    assert divisor.multiplicities == (2, 0, 1)
    assert tuple(point.label for point in divisor.surface.points) == (
        "cusp",
        "off_curve",
        "smooth",
    )


def test_curve_divisor_class_composes_with_existing_intersection_operation() -> None:
    divisor = _compute(_cuspidal_cubic_request())
    other = construct_divisor_class(divisor.surface, 4, (1, 5, 2))
    result = intersect_classes(divisor, other)

    # Independent intersection-form oracle: d*e - sum_i m_i*n_i.
    oracle = 3 * 4 - sum(a * b for a, b in zip((2, 0, 1), (1, 5, 2), strict=True))
    assert oracle == 8
    assert result.value == oracle
    assert tuple(row.product for row in result.exceptional_subtractions) == (2, 0, 2)


def test_unblown_parent_produces_empty_multiplicity_axis() -> None:
    request = PlaneCurveStrictTransformRequest.model_validate(
        {
            "polynomial": {
                "domain": "QQ",
                "variables": ["x", "y", "z"],
                "polynomial": {
                    "terms": [
                        {"coefficient": {"num": 1, "den": 1}, "exponents": [1, 0, 0]},
                    ]
                },
            },
            "surface": {"points": []},
            "projective_coordinate_variables": ["x", "y", "z"],
        }
    )
    result = _compute(request)
    assert (result.degree, result.multiplicities) == (1, ())


def test_nonhomogeneous_source_is_rejected() -> None:
    raw = {
        "polynomial": {
            "domain": "QQ",
            "variables": ["x", "y", "z"],
            "polynomial": {
                "terms": [
                    {"coefficient": {"num": 1, "den": 1}, "exponents": [1, 0, 0]},
                    {"coefficient": {"num": 1, "den": 1}, "exponents": [0, 0, 0]},
                ]
            },
        },
        "surface": {"points": []},
        "projective_coordinate_variables": ["x", "y", "z"],
    }
    with pytest.raises(ValidationError, match="must be homogeneous"):
        PlaneCurveStrictTransformRequest.model_validate(raw)


def test_operation_manifest_uses_the_plane_curve_divisor_id() -> None:
    assert tuple(tool.operation_id for tool in TOOLS) == (
        "algebraic_geometry.plane_curve.strict_transform_class.compute",
    )


def test_native_nonhomogeneous_high_exponent_is_domain_error_before_degree_budget() -> (
    None
):
    request = _cuspidal_cubic_request()
    polynomial = RationalPolynomial.model_construct(
        domain="QQ",
        variables=("x", "y", "z"),
        polynomial=SparseRationalPolynomial.model_construct(
            terms=(
                RationalPolynomialTerm.model_construct(
                    coefficient=CanonicalRational(num=1, den=1), exponents=(13, 0, 0)
                ),
                RationalPolynomialTerm.model_construct(
                    coefficient=CanonicalRational(num=1, den=1), exponents=(0, 0, 0)
                ),
            )
        ),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        plane_curve_strict_transform_class(
            polynomial, request.surface, request.projective_coordinate_variables
        )
    assert error.value.errors()[0]["type"] == "plane_curve_divisor.inhomogeneous"


def test_forged_rational_fields_are_domain_errors_not_helper_exceptions() -> None:
    request = _cuspidal_cubic_request()
    polynomial = RationalPolynomial.model_construct(
        domain="QQ",
        variables=("x", "y", "z"),
        polynomial=SparseRationalPolynomial.model_construct(
            terms=(
                RationalPolynomialTerm.model_construct(
                    coefficient=CanonicalRational.model_construct(num="bad", den=1),
                    exponents=(1, 0, 0),
                ),
            )
        ),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        plane_curve_strict_transform_class(
            polynomial, request.surface, request.projective_coordinate_variables
        )
    assert error.value.errors()[0]["type"] == "plane_curve_divisor.coefficient_type"


def test_native_operation_accepts_values_and_rejects_wire_request_model() -> None:
    request = _cuspidal_cubic_request()
    with pytest.raises(OperationDomainValidationError) as error:
        plane_curve_strict_transform_class(
            request, request.surface, request.projective_coordinate_variables
        )

    assert error.value.errors()[0]["type"] == "plane_curve_divisor.polynomial_type"


def test_native_operation_rejects_malformed_canonical_values_with_stable_error() -> (
    None
):
    malformed_term = RationalPolynomialTerm.model_construct(
        coefficient=CanonicalRational(num=1, den=1),
        exponents=(1, "bad", 0),
    )
    malformed_polynomial = RationalPolynomial.model_construct(
        domain="QQ",
        variables=("x", "y", "z"),
        polynomial=SparseRationalPolynomial.model_construct(terms=(malformed_term,)),
    )
    request = _cuspidal_cubic_request()

    with pytest.raises(OperationDomainValidationError) as error:
        plane_curve_strict_transform_class(
            malformed_polynomial,
            request.surface,
            request.projective_coordinate_variables,
        )

    assert error.value.errors()[0]["type"] == "plane_curve_divisor.term_shape"


def test_native_operation_accepts_the_full_point_cardinality_bound() -> None:
    request = PlaneCurveStrictTransformRequest.model_validate(
        {
            "polynomial": {
                "domain": "QQ",
                "variables": ["x", "y", "z"],
                "polynomial": {
                    "terms": [
                        {"coefficient": {"num": 1, "den": 1}, "exponents": [1, 0, 0]}
                    ]
                },
            },
            "surface": {
                "points": [
                    {
                        "label": f"p{index:02}",
                        "point": _point(0, 1, index),
                    }
                    for index in range(16)
                ]
            },
            "projective_coordinate_variables": ["x", "y", "z"],
        }
    )

    result = _compute(request)

    assert len(result.surface.points) == 16
    assert result.multiplicities == (1,) * 16


def test_native_operation_classifies_forged_large_rationals_as_resource_limits() -> (
    None
):
    request = _cuspidal_cubic_request()
    huge = 10**5000
    forged_coefficient = RationalPolynomial.model_construct(
        domain="QQ",
        variables=("x", "y", "z"),
        polynomial=SparseRationalPolynomial.model_construct(
            terms=(
                RationalPolynomialTerm.model_construct(
                    coefficient=CanonicalRational.model_construct(num=huge, den=1),
                    exponents=(1, 0, 0),
                ),
            )
        ),
    )
    with pytest.raises(OperationResourceAdmissionError) as coefficient_error:
        plane_curve_strict_transform_class(
            forged_coefficient,
            request.surface,
            ("x", "y", "z"),
        )
    assert (
        coefficient_error.value.errors()[0]["type"]
        == "plane_curve_divisor.coefficient_bound"
    )

    point = BlowupPoint.model_construct(
        label="large",
        point=RationalProjectivePoint.model_construct(
            coordinates=(
                CanonicalRational.model_construct(num=huge, den=1),
                CanonicalRational.model_construct(num=0, den=1),
                CanonicalRational.model_construct(num=1, den=1),
            )
        ),
    )
    surface = BlowupP2Surface.model_construct(points=(point,))
    with pytest.raises(OperationResourceAdmissionError) as point_error:
        plane_curve_strict_transform_class(
            request.polynomial,
            surface,
            request.projective_coordinate_variables,
        )
    assert point_error.value.errors()[0]["type"] == "plane_curve_divisor.point_height"


def test_nonzero_constant_is_rejected_as_not_a_curve() -> None:
    request = _cuspidal_cubic_request()
    constant = RationalPolynomial.model_construct(
        domain="QQ",
        variables=("x", "y", "z"),
        polynomial=SparseRationalPolynomial.model_construct(
            terms=(
                RationalPolynomialTerm.model_construct(
                    coefficient=CanonicalRational(num=1, den=1),
                    exponents=(0, 0, 0),
                ),
            )
        ),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        plane_curve_strict_transform_class(constant, request.surface, ("x", "y", "z"))
    assert error.value.errors()[0]["type"] == "plane_curve_divisor.degree_zero"


def test_native_operation_rejects_point_count_before_computation() -> None:
    request = _cuspidal_cubic_request()
    point = BlowupPoint.model_construct(
        label="p",
        point=RationalProjectivePoint.model_construct(
            coordinates=(
                CanonicalRational(num=1, den=1),
                CanonicalRational(num=0, den=1),
                CanonicalRational(num=0, den=1),
            )
        ),
    )
    oversized_surface = BlowupP2Surface.model_construct(points=(point,) * 17)

    with pytest.raises(OperationResourceAdmissionError) as error:
        plane_curve_strict_transform_class(
            request.polynomial,
            oversized_surface,
            request.projective_coordinate_variables,
        )

    assert error.value.errors()[0]["type"] == "plane_curve_divisor.point_bound"
