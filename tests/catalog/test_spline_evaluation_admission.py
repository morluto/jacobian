"""Public exact-output admission for spline evaluation."""

import pytest

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.dispatch import invoke_operation
from jacobian.math.geometry.polytopes._models import (
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes.complexes._models import (
    ComplexPoint,
    SplineEvaluationRequest,
    SplineEvaluationResult,
)
from jacobian.math.geometry.polytopes.complexes.operations import (
    polytopal_complex_closure,
)


def _request(
    coefficients: tuple[CanonicalRational, ...], coordinate: CanonicalRational
) -> SplineEvaluationRequest:
    interval = RationalVPolytope(
        space=RationalCoordinateSpace(axes=("x",)),
        vertices=tuple(
            RationalPolytopeVertex(
                vertex_id=f"v{value}",
                coordinates=(CanonicalRational(num=value, den=1),),
            )
            for value in (0, 1)
        ),
    )
    return SplineEvaluationRequest(
        complex=polytopal_complex_closure((interval,)),
        degree=len(coefficients) - 1,
        smoothness=0,
        basis_coefficients=coefficients,
        point=ComplexPoint(coordinates=(coordinate,)),
    )


@pytest.mark.parametrize("degree", (2, 3))
def test_public_spline_repeated_cancellation_preserves_output_bound(
    degree: int,
) -> None:
    at_limit = 10 ** (MAX_CANONICAL_RATIONAL_DIGITS - 1)
    request = _request(
        (CanonicalRational(num=at_limit, den=1),)
        + (CanonicalRational(num=0, den=1),) * degree,
        CanonicalRational(num=1, den=at_limit),
    )
    payload = request.model_dump(mode="json")
    catalog = Catalog.open()
    if degree == 3:
        with pytest.raises(OperationResourceAdmissionError) as error:
            invoke_operation(
                "polyhedral_complex.spline.evaluate.compute", payload, catalog
            )
        assert error.value.errors()[0]["type"] == (
            "polytopal_complex.spline_evaluation_growth"
        )
    else:
        result = invoke_operation(
            "polyhedral_complex.spline.evaluate.compute", payload, catalog
        )
        decoded = SplineEvaluationResult.model_validate_json(
            encode_strict_json(result.output)
        )
        assert decoded.value == CanonicalRational(num=1, den=at_limit)
        assert decoded.complex == request.complex


@pytest.mark.parametrize("constant_is_zero", (True, False))
def test_public_spline_zero_monomials_do_not_consume_output_budget(
    constant_is_zero: bool,
) -> None:
    at_limit = 10 ** (MAX_CANONICAL_RATIONAL_DIGITS - 1)
    constant = CanonicalRational(num=0 if constant_is_zero else at_limit, den=1)
    request = _request(
        (CanonicalRational(num=1, den=at_limit),) * 2 + (constant,),
        CanonicalRational(num=0, den=1),
    )

    result = invoke_operation(
        "polyhedral_complex.spline.evaluate.compute",
        request.model_dump(mode="json"),
        Catalog.open(),
    )
    decoded = SplineEvaluationResult.model_validate_json(
        encode_strict_json(result.output)
    )
    assert decoded.value == constant
    assert decoded.complex == request.complex
    assert decoded.point == request.point
