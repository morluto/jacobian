"""Cross-owner composition for rational conic coordinate functions."""

from jacobian._exact import CanonicalRational
from jacobian.math.matrices.symbolic._models import (
    SymbolicDeterminantRequest,
)
from jacobian.math.matrices.symbolic._operations import compute_symbolic_determinant
from jacobian.math.plane_algebraic_curves._models import (
    RationalConicParametrizationRequest,
)
from jacobian.math.plane_algebraic_curves._operations import (
    compute_rational_conic_parametrization,
)
from jacobian.math.polynomials.maps._models import EvalRequest, VariablePoint
from jacobian.math.polynomials.maps._operations import evaluate_polynomial
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _integer(value: int) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(value, 1)


def test_conic_coordinate_serialization_is_a_symbolic_matrix_entry() -> None:
    source = RationalPolynomial(
        variables=("x", "y"),
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(coefficient=_integer(1), exponents=(2, 0)),
                RationalPolynomialTerm(coefficient=_integer(1), exponents=(0, 2)),
                RationalPolynomialTerm(coefficient=_integer(-1), exponents=(0, 0)),
            )
        ),
    )
    parametrization = compute_rational_conic_parametrization(
        RationalConicParametrizationRequest(
            polynomial=source,
            point=VariablePoint(
                variables=source.variables,
                values=(_integer(1), _integer(0)),
            ),
            parameter="t",
        )
    )
    coordinate_payload = parametrization.model_dump(mode="json")["coordinates"][0]
    determinant_request = SymbolicDeterminantRequest.model_validate(
        {
            "matrix": {
                "variables": ["t"],
                "entries": [[coordinate_payload]],
            }
        }
    )

    assert determinant_request.matrix.entries[0][0] == parametrization.coordinates[0]
    determinant = compute_symbolic_determinant(determinant_request)
    assert determinant.determinant == parametrization.coordinates[0]


def test_exceptional_point_serialization_evaluates_on_its_source_conic() -> None:
    source = RationalPolynomial(
        variables=("x", "y"),
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(coefficient=_integer(1), exponents=(2, 0)),
                RationalPolynomialTerm(coefficient=_integer(1), exponents=(0, 2)),
                RationalPolynomialTerm(coefficient=_integer(-1), exponents=(0, 0)),
            )
        ),
    )
    point = VariablePoint(
        variables=source.variables,
        values=(_integer(1), _integer(0)),
    )
    request_payload = {
        "polynomial": source.model_dump(mode="json"),
        "point": point.model_dump(mode="json"),
        "parameter": "t",
    }
    request = RationalConicParametrizationRequest.model_validate(request_payload)
    assert request.point == point

    parametrization = compute_rational_conic_parametrization(request)
    exceptional_payload = parametrization.model_dump(mode="json")["exceptional_point"]
    evaluation_request = EvalRequest.model_validate(
        {
            "polynomial": source.model_dump(mode="json"),
            "point": exceptional_payload,
        }
    )
    assert evaluation_request.point == parametrization.exceptional_point
    assert evaluate_polynomial(evaluation_request).value == _integer(0)
