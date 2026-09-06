"""Exact sparse affine character pullback without Boolean-table expansion."""

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.analysis.boolean.fourier.values import (
    BooleanAffineMap,
    RationalWalshPolynomial,
    WalshTerm,
)


class WalshPullbackRequest(StrictModel):
    polynomial: RationalWalshPolynomial
    affine_map: BooleanAffineMap


def affine_pullback(
    polynomial: RationalWalshPolynomial, affine_map: BooleanAffineMap
) -> RationalWalshPolynomial:
    if len(affine_map.rows) != polynomial.variable_count:
        raise OperationDomainValidationError(
            location=("affine_map",),
            code="boolean.affine_map_dimension",
            message="map rows must match the source polynomial dimension",
        )
    # Each source character contributes at most one output character. XOR
    # work is sparse incidence work; sorting emits at most target_dimension
    # coordinates per surviving term. A common denominator divides the product
    # of source denominators, so summed input digit heights bound all sums.
    work = sum(
        len(affine_map.rows[i]) for term in polynomial.terms for i in term.character
    )
    output_incidences = len(polynomial.terms) * affine_map.target_dimension
    digits = sum(
        max(len(term.coefficient.num.lstrip("-")), len(term.coefficient.den))
        for term in polynomial.terms
    )
    if work > 4_194_304 or output_incidences > 65_536 or digits > 8192:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="boolean.walsh_pullback_budget",
            message="sparse pullback exceeds incidence work, output support, or rational height envelope",
        )
    coefficients: dict[tuple[int, ...], Fraction] = {}
    for term in polynomial.terms:
        target: set[int] = set()
        phase = 0
        for i in term.character:
            target.symmetric_difference_update(affine_map.rows[i])
            phase ^= affine_map.offset[i]
        character = tuple(sorted(target))
        value = term.coefficient.as_fraction() * (-1 if phase else 1)
        coefficients[character] = coefficients.get(character, Fraction()) + value
    return RationalWalshPolynomial(
        variable_count=affine_map.target_dimension,
        terms=tuple(
            WalshTerm(
                character=character, coefficient=CanonicalRational.from_fraction(value)
            )
            for character, value in sorted(coefficients.items())
            if value
        ),
    )


def _run_pullback(request: WalshPullbackRequest) -> RationalWalshPolynomial:
    return affine_pullback(request.polynomial, request.affine_map)


WALSH_PULLBACK = MathTool(
    operation_id="boolean.walsh_polynomial.affine_pullback.compute",
    title="Pull back a sparse rational Walsh polynomial along an affine Boolean map",
    description="Compute f(Ay+b) in the rational Boolean character basis by mapping each character to A^T s, applying its phase (-1)^(s dot b), combining collisions and removing zero terms. Retain target ambient dimension without enumerating a truth table.",
    request_type=WalshPullbackRequest,
    result_type=RationalWalshPolynomial,
    run=_run_pullback,
    tags=("boolean", "walsh", "sparse", "affine", "pullback", "rational"),
    examples=(
        OperationExample(
            name="signed_identification",
            description="Identify two Boolean coordinates with opposite signs; map rows and offset have one entry per source coordinate.",
            input={
                "polynomial": {
                    "variable_count": 2,
                    "terms": [
                        {"character": [0, 1], "coefficient": {"num": "1", "den": "1"}}
                    ],
                },
                "affine_map": {
                    "target_dimension": 1,
                    "rows": [[0], [0]],
                    "offset": [0, 1],
                },
            },
        ),
    ),
)
