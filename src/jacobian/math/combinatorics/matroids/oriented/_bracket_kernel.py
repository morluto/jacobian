"""Exact bracket-algebra kernels: canonical atoms, relations, and residuals."""

from __future__ import annotations

from fractions import Fraction
from math import gcd

from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS, CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.matroids.oriented._bracket_models import (
    MAX_BRACKET_COEFFICIENT_DIGITS,
    MAX_BRACKET_CONTRIBUTIONS,
    MAX_BRACKET_FACTORS,
    MAX_BRACKET_OUTPUT_CELLS,
    MAX_BRACKET_RESULT_ALLOCATION_UNITS,
    MAX_BRACKET_TERMS,
    BracketMonomial,
    BracketPolynomial,
    BracketPolynomialTerm,
    BracketSyzygyResidualRequest,
    CanonicalBracket,
    GrassmannPlueckerRelation,
    GrassmannPlueckerRelationRequest,
    GrassmannPlueckerRelationResult,
    ordered_bracket,
)

__all__ = [
    "bracket_polynomial_from_terms",
    "bracket_syzygy_residual",
    "grassmann_pluecker_relation",
]

_RelationTerm = tuple[int, tuple[tuple[int, int, int], ...]]
_MonomialKey = tuple[tuple[tuple[int, int, int], int], ...]
_CoefficientComponent = tuple[Fraction, tuple[int, int]]

# Cache the canonical-integer envelope once. Constructing
# ``10 ** MAX_CANONICAL_INTEGER_DIGITS`` on every pair reduction recreates a
# 32,769-digit integer and dominates otherwise cheap residual admission.
_CANONICAL_INTEGER_LIMIT = 10**MAX_CANONICAL_INTEGER_DIGITS


def _combined_coefficients(
    contributions: list[tuple[Fraction, tuple[tuple[CanonicalBracket, int], ...]]],
) -> dict[_MonomialKey, Fraction]:
    """Combine signed bracket monomials into exact sparse coefficients."""

    combined: dict[_MonomialKey, Fraction] = {}
    for coefficient, brackets in contributions:
        if coefficient == 0:
            continue
        multiplicities: dict[tuple[int, int, int], int] = {}
        for bracket, multiplicity in brackets:
            multiplicities[bracket.indices] = (
                multiplicities.get(bracket.indices, 0) + multiplicity
            )
        canonical_key = tuple(sorted(multiplicities.items()))
        combined[canonical_key] = combined.get(canonical_key, Fraction(0)) + coefficient
    return {
        key: coefficient for key, coefficient in combined.items() if coefficient != 0
    }


def _polynomial_from_coefficients(
    coefficients: dict[_MonomialKey, Fraction], ground_size: int
) -> BracketPolynomial:
    """Build a canonical sparse polynomial from already grouped coefficients."""

    if len(coefficients) > MAX_BRACKET_TERMS:
        raise OperationResourceAdmissionError(
            location=("terms",),
            code="bracket.syzygy_output_term_bound",
            message="the formal residual has too many sparse output terms",
        )
    terms = []
    for monomial_key in sorted(coefficients):
        coefficient = coefficients[monomial_key]
        terms.append(
            BracketPolynomialTerm(
                coefficient=CanonicalRational(
                    num=coefficient.numerator, den=coefficient.denominator
                ),
                monomial=BracketMonomial(
                    factors=tuple(
                        (CanonicalBracket(indices=triples), multiplicity)
                        for triples, multiplicity in monomial_key
                    )
                ),
            )
        )
    return BracketPolynomial(ground_size=ground_size, terms=tuple(terms))


def _combine(
    contributions: list[tuple[Fraction, tuple[tuple[CanonicalBracket, int], ...]]],
    ground_size: int,
) -> BracketPolynomial:
    """Combine signed bracket monomials into a canonical sparse polynomial."""

    return _polynomial_from_coefficients(
        _combined_coefficients(contributions), ground_size
    )


def bracket_polynomial_from_terms(
    ground_size: int,
    terms: list[tuple[Fraction, tuple[tuple[int, int, int], ...]]],
) -> BracketPolynomial:
    """Build a canonical bracket polynomial from signed ordered triples."""

    contributions: list[tuple[Fraction, tuple[tuple[CanonicalBracket, int], ...]]] = []
    for coefficient, ordered_triples in terms:
        brackets: list[CanonicalBracket] = []
        sign = 1
        for ordered in ordered_triples:
            bracket, parity = ordered_bracket(ordered)
            brackets.append(bracket)
            sign *= parity
        contributions.append(
            (coefficient * sign, tuple((bracket, 1) for bracket in brackets))
        )
    return _combine(contributions, ground_size)


def _four_term_relation(indices: tuple[int, ...]) -> list[_RelationTerm]:
    """The four-term rank-3 Grassmann-Pluecker relation on six indices.

    ``[a b c][d e f] - [a b d][c e f] + [a b e][c d f] - [a b f][c d e] = 0``
    holds for every 3xN matrix; it is the Pluecker relation on the six columns.
    """

    a, b, c, d, e, f = indices
    return [
        (1, ((a, b, c), (d, e, f))),
        (-1, ((a, b, d), (c, e, f))),
        (1, ((a, b, e), (c, d, f))),
        (-1, ((a, b, f), (c, d, e))),
    ]


def _shared_index_three_term_relation(indices: tuple[int, ...]) -> list[_RelationTerm]:
    """The three-term Grassmann-Pluecker relation sharing one index.

    For five distinct indices ``(a,b,c,d,e)``, the bisyllabic relation
    ``[a b c][a d e] - [a b d][a c e] + [a b e][a c d] = 0`` holds for every
    3xN matrix: the shared index ``a`` makes it the Pluecker relation on the
    two-dimensional quotient by the column ``a``.
    """

    a, b, c, d, e = indices
    return [
        (1, ((a, b, c), (a, d, e))),
        (-1, ((a, b, d), (a, c, e))),
        (1, ((a, b, e), (a, c, d))),
    ]


def _relation_polynomial(
    request: GrassmannPlueckerRelationRequest,
) -> BracketPolynomial:
    """Build the one canonical polynomial for a validated relation request."""

    ordered_terms = (
        _four_term_relation(request.indices)
        if request.family == "FOUR_TERM"
        else _shared_index_three_term_relation(request.indices)
    )
    contributions: list[tuple[Fraction, tuple[tuple[CanonicalBracket, int], ...]]] = []
    for coefficient, ordered_triples in ordered_terms:
        brackets: list[CanonicalBracket] = []
        sign = coefficient
        for ordered in ordered_triples:
            try:
                bracket, parity = ordered_bracket(ordered)
            except PydanticCustomError as error:
                raise OperationDomainValidationError(
                    location=("indices",),
                    code="bracket.relation_degenerate_indices",
                    message=(
                        "a Grassmann-Pluecker relation requires distinct indices: "
                        f"{error.message()}"
                    ),
                ) from error
            brackets.append(bracket)
            sign *= parity
        contributions.append(
            (Fraction(sign), tuple((bracket, 1) for bracket in brackets))
        )
    return _combine(contributions, request.ground_size)


def _admit_source_relation(relation: GrassmannPlueckerRelation) -> None:
    """Admit the caller's source-bound relation claim once per request."""

    expected = _relation_polynomial(
        GrassmannPlueckerRelationRequest(
            ground_size=relation.ground_size,
            indices=relation.indices,
            family=relation.family,
        )
    )
    if relation.polynomial != expected:
        raise OperationDomainValidationError(
            location=("terms",),
            code="bracket.syzygy_source_relation_mismatch",
            message=(
                "the supplied polynomial does not match its Grassmann-Pluecker "
                "source metadata"
            ),
        )


def _integer_digit_upper_bound(value: int) -> int:
    """Return a cheap sound decimal-digit upper bound without formatting it."""

    if value == 0:
        return 1
    # log10(2) < 30103 / 100000, so this rounds upwards for every bit length.
    return min(
        (abs(value).bit_length() * 30103 + 99_999) // 100_000,
        MAX_BRACKET_COEFFICIENT_DIGITS,
    )


def _rational_component_digit_upper_bound(value: CanonicalRational) -> tuple[int, int]:
    return (
        _integer_digit_upper_bound(value.num),
        _integer_digit_upper_bound(value.den),
    )


def _rational_product_digit_upper_bound(
    left: CanonicalRational, right: CanonicalRational
) -> tuple[int, int]:
    """Bound component widths for a rational product without charging units."""

    left_num_digits, left_den_digits = _rational_component_digit_upper_bound(left)
    right_num_digits, right_den_digits = _rational_component_digit_upper_bound(right)
    if right.den == 1 and abs(right.num) == 1:
        return left_num_digits, left_den_digits
    if left.den == 1 and abs(left.num) == 1:
        return right_num_digits, right_den_digits
    return left_num_digits + right_num_digits, left_den_digits + right_den_digits


def _cancel_opposite_components(
    components: list[_CoefficientComponent],
) -> list[_CoefficientComponent]:
    """Remove equal and opposite exact coefficient components."""

    pending: dict[Fraction, list[tuple[int, int]]] = {}
    for value, widths in components:
        opposite = -value
        opposite_widths = pending.get(opposite)
        if opposite_widths:
            opposite_widths.pop()
            if not opposite_widths:
                del pending[opposite]
        else:
            pending.setdefault(value, []).append(widths)
    return [
        (value, widths)
        for value, component_widths in pending.items()
        for widths in component_widths
    ]


def _exceeds_canonical_integer_bound(value: int) -> bool:
    """Check a nonnegative output integer without formatting huge values."""

    if value.bit_length() <= 3 * MAX_CANONICAL_INTEGER_DIGITS:
        return False
    return bool(value >= _CANONICAL_INTEGER_LIMIT)


def _bounded_integer_sum(left: int, right: int) -> int | None:
    """Add integers only when their exact sum fits the canonical envelope."""

    if left == 0:
        return right
    if right == 0:
        return left
    if (left < 0) == (right < 0):
        if abs(left) >= _CANONICAL_INTEGER_LIMIT - abs(right):
            return None
    return left + right


def _integer_product_digit_upper_bound(left: int, right: int) -> int:
    """Bound a product's digits without charging multiplication by a unit."""

    left_digits = _integer_digit_upper_bound(left)
    right_digits = _integer_digit_upper_bound(right)
    if abs(left) == 1:
        return right_digits
    if abs(right) == 1:
        return left_digits
    return left_digits + right_digits


def _bounded_fraction_pair_sum(left: Fraction, right: Fraction) -> Fraction | None:
    """Return a bounded exact pair sum without oversized cross-products."""

    if left == -right:
        return Fraction(0)
    common_factor = gcd(left.denominator, right.denominator)
    left_scale = right.denominator // common_factor
    right_scale = left.denominator // common_factor
    lcm_digit_bound = _integer_product_digit_upper_bound(
        left.denominator // common_factor, right.denominator
    )
    if lcm_digit_bound > MAX_BRACKET_COEFFICIENT_DIGITS:
        return None
    if (
        _integer_product_digit_upper_bound(left.numerator, left_scale)
        > MAX_BRACKET_COEFFICIENT_DIGITS
        or _integer_product_digit_upper_bound(right.numerator, right_scale)
        > MAX_BRACKET_COEFFICIENT_DIGITS
    ):
        return None
    left_product = left.numerator * left_scale
    right_product = right.numerator * right_scale
    numerator = _bounded_integer_sum(left_product, right_product)
    if numerator is None:
        return None
    denominator = (left.denominator // common_factor) * right.denominator
    return Fraction(numerator, denominator)


def _bounded_component_sum(
    components: list[_CoefficientComponent],
) -> tuple[Fraction, int]:
    """Sum components using only representable exact intermediate values.

    The reduction order is part of admission: an arbitrary left-to-right sum
    can create a numerator or denominator wider than the canonical rational
    envelope even when a cancellation-first order is cheap and exact.
    """

    pending = _cancel_opposite_components(components)
    work_digit_bound = 0
    while pending:
        if len(pending) == 1:
            value, _ = pending[0]
            return value, max(
                work_digit_bound,
                _integer_digit_upper_bound(value.numerator),
                _integer_digit_upper_bound(value.denominator),
            )

        candidate: tuple[int, int, Fraction, int] | None = None
        for left_index, (left, _) in enumerate(pending):
            for right_index in range(left_index + 1, len(pending)):
                right, _ = pending[right_index]
                # An exact opposite was removed above and after every merge,
                # so this fast path mostly documents the cancellation-first
                # invariant for newly created values.
                if left == -right:
                    candidate = (left_index, right_index, Fraction(0), 0)
                    break
                merged: Fraction | None
                if left.denominator == right.denominator:
                    numerator = _bounded_integer_sum(left.numerator, right.numerator)
                    if numerator is None:
                        continue
                    merged = Fraction(numerator, left.denominator)
                else:
                    merged = _bounded_fraction_pair_sum(left, right)
                    if merged is None:
                        continue
                merged_width = (
                    0
                    if merged == 0
                    else max(
                        _integer_digit_upper_bound(merged.numerator),
                        _integer_digit_upper_bound(merged.denominator),
                    )
                )
                score = (merged_width, abs(merged.numerator).bit_length())
                if candidate is None or score < (
                    candidate[3],
                    abs(candidate[2].numerator).bit_length(),
                ):
                    candidate = (left_index, right_index, merged, merged_width)
            if candidate is not None and candidate[3] == 0:
                break

        if candidate is None:
            raise OperationResourceAdmissionError(
                location=("terms",),
                code="bracket.syzygy_coefficient_digit_bound",
                message="exact residual coefficient growth exceeds the supported digit bound",
            )
        left_index, right_index, merged, merged_width = candidate
        pending = [
            component
            for index, component in enumerate(pending)
            if index not in (left_index, right_index)
        ]
        if merged:
            pending.append(
                (
                    merged,
                    (
                        _integer_digit_upper_bound(merged.numerator),
                        _integer_digit_upper_bound(merged.denominator),
                    ),
                )
            )
            work_digit_bound = max(work_digit_bound, merged_width)
        pending = _cancel_opposite_components(pending)
    return Fraction(0), work_digit_bound


def _admit_result_allocation(
    coefficients: dict[_MonomialKey, Fraction],
    ground_size: int,
) -> None:
    """Admit retained scalar digits and structural slots for the residual."""

    index_digits = _integer_digit_upper_bound(ground_size - 1)
    term_cells = len(coefficients)
    factor_cells = sum(len(key) for key in coefficients)
    output_cells = term_cells + factor_cells
    allocation_units = 256 + sum(
        256
        + 2
        * max(
            _integer_digit_upper_bound(coefficient.numerator),
            _integer_digit_upper_bound(coefficient.denominator),
        )
        for coefficient in coefficients.values()
    )
    allocation_units += sum(
        128 + 3 * index_digits + _integer_digit_upper_bound(multiplicity)
        for key in coefficients
        for _, multiplicity in key
    )
    # Keep the cell count explicit: this prevents a future change to the
    # carrier shape from silently dropping a factor from the estimate.
    if output_cells > MAX_BRACKET_OUTPUT_CELLS:
        raise OperationResourceAdmissionError(
            location=("terms",),
            code="bracket.syzygy_output_cell_bound",
            message="the exact sparse residual output exceeds the supported cell bound",
        )
    if allocation_units > MAX_BRACKET_RESULT_ALLOCATION_UNITS:
        raise OperationResourceAdmissionError(
            location=("terms",),
            code="bracket.syzygy_result_allocation_bound",
            message="the exact residual exceeds its retained allocation bound",
        )


def _admit_surviving_components(
    coefficient_components: dict[_MonomialKey, list[_CoefficientComponent]],
) -> tuple[dict[_MonomialKey, Fraction], int]:
    """Return an admitted grouped coefficient plan and its width bound."""

    coefficients: dict[_MonomialKey, Fraction] = {}
    coefficient_digit_bound = 0
    for key, components in coefficient_components.items():
        total, local_digit_bound = _bounded_component_sum(components)
        if total == 0:
            coefficient_digit_bound = max(coefficient_digit_bound, local_digit_bound)
            continue

        if _exceeds_canonical_integer_bound(
            abs(total.numerator)
        ) or _exceeds_canonical_integer_bound(total.denominator):
            raise OperationResourceAdmissionError(
                location=("terms",),
                code="bracket.syzygy_coefficient_digit_bound",
                message="exact residual coefficient growth exceeds the supported digit bound",
            )
        coefficient_digit_bound = max(
            coefficient_digit_bound,
            local_digit_bound,
            _integer_digit_upper_bound(total.numerator),
            _integer_digit_upper_bound(total.denominator),
        )
        coefficients[key] = total
    if len(coefficients) > MAX_BRACKET_TERMS:
        raise OperationResourceAdmissionError(
            location=("terms",),
            code="bracket.syzygy_output_term_bound",
            message="the formal residual has too many sparse output terms",
        )
    return coefficients, coefficient_digit_bound


def _admit_residual_envelope(
    request: BracketSyzygyResidualRequest,
) -> dict[_MonomialKey, Fraction]:
    """Admit source claims, sparse output, and exact coefficient growth up front."""

    active_terms = tuple(
        (scalar, multiplier, relation)
        for scalar, multiplier, relation in request.terms
        if scalar.num != 0
    )
    contribution_count = len(request.target.terms) + sum(
        len(relation.polynomial.terms) for _, _, relation in active_terms
    )
    if contribution_count > MAX_BRACKET_CONTRIBUTIONS:
        raise OperationResourceAdmissionError(
            location=("terms",),
            code="bracket.syzygy_contribution_bound",
            message="the formal residual contribution count exceeds the bound",
        )

    for _, _, relation in active_terms:
        _admit_source_relation(relation)

    def admit_monomial(
        factors: tuple[tuple[CanonicalBracket, int], ...],
    ) -> _MonomialKey:
        multiplicities: dict[tuple[int, int, int], int] = {}
        for factor, multiplicity in factors:
            multiplicities[factor.indices] = (
                multiplicities.get(factor.indices, 0) + multiplicity
            )
        if any(
            _exceeds_canonical_integer_bound(value) for value in multiplicities.values()
        ):
            raise OperationResourceAdmissionError(
                location=("terms",),
                code="bracket.syzygy_multiplicity_digit_bound",
                message=(
                    "assembled bracket-factor multiplicity exceeds the canonical "
                    "integer representation envelope"
                ),
            )
        if len(multiplicities) > MAX_BRACKET_FACTORS:
            raise OperationDomainValidationError(
                location=("terms",),
                code="bracket.syzygy_monomial_factors",
                message="every assembled monomial must fit the bracket-factor bound",
            )
        return tuple(sorted(multiplicities.items()))

    coefficient_components: dict[_MonomialKey, list[_CoefficientComponent]] = {}
    for term in request.target.terms:
        key = admit_monomial(term.monomial.factors)
        coefficient_components.setdefault(key, []).append(
            (
                term.coefficient.as_fraction(),
                _rational_component_digit_upper_bound(term.coefficient),
            )
        )
    for scalar, multiplier, relation in active_terms:
        scalar_value = scalar.as_fraction()
        for term in relation.polynomial.terms:
            key = admit_monomial(multiplier.factors + term.monomial.factors)
            if term.coefficient.num == 1:
                coefficient = -scalar_value
            elif term.coefficient.num == -1:
                coefficient = scalar_value
            else:
                coefficient = -scalar_value * term.coefficient.as_fraction()
            coefficient_components.setdefault(key, []).append(
                (
                    coefficient,
                    _rational_product_digit_upper_bound(scalar, term.coefficient),
                )
            )

    if not coefficient_components:
        return {}
    coefficients, coefficient_digit_bound = _admit_surviving_components(
        coefficient_components
    )
    if coefficient_digit_bound > MAX_BRACKET_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("terms",),
            code="bracket.syzygy_coefficient_digit_bound",
            message="exact residual coefficient growth exceeds the supported digit bound",
        )
    _admit_result_allocation(coefficients, request.target.ground_size)
    return coefficients


def grassmann_pluecker_relation(
    request: GrassmannPlueckerRelationRequest,
) -> GrassmannPlueckerRelationResult:
    """Return the canonical formal expression of one GP relation."""

    polynomial = _relation_polynomial(request)
    return GrassmannPlueckerRelationResult(
        ground_size=request.ground_size,
        indices=request.indices,
        family=request.family,
        polynomial=polynomial,
    )


def bracket_syzygy_residual(request: BracketSyzygyResidualRequest) -> BracketPolynomial:
    """Return ``target - sum(scalar * multiplier * relation)`` exactly.

    The arithmetic is in the free commutative polynomial algebra on canonical
    bracket atoms.  Each supplied relation is admitted against its retained
    indexed source, but this operation does not infer that a nonzero residual
    vanishes on minors.
    """

    coefficients = _admit_residual_envelope(request)
    return _polynomial_from_coefficients(coefficients, request.target.ground_size)
