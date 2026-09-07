"""Exact bounded operations on rational polynomials on the unit circle."""

from __future__ import annotations

from fractions import Fraction
from math import lcm
from typing import Any

import sympy

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices._number_field import (
    EmbeddedNumberFieldRecognitionError,
    field_element_coordinates,
    field_element_from_value,
    field_element_sign,
    recognize_real_simple_number_field,
)
from jacobian.math.number_theory.algebraic_numbers.real import (
    RationalIsolatingInterval,
    RealAlgebraicValue,
)
from jacobian.math.number_theory.number_fields import embeddings
from jacobian.math.number_theory.number_fields._real_embedding_order import (
    NumberFieldRealEmbeddingOrderError,
    admit_real_embedding_difference,
    recognize_real_embedding_record,
)
from jacobian.math.number_theory.number_fields.values import (
    RealNumberFieldEmbedding,
    RealNumberFieldEmbeddingRecord,
    SimpleNumberFieldElement,
    SimpleNumberFieldPresentation,
    SimpleNumberFieldRealEmbeddingBinding,
)
from jacobian.math.polynomials.unit_circle._models import (
    MAX_ARC_ENERGY_CONDUCTOR,
    MAX_ARC_ENERGY_DEGREE,
    MAX_ARC_ENERGY_FIELD_COEFFICIENT_DIGITS,
    MAX_ARC_ENERGY_FIELD_DEGREE,
    MAX_ARC_ENERGY_INPUT_COMPONENT_DIGITS,
    MAX_ARC_ENERGY_TERMS,
    MAX_ARC_ENERGY_TOTAL_DENOMINATOR_DIGITS,
    MAX_FEJER_RIESZ_COMPONENT_DIGITS,
    MAX_FEJER_RIESZ_DERIVED_DIGITS,
    FejerRieszFactored,
    FejerRieszFactorResult,
    FejerRieszNegative,
    FejerRieszZero,
    HermitianLaurentPolynomial,
    RealDegreeOnePolynomialFactor,
    UnitCircleArcEnergyResult,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    require_polynomial_budget,
)

__all__ = [
    "real_symmetric_degree_one_fejer_riesz_factor",
    "unit_circle_arc_energy",
    "verify_real_symmetric_degree_one_fejer_riesz_factor",
    "verify_unit_circle_arc_energy",
]


def _fraction(value: CanonicalRational) -> Fraction:
    return value.as_fraction()


def _admit_polynomial(
    polynomial: RationalPolynomial,
) -> tuple[dict[int, Fraction], int]:
    if len(polynomial.variables) != 1:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.unit_circle.univariate_required",
            message="unit-circle operations require one polynomial variable",
        )
    require_polynomial_budget(
        polynomial,
        maximum_terms=MAX_ARC_ENERGY_TERMS,
        maximum_exponent=MAX_ARC_ENERGY_DEGREE,
        maximum_coefficient_digits=MAX_ARC_ENERGY_FIELD_COEFFICIENT_DIGITS,
        label="unit-circle polynomial",
    )
    coefficients = {
        term.exponents[0]: _fraction(term.coefficient)
        for term in polynomial.polynomial.terms
    }
    return coefficients, max(coefficients, default=0)


def _admit_arc(
    polynomial: RationalPolynomial,
    start_turn: CanonicalRational,
    end_turn: CanonicalRational,
) -> tuple[dict[int, Fraction], int, Fraction, int]:
    coefficients, degree = _admit_polynomial(polynomial)
    start = _fraction(start_turn)
    end = _fraction(end_turn)
    width = end - start
    if width < 0 or width > 1:
        raise OperationDomainValidationError(
            location=("end_turn",),
            code="polynomial.unit_circle.arc_width",
            message="oriented arc width must satisfy 0 <= end_turn-start_turn <= 1",
        )
    conductor = lcm(
        4,
        start_turn.den,
        end_turn.den,
    )
    if conductor > MAX_ARC_ENERGY_CONDUCTOR:
        raise OperationDomainValidationError(
            location=("start_turn", "end_turn"),
            code="polynomial.unit_circle.conductor_bound",
            message="rational turn denominators exceed the cyclotomic conductor bound",
        )
    component_digits = tuple(
        (
            len(format_canonical_integer(abs(term.coefficient.num))),
            len(format_canonical_integer(term.coefficient.den)),
        )
        for term in polynomial.polynomial.terms
    )
    if any(
        numerator_digits > MAX_ARC_ENERGY_INPUT_COMPONENT_DIGITS
        or denominator_digits > MAX_ARC_ENERGY_INPUT_COMPONENT_DIGITS
        for numerator_digits, denominator_digits in component_digits
    ):
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.unit_circle.coefficient_growth_bound",
            message="arc-energy coefficient components exceed the exact-growth bound",
        )
    if sum(denominator_digits for _, denominator_digits in component_digits) > (
        MAX_ARC_ENERGY_TOTAL_DENOMINATOR_DIGITS
    ):
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.unit_circle.denominator_growth_bound",
            message="arc-energy coefficient denominators exceed the exact-growth bound",
        )
    return coefficients, degree, width, conductor


# The admitted conductor is a multiple of four and at most 32.  These are the
# primitive minimal polynomials of alpha_N = 2*cos(2*pi/N), in descending
# order.  Fixing this table makes the standard complex embedding part of the
# value's identity and avoids expression-dependent minimal-polynomial search.
_REAL_CYCLOTOMIC_POLYNOMIALS: dict[int, tuple[int, ...]] = {
    4: (1, 0),
    8: (1, 0, -2),
    12: (1, 0, -3),
    16: (1, 0, -4, 0, 2),
    20: (1, 0, -5, 0, 5),
    24: (1, 0, -4, 0, 1),
    28: (1, 0, -7, 0, 14, 0, -7),
    32: (1, 0, -8, 0, 20, 0, -16, 0, 2),
}


def _real_cyclotomic_record(
    conductor: int,
) -> tuple[SimpleNumberFieldPresentation, RealNumberFieldEmbeddingRecord]:
    coefficients = _REAL_CYCLOTOMIC_POLYNOMIALS[conductor]
    presentation = SimpleNumberFieldPresentation(coefficients_descending=coefficients)
    degree = presentation.degree
    root = RealAlgebraicValue._from_admitted_polynomial(
        polynomial=coefficients,
        real_root_index=degree - 1,
    )
    embedding = RealNumberFieldEmbedding(
        kind="REAL", presentation=presentation, root=root
    )
    if conductor == 4:
        lower = upper = Fraction(0)
    elif conductor <= 12:
        lower, upper = Fraction(1), Fraction(2)
    elif conductor <= 24:
        lower, upper = Fraction(3, 2), Fraction(2)
    else:
        lower, upper = Fraction(7, 4), Fraction(2)
    record = RealNumberFieldEmbeddingRecord._from_kernel(
        embedding=embedding,
        isolating_interval=RationalIsolatingInterval(
            lower=CanonicalRational.from_fraction(lower),
            upper=CanonicalRational.from_fraction(upper),
            interval_type="SINGLETON" if lower == upper else "OPEN",
        ),
    )
    return presentation, record


def _coordinate_add(
    left: tuple[Fraction, ...], right: tuple[Fraction, ...]
) -> tuple[Fraction, ...]:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _coordinate_scale(
    value: tuple[Fraction, ...], scalar: Fraction
) -> tuple[Fraction, ...]:
    return tuple(scalar * coordinate for coordinate in value)


def _coordinate_times_generator(
    value: tuple[Fraction, ...], polynomial: tuple[int, ...]
) -> tuple[Fraction, ...]:
    shifted = [Fraction(0), *value]
    leading = shifted.pop()
    ascending_tail = tuple(int(coefficient) for coefficient in reversed(polynomial[1:]))
    for index, coefficient in enumerate(ascending_tail):
        shifted[index] -= leading * coefficient
    return tuple(shifted)


def _two_cosine_coordinates(exponent: int, conductor: int) -> tuple[Fraction, ...]:
    polynomial = _REAL_CYCLOTOMIC_POLYNOMIALS[conductor]
    degree = len(polynomial) - 1
    reduced = exponent % conductor
    reduced = min(reduced, conductor - reduced)
    previous = (Fraction(2),) + (Fraction(0),) * (degree - 1)
    if reduced == 0:
        return previous
    current = (
        ((Fraction(0), Fraction(1)) + (Fraction(0),) * (degree - 2))
        if degree > 1
        else (Fraction(0),)
    )
    for _ in range(2, reduced + 1):
        following = _coordinate_add(
            _coordinate_times_generator(current, polynomial),
            _coordinate_scale(previous, Fraction(-1)),
        )
        previous, current = current, following
    return current


def _sine_coordinates(
    turn: CanonicalRational, shift: int, conductor: int
) -> tuple[Fraction, ...]:
    denominator = turn.den
    # Only the endpoint phase modulo one enters the sine term.  Reduce before
    # multiplying so an unwrapped turn with a large integer part cannot make
    # recurrence work depend on its decimal height.
    phase = (turn.num % denominator) * (conductor // denominator)
    exponent = (shift * phase - conductor // 4) % conductor
    return _coordinate_scale(
        _two_cosine_coordinates(exponent, conductor), Fraction(1, 2)
    )


def _arc_pi_coordinates(
    correlations: dict[int, Fraction],
    start: CanonicalRational,
    end: CanonicalRational,
    conductor: int,
) -> tuple[Fraction, ...]:
    degree = len(_REAL_CYCLOTOMIC_POLYNOMIALS[conductor]) - 1
    result = (Fraction(0),) * degree
    for shift, correlation in correlations.items():
        if shift == 0 or not correlation:
            continue
        difference = _coordinate_add(
            _sine_coordinates(end, shift, conductor),
            _coordinate_scale(_sine_coordinates(start, shift, conductor), Fraction(-1)),
        )
        result = _coordinate_add(
            result, _coordinate_scale(difference, correlation / shift)
        )
    return result


def _cyclotomic_binding(
    coordinates: tuple[Fraction, ...], conductor: int
) -> SimpleNumberFieldRealEmbeddingBinding:
    presentation, record = _real_cyclotomic_record(conductor)
    if any(
        len(str(abs(component.numerator))) > MAX_ARC_ENERGY_FIELD_COEFFICIENT_DIGITS
        or len(str(component.denominator)) > MAX_ARC_ENERGY_FIELD_COEFFICIENT_DIGITS
        for component in coordinates
    ):
        raise RuntimeError(
            "arc-energy admission failed to bound the exact output coefficient"
        )
    return SimpleNumberFieldRealEmbeddingBinding(
        element=SimpleNumberFieldElement(
            presentation=presentation,
            coefficients_ascending=tuple(
                CanonicalRational.from_fraction(value) for value in coordinates
            ),
        ),
        embedding_record=record,
    )


def _minimal_polynomial(expr: Any) -> tuple[int, ...]:
    variable = sympy.Symbol("x")
    polynomial = sympy.Poly(
        sympy.minimal_polynomial(expr, variable), variable, domain=sympy.QQ
    )
    _denominator, integral = polynomial.clear_denoms(convert=True)
    _content, primitive = integral.primitive()
    if primitive.LC() < 0:
        primitive = -primitive
    coefficients = tuple(int(value) for value in primitive.all_coeffs())
    if len(coefficients) - 1 > MAX_ARC_ENERGY_FIELD_DEGREE:
        raise OperationDomainValidationError(
            location=("pi_inverse_coefficient",),
            code="polynomial.unit_circle.field_degree_bound",
            message="the real cyclotomic coefficient exceeds the field-degree bound",
        )
    if any(
        len(format_canonical_integer(abs(value)))
        > MAX_ARC_ENERGY_FIELD_COEFFICIENT_DIGITS
        for value in coefficients
    ):
        raise OperationDomainValidationError(
            location=("pi_inverse_coefficient",),
            code="polynomial.unit_circle.field_height_bound",
            message="the real cyclotomic coefficient field exceeds the height bound",
        )
    return coefficients


def _binding_in_field(
    expr: Any,
    alpha: Any,
    presentation: SimpleNumberFieldPresentation,
    record: RealNumberFieldEmbeddingRecord,
) -> SimpleNumberFieldRealEmbeddingBinding:
    if expr == 0:
        element_coordinates = (CanonicalRational(num=0, den=1),) * presentation.degree
    else:
        algebraic = sympy.to_number_field(expr, alpha)
        descending = tuple(algebraic.coeffs())
        ascending = tuple(reversed(descending)) + (sympy.Rational(0),) * (
            presentation.degree - len(descending)
        )
        element_coordinates = tuple(
            CanonicalRational.from_fraction(Fraction(int(value.p), int(value.q)))
            for value in ascending
        )
    return SimpleNumberFieldRealEmbeddingBinding(
        element=SimpleNumberFieldElement(
            presentation=presentation,
            coefficients_ascending=element_coordinates,
        ),
        embedding_record=record,
    )


def _correlations(
    coefficients: dict[int, Fraction], degree: int
) -> dict[int, Fraction]:
    return {
        shift: sum(
            (
                coefficients.get(index + shift, Fraction(0))
                * coefficients.get(index, Fraction(0))
                for index in range(degree - shift + 1)
            ),
            Fraction(0),
        )
        for shift in range(degree + 1)
    }


def unit_circle_arc_energy(
    polynomial: RationalPolynomial,
    start_turn: CanonicalRational,
    end_turn: CanonicalRational,
) -> UnitCircleArcEnergyResult:
    """Return ``integral |P(exp(2*pi*i*t))|^2 dt`` on an oriented arc."""
    coefficients, degree, width, conductor = _admit_arc(
        polynomial, start_turn, end_turn
    )
    correlations = _correlations(coefficients, degree)
    rational_part = CanonicalRational.from_fraction(width * correlations[0])
    pi_coordinates = _arc_pi_coordinates(correlations, start_turn, end_turn, conductor)
    return UnitCircleArcEnergyResult(
        polynomial=polynomial,
        start_turn=start_turn,
        end_turn=end_turn,
        cyclotomic_conductor=conductor,
        rational_part=rational_part,
        pi_inverse_coefficient=_cyclotomic_binding(pi_coordinates, conductor),
    )


def verify_unit_circle_arc_energy(claim: UnitCircleArcEnergyResult) -> bool:
    """Verify the exact arc-energy identity against its retained source."""
    try:
        coefficients, _degree, width, conductor = _admit_arc(
            claim.polynomial, claim.start_turn, claim.end_turn
        )
        presentation, record = _real_cyclotomic_record(conductor)
        binding = claim.pi_inverse_coefficient
        if (
            claim.cyclotomic_conductor != conductor
            or binding.element.presentation != presentation
            or binding.embedding_record != record
            or claim.rational_part.as_fraction()
            != width * sum(value * value for value in coefficients.values())
        ):
            return False
        # Verify from the defining double sum rather than replaying the
        # producer's correlation reduction.
        expected = (Fraction(0),) * presentation.degree
        for left_index, left in coefficients.items():
            for right_index, right in coefficients.items():
                shift = left_index - right_index
                if shift <= 0:
                    continue
                difference = _coordinate_add(
                    _sine_coordinates(claim.end_turn, shift, conductor),
                    _coordinate_scale(
                        _sine_coordinates(claim.start_turn, shift, conductor),
                        Fraction(-1),
                    ),
                )
                expected = _coordinate_add(
                    expected,
                    _coordinate_scale(difference, left * right / shift),
                )
        actual = tuple(
            value.as_fraction() for value in binding.element.coefficients_ascending
        )
        return actual == expected
    except (
        AttributeError,
        TypeError,
        OperationDomainValidationError,
        ValueError,
        RuntimeError,
    ):
        return False


def _laurent_coefficients(source: HermitianLaurentPolynomial) -> dict[int, Fraction]:
    for term in source.terms:
        if (
            len(format_canonical_integer(abs(term.coefficient.num)))
            > MAX_FEJER_RIESZ_COMPONENT_DIGITS
            or len(format_canonical_integer(term.coefficient.den))
            > MAX_FEJER_RIESZ_COMPONENT_DIGITS
        ):
            raise OperationResourceAdmissionError(
                location=("source", "terms"),
                code="polynomial.unit_circle.fejer_riesz_component_bound",
                message=(
                    "Fejer-Riesz rational components exceed the "
                    f"{MAX_FEJER_RIESZ_COMPONENT_DIGITS}-digit exact-growth bound"
                ),
            )
    values = {term.exponent: _fraction(term.coefficient) for term in source.terms}
    if any(
        values.get(-exponent, Fraction(0)) != coefficient
        for exponent, coefficient in values.items()
    ):
        raise OperationDomainValidationError(
            location=("source", "terms"),
            code="polynomial.unit_circle.hermitian",
            message="Laurent coefficients must satisfy Hermitian symmetry",
        )
    denominator = lcm(*(value.denominator for value in values.values()), 1)
    lifted = tuple(
        value.numerator * (denominator // value.denominator)
        for value in values.values()
    )
    derived = (
        denominator * denominator,
        *(left * right for left in lifted for right in lifted),
    )
    if any(len(str(abs(value))) > MAX_FEJER_RIESZ_DERIVED_DIGITS for value in derived):
        raise OperationResourceAdmissionError(
            location=("source", "terms"),
            code="polynomial.unit_circle.fejer_riesz_growth_bound",
            message="Fejer-Riesz exact intermediate height exceeds its bound",
        )
    return values


def _largest_real_embedding_record(
    presentation: SimpleNumberFieldPresentation,
) -> RealNumberFieldEmbeddingRecord:
    real_records = tuple(
        record
        for record in embeddings(presentation).records
        if isinstance(record, RealNumberFieldEmbeddingRecord)
    )
    if not real_records:
        raise RuntimeError("a positive real coefficient lost its exact embedding")
    return max(real_records, key=lambda record: record.embedding.root.real_root_index)


def real_symmetric_degree_one_fejer_riesz_factor(
    source: HermitianLaurentPolynomial,
) -> FejerRieszFactorResult:
    """Decide and, when it exists, return the normalized degree-one factor."""
    coefficients = _laurent_coefficients(source)
    c0 = coefficients.get(0, Fraction(0))
    c1 = coefficients.get(1, Fraction(0))
    if not coefficients or (c0 == 0 and c1 == 0):
        return FejerRieszFactorResult(
            source=source,
            conclusion=FejerRieszZero(),
        )
    if c0 < 0 or c0 * c0 < 4 * c1 * c1:
        witness = Fraction(0) if c1 == 0 else Fraction(-1 if c1 > 0 else 1)
        return FejerRieszFactorResult(
            source=source,
            conclusion=FejerRieszNegative(
                cosine_witness=CanonicalRational.from_fraction(witness)
            ),
        )
    radical = (
        sympy.Rational(c0.numerator, c0.denominator) ** 2
        - 4 * sympy.Rational(c1.numerator, c1.denominator) ** 2
    )
    a = sympy.sqrt(
        (sympy.Rational(c0.numerator, c0.denominator) + sympy.sqrt(radical)) / 2
    )
    b = sympy.Rational(c1.numerator, c1.denominator) / a
    alpha = a
    presentation = SimpleNumberFieldPresentation(
        coefficients_descending=_minimal_polynomial(alpha)
    )
    record = _largest_real_embedding_record(presentation)
    factors = (
        _binding_in_field(alpha, alpha, presentation, record),
        _binding_in_field(b, alpha, presentation, record),
    )
    return FejerRieszFactorResult(
        source=source,
        conclusion=FejerRieszFactored(
            factor=RealDegreeOnePolynomialFactor(
                embedding_record=record,
                coefficients_ascending=(factors[0].element, factors[1].element),
            )
        ),
    )


def verify_real_symmetric_degree_one_fejer_riesz_factor(
    claim: FejerRieszFactorResult,
) -> bool:
    """Independently verify the conclusion against its retained Laurent source."""
    try:
        coefficients = _laurent_coefficients(claim.source)
        c0 = coefficients.get(0, Fraction(0))
        c1 = coefficients.get(1, Fraction(0))
        conclusion = claim.conclusion
        if isinstance(conclusion, FejerRieszZero):
            return not coefficients
        if isinstance(conclusion, FejerRieszNegative):
            witness = conclusion.cosine_witness.as_fraction()
            return -1 <= witness <= 1 and c0 + 2 * c1 * witness < 0
        factor = conclusion.factor
        record = factor.embedding_record
        if record.embedding.presentation.degree > 4:
            return False
        for coefficient in factor.coefficients_ascending:
            admit_real_embedding_difference(
                coefficient.presentation,
                tuple(
                    coordinate.as_fraction()
                    for coordinate in coefficient.coefficients_ascending
                ),
            )
        recognize_real_embedding_record(record)
        recognized = recognize_real_simple_number_field(record.embedding)
        q0 = field_element_from_value(factor.coefficients_ascending[0], recognized)
        q1 = field_element_from_value(factor.coefficients_ascending[1], recognized)
        rational_c0 = recognized.field.convert(sympy.QQ(c0.numerator, c0.denominator))
        rational_c1 = recognized.field.convert(sympy.QQ(c1.numerator, c1.denominator))
        outer_difference = q0 * q0 - q1 * q1
        admit_real_embedding_difference(
            record.embedding.presentation,
            field_element_coordinates(outer_difference, recognized),
        )
        return (
            q0 * q0 + q1 * q1 == rational_c0
            and q0 * q1 == rational_c1
            and field_element_sign(q0, recognized) > 0
            and field_element_sign(outer_difference, recognized) >= 0
        )
    except (
        AttributeError,
        EmbeddedNumberFieldRecognitionError,
        NumberFieldRealEmbeddingOrderError,
        OperationDomainValidationError,
        TypeError,
        ValueError,
        RuntimeError,
    ):
        return False
