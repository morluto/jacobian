"""Exact finite function-field operations over GF(p)(x)[y]/(f)."""

from __future__ import annotations

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.function_fields._gfpx import (
    RF,
    ZERO_RF,
    KPoly,
    is_irreducible_over_gf,
    is_irreducible_over_rational_function,
    kp_derivative,
    kp_divmod,
    kp_gcd,
    kp_normalize,
    kp_xgcd,
    rf_add,
    rf_evaluate,
    rf_is_zero,
    rf_mul,
    rf_normalize,
    rf_sub,
)
from jacobian.math.function_fields._models import (
    MAX_EXTENSION_DEGREE,
    MAX_FIELD_ADMISSION_WORK,
    MAX_MULTIPLICATION_WORK,
    MAX_POLYNOMIAL_X_DEGREE,
    FiniteFunctionField,
    FiniteFunctionFieldElement,
    FunctionFieldElementMultiplyResult,
    FunctionFieldProductTerm,
    FunctionFieldReductionStep,
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 1
    return True


def _to_internal_polynomial(polynomial: PrimeFieldPolynomial) -> tuple[int, ...]:
    coefficients = polynomial.coefficients
    if coefficients == (0,):
        return ()
    return coefficients


def _to_internal_rational_function(value: PrimeFieldRationalFunction) -> RF:
    prime = value.characteristic
    return rf_normalize(
        _to_internal_polynomial(value.numerator),
        _to_internal_polynomial(value.denominator),
        prime,
    )


def _from_internal_polynomial(
    coefficients: tuple[int, ...], prime: int
) -> PrimeFieldPolynomial:
    return PrimeFieldPolynomial(
        characteristic=prime,
        coefficients=coefficients if coefficients else (0,),
    )


def _from_internal_rational_function(
    value: RF, prime: int
) -> PrimeFieldRationalFunction:
    return PrimeFieldRationalFunction(
        numerator=_from_internal_polynomial(value[0], prime),
        denominator=_from_internal_polynomial(value[1], prime),
    )


def _canonical_field(field: FiniteFunctionField) -> FiniteFunctionField:
    prime = field.characteristic
    return FiniteFunctionField.model_construct(
        characteristic=prime,
        variable=field.variable,
        generator=field.generator,
        defining_polynomial=tuple(
            _from_internal_rational_function(
                _to_internal_rational_function(coefficient), prime
            )
            for coefficient in field.defining_polynomial
        ),
    )


def _canonical_element(
    element: FiniteFunctionFieldElement, field: FiniteFunctionField
) -> FiniteFunctionFieldElement:
    prime = field.characteristic
    return FiniteFunctionFieldElement.model_construct(
        field=field,
        coordinates=tuple(
            _from_internal_rational_function(
                _to_internal_rational_function(coordinate), prime
            )
            for coordinate in element.coordinates
        ),
    )


def _field_kpoly(field: FiniteFunctionField) -> KPoly:
    return tuple(
        _to_internal_rational_function(coefficient)
        for coefficient in field.defining_polynomial
    )


def _admit_irreducibility(field: FiniteFunctionField, kpoly: KPoly) -> bool:
    """Admit irreducibility by specialization, then exact Gauss-lemma factoring.

    A monic factorization over GF(p)(x) specializes at a point with nonvanishing
    denominators to a monic factorization over GF(p); therefore an irreducible
    specialization proves irreducibility.  If all bounded specializations are
    reducible, clear rational-function denominators, remove the GF(p)[x]
    content, and factor the primitive lift exactly over GF(p)[x,y].
    """

    prime = field.characteristic
    degree_bound = len(kpoly) - 1
    for point in range(prime):
        specialization: list[int] = []
        admissible = True
        for coefficient in kpoly:
            evaluated = rf_evaluate(coefficient, point, prime)
            if evaluated is None:
                admissible = False
                break
            specialization.append(evaluated)
        if not admissible or len(specialization) - 1 != degree_bound:
            continue
        if is_irreducible_over_gf(tuple(specialization), prime):
            return True
    return is_irreducible_over_rational_function(kpoly, prime)


def _admit_field(field: FiniteFunctionField) -> None:
    prime = field.characteristic
    if not _is_prime(prime):
        raise OperationDomainValidationError(
            location=("field", "characteristic"),
            code="function_field.characteristic_not_prime",
            message="the constant field characteristic must be prime",
        )
    degree = field.degree
    if not 1 <= degree <= MAX_EXTENSION_DEGREE:
        raise OperationResourceAdmissionError(
            location=("field", "defining_polynomial"),
            code="function_field.extension_degree_exceeds_envelope",
            message=(
                f"function fields admit extension degree at most {MAX_EXTENSION_DEGREE}"
            ),
        )
    degree_work = 0
    for coefficient in field.defining_polynomial:
        for polynomial in (coefficient.numerator, coefficient.denominator):
            if polynomial.degree > MAX_POLYNOMIAL_X_DEGREE:
                raise OperationResourceAdmissionError(
                    location=("field", "defining_polynomial"),
                    code="function_field.coefficient_degree_exceeds_envelope",
                    message=(
                        "defining-polynomial coefficient degree exceeds the "
                        f"{MAX_POLYNOMIAL_X_DEGREE} envelope"
                    ),
                )
            degree_work += polynomial.degree + 1
    if degree_work * degree * prime > MAX_FIELD_ADMISSION_WORK:
        raise OperationResourceAdmissionError(
            location=("field", "defining_polynomial"),
            code="function_field.admission_work_exceeds_envelope",
            message=(
                "function-field admission work exceeds the "
                f"{MAX_FIELD_ADMISSION_WORK} unit envelope"
            ),
        )
    canonical = _canonical_field(field)
    kpoly = _field_kpoly(canonical)
    derivative = kp_derivative(kpoly, prime)
    if len(kpoly) <= 1 or len(kp_gcd(kpoly, derivative, prime)) > 1:
        raise OperationDomainValidationError(
            location=("field", "defining_polynomial"),
            code="function_field.extension_not_separable",
            message="the defining polynomial must be separable over GF(p)(x)",
        )
    if not _admit_irreducibility(canonical, kpoly):
        raise OperationDomainValidationError(
            location=("field", "defining_polynomial"),
            code="function_field.extension_not_admitted_irreducible",
            message=("the defining polynomial is reducible over GF(p)(x)"),
        )


def _admit_elements(
    left: FiniteFunctionFieldElement, right: FiniteFunctionFieldElement
) -> tuple[FiniteFunctionField, FiniteFunctionFieldElement, FiniteFunctionFieldElement]:
    left_field = _canonical_field(left.field)
    right_field = _canonical_field(right.field)
    if left_field != right_field:
        raise OperationDomainValidationError(
            location=("right", "field"),
            code="function_field.element_field_mismatch",
            message="both elements must be bound to the identical function field",
        )
    _admit_field(left_field)
    canonical_left = _canonical_element(left, left_field)
    canonical_right = _canonical_element(right, left_field)
    max_terms = 1
    total_degree = 0
    max_numerator_degree = 0
    max_denominator_degree = 0
    for element in (canonical_left, canonical_right):
        for coordinate in element.coordinates:
            max_numerator_degree = max(
                max_numerator_degree, coordinate.numerator.degree
            )
            max_denominator_degree = max(
                max_denominator_degree, coordinate.denominator.degree
            )
            for polynomial in (coordinate.numerator, coordinate.denominator):
                max_terms = max(max_terms, len(polynomial.coefficients))
                total_degree += polynomial.degree
    field_coefficient_degree = max(
        coefficient.numerator.degree + coefficient.denominator.degree
        for coefficient in left_field.defining_polynomial
    )
    # A single convolution plus one reduction step can at most square the input
    # denominator degree and add the defining-polynomial coefficient degree.
    growth_bound = (
        2 * (max_numerator_degree + max_denominator_degree) + field_coefficient_degree
    )
    if growth_bound > MAX_POLYNOMIAL_X_DEGREE:
        raise OperationResourceAdmissionError(
            location=("left", "coordinates"),
            code="function_field.coefficient_growth_exceeds_envelope",
            message=(
                "the reduced product can exceed the "
                f"{MAX_POLYNOMIAL_X_DEGREE}-degree coefficient envelope"
            ),
        )
    work = (
        (2 * left_field.degree - 1)
        * max_terms
        * max(1, total_degree + left_field.degree)
    )
    if work > MAX_MULTIPLICATION_WORK:
        raise OperationResourceAdmissionError(
            location=("left", "coordinates"),
            code="function_field.multiplication_work_exceeds_envelope",
            message=(
                "function-field multiplication work exceeds the "
                f"{MAX_MULTIPLICATION_WORK} unit envelope"
            ),
        )
    return left_field, canonical_left, canonical_right


def _internal_coordinates(
    element: FiniteFunctionFieldElement,
) -> tuple[RF, ...]:
    return tuple(
        _to_internal_rational_function(coordinate) for coordinate in element.coordinates
    )


def _multiply_internal(
    left: tuple[RF, ...],
    right: tuple[RF, ...],
    kpoly: KPoly,
    prime: int,
) -> tuple[
    tuple[RF, ...],
    tuple[tuple[int, RF], ...],
    tuple[tuple[int, RF, tuple[tuple[int, RF], ...]], ...],
]:
    degree = len(kpoly) - 1
    raw: list[RF] = [ZERO_RF] * (2 * degree - 1)
    for left_index, left_value in enumerate(left):
        if rf_is_zero(left_value):
            continue
        for right_index, right_value in enumerate(right):
            if rf_is_zero(right_value):
                continue
            position = left_index + right_index
            raw[position] = rf_add(
                raw[position], rf_mul(left_value, right_value, prime), prime
            )
    raw_terms = tuple(
        (position, raw[position])
        for position in range(len(raw))
        if not rf_is_zero(raw[position])
    )
    reduction_steps: list[tuple[int, RF, tuple[tuple[int, RF], ...]]] = []
    for position in range(2 * degree - 2, degree - 1, -1):
        coefficient = raw[position]
        if rf_is_zero(coefficient):
            continue
        raw[position] = ZERO_RF
        replacement: list[tuple[int, RF]] = []
        for index in range(degree):
            field_coefficient = kpoly[index]
            if rf_is_zero(field_coefficient):
                continue
            contribution = rf_mul(coefficient, field_coefficient, prime)
            target = position - degree + index
            raw[target] = rf_sub(raw[target], contribution, prime)
            replacement.append((target, contribution))
        reduction_steps.append((position, coefficient, tuple(replacement)))
    reduced = tuple(raw[index] for index in range(degree))
    reduction_steps.sort(key=lambda step: step[0])
    return reduced, raw_terms, tuple(reduction_steps)


def function_field_element_multiply(
    left: FiniteFunctionFieldElement,
    right: FiniteFunctionFieldElement,
) -> FunctionFieldElementMultiplyResult:
    """Exact product in GF(p)(x)[y]/(f) with a complete reduction ledger."""

    field, left, right = _admit_elements(left, right)
    prime = field.characteristic
    kpoly = _field_kpoly(field)
    reduced, raw_terms, reduction_steps = _multiply_internal(
        _internal_coordinates(left),
        _internal_coordinates(right),
        kpoly,
        prime,
    )
    product = FiniteFunctionFieldElement.model_construct(
        field=field,
        coordinates=tuple(
            _from_internal_rational_function(value, prime) for value in reduced
        ),
    )
    return FunctionFieldElementMultiplyResult._from_kernel(
        field=field,
        left=left,
        right=right,
        raw_product_terms=tuple(
            FunctionFieldProductTerm(
                y_power=position,
                coefficient=_from_internal_rational_function(value, prime),
            )
            for position, value in raw_terms
        ),
        reduction_steps=tuple(
            FunctionFieldReductionStep(
                y_power=position,
                coefficient=_from_internal_rational_function(coefficient, prime),
                replacement_terms=tuple(
                    FunctionFieldProductTerm(
                        y_power=target,
                        coefficient=_from_internal_rational_function(value, prime),
                    )
                    for target, value in replacement
                ),
            )
            for position, coefficient, replacement in reduction_steps
        ),
        product=product,
    )


def _element_inverse(
    element: FiniteFunctionFieldElement,
) -> FiniteFunctionFieldElement:
    """Private exact inverse via extended Euclid in GF(p)(x)[y]/(f)."""

    field, canonical, _ = _admit_elements(element, element)
    prime = field.characteristic
    kpoly = _field_kpoly(field)
    value = _internal_coordinates(canonical)
    if all(rf_is_zero(coordinate) for coordinate in value):
        raise OperationDomainValidationError(
            location=("element",),
            code="function_field.zero_not_invertible",
            message="the zero element has no multiplicative inverse",
        )
    _, inverse, _ = kp_xgcd(kp_normalize(value), kpoly, prime)
    _, remainder = kp_divmod(inverse, kpoly, prime)
    degree = field.degree
    padded = tuple(
        remainder[index] if index < len(remainder) else ZERO_RF
        for index in range(degree)
    )
    return FiniteFunctionFieldElement.model_construct(
        field=field,
        coordinates=tuple(
            _from_internal_rational_function(coordinate, prime) for coordinate in padded
        ),
    )


__all__ = [
    "_element_inverse",
    "function_field_element_multiply",
]
