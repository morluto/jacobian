"""Derived admission for exact rational coordinate Lie derivatives."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import gcd, lcm
from typing import Literal, NoReturn

from jacobian._models import StrictModel
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    strict_json_object_size,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.differential._execution import (
    begin_lie_derivative_deadline,
    require_lie_derivative_deadline,
)
from jacobian.math.geometry.differential._recognition_process import (
    RationalFunctionRecognitionCandidate,
    canonical_recognition_candidates,
    recognize_canonical_rational_functions,
)
from jacobian.math.geometry.differential.values import (
    MAX_RATIONAL_TENSOR_COEFFICIENT_DIGITS,
    MAX_RATIONAL_TENSOR_EXPONENT,
    MAX_RATIONAL_TENSOR_LOCUS_GUARDS,
    MAX_RATIONAL_TENSOR_POLYNOMIAL_TERMS,
    RationalCoordinateTensor,
    canonical_locus_guards,
)
from jacobian.math.polynomials.values import (
    RationalFunction,
    SparseRationalPolynomial,
)

MAX_LIE_DERIVATIVE_WORK_UNITS = 25_000_000
MAX_LIE_DERIVATIVE_RESULT_BYTES = CanonicalLimits().max_output_bytes
MAX_LIE_DERIVATIVE_RAW_POLYNOMIAL_TERMS = 4_096
MAX_LIE_DERIVATIVE_RAW_COEFFICIENT_DIGITS = 4_096

type LieWorkCategory = Literal[
    "recognition",
    "source_conversion",
    "differentiation",
    "multiplication",
    "addition",
    "normalization",
]
_LIE_WORK_CATEGORIES: tuple[LieWorkCategory, ...] = (
    "recognition",
    "source_conversion",
    "differentiation",
    "multiplication",
    "addition",
    "normalization",
)


@dataclass(frozen=True)
class PolynomialBound:
    """Bound ``rational_content * integral_polynomial`` exactly.

    Input bounds start with a primitive integral polynomial. Arithmetic may
    leave its integral factor nonprimitive, while ``coefficient_digits`` still
    bounds every coefficient and ``rational_content`` remains an exact common
    scale. Keeping that scale separate prevents unrelated coefficient
    denominators from multiplying during height admission.
    """

    terms: int
    degrees: tuple[int, ...]
    minimum_exponents: tuple[int, ...]
    coefficient_digits: int
    rational_content: Fraction

    @property
    def is_zero(self) -> bool:
        return self.terms == 0


@dataclass(frozen=True)
class FractionBound:
    numerator: PolynomialBound
    denominator: PolynomialBound

    @property
    def is_zero(self) -> bool:
        return self.numerator.is_zero


@dataclass(frozen=True)
class FactorReference:
    owner: Literal["VECTOR", "TENSOR"]
    component: int
    derivative_axis: int | None


@dataclass(frozen=True)
class LieProductTerm:
    sign: Literal[-1, 1]
    left: FactorReference
    right: FactorReference


@dataclass(frozen=True)
class LieComponentPlan:
    terms: tuple[LieProductTerm, ...]
    raw_result: FractionBound
    canonical_coefficient_digits: int


@dataclass(frozen=True)
class LieDerivativePlan:
    components: tuple[LieComponentPlan, ...]
    recognition_candidates: tuple[RationalFunctionRecognitionCandidate, ...]
    inherited_locus_guards: tuple[SparseRationalPolynomial, ...]
    result_bytes_upper_bound: int
    work_units_by_category: tuple[tuple[LieWorkCategory, int], ...]

    @property
    def work_units(self) -> int:
        return sum(amount for _, amount in self.work_units_by_category)


class _Ledger:
    def __init__(self, *, deadline: float) -> None:
        self.deadline = deadline
        self.work_units = 0
        self._by_category: dict[LieWorkCategory, int] = dict.fromkeys(
            _LIE_WORK_CATEGORIES, 0
        )

    def charge(self, category: LieWorkCategory, amount: int) -> None:
        if amount < 0:
            raise AssertionError("Lie-derivative work charges must be nonnegative")
        require_lie_derivative_deadline(
            self.deadline, f"while charging {category} work"
        )
        self.work_units += amount
        self._by_category[category] += amount
        if self.work_units > MAX_LIE_DERIVATIVE_WORK_UNITS:
            _reject(
                "work_budget",
                "Lie-derivative exact arithmetic exceeds the "
                f"{MAX_LIE_DERIVATIVE_WORK_UNITS}-unit work budget",
            )

    @property
    def by_category(self) -> tuple[tuple[LieWorkCategory, int], ...]:
        return tuple(
            (category, self._by_category[category]) for category in _LIE_WORK_CATEGORIES
        )


def _reject(
    reason: str,
    message: str,
    *,
    location: tuple[str | int, ...] = (),
) -> NoReturn:
    raise OperationDomainValidationError(
        location=location,
        code=f"differential_geometry.lie_derivative.{reason}",
        message=message,
    )


def _dense_term_bound(degrees: tuple[int, ...], *, cap: int | None = None) -> int:
    result = 1
    for degree in degrees:
        result *= degree + 1
        if cap is not None and result > cap:
            return cap + 1
    return result


def _polynomial_admission_work_units(
    polynomial: SparseRationalPolynomial, variable_count: int
) -> int:
    """Price the scalar passes used to derive one exact polynomial bound.

    Each nonzero coefficient is converted, participates in denominator/content
    reduction, is scaled and made primitive, and is inspected for height. Each
    exponent is inspected once for the maximum and once for the minimum. The
    final unit prices construction of the separated rational content. Zero is
    represented by one exact ``Fraction`` construction.
    """

    terms = len(polynomial.terms)
    if terms == 0:
        return 1
    return terms * (6 + 2 * variable_count) + 1


def _polynomial_backend_conversion_work_units(bound: PolynomialBound) -> int:
    """Price sparse coefficient conversion and dense SymPy ``Poly`` creation."""

    dense_coefficients = 1 if bound.is_zero else _dense_term_bound(bound.degrees)
    coefficient_digits = (
        bound.coefficient_digits
        + len(str(abs(bound.rational_content.numerator)))
        + len(str(bound.rational_content.denominator))
    )
    coefficient_chunks = max(1, (coefficient_digits + 31) // 32)
    return bound.terms * coefficient_chunks + dense_coefficients


def _polynomial_bound(polynomial: SparseRationalPolynomial) -> PolynomialBound:
    if not polynomial.terms:
        return PolynomialBound(
            terms=0,
            degrees=(),
            minimum_exponents=(),
            coefficient_digits=1,
            rational_content=Fraction(0),
        )
    variable_count = len(polynomial.terms[0].exponents)
    coefficients = tuple(
        Fraction(*term.coefficient.as_integer_ratio()) for term in polynomial.terms
    )
    common_denominator = lcm(*(coefficient.denominator for coefficient in coefficients))
    integral_coefficients = tuple(
        coefficient.numerator * (common_denominator // coefficient.denominator)
        for coefficient in coefficients
    )
    content_numerator = gcd(
        *(abs(coefficient) for coefficient in integral_coefficients)
    )
    primitive_coefficients = tuple(
        coefficient // content_numerator for coefficient in integral_coefficients
    )
    return PolynomialBound(
        terms=len(polynomial.terms),
        degrees=tuple(
            max(term.exponents[axis] for term in polynomial.terms)
            for axis in range(variable_count)
        ),
        minimum_exponents=tuple(
            min(term.exponents[axis] for term in polynomial.terms)
            for axis in range(variable_count)
        ),
        coefficient_digits=max(
            len(str(abs(coefficient))) for coefficient in primitive_coefficients
        ),
        rational_content=Fraction(content_numerator, common_denominator),
    )


def _zero_polynomial(variable_count: int) -> PolynomialBound:
    return PolynomialBound(
        terms=0,
        degrees=(0,) * variable_count,
        minimum_exponents=(0,) * variable_count,
        coefficient_digits=1,
        rational_content=Fraction(0),
    )


def _one_polynomial(variable_count: int) -> PolynomialBound:
    return PolynomialBound(
        terms=1,
        degrees=(0,) * variable_count,
        minimum_exponents=(0,) * variable_count,
        coefficient_digits=1,
        rational_content=Fraction(1),
    )


def _zero_fraction(variable_count: int) -> FractionBound:
    return FractionBound(
        numerator=_zero_polynomial(variable_count),
        denominator=_one_polynomial(variable_count),
    )


def _fraction_bound(value: RationalFunction, ledger: _Ledger) -> FractionBound:
    variable_count = len(value.variables)
    ledger.charge(
        "source_conversion",
        _polynomial_admission_work_units(value.numerator, variable_count)
        + _polynomial_admission_work_units(value.denominator, variable_count),
    )
    numerator = (
        _zero_polynomial(variable_count)
        if not value.numerator.terms
        else _polynomial_bound(value.numerator)
    )
    result = FractionBound(
        numerator=numerator,
        denominator=_polynomial_bound(value.denominator),
    )
    ledger.charge(
        "source_conversion",
        _polynomial_backend_conversion_work_units(result.numerator)
        + _polynomial_backend_conversion_work_units(result.denominator),
    )
    return result


def _check_raw_polynomial(bound: PolynomialBound) -> None:
    if bound.terms > MAX_LIE_DERIVATIVE_RAW_POLYNOMIAL_TERMS:
        _reject(
            "intermediate_support",
            "Lie-derivative polynomial expansion exceeds the "
            f"{MAX_LIE_DERIVATIVE_RAW_POLYNOMIAL_TERMS}-term intermediate budget",
        )
    scaled_coefficient_digits = max(
        len(str(abs(bound.rational_content.numerator))) + bound.coefficient_digits,
        len(str(bound.rational_content.denominator)),
    )
    if scaled_coefficient_digits > MAX_LIE_DERIVATIVE_RAW_COEFFICIENT_DIGITS:
        _reject(
            "intermediate_height",
            "Lie-derivative coefficient growth exceeds the "
            f"{MAX_LIE_DERIVATIVE_RAW_COEFFICIENT_DIGITS}-digit intermediate budget",
        )


def _differentiate_polynomial(
    source: PolynomialBound,
    axis: int,
    *,
    active_terms: int,
    maximum_axis_exponent: int,
    minimum_exponents: tuple[int, ...],
) -> PolynomialBound:
    if source.is_zero or active_terms == 0:
        return _zero_polynomial(len(source.degrees))
    degrees = list(source.degrees)
    degrees[axis] -= 1
    result = PolynomialBound(
        terms=active_terms,
        degrees=tuple(degrees),
        minimum_exponents=minimum_exponents,
        coefficient_digits=(
            source.coefficient_digits + len(str(maximum_axis_exponent))
        ),
        rational_content=source.rational_content,
    )
    _check_raw_polynomial(result)
    return result


def _multiply_polynomials(
    left: PolynomialBound,
    right: PolynomialBound,
    ledger: _Ledger,
) -> PolynomialBound:
    if left.is_zero or right.is_zero:
        return _zero_polynomial(len(left.degrees))
    pair_count = left.terms * right.terms
    ledger.charge("multiplication", pair_count)
    degrees = tuple(
        left_degree + right_degree
        for left_degree, right_degree in zip(left.degrees, right.degrees, strict=True)
    )
    collision_count = min(left.terms, right.terms)
    product_digits = left.coefficient_digits + right.coefficient_digits
    if collision_count == 1:
        coefficient_digits = product_digits
    else:
        coefficient_digits = product_digits + len(str(collision_count))
    result = PolynomialBound(
        terms=min(pair_count, _dense_term_bound(degrees)),
        degrees=degrees,
        minimum_exponents=tuple(
            left_exponent + right_exponent
            for left_exponent, right_exponent in zip(
                left.minimum_exponents,
                right.minimum_exponents,
                strict=True,
            )
        ),
        coefficient_digits=coefficient_digits,
        rational_content=left.rational_content * right.rational_content,
    )
    _check_raw_polynomial(result)
    return result


def _add_polynomials(
    left: PolynomialBound,
    right: PolynomialBound,
    ledger: _Ledger,
) -> PolynomialBound:
    if left.is_zero:
        return right
    if right.is_zero:
        return left
    ledger.charge("addition", left.terms + right.terms)
    degrees = tuple(
        max(left_degree, right_degree)
        for left_degree, right_degree in zip(left.degrees, right.degrees, strict=True)
    )
    common_content = Fraction(
        gcd(
            abs(left.rational_content.numerator),
            abs(right.rational_content.numerator),
        ),
        lcm(
            left.rational_content.denominator,
            right.rational_content.denominator,
        ),
    )
    left_multiplier = left.rational_content / common_content
    right_multiplier = right.rational_content / common_content
    if left_multiplier.denominator != 1 or right_multiplier.denominator != 1:
        raise AssertionError(
            "rational polynomial content gcd did not divide both inputs"
        )

    def scaled_digits(coefficient_digits: int, multiplier: int) -> int:
        if abs(multiplier) <= 1:
            return coefficient_digits
        return coefficient_digits + len(str(abs(multiplier)))

    result = PolynomialBound(
        terms=min(left.terms + right.terms, _dense_term_bound(degrees)),
        degrees=degrees,
        minimum_exponents=tuple(
            min(left_exponent, right_exponent)
            for left_exponent, right_exponent in zip(
                left.minimum_exponents,
                right.minimum_exponents,
                strict=True,
            )
        ),
        coefficient_digits=max(
            scaled_digits(left.coefficient_digits, left_multiplier.numerator),
            scaled_digits(right.coefficient_digits, right_multiplier.numerator),
        )
        + 1,
        rational_content=common_content,
    )
    _check_raw_polynomial(result)
    return result


def _differentiate_fraction(
    source_value: RationalFunction,
    source_bound: FractionBound,
    axis: int,
    ledger: _Ledger,
) -> FractionBound:
    ledger.charge(
        "differentiation",
        source_bound.numerator.terms + source_bound.denominator.terms,
    )
    numerator_active = tuple(
        term for term in source_value.numerator.terms if term.exponents[axis] > 0
    )
    denominator_active = tuple(
        term for term in source_value.denominator.terms if term.exponents[axis] > 0
    )
    if not numerator_active and not denominator_active:
        return _zero_fraction(len(source_bound.numerator.degrees))
    numerator_derivative = _differentiate_polynomial(
        source_bound.numerator,
        axis,
        active_terms=len(numerator_active),
        maximum_axis_exponent=max(
            (term.exponents[axis] for term in numerator_active), default=1
        ),
        minimum_exponents=tuple(
            min(
                term.exponents[coordinate] - int(coordinate == axis)
                for term in numerator_active
            )
            for coordinate in range(len(source_value.variables))
        )
        if numerator_active
        else (0,) * len(source_value.variables),
    )
    denominator_derivative = _differentiate_polynomial(
        source_bound.denominator,
        axis,
        active_terms=len(denominator_active),
        maximum_axis_exponent=max(
            (term.exponents[axis] for term in denominator_active), default=1
        ),
        minimum_exponents=tuple(
            min(
                term.exponents[coordinate] - int(coordinate == axis)
                for term in denominator_active
            )
            for coordinate in range(len(source_value.variables))
        )
        if denominator_active
        else (0,) * len(source_value.variables),
    )
    first = _multiply_polynomials(
        numerator_derivative, source_bound.denominator, ledger
    )
    second = _multiply_polynomials(
        source_bound.numerator, denominator_derivative, ledger
    )
    numerator = _add_polynomials(first, second, ledger)
    if numerator.is_zero:
        return _zero_fraction(len(source_bound.numerator.degrees))
    denominator = _multiply_polynomials(
        source_bound.denominator, source_bound.denominator, ledger
    )
    return FractionBound(numerator=numerator, denominator=denominator)


def _multiply_fractions(
    left: FractionBound,
    right: FractionBound,
    ledger: _Ledger,
) -> FractionBound:
    if left.is_zero or right.is_zero:
        return _zero_fraction(len(left.numerator.degrees))
    return FractionBound(
        numerator=_multiply_polynomials(left.numerator, right.numerator, ledger),
        denominator=_multiply_polynomials(left.denominator, right.denominator, ledger),
    )


def _add_fractions(
    left: FractionBound,
    right: FractionBound,
    ledger: _Ledger,
) -> FractionBound:
    if left.is_zero:
        return right
    if right.is_zero:
        return left
    left_scaled = _multiply_polynomials(left.numerator, right.denominator, ledger)
    right_scaled = _multiply_polynomials(right.numerator, left.denominator, ledger)
    numerator = _add_polynomials(left_scaled, right_scaled, ledger)
    if numerator.is_zero:
        return _zero_fraction(len(left.numerator.degrees))
    return FractionBound(
        numerator=numerator,
        denominator=_multiply_polynomials(left.denominator, right.denominator, ledger),
    )


def _factor_coefficient_digits(bound: PolynomialBound) -> int:
    """Bound every coefficient of a rational factor of ``bound``.

    Clear source denominators, inject the multivariate polynomial into a
    univariate polynomial by mixed-radix Kronecker substitution, and apply
    Mignotte's ``2**degree * l2_norm`` integer-factor height bound.  A factor's
    degree in each variable cannot exceed the source degree, so the mixed
    radices introduce no carries in a factorization.
    """

    if bound.is_zero:
        return 1
    kronecker_degree = _dense_term_bound(bound.degrees) - 1
    binary_factor_digits = (302 * kronecker_degree + 999) // 1000
    norm_digits = len(str(bound.terms)) + 1
    return bound.coefficient_digits + binary_factor_digits + norm_digits + 2


def _remove_guaranteed_common_monomial(bound: FractionBound) -> FractionBound:
    """Apply exact valuation presolve before the canonical-result proof.

    The coordinatewise minimum exponent is a factor of every monomial. Its
    common part can therefore be canceled without expanding or factoring a
    polynomial, which materially widens monomial-denominator requests.
    """

    if bound.is_zero:
        return bound
    common = tuple(
        min(numerator, denominator)
        for numerator, denominator in zip(
            bound.numerator.minimum_exponents,
            bound.denominator.minimum_exponents,
            strict=True,
        )
    )
    if not any(common):
        return bound

    def divide(polynomial: PolynomialBound) -> PolynomialBound:
        return PolynomialBound(
            terms=polynomial.terms,
            degrees=tuple(
                degree - exponent
                for degree, exponent in zip(polynomial.degrees, common, strict=True)
            ),
            minimum_exponents=tuple(
                minimum - exponent
                for minimum, exponent in zip(
                    polynomial.minimum_exponents, common, strict=True
                )
            ),
            coefficient_digits=polynomial.coefficient_digits,
            rational_content=polynomial.rational_content,
        )

    return FractionBound(
        numerator=divide(bound.numerator), denominator=divide(bound.denominator)
    )


def _canonical_coefficient_digits(bound: FractionBound) -> int:
    if bound.is_zero:
        return 1
    content_ratio = (
        bound.numerator.rational_content / bound.denominator.rational_content
    )
    denominator_is_unit = all(degree == 0 for degree in bound.denominator.degrees)
    numerator_factor_digits = (
        bound.numerator.coefficient_digits
        if denominator_is_unit
        else _factor_coefficient_digits(bound.numerator)
    )
    denominator_factor_digits = (
        bound.denominator.coefficient_digits
        if denominator_is_unit
        else _factor_coefficient_digits(bound.denominator)
    )
    return max(
        len(str(abs(content_ratio.numerator)))
        + bound.numerator.coefficient_digits
        + (0 if denominator_is_unit else numerator_factor_digits),
        len(str(content_ratio.denominator))
        + bound.denominator.coefficient_digits
        + (0 if denominator_is_unit else denominator_factor_digits),
        denominator_factor_digits,
    )


def _validate_canonical_result_bound(bound: FractionBound, ledger: _Ledger) -> int:
    if bound.is_zero:
        ledger.charge("normalization", 1)
        return 1
    for label, polynomial in (
        ("numerator", bound.numerator),
        ("denominator", bound.denominator),
    ):
        if any(degree > MAX_RATIONAL_TENSOR_EXPONENT for degree in polynomial.degrees):
            _reject(
                "result_exponent",
                f"Lie-derivative {label} can exceed the canonical "
                f"exponent bound {MAX_RATIONAL_TENSOR_EXPONENT}",
            )
        dense_terms = _dense_term_bound(
            polynomial.degrees,
            cap=MAX_RATIONAL_TENSOR_POLYNOMIAL_TERMS,
        )
        # When the denominator is the unit polynomial, there can be no
        # cancellation-induced support expansion, so the tracked sparse
        # term count is the accurate support bound.
        denominator_is_unit = all(degree == 0 for degree in bound.denominator.degrees)
        support_terms = polynomial.terms if denominator_is_unit else dense_terms
        if support_terms > MAX_RATIONAL_TENSOR_POLYNOMIAL_TERMS:
            _reject(
                "result_support",
                f"Lie-derivative {label} can exceed the canonical "
                f"{MAX_RATIONAL_TENSOR_POLYNOMIAL_TERMS}-term bound",
            )
    coefficient_digits = _canonical_coefficient_digits(bound)
    if coefficient_digits > MAX_RATIONAL_TENSOR_COEFFICIENT_DIGITS:
        _reject(
            "result_height",
            "Lie-derivative normalization can exceed the canonical "
            f"{MAX_RATIONAL_TENSOR_COEFFICIENT_DIGITS}-digit coefficient bound",
        )
    denominator_is_unit = all(degree == 0 for degree in bound.denominator.degrees)
    numerator_dense = (
        bound.numerator.terms
        if denominator_is_unit
        else _dense_term_bound(bound.numerator.degrees)
    )
    denominator_dense = (
        1 if denominator_is_unit else _dense_term_bound(bound.denominator.degrees)
    )
    normalization_degree = max(numerator_dense + denominator_dense - 2, 0)
    ledger.charge(
        "normalization",
        (numerator_dense + denominator_dense)
        * (normalization_degree + 1)
        * coefficient_digits,
    )
    return coefficient_digits


def _component_offset(index: tuple[int, ...], dimension: int) -> int:
    offset = 0
    for coordinate in index:
        offset = offset * dimension + coordinate
    return offset


def _replace_index(
    index: tuple[int, ...], position: int, replacement: int
) -> tuple[int, ...]:
    return (*index[:position], replacement, *index[position + 1 :])


def _array_size(item_sizes: tuple[int, ...]) -> int:
    return 2 + max(len(item_sizes) - 1, 0) + sum(item_sizes)


def _bounded_string_size(content_digits: int, *, possibly_negative: bool) -> int:
    return content_digits + 2 + int(possibly_negative)


def _polynomial_result_size(bound: PolynomialBound, *, sparse: bool = False) -> int:
    if bound.is_zero:
        return strict_json_object_size((("terms", 2),))
    term_count = bound.terms if sparse else _dense_term_bound(bound.degrees)
    coefficient_size = strict_json_object_size(
        (
            (
                "num",
                _bounded_string_size(bound.coefficient_digits, possibly_negative=True),
            ),
            (
                "den",
                _bounded_string_size(bound.coefficient_digits, possibly_negative=False),
            ),
        )
    )
    exponent_sizes = tuple(len(str(max(degree, 0))) for degree in bound.degrees)
    exponent_array_size = _array_size(exponent_sizes)
    term_size = strict_json_object_size(
        (("coefficient", coefficient_size), ("exponents", exponent_array_size))
    )
    return strict_json_object_size((("terms", _array_size((term_size,) * term_count)),))


def _rational_function_result_size(
    bound: FractionBound,
    coefficient_digits: int,
    *,
    axis_size: int,
) -> int:
    denominator_is_unit = all(degree == 0 for degree in bound.denominator.degrees)
    if bound.is_zero:
        variable_count = len(bound.numerator.degrees)
        numerator = _zero_polynomial(variable_count)
        denominator = _one_polynomial(variable_count)
    else:
        numerator = PolynomialBound(
            terms=(
                bound.numerator.terms
                if denominator_is_unit
                else _dense_term_bound(bound.numerator.degrees)
            ),
            degrees=bound.numerator.degrees,
            minimum_exponents=(0,) * len(bound.numerator.degrees),
            coefficient_digits=coefficient_digits,
            rational_content=Fraction(1),
        )
        denominator = PolynomialBound(
            terms=(
                1
                if denominator_is_unit
                else _dense_term_bound(bound.denominator.degrees)
            ),
            degrees=bound.denominator.degrees,
            minimum_exponents=(0,) * len(bound.denominator.degrees),
            coefficient_digits=coefficient_digits,
            rational_content=Fraction(1),
        )
    return strict_json_object_size(
        (
            ("domain", 4),
            ("variables", axis_size),
            (
                "numerator",
                _polynomial_result_size(numerator, sparse=denominator_is_unit),
            ),
            ("denominator", _polynomial_result_size(denominator)),
        )
    )


def _model_size(value: StrictModel) -> int:
    try:
        payload = value.model_dump(mode="json")
        return len(encode_strict_json(payload))
    except CanonicalizationError:
        _reject(
            "source_bytes",
            "Lie-derivative retained sources exceed the canonical transport envelope",
        )


def _result_bytes_upper_bound(
    vector_field: RationalCoordinateTensor,
    tensor: RationalCoordinateTensor,
    components: tuple[LieComponentPlan, ...],
    inherited_guards: tuple[SparseRationalPolynomial, ...],
) -> int:
    axis_size = len(encode_strict_json(list(tensor.coordinate_axis)))
    variance_size = len(encode_strict_json(list(tensor.variance)))
    component_sizes = tuple(
        _rational_function_result_size(
            component.raw_result,
            component.canonical_coefficient_digits,
            axis_size=axis_size,
        )
        for component in components
    )
    inherited_guard_sizes = tuple(_model_size(guard) for guard in inherited_guards)
    result_guard_sizes = tuple(
        _polynomial_result_size(
            PolynomialBound(
                terms=_dense_term_bound(component.raw_result.denominator.degrees),
                degrees=component.raw_result.denominator.degrees,
                minimum_exponents=(0,) * len(component.raw_result.denominator.degrees),
                coefficient_digits=component.canonical_coefficient_digits,
                rational_content=Fraction(1),
            )
        )
        for component in components
        if not component.raw_result.is_zero
        and any(component.raw_result.denominator.degrees)
    )
    # Deduplicate guards before counting: a result denominator that
    # duplicates an inherited guard should not inflate the cap.
    result_guard_polynomials = tuple(
        component.raw_result.denominator
        for component in components
        if not component.raw_result.is_zero
        and any(component.raw_result.denominator.degrees)
    )
    # Compare compatible canonical representations: the degrees
    # tuple of each PolynomialBound against the degrees tuple
    # computed from each SparseRationalPolynomial's terms.
    # PolynomialBound deliberately retains only admission metadata, not the
    # exact coefficients. Without the full canonical polynomial identity,
    # treating matching degree tuples as duplicates can undercount distinct
    # guards and let result construction exceed its guard budget. Count every
    # possible result guard conservatively; canonical_locus_guards performs the
    # exact deduplication once the backend has produced the polynomials.
    distinct_guards = len(inherited_guards) + len(result_guard_polynomials)
    guard_count_bound = distinct_guards
    if guard_count_bound > MAX_RATIONAL_TENSOR_LOCUS_GUARDS:
        _reject(
            "result_locus_guards",
            "Lie-derivative retained locus can exceed the "
            f"{MAX_RATIONAL_TENSOR_LOCUS_GUARDS}-guard representation budget",
        )
    result_tensor_size = strict_json_object_size(
        (
            ("coordinate_axis", axis_size),
            ("variance", variance_size),
            ("components", _array_size(component_sizes)),
            (
                "retained_nonzero_denominators",
                _array_size(inherited_guard_sizes + result_guard_sizes),
            ),
        )
    )
    return strict_json_object_size(
        (
            ("vector_field", _model_size(vector_field)),
            ("source", _model_size(tensor)),
            ("lie_derivative", result_tensor_size),
        )
    )


def _recognition_work_units(bound: FractionBound) -> int:
    """Charge the dense exact coefficient work exposed to SymPy's GCD.

    SymPy 1.14 represents a multivariate ``Poly`` as a recursively dense DMP.
    The product of per-axis degree ranges therefore bounds the coefficients
    materialized by conversion.  Total degree steps and 32-digit coefficient
    chunks conservatively price the subsequent exact GCD reductions.  A
    killable worker still owns wall time and memory because SymPy exposes no
    cooperative cancellation hook inside ``Poly.gcd``.
    """

    numerator_dense = _dense_term_bound(bound.numerator.degrees)
    denominator_dense = _dense_term_bound(bound.denominator.degrees)
    degree_steps = (
        sum(
            max(numerator, denominator)
            for numerator, denominator in zip(
                bound.numerator.degrees,
                bound.denominator.degrees,
                strict=True,
            )
        )
        + 1
    )
    coefficient_digits = max(
        bound.numerator.coefficient_digits
        + len(str(abs(bound.numerator.rational_content.numerator)))
        + len(str(bound.numerator.rational_content.denominator)),
        bound.denominator.coefficient_digits
        + len(str(abs(bound.denominator.rational_content.numerator)))
        + len(str(bound.denominator.rational_content.denominator)),
    )
    coefficient_chunks = max(1, (coefficient_digits + 31) // 32)
    return (numerator_dense + denominator_dense) * degree_steps * coefficient_chunks


def build_lie_derivative_plan(
    vector_field: RationalCoordinateTensor,
    tensor: RationalCoordinateTensor,
    *,
    deadline: float | None = None,
) -> LieDerivativePlan:
    """Admit one complete Lie derivative and return its reusable plan."""

    if deadline is None:
        deadline = begin_lie_derivative_deadline()

    if vector_field.coordinate_axis != tensor.coordinate_axis:
        _reject(
            "coordinate_axis_mismatch",
            "vector field and tensor must use the same ordered coordinate axis",
            location=("vector_field", "coordinate_axis"),
        )
    if vector_field.variance != ("CONTRAVARIANT",):
        _reject(
            "vector_signature",
            "vector field must have rank one and CONTRAVARIANT variance",
            location=("vector_field", "variance"),
        )
    dimension = len(tensor.coordinate_axis)
    ledger = _Ledger(deadline=deadline)
    vector_bounds = tuple(
        _fraction_bound(value, ledger) for value in vector_field.components
    )
    tensor_bounds = tuple(_fraction_bound(value, ledger) for value in tensor.components)
    vector_derivatives = {
        (component, axis): _zero_fraction(dimension)
        if vector_bounds[component].is_zero
        else _differentiate_fraction(
            vector_field.components[component], vector_bounds[component], axis, ledger
        )
        for component in range(dimension)
        for axis in range(dimension)
    }
    component_plans: list[LieComponentPlan] = []
    for index in product(range(dimension), repeat=len(tensor.variance)):
        component = _component_offset(index, dimension)
        term_plans: list[LieProductTerm] = []
        result_bound = _zero_fraction(dimension)
        for axis in range(dimension):
            if vector_bounds[axis].is_zero:
                continue
            term = LieProductTerm(
                sign=1,
                left=FactorReference("VECTOR", axis, None),
                right=FactorReference("TENSOR", component, axis),
            )
            term_plans.append(term)
            result_bound = _add_fractions(
                result_bound,
                _multiply_fractions(
                    vector_bounds[axis],
                    _differentiate_fraction(
                        tensor.components[component],
                        tensor_bounds[component],
                        axis,
                        ledger,
                    ),
                    ledger,
                ),
                ledger,
            )
        for position, variance in enumerate(tensor.variance):
            component_index = index[position]
            for axis in range(dimension):
                replaced_component = _component_offset(
                    _replace_index(index, position, axis), dimension
                )
                if variance == "CONTRAVARIANT":
                    term = LieProductTerm(
                        sign=-1,
                        left=FactorReference("VECTOR", component_index, axis),
                        right=FactorReference("TENSOR", replaced_component, None),
                    )
                    left_bound = vector_derivatives[(component_index, axis)]
                else:
                    term = LieProductTerm(
                        sign=1,
                        left=FactorReference("VECTOR", axis, component_index),
                        right=FactorReference("TENSOR", replaced_component, None),
                    )
                    left_bound = vector_derivatives[(axis, component_index)]
                if left_bound.is_zero:
                    continue
                term_plans.append(term)
                result_bound = _add_fractions(
                    result_bound,
                    _multiply_fractions(
                        left_bound, tensor_bounds[replaced_component], ledger
                    ),
                    ledger,
                )
        result_bound = _remove_guaranteed_common_monomial(result_bound)
        coefficient_digits = _validate_canonical_result_bound(result_bound, ledger)
        component_plans.append(
            LieComponentPlan(
                terms=tuple(term_plans),
                raw_result=result_bound,
                canonical_coefficient_digits=coefficient_digits,
            )
        )

    inherited_guards = canonical_locus_guards(
        vector_field.retained_nonzero_denominators,
        tensor.retained_nonzero_denominators,
        variable_count=dimension,
    )
    plans = tuple(component_plans)
    result_bytes = _result_bytes_upper_bound(
        vector_field, tensor, plans, inherited_guards
    )
    if result_bytes > MAX_LIE_DERIVATIVE_RESULT_BYTES:
        _reject(
            "result_bytes",
            f"Lie-derivative result estimate of {result_bytes} bytes exceeds the "
            f"{MAX_LIE_DERIVATIVE_RESULT_BYTES}-byte canonical output budget",
        )
    recognition_candidates = canonical_recognition_candidates(vector_field, tensor)
    for candidate in recognition_candidates:
        source_bounds = (
            vector_bounds if candidate.owner == "vector_field" else tensor_bounds
        )
        ledger.charge(
            "recognition",
            _recognition_work_units(source_bounds[candidate.component]),
        )
    plan = LieDerivativePlan(
        components=plans,
        recognition_candidates=recognition_candidates,
        inherited_locus_guards=inherited_guards,
        result_bytes_upper_bound=result_bytes,
        work_units_by_category=ledger.by_category,
    )
    recognition = recognize_canonical_rational_functions(
        plan.recognition_candidates,
        deadline=deadline,
    )
    if recognition.non_coprime is not None:
        failure = recognition.non_coprime
        _reject(
            "component_not_canonical",
            f"{failure.owner} components must have coprime canonical rational-function parts",
            location=(failure.owner, "components", failure.component),
        )
    require_lie_derivative_deadline(deadline, "after coprimality recognition")
    return plan


__all__ = [
    "MAX_LIE_DERIVATIVE_RESULT_BYTES",
    "MAX_LIE_DERIVATIVE_WORK_UNITS",
    "FactorReference",
    "LieComponentPlan",
    "LieDerivativePlan",
    "LieProductTerm",
    "build_lie_derivative_plan",
]
