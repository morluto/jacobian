"""Exact finite class-function operations."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups.characters._cyclotomic import (
    MAX_ARITHMETIC_ORDER,
    MAX_CYCLOTOMIC_REDUCTION_COEFFICIENT_DIGITS,
    add_values,
    conjugate_value,
    euler_phi,
    multiply_values,
    scale_value,
    zero_value,
)
from jacobian.math.groups.characters._models import (
    MAX_CLASS_COUNT,
    MAX_CYCLOTOMIC_ORDER,
    MAX_GROUP_ORDER,
    MAX_INNER_PRODUCT_WORK,
    MAX_VALUE_COEFFICIENT_DIGITS,
    ClassContribution,
    ClassFunctionInnerProductResult,
    CyclotomicValue,
    FiniteClassFunction,
)


def _fractions(value: CyclotomicValue) -> tuple[Fraction, ...]:
    return tuple(coefficient.as_fraction() for coefficient in value.coefficients)


def _make_value(order: int, coefficients: tuple[Fraction, ...]) -> CyclotomicValue:
    return CyclotomicValue._from_kernel(
        order=order,
        coefficients=tuple(
            CanonicalRational.from_fraction(coefficient) for coefficient in coefficients
        ),
    )


@dataclass(frozen=True)
class _CoefficientHeight:
    """Conservative decimal widths for one generated coefficient."""

    numerator_digits: int
    denominator_digits: int

    @property
    def maximum_digits(self) -> int:
        return max(self.numerator_digits, self.denominator_digits)


@dataclass(frozen=True)
class _InputHeight:
    denominator: int | None
    lifted_numerator_digits: int

    @property
    def denominator_digits(self) -> int:
        return (
            MAX_VALUE_COEFFICIENT_DIGITS + 1
            if self.denominator is None
            else len(str(self.denominator))
        )


def _bounded_lcm(values: tuple[int, ...]) -> int | None:
    """Return an input common denominator, capped before it gets unwieldy."""

    result = 1
    for value in values:
        result = (result // gcd(result, value)) * value
        if len(str(result)) > MAX_VALUE_COEFFICIENT_DIGITS:
            return None
    return result


def _bounded_product(left: int | None, right: int | None) -> int | None:
    if left is None or right is None:
        return None
    result = left * right
    if len(str(result)) > MAX_VALUE_COEFFICIENT_DIGITS:
        return None
    return result


def _factor_digits(value: int) -> int:
    """Digits contributed by multiplying by a positive integer factor."""

    return 0 if value == 1 else len(str(value))


def _input_height(value: CyclotomicValue) -> _InputHeight:
    """Bound one value's coefficient numerators and common denominator."""

    denominator = _bounded_lcm(
        tuple(coefficient.den for coefficient in value.coefficients)
    )
    if denominator is None:
        lifted_numerator_digits = MAX_VALUE_COEFFICIENT_DIGITS + 1
    else:
        lifted_numerator_digits = max(
            len(str(abs(coefficient.num)))
            + _factor_digits(denominator // coefficient.den)
            for coefficient in value.coefficients
        )
    return _InputHeight(
        denominator=denominator,
        lifted_numerator_digits=lifted_numerator_digits,
    )


def _reduced_height(
    *, order: int, numerator_digits: int, denominator_digits: int, raw_length: int
) -> _CoefficientHeight:
    """Bound coefficient growth from reduction modulo ``Phi_order``."""

    degree = euler_phi(order)
    reduction_steps = max(0, raw_length - degree)
    return _CoefficientHeight(
        numerator_digits=(
            numerator_digits
            + reduction_steps * MAX_CYCLOTOMIC_REDUCTION_COEFFICIENT_DIGITS
        ),
        denominator_digits=denominator_digits,
    )


def _derived_height(
    *,
    order: int,
    class_sizes: tuple[int, ...],
    group_order: int,
    phi_values: tuple[CyclotomicValue, ...],
    psi_values: tuple[CyclotomicValue, ...],
) -> tuple[_CoefficientHeight, _CoefficientHeight, _CoefficientHeight]:
    """Bound conjugate, weighted-product, and result coefficient heights.

    The bound follows the actual kernel stages without evaluating them: input
    denominators are placed in common least-common-multiple denominators;
    multiplication contributes one coefficient-pair product per raw term,
    cyclotomic
    reduction contributes its fixed integer-coefficient growth, class sizes
    scale numerators, class terms accumulate over a common denominator, and
    division by ``|G|`` enlarges that denominator.  The returned heights cover
    every generated value retained in the result, including contribution rows.
    """

    dimension = euler_phi(order)
    phi_heights = tuple(_input_height(value) for value in phi_values)
    psi_heights = tuple(_input_height(value) for value in psi_values)

    conjugates = tuple(
        _reduced_height(
            order=order,
            numerator_digits=height.lifted_numerator_digits,
            denominator_digits=height.denominator_digits,
            raw_length=order,
        )
        for height in psi_heights
    )
    weighted: list[_CoefficientHeight] = []
    weighted_denominators: list[int | None] = []
    for class_size, phi_height, psi_height, conjugate in zip(
        class_sizes, phi_heights, psi_heights, conjugates, strict=True
    ):
        denominator = _bounded_product(phi_height.denominator, psi_height.denominator)
        denominator_digits = (
            MAX_VALUE_COEFFICIENT_DIGITS + 1
            if denominator is None
            else len(str(denominator))
        )
        product = _reduced_height(
            order=order,
            numerator_digits=(
                phi_height.lifted_numerator_digits
                + conjugate.numerator_digits
                + (len(str(dimension)) if dimension > 1 else 0)
            ),
            denominator_digits=denominator_digits,
            raw_length=(2 * dimension) - 1,
        )
        weighted.append(
            _CoefficientHeight(
                numerator_digits=product.numerator_digits
                + (len(str(class_size)) if class_size != 1 else 0),
                denominator_digits=product.denominator_digits,
            )
        )
        weighted_denominators.append(denominator)

    total_denominator = _bounded_lcm(
        tuple(
            denominator
            for denominator in weighted_denominators
            if denominator is not None
        )
    )
    if total_denominator is None or any(
        denominator is None for denominator in weighted_denominators
    ):
        total = _CoefficientHeight(
            numerator_digits=MAX_VALUE_COEFFICIENT_DIGITS + 1,
            denominator_digits=MAX_VALUE_COEFFICIENT_DIGITS + 1,
        )
    else:
        total_numerator_digits = max(
            height.numerator_digits + _factor_digits(total_denominator // denominator)
            for height, denominator in zip(weighted, weighted_denominators, strict=True)
            if denominator is not None
        ) + (len(str(len(weighted))) if len(weighted) > 1 else 0)
        total = _CoefficientHeight(
            numerator_digits=total_numerator_digits,
            denominator_digits=len(str(total_denominator)),
        )
    inner_denominator = _bounded_product(total_denominator, group_order)
    inner = _CoefficientHeight(
        numerator_digits=total.numerator_digits,
        denominator_digits=(
            MAX_VALUE_COEFFICIENT_DIGITS + 1
            if inner_denominator is None
            else len(str(inner_denominator))
        ),
    )
    return (
        max(conjugates, key=lambda value: value.maximum_digits),
        max(weighted, key=lambda value: value.maximum_digits),
        inner,
    )


def _reject_derived_height(stage: str, height: _CoefficientHeight) -> None:
    if height.maximum_digits > MAX_VALUE_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("phi", "values"),
            code="groups.characters.inner_product_output_digits_exceed_envelope",
            message=(
                f"derived {stage} coefficients exceed the "
                f"{MAX_VALUE_COEFFICIENT_DIGITS}-digit envelope"
            ),
        )


def _admit_inner_product(phi: FiniteClassFunction, psi: FiniteClassFunction) -> None:
    """Shared native/catalog admission for the Hermitian inner product."""

    if phi.axis != psi.axis:
        raise OperationDomainValidationError(
            location=("psi", "axis"),
            code="groups.characters.class_axis_mismatch",
            message=(
                "both class functions must carry the identical class axis "
                "(class sizes, group order, and cyclotomic order)"
            ),
        )
    axis = phi.axis
    class_count = len(axis.class_sizes)
    if class_count > MAX_CLASS_COUNT:
        raise OperationResourceAdmissionError(
            location=("phi", "axis", "class_sizes"),
            code="groups.characters.class_count_exceeds_envelope",
            message=f"class functions admit at most {MAX_CLASS_COUNT} classes",
        )
    if axis.group_order > MAX_GROUP_ORDER:
        raise OperationResourceAdmissionError(
            location=("phi", "axis", "group_order"),
            code="groups.characters.group_order_exceeds_envelope",
            message=f"class functions admit group order at most {MAX_GROUP_ORDER}",
        )
    if axis.cyclotomic_order > MAX_CYCLOTOMIC_ORDER:
        raise OperationResourceAdmissionError(
            location=("phi", "axis", "cyclotomic_order"),
            code="groups.characters.cyclotomic_order_exceeds_envelope",
            message=(
                f"class functions admit cyclotomic order at most {MAX_CYCLOTOMIC_ORDER}"
            ),
        )
    max_digits = 1
    for value in (*phi.values, *psi.values):
        for coefficient in value.coefficients:
            digits = canonical_rational_component_digits(coefficient)
            if digits > max_digits:
                max_digits = digits
    if max_digits > MAX_VALUE_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("phi", "values"),
            code="groups.characters.value_coefficient_digits_exceed_envelope",
            message=(
                "class-function value coefficients exceed the "
                f"{MAX_VALUE_COEFFICIENT_DIGITS}-digit envelope"
            ),
        )
    work = (
        class_count
        * max(1, axis.cyclotomic_order)
        * max(1, max_digits)
        * max(1, max_digits)
    )
    if work > MAX_INNER_PRODUCT_WORK:
        raise OperationResourceAdmissionError(
            location=("phi",),
            code="groups.characters.inner_product_work_exceeds_envelope",
            message=(
                "class-function inner-product work exceeds the "
                f"{MAX_INNER_PRODUCT_WORK} unit envelope"
            ),
        )
    conjugate, weighted, inner = _derived_height(
        order=axis.cyclotomic_order,
        class_sizes=axis.class_sizes,
        group_order=axis.group_order,
        phi_values=phi.values,
        psi_values=psi.values,
    )
    _reject_derived_height("conjugate", conjugate)
    _reject_derived_height("weighted product", weighted)
    _reject_derived_height("inner product", inner)


def class_function_inner_product(
    phi: FiniteClassFunction,
    psi: FiniteClassFunction,
) -> ClassFunctionInnerProductResult:
    """Exact Hermitian inner product with a complete class contribution table."""

    _admit_inner_product(phi, psi)
    axis = phi.axis
    order = axis.cyclotomic_order
    if order > MAX_ARITHMETIC_ORDER:
        raise OperationResourceAdmissionError(
            location=("phi", "axis", "cyclotomic_order"),
            code="groups.characters.arithmetic_order_exceeds_envelope",
            message=(
                f"exact cyclotomic arithmetic admits order at most "
                f"{MAX_ARITHMETIC_ORDER}"
            ),
        )
    total = zero_value(order)
    contributions: list[ClassContribution] = []
    for index, class_size in enumerate(axis.class_sizes):
        phi_value = _fractions(phi.values[index])
        psi_value = _fractions(psi.values[index])
        conjugate = conjugate_value(order, psi_value)
        product = multiply_values(order, phi_value, conjugate)
        weighted = scale_value(order, Fraction(class_size), product)
        total = add_values(order, total, weighted)
        contributions.append(
            ClassContribution(
                class_index=index,
                class_size=class_size,
                phi_value=phi.values[index],
                psi_value=psi.values[index],
                conjugate_psi_value=_make_value(order, conjugate),
                weighted_product=_make_value(order, weighted),
            )
        )
    inner = _make_value(
        order,
        tuple(coefficient / axis.group_order for coefficient in total),
    )
    if len(inner.coefficients) != euler_phi(order):
        raise OperationDomainValidationError(
            location=("inner_product",),
            code="groups.characters.inner_product_degree",
            message="exact inner product did not reduce to the cyclotomic degree",
        )
    return ClassFunctionInnerProductResult._from_kernel(
        axis=axis,
        phi=phi,
        psi=psi,
        contributions=tuple(contributions),
        inner_product=inner,
    )


__all__ = ["class_function_inner_product"]
