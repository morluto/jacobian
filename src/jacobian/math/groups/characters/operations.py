"""Exact finite class-function operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups.characters._cyclotomic import (
    MAX_ARITHMETIC_ORDER,
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
    ClassFunctionInnerProductRequest,
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


def class_function_inner_product(
    request: ClassFunctionInnerProductRequest,
) -> ClassFunctionInnerProductResult:
    """Exact Hermitian inner product with a complete class contribution table."""

    phi = request.phi
    psi = request.psi
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
