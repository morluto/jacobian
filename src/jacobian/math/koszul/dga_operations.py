"""Exact finite-dimensional Koszul differential graded algebras."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from itertools import combinations
from math import comb
from typing import Any

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    canonical_rational_component_digits,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.koszul.module_models import (
    MAX_KOSZUL_DGA_PRODUCT_ENTRIES,
    BasedFiniteModule,
    FiniteCommutativeAlgebra,
    ModuleKoszulDGA,
    ModuleKoszulDGAProduct,
    ModuleKoszulDGARequest,
    ModuleKoszulRequest,
)
from jacobian.math.koszul.module_operations import (
    _admit,
    _build_module_koszul_complex,
    _f,
)

MAX_KOSZUL_DGA_WORK = 2_000_000
MAX_KOSZUL_DGA_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_KOSZUL_DGA_LABEL_BYTES = 256
MAX_KOSZUL_DGA_INTERMEDIATE_DIGITS = 8_192


def _as_request(
    request: ModuleKoszulDGARequest | Mapping[str, Any],
) -> ModuleKoszulDGARequest:
    try:
        payload = (
            request.model_dump()
            if isinstance(request, ModuleKoszulDGARequest)
            else request
        )
        return ModuleKoszulDGARequest.model_validate(payload)
    except Exception as error:
        raise OperationDomainValidationError(
            location=("request",),
            code="koszul.module.dga_request_shape",
            message="the finite-algebra Koszul DGA request is not canonical",
        ) from error


def _admit_dga(value: ModuleKoszulDGARequest) -> None:
    algebra = value.algebra
    dimension = len(algebra.basis)
    sequence_length = len(value.sequence)
    if algebra.unit is None:
        raise OperationDomainValidationError(
            location=("algebra", "unit"),
            code="koszul.module.dga_unit_required",
            message="a unital Koszul DGA requires an explicit algebra unit",
        )
    label_byte_count = 0
    for label in algebra.basis:
        if len(label) > MAX_KOSZUL_DGA_LABEL_BYTES:
            raise OperationResourceAdmissionError(
                location=("algebra", "basis"),
                code="koszul.module.dga_label_budget",
                message="algebra basis labels exceed the admitted DGA output envelope",
            )
        label_size = len(label.encode("utf-8"))
        if label_size > MAX_KOSZUL_DGA_LABEL_BYTES:
            raise OperationResourceAdmissionError(
                location=("algebra", "basis"),
                code="koszul.module.dga_label_budget",
                message="algebra basis labels exceed the admitted DGA output envelope",
            )
        label_byte_count += label_size

    total_basis = dimension * (1 << sequence_length)
    if total_basis > 256:
        raise OperationResourceAdmissionError(
            location=("sequence",),
            code="koszul.module.budget",
            message="finite-algebra Koszul DGA exceeds its admitted chain-basis envelope",
        )

    multiplication_digits = max(
        (
            canonical_rational_component_digits(coefficient)
            for first in algebra.multiplication
            for row in first
            for coefficient in row
        ),
        default=1,
    )
    unit_digits = max(
        (
            canonical_rational_component_digits(coefficient)
            for coefficient in algebra.unit
        ),
        default=1,
    )
    sequence_digits = max(
        (
            canonical_rational_component_digits(coefficient)
            for element in value.sequence
            for coefficient in element
        ),
        default=1,
    )
    log_dimension = len(str(dimension))
    internal_digits = max(
        2 * dimension * multiplication_digits + log_dimension + 4,
        dimension * (unit_digits + multiplication_digits) + log_dimension + 4,
    )
    differential_digits = (
        dimension * (sequence_digits + multiplication_digits) + log_dimension + 4
        if sequence_length
        else 1
    )
    if internal_digits > MAX_KOSZUL_DGA_INTERMEDIATE_DIGITS:
        raise OperationResourceAdmissionError(
            location=("algebra", "multiplication"),
            code="koszul.module.dga_intermediate_digit_budget",
            message="exact DGA identity checks exceed the admitted intermediate digit bound",
        )
    if differential_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationResourceAdmissionError(
            location=("sequence",),
            code="koszul.module.dga_differential_digit_budget",
            message="Koszul differential coefficients may exceed the canonical rational limit",
        )

    nonzero_structure_coefficients = sum(
        coefficient.num != 0
        for first in algebra.multiplication
        for row in first
        for coefficient in row
    )
    product_entries_bound = (3**sequence_length) * nonzero_structure_coefficients
    # Product construction currently inspects every pair of wedges before it
    # skips overlaps, so charge the 4^r scan as well as the 3^r disjoint pairs
    # that reach the algebra structure tensor.
    product_work_bound = (3**sequence_length) * dimension**3 + (4**sequence_length) * (
        sequence_length + 2
    )
    differential_entries_bound = (
        sequence_length * dimension**2 * (1 << (sequence_length - 1))
        if sequence_length
        else 0
    )
    differential_square_work = sum(
        dimension**3
        * comb(sequence_length, degree - 1)
        * comb(sequence_length, degree)
        * comb(sequence_length, degree + 1)
        for degree in range(1, sequence_length)
    )
    algebra_validation_work = (
        3 * dimension**5 + dimension**4 + dimension**3
    ) * internal_digits
    differential_work = (
        differential_entries_bound * dimension * differential_digits
        + differential_square_work * 2 * differential_digits
    )
    product_coefficient_work = product_entries_bound * multiplication_digits
    work_bound = (
        algebra_validation_work
        + differential_work
        + product_work_bound
        + product_coefficient_work
    )
    product_entry_bytes = 2 * multiplication_digits + 80
    differential_entry_bytes = 2 * differential_digits + 64
    rational_items = 4 * dimension**3 + 4 * dimension + 2 * sequence_length * dimension
    context_bytes = rational_items * (
        2 * max(multiplication_digits, unit_digits, sequence_digits) + 24
    )
    context_bytes += 4 * label_byte_count + 2_048
    result_bytes_bound = (
        2_048
        + context_bytes
        + product_entries_bound * product_entry_bytes
        + differential_entries_bound * differential_entry_bytes
    )
    if (
        product_entries_bound > MAX_KOSZUL_DGA_PRODUCT_ENTRIES
        or work_bound > MAX_KOSZUL_DGA_WORK
        or result_bytes_bound > MAX_KOSZUL_DGA_OUTPUT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("sequence",),
            code="koszul.module.dga_output_budget",
            message=(
                "Koszul DGA multiplication, identity work, or complete output "
                "exceeds its admitted bound"
            ),
        )


def _regular_module(algebra: FiniteCommutativeAlgebra) -> BasedFiniteModule:
    dimension = len(algebra.basis)
    return BasedFiniteModule(
        algebra=algebra,
        basis=algebra.basis,
        action=tuple(
            tuple(
                tuple(
                    algebra.multiplication[element][source][target]
                    for source in range(dimension)
                )
                for target in range(dimension)
            )
            for element in range(dimension)
        ),
    )


def _require_unit(algebra: FiniteCommutativeAlgebra) -> None:
    """Check the authored unit against every basis vector on both sides."""
    structure = [
        [[_f(coefficient) for coefficient in cell] for cell in row]
        for row in algebra.multiplication
    ]
    unit = tuple(_f(value) for value in algebra.unit or ())
    dimension = len(algebra.basis)
    for basis_index in range(dimension):
        for target in range(dimension):
            left = sum(
                unit[index] * structure[index][basis_index][target]
                for index in range(dimension)
            )
            right = sum(
                unit[index] * structure[basis_index][index][target]
                for index in range(dimension)
            )
            expected = Fraction(int(target == basis_index))
            if left != expected or right != expected:
                raise OperationDomainValidationError(
                    location=("algebra", "unit"),
                    code="koszul.module.dga_invalid_unit",
                    message="the supplied coordinates are not a two-sided algebra unit",
                )


def _wedge_sign(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    inversions = sum(first > second for first in left for second in right)
    return -1 if inversions % 2 else 1


def _product_table(
    algebra: FiniteCommutativeAlgebra, sequence_length: int
) -> tuple[ModuleKoszulDGAProduct, ...]:
    dimension = len(algebra.basis)
    wedges = tuple(
        tuple(combinations(range(sequence_length), degree))
        for degree in range(sequence_length + 1)
    )
    wedge_indices = tuple(
        {wedge: index for index, wedge in enumerate(degree_wedges)}
        for degree_wedges in wedges
    )
    entries: list[ModuleKoszulDGAProduct] = []
    for left_degree, left_wedges in enumerate(wedges):
        for left_wedge_index, left_wedge in enumerate(left_wedges):
            for left_algebra_index in range(dimension):
                left_index = left_wedge_index * dimension + left_algebra_index
                for right_degree, right_wedges in enumerate(wedges):
                    if left_degree + right_degree > sequence_length:
                        continue
                    result_degree = left_degree + right_degree
                    for right_wedge_index, right_wedge in enumerate(right_wedges):
                        if set(left_wedge).intersection(right_wedge):
                            continue
                        union = tuple(sorted((*left_wedge, *right_wedge)))
                        result_wedge_index = wedge_indices[result_degree][union]
                        sign = _wedge_sign(left_wedge, right_wedge)
                        for right_algebra_index in range(dimension):
                            right_index = (
                                right_wedge_index * dimension + right_algebra_index
                            )
                            for result_algebra_index in range(dimension):
                                coefficient = algebra.multiplication[
                                    left_algebra_index
                                ][right_algebra_index][result_algebra_index]
                                if coefficient.num == 0:
                                    continue
                                if sign < 0:
                                    coefficient = CanonicalRational.from_fraction(
                                        -coefficient.as_fraction()
                                    )
                                entries.append(
                                    ModuleKoszulDGAProduct.model_construct(
                                        left_degree=left_degree,
                                        left_index=left_index,
                                        right_degree=right_degree,
                                        right_index=right_index,
                                        result_index=(
                                            result_wedge_index * dimension
                                            + result_algebra_index
                                        ),
                                        coefficient=coefficient,
                                    )
                                )
    return tuple(entries)


def module_koszul_dga(
    request: ModuleKoszulDGARequest | Mapping[str, Any],
) -> ModuleKoszulDGA:
    """Return the exact unital DGA ``K(f; A)`` for a finite commutative algebra.

    The product combines the algebra multiplication tensor with the exterior
    wedge product. The complete sparse table and all exact validation work are
    admitted before constructing the chain complex or multiplication entries.
    """
    value = _as_request(request)
    _admit_dga(value)
    module = _regular_module(value.algebra)
    _admit(module, value.sequence)
    _require_unit(value.algebra)
    module_request = ModuleKoszulRequest(
        algebra=value.algebra, module=module, sequence=value.sequence
    )
    complex_value = _build_module_koszul_complex(module_request)
    products = _product_table(value.algebra, len(value.sequence))
    return ModuleKoszulDGA._from_kernel(
        algebra=value.algebra,
        sequence=value.sequence,
        complex=complex_value,
        unit_coordinates=value.algebra.unit,
        products=products,
    )


__all__ = ["MAX_KOSZUL_DGA_OUTPUT_BYTES", "MAX_KOSZUL_DGA_WORK", "module_koszul_dga"]
