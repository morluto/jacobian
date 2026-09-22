"""Exact finite class-function operations."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
from jacobian._execution import BackendFailureReason, OperationBackendError
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups._models import GroupConjugacyClassesResult
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
    MAX_CHARACTER_TABLE_CELLS,
    MAX_CHARACTER_TABLE_WORK,
    MAX_CLASS_COUNT,
    MAX_CYCLOTOMIC_ORDER,
    MAX_GROUP_ORDER,
    MAX_INNER_PRODUCT_WORK,
    MAX_VALUE_COEFFICIENT_DIGITS,
    CharacterRow,
    CharacterTableResult,
    ClassAxis,
    ClassContribution,
    ClassFunctionInnerProductResult,
    ConjugacyClassPartition,
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


def _permutation_compose(
    first: tuple[int, ...], second: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(second[first[index]] for index in range(len(first)))


def _cyclic_generator(partition: GroupConjugacyClassesResult) -> tuple[int, ...] | None:
    elements = [
        tuple(element)
        for conjugacy_class in partition.classes
        for element in conjugacy_class
    ]
    identity = tuple(range(partition.source.degree))
    for candidate in elements:
        powers = {identity}
        current = identity
        for _ in range(len(elements)):
            current = _permutation_compose(current, candidate)
            powers.add(current)
            if current == identity:
                break
        if len(powers) == len(elements):
            return candidate
    return None


def _cyclotomic_reduction_work(raw_length: int, degree: int) -> int:
    """Count coefficient updates in the exact reducer's bounded loop."""

    return max(0, raw_length - degree) * (degree + 1)


def _character_table_work_bound(
    *, order: int, class_count: int, row_count: int, max_class_size: int
) -> tuple[int, int]:
    """Bound exact coefficient work and materialized coefficient cells.

    The orthogonality pass is not a scalar ``rows² * classes`` operation.  It
    conjugates, multiplies, reduces, scales, adds, and constructs contribution
    values in ``Q(zeta_order)`` for every class and ordered row pair.  Count
    those vector operations before generating any rows.  The second returned
    quantity accounts for all retained and temporary coefficient vectors,
    including the contribution table that is constructed and immediately
    discarded by the table operation.
    """

    degree = euler_phi(order)
    reduction = _cyclotomic_reduction_work
    conjugate = order + degree + reduction(order, degree)
    product = degree * degree + reduction(2 * degree - 1, degree)
    vector_cells = 3 * degree  # conjugate, product, and weighted product
    pair_class_work = conjugate + product + 2 * degree + vector_cells
    pair_count = row_count * row_count * class_count

    # Rational coefficient height is part of the exact work envelope: reduction
    # can enlarge integer numerators even when the vector length is bounded.
    def reduction_height(raw_length: int) -> int:
        # One reduction update sums up to ``degree + 1`` coefficient terms;
        # account for both the fixed polynomial coefficient width and that
        # accumulation height rather than charging vector length alone.
        per_step_digits = MAX_CYCLOTOMIC_REDUCTION_COEFFICIENT_DIGITS + len(
            str(degree + 1)
        )
        return max(0, raw_length - degree) * per_step_digits

    generated_height = 1 + reduction_height(order)
    conjugate_height = generated_height + reduction_height(order)
    product_height = (
        2 * generated_height
        + (len(str(degree)) if degree > 1 else 0)
        + reduction_height(2 * degree - 1)
    )
    weighted_height = product_height + len(str(max_class_size))
    intermediate_height = max(
        1,
        generated_height,
        conjugate_height,
        product_height,
        weighted_height + len(str(class_count)),
        len(str(order)) + weighted_height,
    )
    work = pair_count * pair_class_work * intermediate_height
    # Final division/addition and the generated character rows are mandatory
    # coefficient work too; output_cells includes every constructed vector.
    work += (
        row_count * row_count * degree * intermediate_height
        + row_count * class_count * degree * generated_height
    )
    output_cells = (
        row_count * class_count * degree
        + pair_count * (vector_cells + degree)  # contribution + inner vectors
    )
    # Decimal serialization of exact coefficient vectors is itself admitted
    # work; do not let a cheap arithmetic estimate hide a large wire result.
    work = max(work, output_cells * intermediate_height)
    return work, output_cells


def _admit_character_table(*, order: int, sizes: tuple[int, ...]) -> int:
    """Admit the complete table, including its aggregate orthogonality pass."""
    class_count = len(sizes)
    if order > MAX_GROUP_ORDER or class_count > MAX_CLASS_COUNT:
        raise OperationResourceAdmissionError(
            location=("partition",),
            code="groups.characters.table_shape_envelope",
            message="character-table classes or group order exceed the bounded envelope",
        )
    if order == 1:
        row_count = 1
    elif order == 6 and sizes == (1, 3, 2):
        row_count = 3
    elif any(size != 1 for size in sizes):
        raise OperationDomainValidationError(
            location=("partition",),
            code="groups.characters.unsupported_group",
            message=(
                "complete tables are admitted for the trivial, cyclic, and S3 "
                "permutation groups"
            ),
        )
    else:
        row_count = order
    # This is a pre-computation bound for the actual coefficient-vector kernel,
    # not a per-pair admission.  It includes reduction and temporary/output
    # coefficient cells, so C31 is refused before row generation.
    work, output_cells = _character_table_work_bound(
        order=order,
        class_count=class_count,
        row_count=row_count,
        max_class_size=max(sizes, default=1),
    )
    if work > MAX_CHARACTER_TABLE_WORK:
        raise OperationResourceAdmissionError(
            location=("partition",),
            code="groups.characters.table_work_exceeds_envelope",
            message=(
                "complete character-table coefficient work exceeds the "
                f"{MAX_CHARACTER_TABLE_WORK}-unit envelope"
            ),
        )
    if output_cells > MAX_CHARACTER_TABLE_CELLS * MAX_CYCLOTOMIC_ORDER:
        raise OperationResourceAdmissionError(
            location=("partition",),
            code="groups.characters.table_output_exceeds_envelope",
            message=(
                "complete character-table coefficient output exceeds the "
                f"{MAX_CHARACTER_TABLE_CELLS * MAX_CYCLOTOMIC_ORDER}-cell envelope"
            ),
        )
    table_cells = row_count * class_count
    if table_cells > MAX_CHARACTER_TABLE_CELLS:
        raise OperationResourceAdmissionError(
            location=("partition",),
            code="groups.characters.table_output_exceeds_envelope",
            message=(
                "character-table cells exceed the "
                f"{MAX_CHARACTER_TABLE_CELLS}-cell envelope"
            ),
        )
    return row_count


def character_table(
    partition: GroupConjugacyClassesResult,
) -> CharacterTableResult:
    """Return a complete exact table for the bounded cyclic/S3 slice.

    The source partition is complete, so the result remains bound to the
    concrete permutation group and class ordering.  The supported family is
    intentionally explicit: trivial groups, cyclic groups, and S3.  Other
    groups are domain-invalid rather than receiving a guessed partial table.
    """
    if not isinstance(partition, GroupConjugacyClassesResult):
        raise OperationDomainValidationError(
            location=("partition",),
            code="groups.characters.partition_type",
            message="partition must be a complete group class partition",
        )
    from jacobian.math.groups.operations import verify_group_conjugacy_classes

    if not verify_group_conjugacy_classes(partition):
        raise OperationDomainValidationError(
            location=("partition",),
            code="groups.characters.partition_not_group_bound",
            message="class rows must be the complete conjugacy partition of their source group",
        )
    order = sum(len(cls) for cls in partition.classes)
    if order > MAX_CYCLOTOMIC_ORDER:
        raise OperationResourceAdmissionError(
            location=("partition",),
            code="groups.characters.cyclotomic_order_exceeds_envelope",
            message=(
                f"complete character tables admit cyclotomic order at most "
                f"{MAX_CYCLOTOMIC_ORDER}"
            ),
        )
    sizes = tuple(len(cls) for cls in partition.classes)
    _admit_character_table(order=order, sizes=sizes)
    parent = ConjugacyClassPartition._from_group_result(partition)
    rows: list[CharacterRow] = []
    if order == 1:
        rows.append(
            CharacterRow(
                label="trivial", degree=1, values=(_make_value(1, (Fraction(1),)),)
            )
        )
    elif order == 6 and sizes == (1, 3, 2):
        # The class order is identity, transpositions, 3-cycles for the
        # canonical permutation-class ordering.
        rows = [
            CharacterRow(
                label="trivial",
                degree=1,
                values=tuple(_make_value(1, (Fraction(v),)) for v in (1, 1, 1)),
            ),
            CharacterRow(
                label="sign",
                degree=1,
                values=tuple(_make_value(1, (Fraction(v),)) for v in (1, -1, 1)),
            ),
            CharacterRow(
                label="standard",
                degree=2,
                values=tuple(_make_value(1, (Fraction(v),)) for v in (2, 0, -1)),
            ),
        ]
    else:
        generator = _cyclic_generator(partition)
        if generator is None or any(len(cls) != 1 for cls in partition.classes):
            raise OperationDomainValidationError(
                location=("partition",),
                code="groups.characters.unsupported_group",
                message="complete tables are admitted for the trivial, cyclic, and S3 permutation groups",
            )
        # Match each canonical singleton class to a unique generator power.
        powers: list[tuple[int, ...]] = []
        current = tuple(range(partition.source.degree))
        for _ in range(order):
            powers.append(current)
            current = _permutation_compose(current, generator)
        power_index = {element: index for index, element in enumerate(powers)}
        from jacobian.math.groups.characters._cyclotomic import value_from_power

        for exponent in range(order):
            values = []
            for cls in partition.classes:
                power = power_index[tuple(cls[0])]
                values.append(
                    _make_value(order, value_from_power(order, exponent * power))
                )
            rows.append(
                CharacterRow(label=f"chi_{exponent}", degree=1, values=tuple(values))
            )
    if sum(row.degree * row.degree for row in rows) != order:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    # Independent orthogonality replay over the complete class axis. This is
    # a defining invariant of the returned table, not a generic result check.
    axis = ClassAxis._from_kernel(
        class_sizes=sizes, cyclotomic_order=rows[0].values[0].order
    )
    functions = tuple(FiniteClassFunction(axis=axis, values=row.values) for row in rows)
    for left_index, left in enumerate(functions):
        for right_index, right in enumerate(functions):
            pairing = _class_function_inner_product_kernel(left, right)
            expected = 1 if left_index == right_index else 0
            coefficients = tuple(
                value.as_fraction() for value in pairing.inner_product.coefficients
            )
            if coefficients[0] != expected or any(
                value != 0 for value in coefficients[1:]
            ):
                raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    return CharacterTableResult._from_kernel(partition=parent, rows=tuple(rows))


def _class_function_inner_product_kernel(
    phi: FiniteClassFunction,
    psi: FiniteClassFunction,
) -> ClassFunctionInnerProductResult:
    """Compute one already-admitted inner product without replaying admission."""
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


def class_function_inner_product(
    phi: FiniteClassFunction,
    psi: FiniteClassFunction,
) -> ClassFunctionInnerProductResult:
    """Exact Hermitian inner product with a complete class contribution table."""
    _admit_inner_product(phi, psi)
    return _class_function_inner_product_kernel(phi, psi)


__all__ = ["character_table", "class_function_inner_product"]
