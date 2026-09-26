"""Exact finite class-function operations."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import cast

from pydantic import ValidationError

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
from jacobian._execution import BackendFailureReason, OperationBackendError
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups._models import (
    MAX_CONJUGACY_CLASSES_GROUP_ORDER,
    MAX_GROUP_DEGREE,
    GroupConjugacyClassesResult,
    PermutationGroup,
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
    MAX_CHARACTER_TABLE_CELLS,
    MAX_CLASS_COUNT,
    MAX_CYCLOTOMIC_ORDER,
    MAX_GROUP_ORDER,
    MAX_INNER_PRODUCT_WORK,
    MAX_VALUE_COEFFICIENT_DIGITS,
    CharacterRow,
    CharacterTableResult,
    CharacterTensorDecompositionRequest,
    CharacterTensorDecompositionResult,
    ClassAxis,
    ClassContribution,
    ClassFunctionInductionResult,
    ClassFunctionInnerProductResult,
    ClassFunctionRestrictionResult,
    ClassPowerMapResult,
    ConjugacyClassPartition,
    CyclicCharacterRestrictionResult,
    CyclotomicValue,
    FiniteClassFunction,
    FrobeniusSchurIndicatorResult,
)

_MAX_ORDER_EIGHT_CHARACTER_PRODUCTS = 512


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


def _reject_derived_height(
    stage: str,
    height: _CoefficientHeight,
    *,
    code: str = "groups.characters.inner_product_output_digits_exceed_envelope",
) -> None:
    if height.maximum_digits > MAX_VALUE_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("phi", "values"),
            code=code,
            message=(
                f"derived {stage} coefficients exceed the "
                f"{MAX_VALUE_COEFFICIENT_DIGITS}-digit envelope"
            ),
        )


def _admit_class_function(
    value: object, *, location: tuple[str, ...], name: str
) -> FiniteClassFunction:
    """Enforce the class-function contract at a native boundary.

    A ``model_construct`` value can bypass the declared axis/value and
    cyclotomic-degree contracts, so every native consumer re-validates its
    class-function inputs before relying on their structure.  The checks are
    explicit rather than a pydantic round-trip: a merely over-envelope but
    self-consistent value remains a resource admission for the caller, while
    a structurally malformed value is domain-invalid.  A group-bound axis also
    validates its concrete parent group and representative coordinates.
    """

    def reject() -> OperationDomainValidationError:
        return OperationDomainValidationError(
            location=location,
            code="groups.characters.invalid_class_function",
            message=f"{name} has malformed axis or exact cyclotomic values",
        )

    if not isinstance(value, FiniteClassFunction):
        raise OperationDomainValidationError(
            location=location,
            code="groups.characters.class_function_type",
            message=f"{name} must be an exact finite class-function value",
        )
    axis = value.axis
    values = value.values
    if not isinstance(axis, ClassAxis) or not isinstance(values, tuple):
        raise reject()
    class_sizes = axis.class_sizes
    order = axis.cyclotomic_order
    group_order = axis.group_order
    if (
        not isinstance(class_sizes, tuple)
        or not class_sizes
        or type(order) is not int
        or order < 1
        or type(group_order) is not int
        or any(type(size) is not int or size < 1 for size in class_sizes)
        or sum(class_sizes) != group_order
        or len(values) != len(class_sizes)
    ):
        raise reject()
    degree = euler_phi(order) if order <= MAX_CYCLOTOMIC_ORDER else None
    for one in values:
        if (
            not isinstance(one, CyclotomicValue)
            or one.order != order
            or not isinstance(one.coefficients, tuple)
            or (degree is not None and len(one.coefficients) != degree)
            or any(
                not isinstance(coefficient, CanonicalRational)
                or type(coefficient.num) is not int
                or type(coefficient.den) is not int
                or coefficient.den < 1
                for coefficient in one.coefficients
            )
        ):
            raise reject()
    if (axis.group is None) != (axis.class_representatives is None):
        raise reject()
    if axis.group is not None:
        group = _admit_permutation_group(
            axis.group, location=(*location, "axis", "group")
        )
        representatives = axis.class_representatives
        if (
            not isinstance(representatives, tuple)
            or len(representatives) != len(class_sizes)
            or any(
                not isinstance(representative, tuple)
                or len(representative) != group.degree
                or any(type(point) is not int for point in representative)
                for representative in representatives
            )
        ):
            raise reject()
    return value


def _admit_permutation_group(
    value: object, *, location: tuple[str, ...]
) -> PermutationGroup:
    """Re-admit a concrete permutation group at a native boundary."""

    if not isinstance(value, PermutationGroup):
        raise OperationDomainValidationError(
            location=location,
            code="groups.characters.permutation_group_type",
            message="group input must be a typed permutation group",
        )
    try:
        return PermutationGroup.model_validate(value.model_dump())
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=location,
            code="groups.characters.permutation_group_shape",
            message="permutation group has malformed degree or generators",
        ) from exc


def _admit_cyclotomic_value(
    value: object, *, location: tuple[str, ...], name: str
) -> CyclotomicValue:
    """Re-admit one exact cyclotomic value at a native boundary."""

    if not isinstance(value, CyclotomicValue):
        raise OperationDomainValidationError(
            location=location,
            code="groups.characters.scalar_type",
            message=f"{name} must be an exact cyclotomic value",
        )
    try:
        return CyclotomicValue.model_validate(value.model_dump())
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=location,
            code="groups.characters.invalid_cyclotomic_value",
            message=f"{name} has malformed cyclotomic coefficients",
        ) from exc


def _admit_inner_product(phi: FiniteClassFunction, psi: FiniteClassFunction) -> None:
    """Shared native/catalog admission for the Hermitian inner product."""

    phi = _admit_class_function(phi, location=("phi",), name="phi")
    psi = _admit_class_function(psi, location=("psi",), name="psi")
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


def _order_eight_sign_map(
    multiplication: tuple[tuple[int, ...], ...],
    identity: int,
    generators: tuple[int, int],
    generator_signs: tuple[int, int],
) -> tuple[int, ...] | None:
    values: list[int | None] = [None] * 8
    values[identity] = 1
    pending = [identity]
    while pending:
        current = pending.pop()
        current_sign = values[current]
        if current_sign is None:
            raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
        for generator, generator_sign in zip(generators, generator_signs, strict=True):
            product = multiplication[current][generator]
            proposed_sign = current_sign * generator_sign
            if values[product] is None:
                values[product] = proposed_sign
                pending.append(product)
            elif values[product] != proposed_sign:
                return None
    if any(value is None for value in values):
        return None
    return tuple(value for value in values if value is not None)


def _order_eight_linear_characters(
    partition: GroupConjugacyClassesResult,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate the four linear characters of a nonabelian group of order 8.

    Two noncommuting elements generate every nonabelian group of order 8:
    their generated subgroup is nonabelian, while every group of order at most
    4 is abelian. Each character is determined by its values on that pair.
    """
    elements = tuple(
        sorted(tuple(element) for cls in partition.classes for element in cls)
    )
    element_index = {element: index for index, element in enumerate(elements)}
    identity_index = element_index.get(tuple(range(partition.source.degree)))
    if len(elements) != 8 or identity_index is None:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    multiplication = tuple(
        tuple(
            element_index.get(_permutation_compose(left, right), -1)
            for right in elements
        )
        for left in elements
    )
    if any(index < 0 for row in multiplication for index in row):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)

    generators = next(
        (
            (left, right)
            for left in range(8)
            for right in range(left + 1, 8)
            if multiplication[left][right] != multiplication[right][left]
        ),
        None,
    )
    if generators is None:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)

    rows: set[tuple[int, ...]] = set()
    for first_sign in (-1, 1):
        for second_sign in (-1, 1):
            complete = _order_eight_sign_map(
                multiplication,
                identity_index,
                generators,
                (first_sign, second_sign),
            )
            if complete is None:
                continue
            if not all(
                complete[multiplication[left][right]]
                == complete[left] * complete[right]
                for left in range(8)
                for right in range(8)
            ):
                continue
            rows.add(
                tuple(
                    complete[element_index[tuple(cls[0])]] for cls in partition.classes
                )
            )

    ordered_rows = tuple(sorted(rows, key=lambda row: tuple(-value for value in row)))
    if len(ordered_rows) != 4 or ordered_rows[0] != (1,) * len(partition.classes):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    return ordered_rows


def _admit_character_table(
    *, order: int, class_count: int, cyclotomic_order: int, row_count: int
) -> None:
    """Admit complete table construction and exact output together.

    The cyclic construction reduces each distinct root-of-unity power once;
    the output cell count separately bounds the retained table. Admission
    charges both before any row or contribution is materialized.
    """
    table_cells = row_count * class_count
    if table_cells > MAX_CHARACTER_TABLE_CELLS:
        raise OperationResourceAdmissionError(
            location=("partition",),
            code="groups.characters.table_cells_exceed_envelope",
            message=(
                "complete character-table cells exceed the "
                f"{MAX_CHARACTER_TABLE_CELLS:,}-cell envelope"
            ),
        )
    # Each cell carries phi(order) exact coefficients.  This is intentionally
    # separate from the source group-order bound: a compact group can still
    # produce an oversized serialized table.
    output_cells = table_cells * euler_phi(cyclotomic_order)
    if output_cells > MAX_CHARACTER_TABLE_CELLS:
        raise OperationResourceAdmissionError(
            location=("partition",),
            code="groups.characters.table_output_exceeds_envelope",
            message="complete character-table exact output exceeds its envelope",
        )


def _admit_partition_source(source: object) -> tuple[int, tuple[tuple[int, ...], ...]]:
    if not isinstance(source, PermutationGroup):
        raise OperationDomainValidationError(
            location=("partition", "source"),
            code="groups.characters.partition_source",
            message="partition source must be a typed permutation group",
        )
    degree = getattr(source, "degree", None)
    generators = getattr(source, "generators", None)
    if type(degree) is not int or not 1 <= degree <= MAX_GROUP_DEGREE:
        raise OperationDomainValidationError(
            location=("partition", "source"),
            code="groups.characters.partition_source",
            message="partition source degree is malformed",
        )
    if (
        not isinstance(generators, tuple)
        or not 1 <= len(generators) <= MAX_GROUP_DEGREE
    ):
        raise OperationDomainValidationError(
            location=("partition", "source"),
            code="groups.characters.partition_source",
            message="partition source generators are malformed",
        )
    for generator in generators:
        if (
            not isinstance(generator, tuple)
            or len(generator) != degree
            or any(type(value) is not int for value in generator)
            or sorted(generator) != list(range(degree))
        ):
            raise OperationDomainValidationError(
                location=("partition", "source"),
                code="groups.characters.partition_source",
                message="partition source generators are malformed",
            )
    return degree, generators


def _admit_partition_classes(
    classes: object, degree: int
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if not isinstance(classes, tuple) or not 1 <= len(classes) <= MAX_CLASS_COUNT:
        raise OperationDomainValidationError(
            location=("partition", "classes"),
            code="groups.characters.partition_shape",
            message="partition classes are malformed",
        )
    total = 0
    previous_class: tuple[tuple[int, ...], ...] | None = None
    seen: set[tuple[int, ...]] = set()
    for conjugacy_class in classes:
        if not isinstance(conjugacy_class, tuple) or not conjugacy_class:
            raise OperationDomainValidationError(
                location=("partition", "classes"),
                code="groups.characters.partition_shape",
                message="partition classes must be nonempty tuples",
            )
        total += len(conjugacy_class)
        if total > MAX_CONJUGACY_CLASSES_GROUP_ORDER:
            raise OperationResourceAdmissionError(
                location=("partition", "classes"),
                code="groups.characters.partition_over_envelope",
                message="partition contains too many group elements",
            )
        previous_member: tuple[int, ...] | None = None
        for member in conjugacy_class:
            if (
                not isinstance(member, tuple)
                or len(member) != degree
                or any(type(value) is not int for value in member)
                or sorted(member) != list(range(degree))
                or member in seen
                or (previous_member is not None and member <= previous_member)
            ):
                raise OperationDomainValidationError(
                    location=("partition", "classes"),
                    code="groups.characters.partition_shape",
                    message="partition members must be distinct canonical permutations",
                )
            seen.add(member)
            previous_member = member
        if previous_class is not None and conjugacy_class[0] <= previous_class[0]:
            raise OperationDomainValidationError(
                location=("partition", "classes"),
                code="groups.characters.partition_shape",
                message="partition classes must be canonically ordered",
            )
        previous_class = conjugacy_class
    return classes


def _admit_character_partition(partition: object) -> GroupConjugacyClassesResult:
    """Re-admit a complete group partition at this native owner boundary."""
    if not isinstance(partition, GroupConjugacyClassesResult):
        raise OperationDomainValidationError(
            location=("partition",),
            code="groups.characters.partition_type",
            message="partition must be a complete group class partition",
        )
    source = getattr(partition, "source", None)
    classes = getattr(partition, "classes", None)
    degree, generators = _admit_partition_source(source)
    source = cast(PermutationGroup, source)
    classes = _admit_partition_classes(classes, degree)
    from jacobian.math.groups.operations import group_conjugacy_classes, group_order

    try:
        source_order = group_order(source)
    except (TypeError, ValueError, AttributeError, KeyError, IndexError) as exc:
        raise OperationDomainValidationError(
            location=("partition", "source"),
            code="groups.characters.partition_source",
            message="partition source could not be admitted",
        ) from exc
    if source_order > MAX_CONJUGACY_CLASSES_GROUP_ORDER:
        raise OperationResourceAdmissionError(
            location=("partition", "source"),
            code="groups.characters.partition_over_envelope",
            message="source group exceeds the complete class-partition envelope",
        )
    try:
        expected = group_conjugacy_classes(
            degree, [list(generator) for generator in generators]
        )
    except OperationDomainValidationError as exc:
        raise OperationResourceAdmissionError(
            location=("partition", "source"),
            code="groups.characters.partition_over_envelope",
            message="source group exceeds the complete class-partition envelope",
        ) from exc
    actual = [
        [list(member) for member in conjugacy_class] for conjugacy_class in classes
    ]
    if expected != actual:
        raise OperationDomainValidationError(
            location=("partition",),
            code="groups.characters.partition_not_group_bound",
            message="class rows must be the complete conjugacy partition of their source group",
        )
    return partition


def character_table(
    partition: GroupConjugacyClassesResult,
) -> CharacterTableResult:
    """Return a complete exact table for the bounded supported group families.

    The source partition is complete, so the result remains bound to the
    concrete permutation group and class ordering.  The supported family is
    intentionally explicit: trivial groups, cyclic groups, S3, and nonabelian
    groups of order eight. Other groups are domain-invalid rather than
    receiving a guessed partial table.
    """
    partition = _admit_character_partition(partition)
    return _character_table_from_admitted_partition(partition)


def _character_table_from_admitted_partition(
    partition: GroupConjugacyClassesResult,
) -> CharacterTableResult:
    """Build a table from a partition authenticated by the caller."""
    order = sum(len(cls) for cls in partition.classes)
    if (
        order > MAX_GROUP_ORDER
        or len(partition.classes) > MAX_CLASS_COUNT
        or order > MAX_CYCLOTOMIC_ORDER
    ):
        raise OperationResourceAdmissionError(
            location=("partition",),
            code="groups.characters.table_envelope",
            message="character table exceeds the bounded class/group envelope",
        )
    parent = ConjugacyClassPartition._from_group_result(partition)
    sizes = tuple(len(cls) for cls in partition.classes)
    rows: list[CharacterRow] = []
    if order == 1:
        _admit_character_table(
            order=order,
            class_count=len(sizes),
            cyclotomic_order=1,
            row_count=1,
        )
        rows.append(
            CharacterRow(
                label="trivial", degree=1, values=(_make_value(1, (Fraction(1),)),)
            )
        )
    elif order == 6 and len(sizes) == 3 and set(sizes) == {1, 2, 3}:
        _admit_character_table(
            order=order,
            class_count=len(sizes),
            cyclotomic_order=1,
            row_count=3,
        )
        # Canonical conjugacy ordering may permute the transposition and
        # 3-cycle classes for non-natural embeddings; identify by class size.
        labels = ("trivial", "sign", "standard")
        degrees = (1, 1, 2)
        rows = [
            CharacterRow(
                label=label,
                degree=degree,
                values=tuple(
                    _make_value(
                        order,
                        (
                            Fraction(
                                
                                    {"trivial": 1, "sign": 1, "standard": 2}[label]
                                    if size == 1
                                    else {"trivial": 1, "sign": -1, "standard": 0}[
                                        label
                                    ]
                                    if size == 3
                                    else {"trivial": 1, "sign": 1, "standard": -1}[
                                        label
                                    ]
                                
                            ),
                            Fraction(0),
                        ),
                    )
                    for size in sizes
                ),
            )
            for label, degree in zip(labels, degrees, strict=True)
        ]
    elif order == 8 and len(sizes) == 5 and sorted(sizes) == [1, 1, 2, 2, 2]:
        _admit_character_table(
            order=order,
            class_count=len(sizes),
            cyclotomic_order=1,
            row_count=5,
        )
        estimated_products = (8 * 8) + (4 * 8 * 2) + (4 * 8 * 8)
        if estimated_products > _MAX_ORDER_EIGHT_CHARACTER_PRODUCTS:
            raise OperationResourceAdmissionError(
                location=("partition",),
                code="groups.characters.order_eight_work_exceeds_envelope",
                message=(
                    "order-eight character construction exceeds its "
                    f"{_MAX_ORDER_EIGHT_CHARACTER_PRODUCTS}-product envelope"
                ),
            )
        linear_rows = _order_eight_linear_characters(partition)
        identity = tuple(range(partition.source.degree))
        identity_class = next(
            index
            for index, conjugacy_class in enumerate(partition.classes)
            if identity in conjugacy_class
        )
        central_singletons = tuple(
            index for index, size in enumerate(sizes) if size == 1
        )
        if len(central_singletons) != 2:
            raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
        nonidentity_center = next(
            index for index in central_singletons if index != identity_class
        )
        rows.extend(
            CharacterRow(
                label="trivial" if row_index == 0 else f"linear_{row_index}",
                degree=1,
                values=tuple(_make_value(1, (Fraction(value),)) for value in values),
            )
            for row_index, values in enumerate(linear_rows)
        )
        nonlinear_values = tuple(
            _make_value(
                1,
                (
                    Fraction(2)
                    if index == identity_class
                    else Fraction(-2)
                    if index == nonidentity_center
                    else Fraction(0),
                ),
            )
            for index in range(len(sizes))
        )
        rows.append(CharacterRow(label="degree_two", degree=2, values=nonlinear_values))
        # Check the exact orthogonality relations before publishing the table.
        # For nonabelian groups of order 8, the two singleton classes are the
        # identity and the nontrivial central element; the other three classes
        # have size two.
        for row_index, left in enumerate(rows):
            for right_index, right in enumerate(rows[row_index:], start=row_index):
                inner = (
                    sum(
                        (
                            sizes[index]
                            * left.values[index].coefficients[0].as_fraction()
                            * right.values[index].coefficients[0].as_fraction()
                            for index in range(len(sizes))
                        ),
                        Fraction(0),
                    )
                    / order
                )
                if inner != int(row_index == right_index):
                    raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    else:
        generator = _cyclic_generator(partition)
        if generator is None or any(len(cls) != 1 for cls in partition.classes):
            raise OperationDomainValidationError(
                location=("partition",),
                code="groups.characters.unsupported_group",
                message="complete tables are admitted for the trivial, cyclic, and S3 permutation groups",
            )
        _admit_character_table(
            order=order,
            class_count=len(sizes),
            cyclotomic_order=order,
            row_count=order,
        )
        # Match each canonical singleton class to a unique generator power.
        powers: list[tuple[int, ...]] = []
        current = tuple(range(partition.source.degree))
        for _ in range(order):
            powers.append(current)
            current = _permutation_compose(current, generator)
        power_index = {element: index for index, element in enumerate(powers)}
        from jacobian.math.groups.characters._cyclotomic import value_from_power

        # These n powers determine all n^2 Fourier-table entries. Reduce each
        # power once and reuse its immutable exact value across the rows.
        power_values = tuple(
            _make_value(order, value_from_power(order, power)) for power in range(order)
        )
        for exponent in range(order):
            values = []
            for cls in partition.classes:
                power = power_index[tuple(cls[0])]
                values.append(power_values[(exponent * power) % order])
            rows.append(
                CharacterRow(label=f"chi_{exponent}", degree=1, values=tuple(values))
            )
    if sum(row.degree * row.degree for row in rows) != order:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    # The supported branches construct the trivial character, the explicit
    # S3 irreducibles, or the cyclic Fourier characters. Their formulas give
    # orthogonality directly; pairwise inner products remain owner-test
    # evidence rather than repeated production work.
    table_axis = ClassAxis._from_kernel(
        class_sizes=sizes,
        cyclotomic_order=rows[0].values[0].order,
        group=parent.source,
        class_representatives=tuple(cls[0] for cls in parent.classes),
    )
    return CharacterTableResult._from_kernel(
        partition=parent,
        rows=tuple(rows),
        axis=table_axis,
    )


MAX_CHARACTER_TENSOR_INNER_WORK = 50_000_000
MAX_CHARACTER_TENSOR_OUTPUT_BYTES = 2_000_000


def character_tensor_decomposition(
    request: CharacterTensorDecompositionRequest,
) -> CharacterTensorDecompositionResult:
    """Decompose an S3 character tensor product in its canonical irreducibles."""
    if not isinstance(request, CharacterTensorDecompositionRequest):
        raise OperationDomainValidationError(
            location=("request",),
            code="groups.characters.tensor_request_type",
            message="request must be a character tensor-decomposition request",
        )
    request = CharacterTensorDecompositionRequest.model_validate(request.model_dump())
    partition = _admit_character_partition(request.partition)
    table = _character_table_from_admitted_partition(partition)
    if (
        table.axis.group_order != 6
        or table.axis.class_sizes != (1, 3, 2)
        or table.axis.cyclotomic_order != 6
        or tuple(row.label for row in table.rows) != ("trivial", "sign", "standard")
    ):
        raise OperationDomainValidationError(
            location=("partition",),
            code="groups.characters.tensor_group_unsupported",
            message="tensor decomposition currently supports the canonical S3 table",
        )
    row_count = len(table.rows)
    if request.left_row_index >= row_count or request.right_row_index >= row_count:
        raise OperationDomainValidationError(
            location=("row_index",),
            code="groups.characters.tensor_row_index",
            message="tensor-product row index is outside the canonical table",
        )

    axis = table.axis
    left = FiniteClassFunction._from_kernel(
        axis=axis, values=table.rows[request.left_row_index].values
    )
    right = FiniteClassFunction._from_kernel(
        axis=axis, values=table.rows[request.right_row_index].values
    )
    product_axis, order, dimension = _admit_pointwise_axis(left, right)
    pointwise_inputs = _admit_pointwise_inputs(
        left, right, class_count=3, dimension=dimension
    )
    product_heights = _admit_pointwise_output(
        pointwise_inputs, dimension=dimension, output_cells=3 * dimension
    )
    if any(height.denominator != 1 for height in product_heights):
        raise OperationResourceAdmissionError(
            location=("partition",),
            code="groups.characters.tensor_product_height",
            message="canonical S3 product estimate unexpectedly has a denominator",
        )
    predicted_product = FiniteClassFunction._from_kernel(
        axis=product_axis,
        values=tuple(
            _make_value(
                order,
                (
                    Fraction(10 ** max(1, height.lifted_numerator_digits) - 1),
                    Fraction(0),
                ),
            )
            for height in product_heights
        ),
    )

    aggregate_inner_work = 0
    for row in table.rows:
        basis_function = FiniteClassFunction._from_kernel(axis=axis, values=row.values)
        _admit_inner_product(predicted_product, basis_function)
        maximum_digits = max(
            canonical_rational_component_digits(coefficient)
            for value in (*predicted_product.values, *basis_function.values)
            for coefficient in value.coefficients
        )
        aggregate_inner_work += 3 * order * maximum_digits * maximum_digits
    product_max_digits = max(
        canonical_rational_component_digits(coefficient)
        for value in (*left.values, *right.values)
        for coefficient in value.coefficients
    )
    aggregate_work = (
        3 * dimension * dimension * product_max_digits * product_max_digits
        + aggregate_inner_work
    )
    if aggregate_work > MAX_CHARACTER_TENSOR_INNER_WORK:
        raise OperationResourceAdmissionError(
            location=("partition",),
            code="groups.characters.tensor_work_exceeds_envelope",
            message="tensor-product and multiplicity arithmetic exceeds its work envelope",
        )

    # At most three copies of the bounded degree-256 permutation group are
    # retained. Values use at most 512 digits; this estimate stays below both
    # the operation's 2 MB cap and the canonical 10 MB transport limit.
    group = axis.group
    assert group is not None
    degree = group.degree
    generator_count = len(group.generators)
    group_bytes = 512 + generator_count * (degree * 8 + 16)
    structure_bytes = 3 * group_bytes + 12 * degree * 8 + 128_000
    value_bytes = 12 * dimension * (2 * MAX_VALUE_COEFFICIENT_DIGITS + 24)
    if structure_bytes + value_bytes > MAX_CHARACTER_TENSOR_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("partition",),
            code="groups.characters.tensor_output_exceeds_envelope",
            message="tensor-decomposition result exceeds its bounded output envelope",
        )

    # Whole-operation admission is complete before any product or pairing.
    tensor_values = tuple(
        _make_value(order, multiply_values(order, _fractions(a), _fractions(b)))
        for a, b in zip(left.values, right.values, strict=True)
    )
    tensor_product = FiniteClassFunction._from_kernel(axis=axis, values=tensor_values)
    multiplicities: list[int] = []
    for row in table.rows:
        total = zero_value(order)
        for class_size, tensor_value, row_value in zip(
            axis.class_sizes, tensor_values, row.values, strict=True
        ):
            conjugate = conjugate_value(order, _fractions(row_value))
            product = multiply_values(order, _fractions(tensor_value), conjugate)
            weighted = scale_value(order, Fraction(class_size), product)
            total = add_values(order, total, weighted)
        inner = _make_value(
            order,
            tuple(coefficient / axis.group_order for coefficient in total),
        )
        if (
            inner.coefficients[0].den != 1
            or inner.coefficients[0].num < 0
            or any(coefficient.num != 0 for coefficient in inner.coefficients[1:])
        ):
            raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
        multiplicities.append(inner.coefficients[0].num)
    if any(value > 4 for value in multiplicities):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    return CharacterTensorDecompositionResult._from_kernel(
        table=table,
        left_row_index=request.left_row_index,
        right_row_index=request.right_row_index,
        tensor_product=tensor_product,
        multiplicities=tuple(multiplicities),
    )


def restrict_cyclic_character(
    partition: GroupConjugacyClassesResult,
    row_index: int,
    subgroup_order: int,
) -> CyclicCharacterRestrictionResult:
    """Restrict an irreducible character of C_n to its unique subgroup C_d."""
    partition = _admit_character_partition(partition)
    order = sum(len(cls) for cls in partition.classes)
    if order > MAX_CYCLOTOMIC_ORDER or len(partition.classes) != order:
        raise OperationDomainValidationError(
            location=("partition",),
            code="groups.characters.restriction_requires_cyclic_group",
            message="this restriction operation admits only supported cyclic groups",
        )
    if (
        type(subgroup_order) is not int
        or not 1 <= subgroup_order <= MAX_CYCLOTOMIC_ORDER
    ):
        raise OperationDomainValidationError(
            location=("subgroup_order",),
            code="groups.characters.subgroup_order_not_divisor",
            message="subgroup_order must divide the cyclic source group order",
        )
    if order % subgroup_order:
        raise OperationDomainValidationError(
            location=("subgroup_order",),
            code="groups.characters.subgroup_order_not_divisor",
            message="subgroup_order must divide the cyclic source group order",
        )
    target_cells = subgroup_order * euler_phi(order)
    if target_cells > MAX_CHARACTER_TABLE_CELLS:
        raise OperationResourceAdmissionError(
            location=("subgroup_order",),
            code="groups.characters.restriction_output_exceeds_envelope",
            message="restricted character output exceeds its coefficient-cell envelope",
        )
    estimated_work = _cyclic_restriction_work(
        order, partition.source.degree, subgroup_order
    )
    if estimated_work > MAX_CYCLIC_RESTRICTION_WORK:
        raise OperationResourceAdmissionError(
            location=("partition",),
            code="groups.characters.restriction_work_exceeds_envelope",
            message=(
                "cyclic subgroup restriction work exceeds its "
                f"{MAX_CYCLIC_RESTRICTION_WORK:,}-unit envelope"
            ),
        )
    if type(row_index) is not int or not 0 <= row_index < order:
        raise OperationDomainValidationError(
            location=("row_index",),
            code="groups.characters.restriction_row_out_of_range",
            message="row_index must select an irreducible row in the complete table",
        )
    table = character_table(partition)
    # The unique C_d in C_n is {g : g^d = 1}, independent of a choice of
    # generator. The source and target class representatives retain inclusion.
    identity = tuple(range(partition.source.degree))
    source_classes = partition.classes
    source_index = {
        tuple(element): index
        for index, conjugacy_class in enumerate(source_classes)
        for element in conjugacy_class
    }
    subgroup_elements = tuple(
        sorted(
            element
            for element in source_index
            if _permutation_power(element, subgroup_order) == identity
        )
    )
    if len(subgroup_elements) != subgroup_order:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    target_group = PermutationGroup(
        degree=partition.source.degree,
        generators=subgroup_elements,
    )
    # C_d is abelian, so every class is a singleton. subgroup_elements are
    # lexicographically sorted, which is exactly the canonical class order:
    # each singleton is sorted and its representative sequence is increasing.
    target_classes_data = tuple((element,) for element in subgroup_elements)
    target_partition = ConjugacyClassPartition._from_group_result(
        GroupConjugacyClassesResult._from_kernel(
            target_group,
            target_classes_data,
        )
    )
    class_map = tuple(
        source_index[tuple(target_class[0])]
        for target_class in target_partition.classes
    )
    target_axis = ClassAxis._from_kernel(
        class_sizes=tuple(len(cls) for cls in target_partition.classes),
        cyclotomic_order=table.axis.cyclotomic_order,
        group=target_group,
        class_representatives=tuple(cls[0] for cls in target_partition.classes),
    )
    row = table.rows[row_index]
    restricted = FiniteClassFunction._from_kernel(
        axis=target_axis,
        values=tuple(row.values[index] for index in class_map),
    )
    return CyclicCharacterRestrictionResult(
        source_table=table,
        target_partition=target_partition,
        row_index=row_index,
        source_class_indices=class_map,
        restricted_character=restricted,
    )


def _permutation_power(permutation: tuple[int, ...], exponent: int) -> tuple[int, ...]:
    result = tuple(range(len(permutation)))
    base = permutation
    while exponent:
        if exponent & 1:
            result = _permutation_compose(result, base)
        exponent >>= 1
        if exponent:
            base = _permutation_compose(base, base)
    return result


MAX_CYCLIC_RESTRICTION_WORK = 1_000_000


def _cyclic_restriction_work(order: int, degree: int, subgroup_order: int) -> int:
    """Conservative coordinate/cell bound for restriction after source admission.

    Charge binary exponentiation for every source element; scanning/indexing the
    source; quadratic coordinate comparisons as an upper bound for ordering the
    target elements; target parent/class/map construction and validation; and
    every exact coefficient cell in the restricted row. The degree factors
    charge permutation-coordinate operations, including pydantic's permutation
    validation sorts.
    """
    exponent_bits = max(1, subgroup_order.bit_length())
    degree_bits = max(1, degree.bit_length())
    permutation_coordinate_work = degree * (
        2 * order * exponent_bits
        + 3 * order
        + subgroup_order * subgroup_order
        + 4 * subgroup_order
    )
    permutation_validation_work = 2 * subgroup_order * degree * degree_bits
    output_cells = subgroup_order * euler_phi(order)
    return permutation_coordinate_work + permutation_validation_work + output_cells


MAX_CLASS_POWER_MAP_EXPONENT = 1_000_000


def class_power_map(
    partition: GroupConjugacyClassesResult, exponent: int
) -> ClassPowerMapResult:
    """Compute the class image induced by ``g -> g**k``."""
    from jacobian.math.groups.operations import group_order

    if not isinstance(partition, GroupConjugacyClassesResult):
        raise OperationDomainValidationError(
            location=("partition",),
            code="groups.characters.partition_type",
            message="partition must be a complete group class partition",
        )
    if type(exponent) is not int or not 1 <= exponent <= MAX_CLASS_POWER_MAP_EXPONENT:
        raise OperationDomainValidationError(
            location=("exponent",),
            code="groups.characters.power_map_exponent",
            message=(
                "exponent must be an integer between 1 and "
                f"{MAX_CLASS_POWER_MAP_EXPONENT:,}"
            ),
        )
    _admit_partition_source(partition.source)
    source_order = group_order(partition.source)
    if source_order > 256:
        raise OperationResourceAdmissionError(
            location=("partition", "source"),
            code="groups.characters.power_map_group_order_exceeds_envelope",
            message="class power maps admit groups of order at most 256",
        )
    if len(partition.classes) > MAX_CLASS_COUNT:
        raise OperationResourceAdmissionError(
            location=("partition", "classes"),
            code="groups.characters.power_map_class_count_exceeds_envelope",
            message=f"class power maps admit at most {MAX_CLASS_COUNT} classes",
        )
    partition = _admit_character_partition(partition)
    elements = {
        tuple(element): class_index
        for class_index, conjugacy_class in enumerate(partition.classes)
        for element in conjugacy_class
    }
    identity = tuple(range(partition.source.degree))
    images: list[int] = []
    for conjugacy_class in partition.classes:
        base = tuple(conjugacy_class[0])
        remaining = exponent
        result = identity
        while remaining:
            if remaining & 1:
                result = _permutation_compose(result, base)
            remaining >>= 1
            if remaining:
                base = _permutation_compose(base, base)
        try:
            images.append(elements[result])
        except KeyError as exc:
            raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT) from exc
    parent = ConjugacyClassPartition._from_group_result(partition)
    return ClassPowerMapResult._from_kernel(
        partition=parent,
        exponent=exponent,
        image_class_indices=tuple(images),
    )


def frobenius_schur_indicator(
    table: CharacterTableResult,
    row_index: int,
) -> FrobeniusSchurIndicatorResult:
    """Compute the ordinary second Frobenius-Schur indicator of one row.

    The class formula is nu_2(chi) = |G|^-1 sum_C |C| chi(c^2). This is
    deliberately restricted to ordinary irreducible rows, not modular or
    higher indicators.
    """
    supplied = table
    if not isinstance(supplied, CharacterTableResult):
        raise OperationDomainValidationError(
            location=("table",),
            code="groups.characters.indicator_table_type",
            message="table must be a complete exact character-table value",
        )
    # Rebuild the table from a re-established complete group partition. This
    # checks caller-supplied rows against the exact-table owner's contract.
    partition = _admit_character_partition(supplied.partition)
    table = character_table(partition)
    if supplied != table:
        raise OperationDomainValidationError(
            location=("table",),
            code="groups.characters.indicator_incomplete_table",
            message="indicator input must be the complete canonical table of its group",
        )
    if type(row_index) is not int or not 0 <= row_index < len(table.rows):
        raise OperationDomainValidationError(
            location=("row_index",),
            code="groups.characters.indicator_row_out_of_range",
            message="row_index must select one irreducible row of the complete table",
        )
    order = table.axis.group_order
    class_count = len(table.partition.classes)
    if order > 256 or class_count > MAX_CLASS_COUNT:
        raise OperationResourceAdmissionError(
            location=("table",),
            code="groups.characters.indicator_work_exceeds_envelope",
            message="ordinary second indicators admit groups of order at most 256",
        )
    row = table.rows[row_index]
    value_order = table.axis.cyclotomic_order
    if value_order > MAX_ARITHMETIC_ORDER:
        raise OperationResourceAdmissionError(
            location=("table", "axis", "cyclotomic_order"),
            code="groups.characters.arithmetic_order_exceeds_envelope",
            message="exact cyclotomic arithmetic order exceeds its envelope",
        )
    # Bound class lookups, cyclotomic coefficient cells, and exact coefficient
    # growth before requesting the power map or accumulating the sum.
    degree = table.partition.source.degree
    work = order + class_count * degree + class_count * euler_phi(value_order)
    if work > MAX_INNER_PRODUCT_WORK:
        raise OperationResourceAdmissionError(
            location=("table",),
            code="groups.characters.indicator_work_exceeds_envelope",
            message="ordinary second indicator work exceeds its exact envelope",
        )
    denominators = tuple(
        coefficient.den for value in row.values for coefficient in value.coefficients
    )
    common_denominator = _bounded_lcm(denominators)
    if common_denominator is None:
        raise OperationResourceAdmissionError(
            location=("table", "rows", row_index),
            code="groups.characters.indicator_height_exceeds_envelope",
            message="indicator input coefficient denominators exceed the height envelope",
        )
    max_scaled_digits = max(
        len(str(abs(coefficient.num)))
        + _factor_digits(common_denominator // coefficient.den)
        for value in row.values
        for coefficient in value.coefficients
    )
    weighted_digits = max_scaled_digits + _factor_digits(order) + len(str(class_count))
    if weighted_digits > MAX_VALUE_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("table", "rows", row_index),
            code="groups.characters.indicator_height_exceeds_envelope",
            message="indicator exact coefficient growth exceeds its height envelope",
        )
    class_index_by_element = {
        tuple(element): index
        for index, conjugacy_class in enumerate(table.partition.classes)
        for element in conjugacy_class
    }
    square_class_indices: list[int] = []
    for conjugacy_class in table.partition.classes:
        representative = tuple(conjugacy_class[0])
        square = _permutation_compose(representative, representative)
        try:
            square_class_indices.append(class_index_by_element[square])
        except KeyError as exc:
            raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT) from exc
    sums = [Fraction(0) for _ in range(euler_phi(value_order))]
    for class_index, target_index in enumerate(square_class_indices):
        class_size = table.axis.class_sizes[class_index]
        for coefficient_index, coefficient in enumerate(
            _fractions(row.values[target_index])
        ):
            sums[coefficient_index] += class_size * coefficient
    result_coefficients = tuple(coefficient / order for coefficient in sums)
    if any(result_coefficients[index] for index in range(1, len(result_coefficients))):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    scalar = result_coefficients[0]
    if scalar.denominator != 1 or scalar not in (
        Fraction(-1),
        Fraction(0),
        Fraction(1),
    ):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    return FrobeniusSchurIndicatorResult(
        source_table=table, row_index=row_index, indicator=scalar.numerator
    )


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


def class_function_restrict_to_subgroup(
    class_function: FiniteClassFunction,
    subgroup: PermutationGroup,
) -> ClassFunctionRestrictionResult:
    """Restrict one exact class function along an explicit subgroup inclusion."""
    from sympy.combinatorics import Permutation

    from jacobian.math.groups.operations import group_conjugacy_classes, group_order

    function = _admit_class_function(
        class_function, location=("class_function",), name="class_function"
    )
    subgroup = _admit_permutation_group(subgroup, location=("subgroup",))
    axis = function.axis
    source = axis.group
    if source is None or axis.class_representatives is None:
        raise OperationDomainValidationError(
            location=("class_function", "axis"),
            code="groups.characters.restriction_requires_group",
            message="restriction requires a class axis bound to a concrete group",
        )
    if subgroup.degree != source.degree:
        raise OperationDomainValidationError(
            location=("subgroup", "degree"),
            code="groups.characters.restriction_degree",
            message="subgroup generators must use the source permutation domain",
        )

    source_order = group_order(source)
    subgroup_order = group_order(subgroup)
    restriction_order_bound = 256
    if (
        source_order > restriction_order_bound
        or subgroup_order > restriction_order_bound
    ):
        raise OperationResourceAdmissionError(
            location=("class_function", "axis", "group"),
            code="groups.characters.restriction_order_bound",
            message=(
                "class-function restriction admits source and subgroup orders "
                f"at most {restriction_order_bound}"
            ),
        )
    if subgroup_order > MAX_CLASS_COUNT:
        raise OperationResourceAdmissionError(
            location=("subgroup",),
            code="groups.characters.restriction_class_bound",
            message=(
                f"subgroup order exceeds the {MAX_CLASS_COUNT}-class output envelope"
            ),
        )
    if subgroup_order > source_order:
        raise OperationDomainValidationError(
            location=("subgroup",),
            code="groups.characters.restriction_not_subgroup",
            message="subgroup order cannot exceed source group order",
        )

    partition_work_bound = (
        source_order * len(source.generators) * source.degree
        + subgroup_order * len(subgroup.generators) * subgroup.degree
        + len(subgroup.generators) * source_order * source.degree
    )
    partition_work_limit = 3_150_000
    largest_component_digits = max(
        canonical_rational_component_digits(coefficient)
        for value in function.values
        for coefficient in value.coefficients
    )
    output_cells = (
        len(function.values) + min(subgroup_order, MAX_CLASS_COUNT)
    ) * euler_phi(axis.cyclotomic_order)
    if (
        partition_work_bound > partition_work_limit
        or output_cells > MAX_CHARACTER_TABLE_CELLS
        or largest_component_digits > MAX_VALUE_COEFFICIENT_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=("subgroup",),
            code="groups.characters.restriction_work_bound",
            message=(
                "restricted class partitions or exact class-function values exceed "
                "the admitted work/output envelope"
            ),
        )

    from jacobian.math.groups.operations import _backend_group

    ambient_backend = _backend_group(source)
    for index, generator in enumerate(subgroup.generators):
        if not ambient_backend.contains(Permutation(list(generator)), strict=True):
            raise OperationDomainValidationError(
                location=("subgroup", "generators", index),
                code="groups.characters.restriction_not_subgroup",
                message="every subgroup generator must belong to the source group",
            )

    source_classes = group_conjugacy_classes(
        source.degree, [list(generator) for generator in source.generators]
    )
    if (
        len(source_classes) != len(axis.class_sizes)
        or tuple(len(cls) for cls in source_classes) != axis.class_sizes
        or tuple(tuple(cls[0]) for cls in source_classes) != axis.class_representatives
        or axis.group_order != source_order
    ):
        raise OperationDomainValidationError(
            location=("class_function", "axis"),
            code="groups.characters.restriction_axis_mismatch",
            message=(
                "class-function axis must match the canonical source conjugacy "
                "classes and group order"
            ),
        )

    target_classes = group_conjugacy_classes(
        subgroup.degree, [list(generator) for generator in subgroup.generators]
    )
    source_class_index = {
        tuple(element): index
        for index, conjugacy_class in enumerate(source_classes)
        for element in conjugacy_class
    }
    class_map: list[int] = []
    restricted_values = []
    for target_class in target_classes:
        source_index = source_class_index.get(tuple(target_class[0]))
        if source_index is None:
            raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
        class_map.append(source_index)
        restricted_values.append(function.values[source_index])

    target_result = GroupConjugacyClassesResult._from_kernel(
        subgroup,
        tuple(tuple(tuple(element) for element in cls) for cls in target_classes),
    )
    target_partition = ConjugacyClassPartition._from_group_result(target_result)
    target_axis = ClassAxis._from_kernel(
        class_sizes=tuple(len(cls) for cls in target_classes),
        cyclotomic_order=axis.cyclotomic_order,
        group=subgroup,
        class_representatives=tuple(tuple(cls[0]) for cls in target_classes),
    )
    restricted = FiniteClassFunction._from_kernel(
        axis=target_axis, values=tuple(restricted_values)
    )
    return ClassFunctionRestrictionResult._from_kernel(
        source_class_function=function,
        subgroup_partition=target_partition,
        target_class_to_source_class=tuple(class_map),
        restricted=restricted,
    )


MAX_CLASS_FUNCTION_INDUCTION_WORK = 50_000_000


def _admit_class_function_induction(
    class_function: FiniteClassFunction,
    parent_group: PermutationGroup,
) -> tuple[FiniteClassFunction, PermutationGroup, int, int]:
    from jacobian.math.groups.operations import group_order

    function = _admit_class_function(
        class_function, location=("class_function",), name="class_function"
    )
    parent = _admit_permutation_group(parent_group, location=("parent_group",))
    subgroup = function.axis.group
    axis = function.axis
    if subgroup is None or axis.class_representatives is None:
        raise OperationDomainValidationError(
            location=("class_function", "axis"),
            code="groups.characters.induction_requires_group",
            message="induction requires a class axis bound to a concrete subgroup",
        )
    if subgroup.degree != parent.degree:
        raise OperationDomainValidationError(
            location=("parent_group", "degree"),
            code="groups.characters.induction_degree",
            message="subgroup and parent must use the same permutation domain",
        )

    subgroup_order = group_order(subgroup)
    parent_order = group_order(parent)
    if subgroup_order > MAX_CLASS_COUNT or parent_order > 256:
        raise OperationResourceAdmissionError(
            location=("class_function", "axis", "group"),
            code="groups.characters.induction_group_order_bound",
            message=(
                f"induction admits subgroup order at most {MAX_CLASS_COUNT} "
                "and parent order at most 256"
            ),
        )
    if subgroup_order > parent_order:
        raise OperationDomainValidationError(
            location=("parent_group",),
            code="groups.characters.induction_not_subgroup",
            message="subgroup order cannot exceed parent group order",
        )

    degree = parent.degree
    subgroup_generator_count = len(subgroup.generators)
    parent_generator_count = len(parent.generators)
    group_work_bound = (
        subgroup_order * subgroup_generator_count * degree
        + parent_order * parent_generator_count * degree
        + subgroup_generator_count * parent_order * degree
    )
    order = axis.cyclotomic_order
    dimension = euler_phi(order)
    denominator_work_bound = (
        len(function.values)
        * dimension
        * max(1, (2 * MAX_VALUE_COEFFICIENT_DIGITS + 31) // 32) ** 2
    )
    if denominator_work_bound > MAX_CLASS_FUNCTION_INDUCTION_WORK:
        raise OperationResourceAdmissionError(
            location=("class_function", "values"),
            code="groups.characters.induction_work_bound",
            message=(
                "class-function denominator analysis exceeds its "
                f"{MAX_CLASS_FUNCTION_INDUCTION_WORK}-unit work envelope"
            ),
        )

    largest_output_digits = _admit_induction_coefficient_growth(
        function, parent_order=parent_order, subgroup_order=subgroup_order
    )
    output_limbs = max(1, (largest_output_digits + 31) // 32)
    # At most |G| exact additions are made for each parent class.
    arithmetic_work = (
        denominator_work_bound
        + MAX_CLASS_COUNT * parent_order * dimension * output_limbs**2
    )
    if group_work_bound + arithmetic_work > MAX_CLASS_FUNCTION_INDUCTION_WORK:
        raise OperationResourceAdmissionError(
            location=("class_function", "values"),
            code="groups.characters.induction_work_bound",
            message=(
                "class-function induction exceeds its "
                f"{MAX_CLASS_FUNCTION_INDUCTION_WORK}-unit work envelope"
            ),
        )
    output_cells = (len(function.values) + MAX_CLASS_COUNT) * dimension
    if (
        output_cells > MAX_CHARACTER_TABLE_CELLS
        or largest_output_digits > MAX_VALUE_COEFFICIENT_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=("class_function", "values"),
            code="groups.characters.induction_output_bound",
            message="predicted induced class function exceeds the output envelope",
        )
    return function, parent, subgroup_order, parent_order


def _admit_induction_coefficient_growth(
    function: FiniteClassFunction,
    *,
    parent_order: int,
    subgroup_order: int,
) -> int:
    dimension = euler_phi(function.axis.cyclotomic_order)
    largest_output_digits = 1
    for coordinate in range(dimension):
        denominators = tuple(
            value.coefficients[coordinate].den for value in function.values
        )
        common_denominator = _bounded_lcm(denominators)
        if common_denominator is None:
            raise OperationResourceAdmissionError(
                location=("class_function", "values"),
                code="groups.characters.induction_denominator_bound",
                message=(
                    "the common input coefficient denominator exceeds the "
                    "exact arithmetic envelope"
                ),
            )
        lifted_numerator_digits = max(
            len(
                str(
                    abs(value.coefficients[coordinate].num)
                    * (common_denominator // value.coefficients[coordinate].den)
                )
            )
            for value in function.values
        )
        numerator_digits = lifted_numerator_digits + len(str(parent_order))
        output_denominator = _bounded_product(common_denominator, subgroup_order)
        if output_denominator is None:
            raise OperationResourceAdmissionError(
                location=("class_function", "values"),
                code="groups.characters.induction_denominator_bound",
                message="the predicted induced-value denominator exceeds its envelope",
            )
        denominator_digits = len(str(output_denominator))
        _reject_derived_height(
            "class-function induction",
            _CoefficientHeight(numerator_digits, denominator_digits),
            code="groups.characters.induction_coefficient_bound",
        )
        largest_output_digits = max(
            largest_output_digits, numerator_digits, denominator_digits
        )
    return largest_output_digits


def _induction_conjugacy_classes(
    function: FiniteClassFunction,
    subgroup: PermutationGroup,
    parent: PermutationGroup,
    subgroup_order: int,
) -> tuple[list[list[list[int]]], list[list[list[int]]]]:
    from jacobian.math.groups.operations import group_conjugacy_classes

    axis = function.axis
    subgroup_classes = group_conjugacy_classes(
        subgroup.degree, [list(generator) for generator in subgroup.generators]
    )
    if (
        len(subgroup_classes) != len(axis.class_sizes)
        or tuple(len(cls) for cls in subgroup_classes) != axis.class_sizes
        or tuple(tuple(cls[0]) for cls in subgroup_classes)
        != axis.class_representatives
        or axis.group_order != subgroup_order
    ):
        raise OperationDomainValidationError(
            location=("class_function", "axis"),
            code="groups.characters.induction_axis_mismatch",
            message=(
                "class-function axis must match the canonical subgroup classes "
                "and exact subgroup order"
            ),
        )
    parent_classes = group_conjugacy_classes(
        parent.degree, [list(generator) for generator in parent.generators]
    )
    if len(parent_classes) > MAX_CLASS_COUNT:
        raise OperationResourceAdmissionError(
            location=("parent_group",),
            code="groups.characters.induction_class_count_bound",
            message=(
                f"induced class functions admit at most {MAX_CLASS_COUNT} parent classes"
            ),
        )
    return subgroup_classes, parent_classes


def _induction_class_map(
    subgroup_classes: list[list[list[int]]],
    parent_classes: list[list[list[int]]],
) -> tuple[dict[tuple[int, ...], int], tuple[int, ...]]:
    parent_element_class = {
        tuple(element): class_index
        for class_index, conjugacy_class in enumerate(parent_classes)
        for element in conjugacy_class
    }
    subgroup_to_parent = []
    for conjugacy_class in subgroup_classes:
        parent_class = parent_element_class.get(tuple(conjugacy_class[0]))
        if parent_class is None:
            raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
        subgroup_to_parent.append(parent_class)
    subgroup_element_class = {
        tuple(element): class_index
        for class_index, conjugacy_class in enumerate(subgroup_classes)
        for element in conjugacy_class
    }
    return subgroup_element_class, tuple(subgroup_to_parent)


def _induced_values(
    function: FiniteClassFunction,
    parent_classes: list[list[list[int]]],
    subgroup_element_class: dict[tuple[int, ...], int],
    subgroup_order: int,
) -> tuple[CyclotomicValue, ...]:
    order = function.axis.cyclotomic_order
    degree = len(parent_classes[0][0])
    elements = tuple(
        tuple(element)
        for conjugacy_class in parent_classes
        for element in conjugacy_class
    )
    inverses: dict[tuple[int, ...], tuple[int, ...]] = {}
    for element in elements:
        inverse = [0] * degree
        for point, image in enumerate(element):
            inverse[image] = point
        inverses[element] = tuple(inverse)

    result = []
    for parent_class in parent_classes:
        representative = tuple(parent_class[0])
        total = zero_value(order)
        for element in elements:
            conjugate = _permutation_compose(
                _permutation_compose(element, representative), inverses[element]
            )
            subgroup_class = subgroup_element_class.get(conjugate)
            if subgroup_class is not None:
                total = add_values(
                    order,
                    total,
                    _fractions(function.values[subgroup_class]),
                )
        result.append(
            _make_value(
                order,
                tuple(coefficient / subgroup_order for coefficient in total),
            )
        )
    return tuple(result)


def _partition_from_classes(
    group: PermutationGroup,
    classes: list[list[list[int]]],
) -> ConjugacyClassPartition:
    result = GroupConjugacyClassesResult._from_kernel(
        group,
        tuple(tuple(tuple(element) for element in cls) for cls in classes),
    )
    return ConjugacyClassPartition._from_group_result(result)


def class_function_induce_from_subgroup(
    class_function: FiniteClassFunction,
    parent_group: PermutationGroup,
) -> ClassFunctionInductionResult:
    """Induce an exact class function along a same-domain subgroup inclusion.

    For each parent-class representative ``g``, this computes
    ``(1 / |H|) * sum_x phi(x^-1*g*x)`` over parent elements ``x`` whose
    conjugate belongs to ``H``. The subgroup-to-parent class map is retained
    alongside the induced class function.
    """
    function, parent, subgroup_order, _parent_order = _admit_class_function_induction(
        class_function, parent_group
    )
    subgroup = cast(PermutationGroup, function.axis.group)
    axis = function.axis
    order = axis.cyclotomic_order
    from sympy.combinatorics import Permutation

    from jacobian.math.groups.operations import _backend_group

    parent_backend = _backend_group(parent)
    for generator_index, generator in enumerate(subgroup.generators):
        if not parent_backend.contains(Permutation(list(generator)), strict=True):
            raise OperationDomainValidationError(
                location=(
                    "class_function",
                    "axis",
                    "group",
                    "generators",
                    generator_index,
                ),
                code="groups.characters.induction_not_subgroup",
                message="every source subgroup generator must belong to the parent group",
            )

    subgroup_classes, parent_classes = _induction_conjugacy_classes(
        function, subgroup, parent, subgroup_order
    )
    subgroup_element_class, subgroup_to_parent = _induction_class_map(
        subgroup_classes, parent_classes
    )
    induced_values = _induced_values(
        function, parent_classes, subgroup_element_class, subgroup_order
    )
    subgroup_partition = _partition_from_classes(subgroup, subgroup_classes)
    parent_partition = _partition_from_classes(parent, parent_classes)
    parent_axis = ClassAxis._from_kernel(
        class_sizes=tuple(len(cls) for cls in parent_classes),
        cyclotomic_order=order,
        group=parent,
        class_representatives=tuple(tuple(cls[0]) for cls in parent_classes),
    )
    induced = FiniteClassFunction._from_kernel(axis=parent_axis, values=induced_values)
    return ClassFunctionInductionResult._from_kernel(
        source_class_function=function,
        subgroup_partition=subgroup_partition,
        parent_partition=parent_partition,
        subgroup_class_to_parent_class=subgroup_to_parent,
        induced=induced,
    )


MAX_POINTWISE_PRODUCT_WORK = 50_000_000
MAX_POINTWISE_PRODUCT_OUTPUT_DIGITS = 2_000_000
MAX_CLASS_FUNCTION_CONJUGATE_WORK = 50_000_000
MAX_CLASS_FUNCTION_CONJUGATE_OUTPUT_DIGITS = 2_000_000
MAX_CLASS_FUNCTION_ADD_WORK = 50_000_000
MAX_CLASS_FUNCTION_ADD_OUTPUT_DIGITS = 2_000_000


def class_function_conjugate(function: FiniteClassFunction) -> FiniteClassFunction:
    """Apply exact complex conjugation to every value on the retained axis.

    In the distinguished cyclotomic power basis this is the involution
    ``zeta -> zeta^-1``. It preserves the class coordinate axis and does not
    assert that the input is a character or irreducible character.
    """

    function = _admit_class_function(function, location=("function",), name="function")
    axis = function.axis
    class_count = len(axis.class_sizes)
    order = axis.cyclotomic_order
    if class_count > MAX_CLASS_COUNT:
        raise OperationResourceAdmissionError(
            location=("function", "axis", "class_sizes"),
            code="groups.characters.class_count_exceeds_envelope",
            message=f"class functions admit at most {MAX_CLASS_COUNT} classes",
        )
    if axis.group_order > MAX_GROUP_ORDER:
        raise OperationResourceAdmissionError(
            location=("function", "axis", "group_order"),
            code="groups.characters.group_order_exceeds_envelope",
            message=f"class functions admit group order at most {MAX_GROUP_ORDER}",
        )
    if order > MAX_ARITHMETIC_ORDER:
        raise OperationResourceAdmissionError(
            location=("function", "axis", "cyclotomic_order"),
            code="groups.characters.arithmetic_order_exceeds_envelope",
            message=(
                f"exact cyclotomic arithmetic admits order at most "
                f"{MAX_ARITHMETIC_ORDER}"
            ),
        )

    dimension = euler_phi(order)
    output_cells = class_count * dimension
    if output_cells > MAX_CHARACTER_TABLE_CELLS:
        raise OperationResourceAdmissionError(
            location=("function", "axis"),
            code="groups.characters.conjugate_output_cells_exceed_envelope",
            message="conjugated class-function coefficient output exceeds its envelope",
        )

    heights: list[_InputHeight] = []
    output_digits = 1
    maximum_input_digits = 1
    for value in function.values:
        height = _input_height(value)
        if (
            height.denominator is None
            or height.lifted_numerator_digits > MAX_VALUE_COEFFICIENT_DIGITS
        ):
            raise OperationResourceAdmissionError(
                location=("function", "values"),
                code="groups.characters.conjugate_input_height_exceeds_envelope",
                message="class-function input height exceeds the exact arithmetic envelope",
            )
        heights.append(height)
        for coefficient in value.coefficients:
            maximum_input_digits = max(
                maximum_input_digits,
                canonical_rational_component_digits(coefficient),
            )
        conjugate_height = _reduced_height(
            order=order,
            numerator_digits=height.lifted_numerator_digits,
            denominator_digits=height.denominator_digits,
            raw_length=order,
        )
        _reject_derived_height(
            "class-function conjugate",
            conjugate_height,
            code="groups.characters.conjugate_output_digits_exceed_envelope",
        )
        output_digits = max(
            output_digits,
            conjugate_height.numerator_digits,
            conjugate_height.denominator_digits,
        )

    work = (
        class_count
        * order
        * max(1, dimension)
        * max(1, (maximum_input_digits + 31) // 32) ** 2
    )
    if work > MAX_CLASS_FUNCTION_CONJUGATE_WORK:
        raise OperationResourceAdmissionError(
            location=("function", "values"),
            code="groups.characters.conjugate_work_exceeds_envelope",
            message=(
                "exact class-function conjugation work exceeds its "
                f"{MAX_CLASS_FUNCTION_CONJUGATE_WORK} unit envelope"
            ),
        )
    # Every retained coefficient contributes its numerator and denominator
    # decimal widths; the envelope bounds the exact output digits before
    # cyclotomic arithmetic.
    if output_cells * 2 * output_digits > MAX_CLASS_FUNCTION_CONJUGATE_OUTPUT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("function", "values"),
            code="groups.characters.conjugate_output_exceeds_envelope",
            message="predicted conjugated class-function output exceeds its digit envelope",
        )

    values = tuple(
        _make_value(order, conjugate_value(order, _fractions(value)))
        for value in function.values
    )
    if any(
        canonical_rational_component_digits(coefficient) > MAX_VALUE_COEFFICIENT_DIGITS
        for value in values
        for coefficient in value.coefficients
    ):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    return FiniteClassFunction._from_kernel(axis=axis, values=values)


def class_function_add(
    phi: FiniteClassFunction, psi: FiniteClassFunction
) -> FiniteClassFunction:
    """Add exact finite-group class functions coordinatewise on one axis.

    Addition is defined for arbitrary class functions and retains their shared
    class axis and cyclotomic parent. Input, coefficient growth, work, and
    exact output digits are admitted before exact rational addition.
    """
    phi = _admit_class_function(phi, location=("phi",), name="phi")
    psi = _admit_class_function(psi, location=("psi",), name="psi")
    if phi.axis != psi.axis:
        raise OperationDomainValidationError(
            location=("psi", "axis"),
            code="groups.characters.class_axis_mismatch",
            message="both class functions must carry the identical class axis",
        )
    axis = phi.axis
    class_count = len(axis.class_sizes)
    order = axis.cyclotomic_order
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
    if order > MAX_ARITHMETIC_ORDER:
        raise OperationResourceAdmissionError(
            location=("phi", "axis", "cyclotomic_order"),
            code="groups.characters.arithmetic_order_exceeds_envelope",
            message=(
                f"exact cyclotomic arithmetic admits order at most "
                f"{MAX_ARITHMETIC_ORDER}"
            ),
        )

    dimension = euler_phi(order)
    output_cells = class_count * dimension
    if output_cells > MAX_CHARACTER_TABLE_CELLS:
        raise OperationResourceAdmissionError(
            location=("phi", "axis"),
            code="groups.characters.add_output_cells_exceed_envelope",
            message="class-function sum coefficient output exceeds its envelope",
        )

    output_digits = 1
    max_input_digits = 1
    for left, right in zip(phi.values, psi.values, strict=True):
        for value in (left, right):
            for coefficient in value.coefficients:
                digits = canonical_rational_component_digits(coefficient)
                max_input_digits = max(max_input_digits, digits)
                if digits > MAX_VALUE_COEFFICIENT_DIGITS:
                    raise OperationResourceAdmissionError(
                        location=("phi", "values"),
                        code="groups.characters.add_input_digits_exceed_envelope",
                        message="class-function input coefficients exceed the exact arithmetic envelope",
                    )
        left_height, right_height = _input_height(left), _input_height(right)
        denominator = _bounded_product(
            left_height.denominator, right_height.denominator
        )
        if denominator is None:
            raise OperationResourceAdmissionError(
                location=("phi", "values"),
                code="groups.characters.add_denominator_exceeds_envelope",
                message="predicted class-function sum denominator exceeds the exact coefficient envelope",
            )
        numerator_digits = (
            max(
                left_height.lifted_numerator_digits
                + len(str(right_height.denominator)),
                right_height.lifted_numerator_digits
                + len(str(left_height.denominator)),
            )
            + 1
        )
        _reject_derived_height(
            "class-function sum",
            _CoefficientHeight(numerator_digits, len(str(denominator))),
            code="groups.characters.add_output_digits_exceed_envelope",
        )
        output_digits = max(output_digits, numerator_digits, len(str(denominator)))

    work = class_count * dimension * max(1, (max_input_digits + 31) // 32) ** 2
    if work > MAX_CLASS_FUNCTION_ADD_WORK:
        raise OperationResourceAdmissionError(
            location=("phi", "values"),
            code="groups.characters.add_work_exceeds_envelope",
            message=(
                "class-function addition work exceeds its "
                f"{MAX_CLASS_FUNCTION_ADD_WORK} unit envelope"
            ),
        )
    if output_cells * 2 * output_digits > MAX_CLASS_FUNCTION_ADD_OUTPUT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("phi", "values"),
            code="groups.characters.add_output_exceeds_envelope",
            message="predicted exact class-function sum exceeds its 2,000,000-digit envelope",
        )

    values = tuple(
        _make_value(order, add_values(order, _fractions(left), _fractions(right)))
        for left, right in zip(phi.values, psi.values, strict=True)
    )
    if any(
        canonical_rational_component_digits(coefficient) > MAX_VALUE_COEFFICIENT_DIGITS
        for value in values
        for coefficient in value.coefficients
    ):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    return FiniteClassFunction._from_kernel(axis=axis, values=values)


def _admit_pointwise_axis(
    phi: FiniteClassFunction, psi: FiniteClassFunction
) -> tuple[ClassAxis, int, int]:
    if phi.axis != psi.axis:
        raise OperationDomainValidationError(
            location=("psi", "axis"),
            code="groups.characters.class_axis_mismatch",
            message="both class functions must carry the identical class axis",
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
    order = axis.cyclotomic_order
    if order > MAX_ARITHMETIC_ORDER:
        raise OperationResourceAdmissionError(
            location=("phi", "axis", "cyclotomic_order"),
            code="groups.characters.arithmetic_order_exceeds_envelope",
            message=f"exact cyclotomic arithmetic admits order at most {MAX_ARITHMETIC_ORDER}",
        )
    dimension = euler_phi(order)
    output_cells = class_count * dimension
    if output_cells > MAX_CHARACTER_TABLE_CELLS:
        raise OperationResourceAdmissionError(
            location=("phi", "axis"),
            code="groups.characters.product_output_exceeds_envelope",
            message="pointwise-product coefficient output exceeds its envelope",
        )
    return axis, order, dimension


def _admit_pointwise_inputs(
    phi: FiniteClassFunction,
    psi: FiniteClassFunction,
    *,
    class_count: int,
    dimension: int,
) -> list[tuple[_InputHeight, _InputHeight]]:
    heights: list[tuple[_InputHeight, _InputHeight]] = []
    max_digits = 1
    for left, right in zip(phi.values, psi.values, strict=True):
        for value in (left, right):
            for coefficient in value.coefficients:
                max_digits = max(
                    max_digits, canonical_rational_component_digits(coefficient)
                )
        left_height, right_height = _input_height(left), _input_height(right)
        if (
            left_height.denominator is None
            or right_height.denominator is None
            or left_height.lifted_numerator_digits > MAX_VALUE_COEFFICIENT_DIGITS
            or right_height.lifted_numerator_digits > MAX_VALUE_COEFFICIENT_DIGITS
        ):
            raise OperationResourceAdmissionError(
                location=("phi", "values"),
                code="groups.characters.product_input_height_exceeds_envelope",
                message="class-function input height exceeds the exact arithmetic envelope",
            )
        heights.append((left_height, right_height))
    if max_digits > MAX_VALUE_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("phi", "values"),
            code="groups.characters.value_coefficient_digits_exceed_envelope",
            message="class-function value coefficients exceed the 512-digit envelope",
        )
    work = class_count * dimension * dimension * max(1, (max_digits + 31) // 32) ** 2
    if work > MAX_POINTWISE_PRODUCT_WORK:
        raise OperationResourceAdmissionError(
            location=("phi",),
            code="groups.characters.product_work_exceeds_envelope",
            message="class-function product work exceeds its 50,000,000 unit envelope",
        )
    return heights


def _admit_pointwise_output(
    heights: list[tuple[_InputHeight, _InputHeight]],
    *,
    dimension: int,
    output_cells: int,
) -> tuple[_InputHeight, ...]:
    # Each raw convolution coefficient sums at most dimension rational
    # products. Reduction uses at most dimension-1 monic cyclotomic steps;
    # the owner envelope bounds every reduction coefficient to one digit.
    reduction_digits = MAX_CYCLOTOMIC_REDUCTION_COEFFICIENT_DIGITS
    maximum_output_digits = 1
    output_heights: list[_InputHeight] = []
    for left_height, right_height in heights:
        denominator = _bounded_product(
            left_height.denominator, right_height.denominator
        )
        if denominator is None:
            raise OperationResourceAdmissionError(
                location=("phi", "values"),
                code="groups.characters.product_denominator_exceeds_envelope",
                message="predicted product denominator exceeds the exact coefficient envelope",
            )
        output_numerator_digits = (
            left_height.lifted_numerator_digits
            + right_height.lifted_numerator_digits
            + (len(str(dimension - 1)) if dimension > 1 else 0)
            + max(0, dimension - 1) * reduction_digits
        )
        output_denominator_digits = len(str(denominator))
        output_heights.append(
            _InputHeight(
                denominator=denominator,
                lifted_numerator_digits=output_numerator_digits,
            )
        )
        _reject_derived_height(
            "pointwise product",
            _CoefficientHeight(output_numerator_digits, output_denominator_digits),
            code="groups.characters.pointwise_product_output_digits_exceed_envelope",
        )
        maximum_output_digits = max(
            maximum_output_digits, output_numerator_digits, output_denominator_digits
        )

    # Every retained coefficient contributes its numerator and denominator
    # decimal widths; the envelope bounds the exact output digits before
    # cyclotomic arithmetic.
    if output_cells * 2 * maximum_output_digits > MAX_POINTWISE_PRODUCT_OUTPUT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("phi", "values"),
            code="groups.characters.product_output_exceeds_envelope",
            message="predicted exact class-function product exceeds the 2,000,000-digit envelope",
        )
    return tuple(output_heights)


def class_function_pointwise_product(
    phi: FiniteClassFunction, psi: FiniteClassFunction
) -> FiniteClassFunction:
    """Multiply two class functions valuewise on their shared class axis.

    For characters, this is the character of the tensor product. The exact
    power-basis product is admitted for coefficient growth and work before
    cyclotomic expansion; its class axis and order are retained unchanged.
    """
    phi = _admit_class_function(phi, location=("phi",), name="phi")
    psi = _admit_class_function(psi, location=("psi",), name="psi")
    axis, order, dimension = _admit_pointwise_axis(phi, psi)
    output_cells = len(axis.class_sizes) * dimension
    heights = _admit_pointwise_inputs(
        phi,
        psi,
        class_count=len(axis.class_sizes),
        dimension=dimension,
    )
    _admit_pointwise_output(
        heights,
        dimension=dimension,
        output_cells=output_cells,
    )

    values = tuple(
        _make_value(
            order,
            multiply_values(order, _fractions(left), _fractions(right)),
        )
        for left, right in zip(phi.values, psi.values, strict=True)
    )
    if any(
        canonical_rational_component_digits(coefficient) > MAX_VALUE_COEFFICIENT_DIGITS
        for value in values
        for coefficient in value.coefficients
    ):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    return FiniteClassFunction._from_kernel(axis=axis, values=values)


def class_function_scale(
    function: FiniteClassFunction, scalar: CyclotomicValue
) -> FiniteClassFunction:
    """Scale every class value by one exact element of its cyclotomic field.

    The existing pointwise-product admission owns convolution work and
    coefficient/output growth; representing the scalar as a constant class
    function applies exactly that admitted multiplication kernel.
    """
    function = _admit_class_function(function, location=("function",), name="function")
    scalar = _admit_cyclotomic_value(scalar, location=("scalar",), name="scalar")
    if scalar.order != function.axis.cyclotomic_order:
        raise OperationDomainValidationError(
            location=("scalar", "order"),
            code="groups.characters.scalar_field_mismatch",
            message="scalar must belong to the class axis cyclotomic field",
        )
    constant = FiniteClassFunction(
        axis=function.axis,
        values=(scalar,) * len(function.axis.class_sizes),
    )
    return class_function_pointwise_product(function, constant)


__all__ = [
    "class_function_add",
    "class_function_conjugate",
    "class_function_induce_from_subgroup",
    "class_function_inner_product",
    "class_function_pointwise_product",
    "class_function_restrict_to_subgroup",
    "class_function_scale",
]
