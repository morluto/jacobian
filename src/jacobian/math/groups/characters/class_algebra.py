"""Exact, bounded class-algebra operations."""

from __future__ import annotations

from jacobian._execution import BackendFailureReason, OperationBackendError
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups._models import GroupConjugacyClassesResult
from jacobian.math.groups.characters._models import (
    ClassMultiplicationConstantsRequest,
    ClassMultiplicationConstantsResult,
    ConjugacyClassPartition,
)
from jacobian.math.groups.characters.operations import _admit_partition_source

MAX_CLASS_ALGEBRA_GROUP_ORDER = 256
MAX_CLASS_ALGEBRA_CLASS_COUNT = 64
MAX_CLASS_ALGEBRA_MULTIPLICATION_STEPS = 4_194_304
MAX_CLASS_ALGEBRA_OUTPUT_CELLS = 262_144


def _compose(first: tuple[int, ...], second: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(second[first[index]] for index in range(len(first)))


def class_multiplication_constants(
    request: ClassMultiplicationConstantsRequest,
) -> ClassMultiplicationConstantsResult:
    """Compute the complete integral class-sum multiplication tensor."""
    claim = request.partition
    claimed_order = sum(len(conjugacy_class) for conjugacy_class in claim.classes)
    claimed_count = len(claim.classes)
    # Reject from the bounded wire shape before group closure or class
    # expansion. A forged short partition still cannot lower the actual group
    # order check below.
    _admit_class_algebra_size(claimed_order, claimed_count, claim.source.degree)

    from jacobian.math.groups.operations import group_conjugacy_classes, group_order

    degree, generators = _admit_partition_source(claim.source)
    source_order = group_order(claim.source)
    if source_order > MAX_CLASS_ALGEBRA_GROUP_ORDER:
        raise OperationResourceAdmissionError(
            location=("partition", "source"),
            code="groups.characters.class_algebra_group_order_exceeds_envelope",
            message=(
                "class multiplication constants admit groups of order at most "
                f"{MAX_CLASS_ALGEBRA_GROUP_ORDER}"
            ),
        )
    try:
        expected = group_conjugacy_classes(
            degree, [list(generator) for generator in generators]
        )
    except OperationDomainValidationError as exc:
        raise OperationResourceAdmissionError(
            location=("partition", "source"),
            code="groups.characters.class_algebra_partition_exceeds_envelope",
            message="source group exceeds the class algebra envelope",
        ) from exc
    actual = [
        [list(member) for member in conjugacy_class]
        for conjugacy_class in claim.classes
    ]
    if actual != expected:
        raise OperationDomainValidationError(
            location=("partition",),
            code="groups.characters.partition_not_group_bound",
            message="class rows must be the complete conjugacy partition of their source group",
        )

    class_count = len(expected)
    _admit_class_algebra_size(source_order, class_count, degree)
    class_sizes = tuple(len(conjugacy_class) for conjugacy_class in expected)
    class_by_element = {
        tuple(element): class_index
        for class_index, conjugacy_class in enumerate(expected)
        for element in conjugacy_class
    }
    tensor: list[list[tuple[int, ...]]] = []
    for left_class in expected:
        plane: list[tuple[int, ...]] = []
        for right_class in expected:
            counts = [0] * class_count
            for left in left_class:
                for right in right_class:
                    product = _compose(tuple(left), tuple(right))
                    try:
                        counts[class_by_element[product]] += 1
                    except KeyError as exc:
                        raise OperationBackendError(
                            BackendFailureReason.INVALID_OUTPUT
                        ) from exc
            coefficients: list[int] = []
            for class_index, count in enumerate(counts):
                class_size = class_sizes[class_index]
                coefficient, remainder = divmod(count, class_size)
                if remainder:
                    raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
                # Conjugation acts transitively on the target class and
                # bijectively on factorizations, so this quotient is the
                # number of factorizations of any one fixed target element.
                # For that target, each left factor determines at most one
                # right factor and vice versa. Hence the coefficient is at
                # most min(|C_left|, |C_right|) <= |G|, which is at most three
                # decimal digits under the admitted order cap.
                coefficients.append(coefficient)
            plane.append(tuple(coefficients))
        tensor.append(plane)

    partition = ConjugacyClassPartition._from_group_result(
        GroupConjugacyClassesResult._from_kernel(
            claim.source,
            tuple(tuple(tuple(member) for member in cls) for cls in expected),
        )
    )
    return ClassMultiplicationConstantsResult._from_kernel(
        partition=partition,
        constants=tuple(tuple(plane) for plane in tensor),
    )


def _admit_class_algebra_size(order: int, class_count: int, degree: int) -> None:
    """Admit work and exact result shape before class-algebra allocation.

    The tensor has exactly ``class_count**3`` integer cells. For any fixed
    target element, its structure constant is at most ``order`` by the
    per-left-factor uniqueness argument in the kernel, so the order bound also
    bounds exact scalar height. The retained partition has ``order * degree``
    point coordinates, each in ``0..degree - 1``.
    """
    if order > MAX_CLASS_ALGEBRA_GROUP_ORDER:
        raise OperationResourceAdmissionError(
            location=("partition",),
            code="groups.characters.class_algebra_group_order_exceeds_envelope",
            message=(
                "class multiplication constants admit groups of order at most "
                f"{MAX_CLASS_ALGEBRA_GROUP_ORDER}"
            ),
        )
    if class_count > MAX_CLASS_ALGEBRA_CLASS_COUNT:
        raise OperationResourceAdmissionError(
            location=("partition", "classes"),
            code="groups.characters.class_algebra_class_count_exceeds_envelope",
            message=(
                "class multiplication constants admit at most "
                f"{MAX_CLASS_ALGEBRA_CLASS_COUNT} conjugacy classes"
            ),
        )
    work = order * order * degree
    if work > MAX_CLASS_ALGEBRA_MULTIPLICATION_STEPS:
        raise OperationResourceAdmissionError(
            location=("partition",),
            code="groups.characters.class_algebra_work_exceeds_envelope",
            message="class multiplication exceeds its admitted permutation-work envelope",
        )
    output_cells = class_count**3
    if output_cells > MAX_CLASS_ALGEBRA_OUTPUT_CELLS:
        raise OperationResourceAdmissionError(
            location=("partition", "classes"),
            code="groups.characters.class_algebra_output_exceeds_envelope",
            message="class multiplication tensor exceeds its output-cell envelope",
        )
