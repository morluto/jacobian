"""Exact dynamic programming over finite abelian product groups."""

from __future__ import annotations

import time
from itertools import product

from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.additive.finite_abelian_subset_sum._models import (
    MAX_FINITE_ABELIAN_SUBSET_SUM_COORDINATE_SLOTS,
    MAX_FINITE_ABELIAN_SUBSET_SUM_ITEMS,
    MAX_FINITE_ABELIAN_SUBSET_SUM_ORDER,
    MAX_FINITE_ABELIAN_SUBSET_SUM_TRANSITIONS,
    FiniteAbelianSubsetSumResult,
    FiniteAbelianSubsetSumRow,
)
from jacobian.math.groups.finite_abelian import (
    FiniteAbelianGroupElement,
    FiniteAbelianProductGroup,
)


def _reject(code: str, message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("group", "sequence"),
        code=f"additive.finite_abelian_subset_sum.{code}",
        message=message,
    )


def finite_abelian_subset_sum_profile(
    group: FiniteAbelianProductGroup,
    sequence: tuple[FiniteAbelianGroupElement, ...],
) -> FiniteAbelianSubsetSumResult:
    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return finite_abelian_subset_sum_profile(group, sequence)
    deadline = execution.started_at + 60.0
    if execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    request_checkpoint("before finite abelian subset-sum admission")
    order = group.order
    if not group.moduli or order > MAX_FINITE_ABELIAN_SUBSET_SUM_ORDER:
        _reject("group_order", "finite abelian group order must be between 2 and 4,096")
    if any(element.group != group for element in sequence):
        _reject("group_binding", "every sequence element must use the supplied group")
    coordinate_slots = (len(sequence) + order) * len(group.moduli)
    if coordinate_slots > MAX_FINITE_ABELIAN_SUBSET_SUM_COORDINATE_SLOTS:
        _reject(
            "coordinates",
            "finite abelian subset-sum coordinates exceed their admitted bound",
        )
    if len(sequence) > MAX_FINITE_ABELIAN_SUBSET_SUM_ITEMS:
        _reject("input_length", "finite abelian subset-sum sequence is too long")
    transitions = len(sequence) * order
    if transitions > MAX_FINITE_ABELIAN_SUBSET_SUM_TRANSITIONS:
        _reject("work", "finite abelian subset-sum DP exceeds its transition bound")
    elements = tuple(
        FiniteAbelianGroupElement(group=group, coordinates=coordinates)
        for coordinates in product(*(range(modulus) for modulus in group.moduli))
    )
    index = {element.coordinates: position for position, element in enumerate(elements)}
    counts = [0] * order
    counts[0] = 1
    for position, element in enumerate(sequence):
        request_checkpoint(f"during finite abelian subset-sum transition {position}")
        next_counts = counts.copy()
        for prior, multiplicity in enumerate(counts):
            coordinates = elements[prior].coordinates
            target = tuple(
                (left + right) % modulus
                for left, right, modulus in zip(
                    coordinates, element.coordinates, group.moduli, strict=True
                )
            )
            next_counts[index[target]] += multiplicity
        counts = next_counts
    rows = tuple(
        FiniteAbelianSubsetSumRow(element=element, multiplicity=multiplicity)
        for element, multiplicity in zip(elements, counts, strict=True)
    )
    support_size = sum(multiplicity > 0 for multiplicity in counts)
    return FiniteAbelianSubsetSumResult(
        group=group,
        sequence=sequence,
        rows=rows,
        support_size=support_size,
        covers_group=support_size == order,
        total_subsets=1 << len(sequence),
    )


__all__ = ["finite_abelian_subset_sum_profile"]
