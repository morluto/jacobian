"""Exact additive-combinatorics decisions for finite difference sets."""

from __future__ import annotations

import math
from collections import Counter

from jacobian.contracts.combinatorics import (
    CyclicDifferenceMultiplicity,
    CyclicDifferenceSetExtensionRequest,
    CyclicDifferenceSetExtensionResult,
    CyclicPerfectDifferenceSetRequest,
    CyclicPerfectDifferenceSetResult,
    IntegerSidonRequest,
    IntegerSidonResult,
    OrderedIntegerDifference,
)
from jacobian.domains._examples import example
from jacobian.domains.combinatorics._support import (
    combinatorics_operation,
    materialized_combinatorics_operation,
)


def decide_integer_sidon(request: IntegerSidonRequest) -> IntegerSidonResult:
    elements = tuple(sorted(int(value) for value in request.elements))
    differences = tuple(
        OrderedIntegerDifference(
            minuend=str(left),
            subtrahend=str(right),
            difference=str(left - right),
        )
        for left in elements
        for right in elements
        if left != right
    )
    values = tuple(int(record.difference) for record in differences)
    return IntegerSidonResult(
        semantics_version="integer-sidon.ordered-differences.v1",
        normalized_elements=tuple(str(value) for value in elements),
        ordered_differences=differences,
        is_sidon=len(set(values)) == len(values),
    )


def _difference_counts(residues: tuple[int, ...], modulus: int) -> Counter[int]:
    return Counter(
        (left - right) % modulus
        for left in residues
        for right in residues
        if left != right
    )


def decide_cyclic_perfect_difference_set(
    request: CyclicPerfectDifferenceSetRequest,
) -> CyclicPerfectDifferenceSetResult:
    residues = tuple(sorted(request.residues))
    counts = _difference_counts(residues, request.modulus)
    profile = tuple(
        CyclicDifferenceMultiplicity(
            residue=residue,
            multiplicity=counts.get(residue, 0),
        )
        for residue in range(1, request.modulus)
    )
    missing = tuple(item.residue for item in profile if item.multiplicity == 0)
    repeated = tuple(item.residue for item in profile if item.multiplicity > 1)
    order = len(residues)
    expected_modulus = order * (order - 1) + 1
    return CyclicPerfectDifferenceSetResult(
        semantics_version="cyclic-perfect-difference-set.v1",
        modulus=request.modulus,
        normalized_residues=residues,
        order=order,
        expected_modulus=expected_modulus,
        difference_multiplicities=profile,
        missing_residues=missing,
        repeated_residues=repeated,
        is_perfect=(
            request.modulus == expected_modulus and not missing and not repeated
        ),
    )


def _initial_difference_mask(residues: tuple[int, ...], modulus: int) -> int | None:
    mask = 0
    for left in residues:
        for right in residues:
            if left == right:
                continue
            residue = (left - right) % modulus
            bit = 1 << residue
            if residue == 0 or mask & bit:
                return None
            mask |= bit
    return mask


def _extended_difference_mask(
    selected: tuple[int, ...],
    mask: int,
    candidate: int,
    modulus: int,
) -> int | None:
    result = mask
    for existing in selected:
        for difference in (
            (candidate - existing) % modulus,
            (existing - candidate) % modulus,
        ):
            bit = 1 << difference
            if difference == 0 or result & bit:
                return None
            result |= bit
    return result


def _visit_extensions(
    selected: tuple[int, ...],
    mask: int,
    start: int,
    available: tuple[int, ...],
    target_order: int,
    modulus: int,
) -> tuple[int, ...] | None:
    remaining = target_order - len(selected)
    if remaining == 0:
        return selected
    if len(available) - start < remaining:
        return None
    for position in range(start, len(available)):
        candidate = available[position]
        local_mask = _extended_difference_mask(selected, mask, candidate, modulus)
        if local_mask is None:
            continue
        found = _visit_extensions(
            tuple(sorted((*selected, candidate))),
            local_mask,
            position + 1,
            available,
            target_order,
            modulus,
        )
        if found is not None:
            return found
    return None


def _find_extension(
    base: tuple[int, ...], target_order: int, modulus: int
) -> tuple[int, ...] | None:
    initial_mask = _initial_difference_mask(base, modulus)
    if initial_mask is None:
        return None
    needed = target_order - len(base)
    if needed == 0:
        return base
    available = tuple(value for value in range(modulus) if value not in set(base))
    return _visit_extensions(
        base,
        initial_mask,
        0,
        available,
        target_order,
        modulus,
    )


def decide_cyclic_difference_set_extension(
    request: CyclicDifferenceSetExtensionRequest,
) -> CyclicDifferenceSetExtensionResult:
    order = request.target_order
    modulus = order * (order - 1) + 1
    base = tuple(sorted({int(value) % modulus for value in request.base_elements}))
    additional = order - len(base)
    candidate_count = math.comb(modulus - len(base), additional)
    extension = _find_extension(base, order, modulus)
    return CyclicDifferenceSetExtensionResult(
        semantics_version="cyclic-pds-extension.fixed-order.v1",
        target_order=order,
        modulus=modulus,
        base_residues=base,
        candidate_space_size=candidate_count,
        decision="EXTENDS" if extension is not None else "DOES_NOT_EXTEND",
        extension=extension or (),
        coverage="WITNESS" if extension is not None else "ALL_CANDIDATES",
    )


DIFFERENCE_SET_CAPABILITIES = (
    combinatorics_operation(
        "combinatorics.integer_set.sidon.decide",
        "Decide the integer Sidon property",
        (
            "Materialize every ordered nonzero integer difference of one bounded "
            "finite set and decide whether all such differences are distinct."
        ),
        IntegerSidonRequest,
        IntegerSidonResult,
        decide_integer_sidon,
        "combinatorics",
        "additive-combinatorics",
        "sidon-set",
        "ordered-differences",
        invocation_examples=(
            example(
                "mian_chowla_prefix",
                "Decide whether 1, 2, 4, 8, 13 is Sidon over the integers.",
                {"elements": ["1", "2", "4", "8", "13"]},
            ),
        ),
    ),
    combinatorics_operation(
        "combinatorics.cyclic_difference_set.perfect.decide",
        "Decide the cyclic perfect-difference-set property",
        (
            "Compute the complete nonzero residue-difference multiplicity profile "
            "and decide whether one finite residue set is perfect."
        ),
        CyclicPerfectDifferenceSetRequest,
        CyclicPerfectDifferenceSetResult,
        decide_cyclic_perfect_difference_set,
        "combinatorics",
        "additive-combinatorics",
        "difference-set",
        "finite-design",
        invocation_examples=(
            example(
                "fano_difference_set",
                "Decide whether 0, 1, 3 is a perfect difference set modulo 7.",
                {"modulus": 7, "residues": [0, 1, 3]},
            ),
        ),
    ),
    materialized_combinatorics_operation(
        "combinatorics.cyclic_difference_set.extension.decide",
        "Decide fixed-order perfect-difference-set extension",
        (
            "Completely decide whether the reduced residues of one bounded integer "
            "set extend to a cyclic perfect difference set of one fixed order."
        ),
        CyclicDifferenceSetExtensionRequest,
        CyclicDifferenceSetExtensionResult,
        decide_cyclic_difference_set_extension,
        "combinatorics",
        "additive-combinatorics",
        "difference-set",
        "bounded-completion",
        invocation_examples=(
            example(
                "mian_chowla_order_six",
                "Decide fixed-order extension of 1, 2, 4, 8, 13 at order 6.",
                {
                    "base_elements": ["1", "2", "4", "8", "13"],
                    "target_order": 6,
                },
            ),
        ),
        preview=lambda result: result,
        preview_complete=True,
    ),
)


__all__ = [
    "DIFFERENCE_SET_CAPABILITIES",
    "decide_cyclic_difference_set_extension",
    "decide_cyclic_perfect_difference_set",
    "decide_integer_sidon",
]
