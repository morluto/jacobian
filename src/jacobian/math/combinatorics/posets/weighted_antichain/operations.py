"""Maximum weight antichain kernel using exhaustive search."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

from jacobian._exact import (
    CanonicalRational,
    canonical_rational_component_digits,
)
from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    strict_json_object_size,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.posets.core._models import FinitePoset
from jacobian.math.combinatorics.posets.weighted_antichain._models import (
    MAX_ENUMERATION_WORK,
    MaximumWeightAntichainResult,
)

__all__ = ["compute_maximum_weight_antichain"]


@dataclass(frozen=True, slots=True)
class _MaximumWeightAntichainAdmission:
    weights: tuple[Fraction, ...]


def _admit_maximum_weight_antichain(
    poset: FinitePoset, weights: tuple[CanonicalRational, ...]
) -> _MaximumWeightAntichainAdmission:
    if not isinstance(poset, FinitePoset):
        raise OperationDomainValidationError(
            location=("poset",),
            code="weighted_antichain.invalid_poset",
            message="poset must be a FinitePoset value",
        )
    if not isinstance(weights, tuple) or len(weights) != len(poset.elements):
        raise OperationDomainValidationError(
            location=("weights",),
            code="weighted_antichain.weight_count_mismatch",
            message="weights must have exactly one entry per poset element",
        )
    if any(not isinstance(weight, CanonicalRational) for weight in weights):
        raise OperationDomainValidationError(
            location=("weights",),
            code="weighted_antichain.invalid_weight",
            message="weights must be CanonicalRational values",
        )
    weight_fracs = tuple(weight.as_fraction() for weight in weights)
    if any(weight < 0 for weight in weight_fracs):
        raise OperationDomainValidationError(
            location=("weights",),
            code="weighted_antichain.negative_weight",
            message="all weights must be nonnegative",
        )
    n = len(poset.elements)
    pair_checks = n * max(n - 1, 0) // 2
    if (1 << n) * pair_checks > MAX_ENUMERATION_WORK:
        raise OperationDomainValidationError(
            location=("poset", "elements"),
            code="weighted_antichain.work_bound_exceeded",
            message="the exhaustive antichain work envelope is exceeded",
        )
    width = _poset_width(poset, list(poset.elements))
    max_digits = max(
        (canonical_rational_component_digits(weight) for weight in weights),
        default=1,
    )
    arithmetic_work = (1 << n) * max(width, 1) * max_digits
    if arithmetic_work > MAX_ENUMERATION_WORK:
        raise OperationDomainValidationError(
            location=("weights",),
            code="weighted_antichain.arithmetic_work_bound_exceeded",
            message="rational antichain summation exceeds the admitted work envelope",
        )
    # A singleton antichain returns the input weight unchanged; do not charge
    # an extra addition digit at that exact canonical boundary.
    max_sum_digits = max_digits if width <= 1 else width * max_digits + len(str(width))
    if max_sum_digits > 32_768:
        raise OperationDomainValidationError(
            location=("weights",),
            code="weighted_antichain.result_growth_exceeded",
            message="maximum-weight rational growth exceeds the canonical digit envelope",
        )
    rational_size = strict_json_object_size(
        (
            ("num", len(encode_strict_json("9" * max_sum_digits))),
            ("den", len(encode_strict_json("9" * max_sum_digits))),
        )
    )
    labels_size = (
        2
        + max(n - 1, 0)
        + sum(len(encode_strict_json(element)) for element in poset.elements)
    )
    result_bytes = strict_json_object_size(
        (
            ("poset", len(encode_strict_json(poset.model_dump(mode="json")))),
            (
                "weights",
                len(
                    encode_strict_json(
                        [weight.model_dump(mode="json") for weight in weights]
                    )
                ),
            ),
            ("maximum_weight", rational_size),
            ("antichain", labels_size),
        )
    )
    if result_bytes > CanonicalLimits().max_output_bytes:
        raise OperationDomainValidationError(
            location=("poset", "weights"),
            code="weighted_antichain.result_too_large",
            message="maximum-weight antichain result exceeds the canonical output envelope",
        )
    return _MaximumWeightAntichainAdmission(weights=weight_fracs)


def compute_maximum_weight_antichain(
    poset: FinitePoset,
    weights: tuple[CanonicalRational, ...],
) -> MaximumWeightAntichainResult:
    """Return the exact maximum weight antichain and a witness.

    Uses exhaustive search over all subsets within the admitted work envelope.
    """
    admission = _admit_maximum_weight_antichain(poset, weights)
    elements = list(poset.elements)
    n = len(elements)
    weight_fracs = admission.weights

    comparable = _build_comparable(poset, elements)

    best_weight = Fraction(0)
    best_antichain: tuple[str, ...] = ()

    for subset in _all_subsets(n):
        if _is_antichain(subset, comparable):
            total = Fraction(0)
            for index in subset:
                total += weight_fracs[index]
            if total > best_weight or (
                total == best_weight
                and _subset_to_elements(subset, elements) < best_antichain
            ):
                best_weight = total
                best_antichain = _subset_to_elements(subset, elements)

    return MaximumWeightAntichainResult(
        poset=poset,
        weights=weights,
        maximum_weight=CanonicalRational.from_fraction(best_weight),
        antichain=best_antichain,
    )


def _build_comparable(poset: FinitePoset, elements: list[str]) -> set[tuple[int, int]]:
    idx = {e: i for i, e in enumerate(elements)}
    comparable: set[tuple[int, int]] = set()
    for pair in poset.strict_order_pairs:
        i, j = idx[pair.lower], idx[pair.upper]
        comparable.add((i, j))
        comparable.add((j, i))
    return comparable


def _poset_width(poset: FinitePoset, elements: list[str]) -> int:
    """Return the maximum antichain size via the bipartite matching theorem."""
    index = {element: position for position, element in enumerate(elements)}
    adjacent: list[list[int]] = [[] for _ in elements]
    for pair in poset.strict_order_pairs:
        adjacent[index[pair.lower]].append(index[pair.upper])

    matched_upper = [-1] * len(elements)

    def augment(lower: int, seen: set[int]) -> bool:
        for upper in adjacent[lower]:
            if upper in seen:
                continue
            seen.add(upper)
            if matched_upper[upper] == -1 or augment(matched_upper[upper], seen):
                matched_upper[upper] = lower
                return True
        return False

    matching = sum(augment(lower, set()) for lower in range(len(elements)))
    return len(elements) - matching


def _all_subsets(n: int) -> Iterator[tuple[int, ...]]:
    yield from (subset for r in range(n + 1) for subset in combinations(range(n), r))


def _is_antichain(subset: tuple[int, ...], comparable: set[tuple[int, int]]) -> bool:
    for i in range(len(subset)):
        for j in range(i + 1, len(subset)):
            if (subset[i], subset[j]) in comparable:
                return False
    return True


def _subset_to_elements(
    subset: tuple[int, ...], elements: list[str]
) -> tuple[str, ...]:
    return tuple(elements[i] for i in subset)
