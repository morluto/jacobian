"""Domain adapter for numerical semigroup operations."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.contracts.numerical_semigroups import (
    NumericalSemigroupSummaryRequest,
    NumericalSemigroupSummaryResult,
    SemigroupMembershipRequest,
    SemigroupMembershipResult,
)


def _normalize_generators(gens: tuple[str, ...]) -> list[int]:
    """Return sorted unique positive generators."""
    return sorted({parse_canonical_integer(generator) for generator in gens})


def _compute_summary(gens: list[int]) -> NumericalSemigroupSummaryResult:
    multiplicity = gens[0]
    if multiplicity == 1:
        return NumericalSemigroupSummaryResult(
            minimal_generators=("1",),
            multiplicity="1",
            embedding_dimension=1,
            frobenius_number="-1",
            conductor="0",
            genus=0,
            gaps=(),
        )

    limit = (multiplicity - 1) * max(gens)
    in_semigroup = [False] * (limit + 1)
    in_semigroup[0] = True
    run = 0
    conductor = limit + 1
    for value in range(1, limit + 1):
        in_semigroup[value] = any(
            value >= generator and in_semigroup[value - generator] for generator in gens
        )
        if in_semigroup[value]:
            run += 1
            if run == multiplicity:
                conductor = value - multiplicity + 1
                break
        else:
            run = 0

    gaps = [
        value
        for value in range(1, conductor)
        if value <= limit and not in_semigroup[value]
    ]
    frobenius = max(gaps) if gaps else -1

    min_gens = []
    for generator in gens:
        others = [other for other in gens if other != generator]
        if not others:
            min_gens.append(generator)
            continue
        can_reach = [False] * (generator + 1)
        can_reach[0] = True
        for value in range(1, generator + 1):
            can_reach[value] = any(
                value >= other and can_reach[value - other] for other in others
            )
        if not can_reach[generator]:
            min_gens.append(generator)

    return NumericalSemigroupSummaryResult(
        minimal_generators=tuple(
            format_canonical_integer(generator) for generator in min_gens
        ),
        multiplicity=format_canonical_integer(multiplicity),
        embedding_dimension=len(min_gens),
        frobenius_number=format_canonical_integer(frobenius),
        conductor=format_canonical_integer(conductor),
        genus=len(gaps),
        gaps=tuple(format_canonical_integer(gap) for gap in gaps),
    )


def compute_summary(
    request: NumericalSemigroupSummaryRequest,
) -> NumericalSemigroupSummaryResult:
    return _compute_summary(_normalize_generators(request.generators))


def compute_membership(
    request: SemigroupMembershipRequest,
) -> SemigroupMembershipResult:
    gens = _normalize_generators(request.generators)
    value = parse_canonical_integer(request.value)
    if value < 0:
        return SemigroupMembershipResult(value=request.value, in_semigroup=False)
    if value == 0:
        return SemigroupMembershipResult(value=request.value, in_semigroup=True)
    can_reach = [False] * (value + 1)
    can_reach[0] = True
    for index in range(1, value + 1):
        can_reach[index] = any(
            index >= generator and can_reach[index - generator] for generator in gens
        )
    return SemigroupMembershipResult(
        value=request.value,
        in_semigroup=can_reach[value],
    )


__all__ = ["compute_membership", "compute_summary"]
