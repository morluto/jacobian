"""Summary and membership operations for numerical semigroups."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.numerical_semigroups._models import (
    NumericalSemigroupSummaryRequest,
    NumericalSemigroupSummaryResult,
    SemigroupMembershipRequest,
    SemigroupMembershipResult,
)


def _normalize_generators(generators: tuple[str, ...]) -> list[int]:
    """Return sorted unique positive generators."""

    return sorted({parse_canonical_integer(generator) for generator in generators})


def _compute_summary(generators: list[int]) -> NumericalSemigroupSummaryResult:
    multiplicity = generators[0]
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

    limit = (multiplicity - 1) * max(generators)
    in_semigroup = [False] * (limit + 1)
    in_semigroup[0] = True
    run = 0
    conductor = limit + 1
    for value in range(1, limit + 1):
        in_semigroup[value] = any(
            value >= generator and in_semigroup[value - generator]
            for generator in generators
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

    minimal_generators = []
    for generator in generators:
        others = [other for other in generators if other != generator]
        if not others:
            minimal_generators.append(generator)
            continue
        can_reach = [False] * (generator + 1)
        can_reach[0] = True
        for value in range(1, generator + 1):
            can_reach[value] = any(
                value >= other and can_reach[value - other] for other in others
            )
        if not can_reach[generator]:
            minimal_generators.append(generator)

    return NumericalSemigroupSummaryResult(
        minimal_generators=tuple(
            format_canonical_integer(generator) for generator in minimal_generators
        ),
        multiplicity=format_canonical_integer(multiplicity),
        embedding_dimension=len(minimal_generators),
        frobenius_number=format_canonical_integer(frobenius),
        conductor=format_canonical_integer(conductor),
        genus=len(gaps),
        gaps=tuple(format_canonical_integer(gap) for gap in gaps),
    )


def compute_summary(
    request: NumericalSemigroupSummaryRequest,
) -> NumericalSemigroupSummaryResult:
    """Compute the exact summary on the canonical minimal generator axis."""

    return _compute_summary(_normalize_generators(request.generators))


def compute_membership(
    request: SemigroupMembershipRequest,
) -> SemigroupMembershipResult:
    """Check whether one admitted integer belongs to the generated semigroup."""

    generators = _normalize_generators(request.generators)
    value = parse_canonical_integer(request.value)
    if value < 0:
        return SemigroupMembershipResult(value=request.value, in_semigroup=False)
    if value == 0:
        return SemigroupMembershipResult(value=request.value, in_semigroup=True)
    can_reach = [False] * (value + 1)
    can_reach[0] = True
    for index in range(1, value + 1):
        can_reach[index] = any(
            index >= generator and can_reach[index - generator]
            for generator in generators
        )
    return SemigroupMembershipResult(
        value=request.value,
        in_semigroup=can_reach[value],
    )


__all__ = [
    "compute_membership",
    "compute_summary",
]
