"""Summary and membership contracts for numerical semigroups."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.numerical_semigroups._models import (
    MAX_ELEMENT,
    MAX_GENERATOR,
    MAX_GENERATORS,
    _require_canonical_minimal_axis,
    _require_positive_bounded_generators,
    _summary_invariants,
    _validation_error,
)


class NumericalSemigroupSummaryRequest(StrictModel):
    """Compute the full summary of a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            f"Positive generators with gcd 1, each at most {MAX_GENERATOR}. "
            "The presentation may be reordered or redundant; the summary uses "
            "its increasing minimal generator axis."
        ),
    )

    @model_validator(mode="after")
    def require_positive_generators(self) -> Self:
        _require_positive_bounded_generators(self.generators)
        return self


class NumericalSemigroupSummaryResult(StrictModel):
    """Summary of a numerical semigroup."""

    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description="Increasing minimal generator axis of the numerical semigroup.",
    )
    multiplicity: CanonicalInteger
    embedding_dimension: int = Field(ge=1)
    frobenius_number: str
    conductor: str
    genus: int = Field(ge=0)
    gaps: tuple[CanonicalInteger, ...]

    @model_validator(mode="after")
    def require_exact_summary(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        (
            multiplicity,
            embedding_dimension,
            frobenius_number,
            conductor,
            genus,
            gaps,
        ) = _summary_invariants(generators)
        if parse_canonical_integer(self.multiplicity) != multiplicity:
            raise _validation_error(
                "multiplicity does not match the minimal generators"
            )
        if self.embedding_dimension != embedding_dimension:
            raise _validation_error(
                "embedding_dimension does not match the minimal generators"
            )
        if parse_canonical_integer(self.frobenius_number) != frobenius_number:
            raise _validation_error(
                "frobenius_number does not match the minimal generators"
            )
        if parse_canonical_integer(self.conductor) != conductor:
            raise _validation_error("conductor does not match the minimal generators")
        if self.genus != genus:
            raise _validation_error("genus does not match the minimal generators")
        if tuple(map(parse_canonical_integer, self.gaps)) != gaps:
            raise _validation_error("gaps do not match the minimal generators")
        return self


class SemigroupMembershipRequest(StrictModel):
    """Check membership of an integer in a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            f"Positive generators with gcd 1, each at most {MAX_GENERATOR}. "
            "The presentation may be reordered or redundant."
        ),
    )
    value: CanonicalInteger = Field(
        description=f"Integer to test for membership, at most {MAX_ELEMENT}."
    )

    @model_validator(mode="after")
    def require_positive_generators_and_bounded_value(self) -> Self:
        _require_positive_bounded_generators(self.generators)
        if parse_canonical_integer(self.value) > MAX_ELEMENT:
            raise _validation_error(f"membership value must be at most {MAX_ELEMENT}")
        return self


class SemigroupMembershipResult(StrictModel):
    """Whether the value is in the semigroup."""

    value: CanonicalInteger
    in_semigroup: bool


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
    "NumericalSemigroupSummaryRequest",
    "NumericalSemigroupSummaryResult",
    "SemigroupMembershipRequest",
    "SemigroupMembershipResult",
    "compute_membership",
    "compute_summary",
]
