"""Profiles of immediate degeneracies in finite simplicial-set prefixes."""

from __future__ import annotations

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.simplicial_sets._models import (
    MAX_TOTAL_SIMPLICES,
    FiniteTruncatedSimplicialSet,
)
from jacobian.math.topology.simplicial_sets.operations import from_tables

_MAX_IDENTITY_ROW_WORK = 100_000
_MAX_OUTPUT_BYTES = 65_536


class DegeneracyWitness(StrictModel):
    """The chosen one-step presentation ``s_i(source_simplex)``."""

    degeneracy_index: int = Field(ge=0)
    source_simplex_index: int = Field(ge=0)


class DegeneracyDegreeProfile(StrictModel):
    """A profile row is parallel to the source degree's simplex axis."""

    nondegenerate_indices: tuple[int, ...]
    immediate_witnesses: tuple[DegeneracyWitness | None, ...]

    @model_validator(mode="after")
    def require_partition(self) -> DegeneracyDegreeProfile:
        if tuple(sorted(set(self.nondegenerate_indices))) != self.nondegenerate_indices:
            raise ValueError("nondegenerate indices must be sorted and unique")
        if any(
            (witness is None) != (index in self.nondegenerate_indices)
            for index, witness in enumerate(self.immediate_witnesses)
        ):
            raise ValueError("indices and immediate witnesses must partition the axis")
        return self


class DegeneracyProfileResult(StrictModel):
    """Dimension-indexed nondegenerate indices and first immediate witnesses.

    A witness is the lexicographically first (degeneracy index, source index)
    whose degeneracy map reaches that simplex. It makes no claim to encode the
    full epi-mono normal form.
    """

    simplicial_set: FiniteTruncatedSimplicialSet
    degrees: tuple[DegeneracyDegreeProfile, ...]

    @model_validator(mode="after")
    def require_source_axes(self) -> DegeneracyProfileResult:
        if len(self.degrees) != self.simplicial_set.max_degree + 1:
            raise ValueError("profiles must cover every source degree")
        if any(
            len(profile.immediate_witnesses) != len(self.simplicial_set.sets[degree])
            for degree, profile in enumerate(self.degrees)
        ):
            raise ValueError("profile rows must retain their source simplex axes")
        return self


def _preflight(source: FiniteTruncatedSimplicialSet) -> tuple[int, ...]:
    sizes = tuple(len(level) for level in source.sets)
    # from_tables composes at most this many rows while checking all visible
    # identities. This upper bound is computed before any identity traversal.
    work = 0
    for n in range(1, source.max_degree + 1):
        work += (n + 1) * sizes[n]  # face-table axis admission
    for n in range(source.max_degree):
        work += (n + 1) * sizes[n]  # degeneracy-table axis admission
    for n in range(2, source.max_degree + 1):
        work += n * (n + 1) // 2 * sizes[n]
    for n in range(source.max_degree - 1):
        work += (n + 1) * (n + 2) // 2 * sizes[n]
    for n in range(source.max_degree):
        work += (n + 1) * (n + 2) * sizes[n]
        work += (n + 1) * sizes[n]  # profile's degeneracy-image scan
    # Profile output is at most one index plus one optional witness per input
    # simplex. Include a conservative bound for the retained source and JSON
    # framing; the source schema separately caps it at 96 simplices / degree 4.
    output_bound = len(source.model_dump_json()) + 160 * sum(sizes) + 256
    if work > _MAX_IDENTITY_ROW_WORK:
        raise OperationResourceAdmissionError(
            location=("simplicial_set",),
            code="simplicial_set.degeneracy_profile_work_budget_exceeded",
            message="simplicial identity checks exceed the profile work bound",
        )
    if sum(sizes) > MAX_TOTAL_SIMPLICES or output_bound > _MAX_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("simplicial_set",),
            code="simplicial_set.degeneracy_profile_output_budget_exceeded",
            message="degeneracy profile exceeds the bounded output size",
        )
    return sizes


def degeneracy_profile(
    source: FiniteTruncatedSimplicialSet,
) -> DegeneracyProfileResult:
    """Classify each simplex by its immediate degeneracy status."""
    sizes = _preflight(source)
    checked = from_tables(
        source.max_degree, source.sets, source.face_maps, source.degeneracy_maps
    )
    if checked.simplicial_set is None:
        raise OperationDomainValidationError(
            location=("simplicial_set",),
            code="simplicial_set.degeneracy_profile_source_invalid",
            message="source tables fail a visible simplicial identity",
        )
    admitted_source = checked.simplicial_set
    profiles: list[DegeneracyDegreeProfile] = []
    for degree, size in enumerate(sizes):
        witnesses: list[DegeneracyWitness | None] = [None] * size
        if degree:
            for degeneracy_index, row in enumerate(
                admitted_source.degeneracy_maps[degree - 1]
            ):
                for source_index, target_index in enumerate(row):
                    if witnesses[target_index] is None:
                        witnesses[target_index] = DegeneracyWitness(
                            degeneracy_index=degeneracy_index,
                            source_simplex_index=source_index,
                        )
        nondegenerate = tuple(
            index for index, witness in enumerate(witnesses) if witness is None
        )
        profiles.append(
            DegeneracyDegreeProfile(
                nondegenerate_indices=nondegenerate,
                immediate_witnesses=tuple(witnesses),
            )
        )
    return DegeneracyProfileResult(
        simplicial_set=admitted_source, degrees=tuple(profiles)
    )


__all__ = [
    "DegeneracyDegreeProfile",
    "DegeneracyProfileResult",
    "DegeneracyWitness",
    "degeneracy_profile",
]
