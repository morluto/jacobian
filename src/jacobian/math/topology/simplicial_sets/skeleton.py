"""Finite-prefix simplicial-set skeleta and their inclusion maps."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, ValidationError, model_validator

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.simplicial_sets._models import (
    MAX_SIMPLICIAL_SET_DEGREE,
    MAX_TOTAL_SIMPLICES,
    FiniteTruncatedSimplicialSet,
)
from jacobian.math.topology.simplicial_sets.maps import TruncatedSimplicialMap
from jacobian.math.topology.simplicial_sets.operations import (
    _from_admitted_tables,
    admit_tables,
)
from jacobian.math.topology.simplicial_sets.subset_models import SimplicialSubsetPrefix

MAX_SKELETON_WORK = 100_000
MAX_SKELETON_OUTPUT_CELLS = 10_000


class SimplicialSetSkeletonRequest(StrictModel):
    """The k-skeleton of a source prefix through its existing maximum degree."""

    simplicial_set: FiniteTruncatedSimplicialSet
    k: StrictInt = Field(
        ge=0,
        le=MAX_SIMPLICIAL_SET_DEGREE,
        description=(
            "Skeleton degree; it must also satisfy k <= "
            "simplicial_set.max_degree for the supplied prefix."
        ),
    )


class SimplicialSetSkeletonResult(StrictModel):
    """A source-bound simplicial subset and its degreewise inclusion."""

    k: StrictInt = Field(ge=0, le=MAX_SIMPLICIAL_SET_DEGREE)
    skeleton: FiniteTruncatedSimplicialSet
    inclusion: TruncatedSimplicialMap

    @model_validator(mode="after")
    def require_inclusion_source(self) -> Self:
        try:
            skeleton = FiniteTruncatedSimplicialSet.model_validate(
                self.skeleton.model_dump()
            )
            inclusion_source = FiniteTruncatedSimplicialSet.model_validate(
                self.inclusion.source.model_dump()
            )
            target = FiniteTruncatedSimplicialSet.model_validate(
                self.inclusion.target.model_dump()
            )
            if inclusion_source != skeleton:
                raise ValueError(
                    "the inclusion source must equal the returned skeleton"
                )
            inclusion = TruncatedSimplicialMap.model_validate(
                {
                    "source": inclusion_source,
                    "target": target,
                    "maps": self.inclusion.maps,
                }
            )
            SimplicialSubsetPrefix(inclusion=inclusion)
        except (
            ValidationError,
            AttributeError,
            TypeError,
            ValueError,
            IndexError,
        ) as error:
            raise ValueError(
                "the result must retain canonical simplicial sets and an injective inclusion"
            ) from error
        if inclusion.source != skeleton:
            raise ValueError("the inclusion source must be the returned skeleton")
        if self.k > target.max_degree:
            raise ValueError("k must be visible in the source prefix")
        expected = [
            set(range(len(target.sets[degree]))) if degree <= self.k else set()
            for degree in range(target.max_degree + 1)
        ]
        for degree in range(target.max_degree):
            for degeneracy in target.degeneracy_maps[degree]:
                for source_index in tuple(expected[degree]):
                    expected[degree + 1].add(degeneracy[source_index])
        expected_maps = tuple(tuple(sorted(level)) for level in expected)
        if inclusion.maps != expected_maps:
            raise ValueError(
                "the retained simplices must equal degrees through k and their degeneracy closure"
            )
        return self


def _preflight(
    source: FiniteTruncatedSimplicialSet, k: int, sizes: tuple[int, ...]
) -> None:
    if type(k) is not int or not 0 <= k <= source.max_degree:
        raise OperationDomainValidationError(
            location=("k",),
            code="simplicial_set.skeleton_degree_out_of_bounds",
            message="k must be a degree visible in the source prefix",
        )
    total = sum(sizes)
    table_cells = sum(
        (degree + 1) * sizes[degree] for degree in range(1, source.max_degree + 1)
    )
    table_cells += sum(
        (degree + 1) * sizes[degree] for degree in range(source.max_degree)
    )
    identity_rows = sum(
        (degree + 1) * degree // 2 * sizes[degree]
        for degree in range(2, source.max_degree + 1)
    )
    identity_rows += sum(
        (degree + 1) * (degree + 2) // 2 * sizes[degree]
        for degree in range(source.max_degree - 1)
    )
    identity_rows += sum(
        (degree + 1) * (degree + 2) * sizes[degree]
        for degree in range(source.max_degree)
    )
    degeneracy_cells = sum(
        (degree + 1) * sizes[degree] for degree in range(source.max_degree)
    )
    # Check identities once at the untrusted source boundary. Restriction and
    # inclusion are constructed from those admitted tables without replay.
    work = identity_rows + 2 * table_cells + degeneracy_cells
    map_count = sum(degree + 1 for degree in range(1, source.max_degree + 1))
    map_count += sum(degree + 1 for degree in range(source.max_degree))
    # Bound the in-memory collections created by closure, restriction,
    # inclusion, and the returned skeleton by cardinality. Wire encoding owns
    # its independent byte ceiling at the delivery boundary.
    output_cells = 4 * total + 3 * table_cells + 3 * map_count
    if work > MAX_SKELETON_WORK:
        raise OperationResourceAdmissionError(
            location=("simplicial_set",),
            code="simplicial_set.skeleton_work_budget_exceeded",
            message="the admitted face/degeneracy scan exceeds the skeleton work bound",
        )
    if total > MAX_TOTAL_SIMPLICES or output_cells > MAX_SKELETON_OUTPUT_CELLS:
        raise OperationResourceAdmissionError(
            location=("simplicial_set",),
            code="simplicial_set.skeleton_output_budget_exceeded",
            message="the source-bound skeleton result exceeds its allocation bound",
        )


def simplicial_set_skeleton(
    simplicial_set: FiniteTruncatedSimplicialSet, k: int
) -> SimplicialSetSkeletonResult:
    """Generate the smallest finite simplicial subset containing degrees <= k.

    The degreewise closure is obtained by applying every visible degeneracy to
    the already admitted lower-degree members. Since degeneracies raise degree
    by one, a single ascending pass computes the full closure.
    """
    if not isinstance(simplicial_set, FiniteTruncatedSimplicialSet):
        raise OperationDomainValidationError(
            location=("simplicial_set",),
            code="simplicial_set.skeleton_source_invalid",
            message="simplicial_set must be a finite truncated simplicial set",
        )
    source = simplicial_set
    sizes = admit_tables(
        source.max_degree,
        source.sets,
        source.face_maps,
        source.degeneracy_maps,
    )
    admitted_total = sum(sizes)
    if (
        type(source.total_simplices) is not int
        or source.total_simplices != admitted_total
    ):
        raise OperationDomainValidationError(
            location=("simplicial_set", "total_simplices"),
            code="simplicial_set.total_simplex_count_mismatch",
            message="total_simplices must equal the admitted degree sizes",
        )
    _preflight(source, k, sizes)
    checked_source = _from_admitted_tables(
        source.max_degree,
        source.sets,
        source.face_maps,
        source.degeneracy_maps,
        sizes,
    )
    if checked_source.simplicial_set is None:
        raise OperationDomainValidationError(
            location=("simplicial_set",),
            code="simplicial_set.skeleton_source_invalid",
            message="the source tables fail a visible simplicial identity",
        )
    source = checked_source.simplicial_set

    included = [[degree <= k] * size for degree, size in enumerate(sizes)]
    for degree in range(source.max_degree):
        for row in source.degeneracy_maps[degree]:
            for simplex_index, image in enumerate(row):
                if included[degree][simplex_index]:
                    included[degree + 1][image] = True

    source_indices = tuple(
        tuple(index for index, keep in enumerate(level) if keep) for level in included
    )
    skeleton_sets, faces, degeneracies = _restrict_tables(source, source_indices)
    skeleton_sizes = tuple(map(len, skeleton_sets))
    skeleton_identities = (
        sum((degree + 1) * degree // 2 for degree in range(2, source.max_degree + 1))
        + sum(
            (degree + 1) * (degree + 2) // 2 for degree in range(source.max_degree - 1)
        )
        + sum((degree + 1) * (degree + 2) for degree in range(source.max_degree))
    )
    # Degeneracy closure preserves restricted identities, and restriction
    # indices intertwine every source map by construction. These trusted kernel
    # constructors avoid repeating those admitted scans during production.
    skeleton = FiniteTruncatedSimplicialSet._from_kernel(
        max_degree=source.max_degree,
        sets=skeleton_sets,
        face_maps=faces,
        degeneracy_maps=degeneracies,
        total_simplices=sum(skeleton_sizes),
        checked_identities=skeleton_identities,
    )
    inclusion = TruncatedSimplicialMap.model_construct(
        source=skeleton, target=source, maps=source_indices
    )

    return SimplicialSetSkeletonResult(k=k, skeleton=skeleton, inclusion=inclusion)


def _restrict_tables(
    source: FiniteTruncatedSimplicialSet,
    source_indices: tuple[tuple[int, ...], ...],
) -> tuple[
    tuple[tuple[str, ...], ...],
    tuple[tuple[tuple[int, ...], ...], ...],
    tuple[tuple[tuple[int, ...], ...], ...],
]:
    positions = tuple(
        {source_index: target_index for target_index, source_index in enumerate(level)}
        for level in source_indices
    )
    skeleton_sets = tuple(
        tuple(source.sets[degree][index] for index in source_indices[degree])
        for degree in range(source.max_degree + 1)
    )
    faces = tuple(
        tuple(
            tuple(
                positions[degree - 1][source.face_maps[degree - 1][face][index]]
                for index in source_indices[degree]
            )
            for face in range(degree + 1)
        )
        for degree in range(1, source.max_degree + 1)
    )
    degeneracies = tuple(
        tuple(
            tuple(
                positions[degree + 1][source.degeneracy_maps[degree][degeneracy][index]]
                for index in source_indices[degree]
            )
            for degeneracy in range(degree + 1)
        )
        for degree in range(source.max_degree)
    )
    return skeleton_sets, faces, degeneracies


__all__ = [
    "SimplicialSetSkeletonRequest",
    "SimplicialSetSkeletonResult",
    "simplicial_set_skeleton",
]
