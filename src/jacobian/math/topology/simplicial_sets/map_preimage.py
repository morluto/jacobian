"""Finite-prefix inverse images of simplicial subobjects."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from jacobian._execution import request_checkpoint
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.simplicial_sets._models import FiniteTruncatedSimplicialSet
from jacobian.math.topology.simplicial_sets.maps import (
    TruncatedSimplicialMap,
    _require_naturality,
)
from jacobian.math.topology.simplicial_sets.operations import from_tables
from jacobian.math.topology.simplicial_sets.subset_models import (
    SimplicialSubsetPrefix,
)

_MAX_PREIMAGE_WORK = 100_000
_MAX_PREIMAGE_OUTPUT_CELLS = 262_144


class SimplicialMapPreimageRequest(StrictModel):
    """Pull a complete finite-prefix subobject back along a simplicial map."""

    simplicial_map: TruncatedSimplicialMap
    target_subset: SimplicialSubsetPrefix


class SimplicialMapPreimageResult(StrictModel):
    """The preimage subobject and the induced map into the selected target."""

    simplicial_map: TruncatedSimplicialMap
    target_subset: SimplicialSubsetPrefix
    preimage: SimplicialSubsetPrefix
    restricted_map: TruncatedSimplicialMap

    @model_validator(mode="after")
    def require_pullback_axes(self) -> Self:
        if (
            self.target_subset.inclusion.target != self.simplicial_map.target
            or self.preimage.inclusion.target != self.simplicial_map.source
            or self.restricted_map.source != self.preimage.inclusion.source
            or self.restricted_map.target != self.target_subset.inclusion.source
        ):
            raise ValueError("preimage factors must retain the original map axes")
        return self


def _identity_work(value: FiniteTruncatedSimplicialSet) -> int:
    sizes = tuple(map(len, value.sets))
    top = value.max_degree
    identities = sum(
        size
        * (
            ((degree + 1) * degree // 2 if degree >= 2 else 0)
            + ((degree + 1) * (degree + 2) // 2 if degree + 2 <= top else 0)
            + ((degree + 1) * (degree + 2) if degree < top else 0)
        )
        for degree, size in enumerate(sizes)
    )
    # Each checked identity compares two map composites.
    return 4 * identities


def _carrier_admission_work(value: FiniteTruncatedSimplicialSet) -> int:
    sizes = tuple(map(len, value.sets))
    table_entries = sum(
        (degree + 1) * sizes[degree] for degree in range(1, value.max_degree + 1)
    ) + sum((degree + 1) * sizes[degree] for degree in range(value.max_degree))
    return table_entries + sum(len(label) for level in value.sets for label in level)


def _carrier_cells(value: FiniteTruncatedSimplicialSet) -> int:
    sizes = tuple(map(len, value.sets))
    table_entries = sum(
        (degree + 1) * sizes[degree] for degree in range(1, value.max_degree + 1)
    ) + sum((degree + 1) * sizes[degree] for degree in range(value.max_degree))
    labels = sum(len(label) for level in value.sets for label in level)
    return sum(sizes) + table_entries + labels


def _naturality_work(value: TruncatedSimplicialMap) -> int:
    sizes = tuple(map(len, value.source.sets))
    return sum(
        size * (degree + 1) * (int(degree > 0) + int(degree < value.source.max_degree))
        for degree, size in enumerate(sizes)
    )


def _checked_carrier(
    value: FiniteTruncatedSimplicialSet, *, location: str
) -> FiniteTruncatedSimplicialSet:
    checked = from_tables(
        value.max_degree, value.sets, value.face_maps, value.degeneracy_maps
    )
    if checked.status != "SIMPLICIAL_SET" or checked.simplicial_set is None:
        raise OperationDomainValidationError(
            location=(location,),
            code="simplicial_map.preimage_carrier_invalid",
            message=f"{location} does not satisfy every visible simplicial identity",
        )
    return checked.simplicial_set


def _preflight(request: SimplicialMapPreimageRequest) -> None:
    value = request.simplicial_map
    target_inclusion = request.target_subset.inclusion
    source, target = value.source, value.target
    selected_target = target_inclusion.source
    if target_inclusion.target != target:
        raise OperationDomainValidationError(
            location=("target_subset", "inclusion", "target"),
            code="simplicial_map.preimage_target_mismatch",
            message="the target subobject must be included in the map target",
        )

    sizes = tuple(map(len, source.sets))
    subset_sizes = tuple(map(len, selected_target.sets))
    map_cells = sum(sizes)
    inclusion_cells = sum(subset_sizes)
    identity_work = (
        _identity_work(source)
        + _identity_work(target)
        + _identity_work(selected_target)
    )
    naturality_work = _naturality_work(value) + _naturality_work(target_inclusion)
    preimage_incidence = sum(
        (degree + 1)
        * sizes[degree]
        * (int(degree > 0) + int(degree < source.max_degree))
        for degree in range(source.max_degree + 1)
    )
    work = (
        identity_work
        + _carrier_admission_work(source)
        + _carrier_admission_work(target)
        + _carrier_admission_work(selected_target)
        + naturality_work
        + map_cells
        + inclusion_cells
        + preimage_incidence
        + 2 * sum(sizes)
    )
    if work > _MAX_PREIMAGE_WORK:
        raise OperationResourceAdmissionError(
            location=("simplicial_map",),
            code="simplicial_map.preimage_work_budget",
            message="the pullback and its input relation checks exceed the work bound",
        )

    source_cells = _carrier_cells(source)
    target_cells = _carrier_cells(target)
    subset_cells = _carrier_cells(selected_target)
    input_map_cells = sum(map(len, value.maps)) + sum(map(len, target_inclusion.maps))
    # The result retains the source and target ambient carriers twice, the
    # selected subset twice, and the two input maps plus the induced preimage
    # and restriction rows. Index rows cannot be wider than the already-admitted
    # source, target, or subset axes.
    output_cells = (
        4 * source_cells
        + 4 * target_cells
        + 2 * subset_cells
        + 2 * input_map_cells
        + 2 * (sum(sizes) + sum(subset_sizes))
        + 4_096
    )
    if output_cells > _MAX_PREIMAGE_OUTPUT_CELLS:
        raise OperationResourceAdmissionError(
            location=("simplicial_map",),
            code="simplicial_map.preimage_output_budget",
            message="the simplicial-map preimage result exceeds the admitted cell bound",
        )


def simplicial_map_preimage(
    request: SimplicialMapPreimageRequest,
) -> SimplicialMapPreimageResult:
    """Compute the degreewise inverse image of a finite simplicial subobject.

    The preimage in each degree consists of source simplices whose image lies
    in the selected target family. Naturality of the map and closure of the
    selected subobject make these families closed under every visible face and
    degeneracy; the returned map is the original map restricted to them.
    """
    _preflight(request)
    value = request.simplicial_map
    source = _checked_carrier(value.source, location="simplicial_map.source")
    target = _checked_carrier(value.target, location="simplicial_map.target")
    subset = _checked_carrier(
        request.target_subset.inclusion.source,
        location="target_subset.inclusion.source",
    )
    checked_map = TruncatedSimplicialMap(source=source, target=target, maps=value.maps)
    checked_inclusion = TruncatedSimplicialMap(
        source=subset,
        target=target,
        maps=request.target_subset.inclusion.maps,
    )
    _require_naturality(checked_map, location="simplicial_map")
    _require_naturality(checked_inclusion, location="target_subset.inclusion")

    inverse_target_indices = tuple(
        {ambient_index: local_index for local_index, ambient_index in enumerate(row)}
        for row in checked_inclusion.maps
    )
    preimage_indices_list: list[tuple[int, ...]] = []
    restricted_rows: list[tuple[int, ...]] = []
    for degree, row in enumerate(checked_map.maps):
        request_checkpoint("during finite simplicial map preimage construction")
        local_map = inverse_target_indices[degree]
        indices = tuple(
            source_index
            for source_index, target_index in enumerate(row)
            if target_index in local_map
        )
        preimage_indices_list.append(indices)
        restricted_rows.append(tuple(local_map[row[index]] for index in indices))
    preimage_indices = tuple(preimage_indices_list)
    preimage_positions = tuple(
        {source_index: local_index for local_index, source_index in enumerate(row)}
        for row in preimage_indices
    )
    preimage_sets = tuple(
        tuple(source.sets[degree][index] for index in indices)
        for degree, indices in enumerate(preimage_indices)
    )
    preimage_faces = tuple(
        tuple(
            tuple(
                preimage_positions[degree - 1][face[source_index]]
                for source_index in preimage_indices[degree]
            )
            for face in source.face_maps[degree - 1]
        )
        for degree in range(1, source.max_degree + 1)
    )
    preimage_degeneracies = tuple(
        tuple(
            tuple(
                preimage_positions[degree + 1][degeneracy[source_index]]
                for source_index in preimage_indices[degree]
            )
            for degeneracy in source.degeneracy_maps[degree]
        )
        for degree in range(source.max_degree)
    )
    preimage_carrier = FiniteTruncatedSimplicialSet._from_kernel(
        max_degree=source.max_degree,
        sets=preimage_sets,
        face_maps=preimage_faces,
        degeneracy_maps=preimage_degeneracies,
        total_simplices=sum(map(len, preimage_sets)),
        checked_identities=source.checked_identities,
    )
    preimage = SimplicialSubsetPrefix(
        inclusion=TruncatedSimplicialMap(
            source=preimage_carrier,
            target=source,
            maps=preimage_indices,
        )
    )
    restricted_map = TruncatedSimplicialMap(
        source=preimage_carrier,
        target=subset,
        maps=tuple(restricted_rows),
    )
    return SimplicialMapPreimageResult(
        simplicial_map=checked_map,
        target_subset=SimplicialSubsetPrefix(inclusion=checked_inclusion),
        preimage=preimage,
        restricted_map=restricted_map,
    )


__all__ = [
    "SimplicialMapPreimageRequest",
    "SimplicialMapPreimageResult",
    "simplicial_map_preimage",
]
