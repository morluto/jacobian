"""Bounded construction of finite simplicial subobject prefixes."""

from __future__ import annotations

from typing import NoReturn

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.simplicial_sets._models import (
    MAX_SIMPLICES_PER_DEGREE,
    MAX_TOTAL_SIMPLICES,
    FiniteTruncatedSimplicialSet,
)
from jacobian.math.topology.simplicial_sets.maps import TruncatedSimplicialMap
from jacobian.math.topology.simplicial_sets.operations import (
    _from_admitted_tables,
    admit_tables,
)
from jacobian.math.topology.simplicial_sets.subset_models import (
    SimplicialSubsetPrefix,
    SimplicialSubsetRequest,
)

MAX_SUBSET_CHECK_WORK = 100_000
MAX_SUBSET_CLOSURE_INCIDENCE = 4_096
MAX_SUBSET_RESULT_CELLS = 256_000


def _invalid(reason: str, message: str, *location: str | int) -> NoReturn:
    raise OperationDomainValidationError(
        location=location or ("degree_indices",),
        code=f"simplicial_set.subset.{reason}",
        message=message,
    )


def _identity_work(max_degree: int, sizes: tuple[int, ...]) -> int:
    identities = sum(
        degree * (degree + 1) // 2 * sizes[degree]
        for degree in range(2, max_degree + 1)
    )
    identities += sum(
        (degree + 1) * (degree + 2) // 2 * sizes[degree]
        for degree in range(max_degree - 1)
    )
    identities += sum(
        (degree + 1) * (degree + 2) * sizes[degree] for degree in range(max_degree)
    )
    # Each identity compares two constructed composite rows. Both can scan
    # every simplex on the source degree axis before they are found equal.
    return 2 * identities


def _output_cell_bound(
    ambient: FiniteTruncatedSimplicialSet,
    degree_indices: tuple[tuple[int, ...], ...],
    subset_incidence: int,
) -> int:
    source_sizes = tuple(map(len, ambient.sets))
    source_labels = sum(len(label) for level in ambient.sets for label in level)
    subset_labels = sum(
        len(ambient.sets[degree][index])
        for degree, indices in enumerate(degree_indices)
        for index in indices
    )
    source_map_entries = sum(
        (degree + 1) * source_sizes[degree]
        for degree in range(1, ambient.max_degree + 1)
    ) + sum((degree + 1) * source_sizes[degree] for degree in range(ambient.max_degree))
    # Target source tables occur once, the restricted prefix contributes at
    # most one index per admitted selected incidence, and the inclusion map
    # has at most one target index per selected simplex.
    return (
        source_labels
        + subset_labels
        + 3 * source_map_entries
        + 3 * subset_incidence
        + 3 * sum(map(len, degree_indices))
        + 4_096
    )


def _preflight_request(
    request: SimplicialSubsetRequest,
) -> tuple[
    FiniteTruncatedSimplicialSet,
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
]:
    if type(request) is not SimplicialSubsetRequest:
        _invalid("request_type", "request must be a typed simplicial subset request")
    ambient = request.simplicial_set
    if type(ambient) is not FiniteTruncatedSimplicialSet:
        _invalid("source_type", "source must be a finite truncated simplicial set")
    sizes = admit_tables(
        ambient.max_degree,
        ambient.sets,
        ambient.face_maps,
        ambient.degeneracy_maps,
    )
    degree_indices = request.degree_indices
    if (
        type(degree_indices) is not tuple
        or len(degree_indices) != ambient.max_degree + 1
    ):
        _invalid(
            "degree_coverage",
            "degree_indices must provide one family for every source degree",
        )
    total_selected = 0
    for degree, indices in enumerate(degree_indices):
        if (
            type(indices) is not tuple
            or len(indices) > MAX_SIMPLICES_PER_DEGREE
            or any(type(index) is not int for index in indices)
            or indices != tuple(sorted(set(indices)))
            or any(not 0 <= index < sizes[degree] for index in indices)
        ):
            _invalid(
                "degree_indices_invalid",
                f"degree {degree} indices must be an increasing subset of the source axis",
                "degree_indices",
                degree,
            )
        total_selected += len(indices)
    if total_selected > MAX_TOTAL_SIMPLICES:
        raise OperationResourceAdmissionError(
            location=("degree_indices",),
            code="simplicial_set.subset.total_simplex_budget_exceeded",
            message=(
                f"the selected prefix contains {total_selected} simplices, above "
                f"the {MAX_TOTAL_SIMPLICES}-simplex envelope"
            ),
        )

    max_degree = ambient.max_degree
    subset_incidence = sum(
        (degree + 1) * len(degree_indices[degree])
        for degree in range(1, max_degree + 1)
    ) + sum((degree + 1) * len(degree_indices[degree]) for degree in range(max_degree))
    identity_work = _identity_work(max_degree, sizes)
    source_map_incidence = sum(
        (degree + 1) * sizes[degree] for degree in range(1, max_degree + 1)
    ) + sum((degree + 1) * sizes[degree] for degree in range(max_degree))
    source_label_characters = sum(
        len(label) for level in ambient.sets for label in level
    )
    subset_work = 3 * subset_incidence + 3 * total_selected
    if (
        identity_work + source_map_incidence + source_label_characters + subset_work
        > MAX_SUBSET_CHECK_WORK
    ):
        raise OperationResourceAdmissionError(
            location=("simplicial_set",),
            code="simplicial_set.subset.work_budget_exceeded",
            message="source identity replay and subset closure exceed the work envelope",
        )
    if subset_incidence > MAX_SUBSET_CLOSURE_INCIDENCE:
        raise OperationResourceAdmissionError(
            location=("degree_indices",),
            code="simplicial_set.subset.incidence_budget_exceeded",
            message=(
                f"the selected family has {subset_incidence} face/degeneracy "
                f"incidences, above {MAX_SUBSET_CLOSURE_INCIDENCE}"
            ),
        )
    output_cells = _output_cell_bound(ambient, degree_indices, subset_incidence)
    if output_cells > MAX_SUBSET_RESULT_CELLS:
        raise OperationResourceAdmissionError(
            location=("degree_indices",),
            code="simplicial_set.subset.output_budget_exceeded",
            message=(
                f"estimated subobject output {output_cells} cells exceeds "
                f"{MAX_SUBSET_RESULT_CELLS}"
            ),
        )
    return ambient, degree_indices, sizes


def simplicial_subset(
    request: SimplicialSubsetRequest,
) -> SimplicialSubsetPrefix:
    """Build the selected degreewise family when it is closed under all maps."""

    ambient, degree_indices, sizes = _preflight_request(request)
    checked = _from_admitted_tables(
        ambient.max_degree,
        ambient.sets,
        ambient.face_maps,
        ambient.degeneracy_maps,
        sizes,
    )
    if checked.status != "SIMPLICIAL_SET" or checked.simplicial_set is None:
        _invalid(
            "source_invalid",
            "source tables do not satisfy all visible simplicial identities",
            "simplicial_set",
        )
    source = checked.simplicial_set
    selected = tuple(set(indices) for indices in degree_indices)
    for degree in range(source.max_degree + 1):
        request_checkpoint("during finite simplicial subset closure check")
        for local_index, source_index in enumerate(degree_indices[degree]):
            if degree:
                for face_index, face_row in enumerate(source.face_maps[degree - 1]):
                    if face_row[source_index] not in selected[degree - 1]:
                        _invalid(
                            "not_face_closed",
                            f"selected degree-{degree} simplex is missing face d_{face_index}",
                            "degree_indices",
                            degree,
                            local_index,
                        )
            if degree < source.max_degree:
                for degeneracy_index, degeneracy_row in enumerate(
                    source.degeneracy_maps[degree]
                ):
                    if degeneracy_row[source_index] not in selected[degree + 1]:
                        _invalid(
                            "not_degeneracy_closed",
                            "selected simplex is missing degeneracy "
                            f"s_{degeneracy_index} in degree {degree + 1}",
                            "degree_indices",
                            degree,
                            local_index,
                        )

    position_maps = tuple(
        {source_index: local_index for local_index, source_index in enumerate(indices)}
        for indices in degree_indices
    )
    subset_sets = tuple(
        tuple(source.sets[degree][index] for index in indices)
        for degree, indices in enumerate(degree_indices)
    )
    subset_faces = tuple(
        tuple(
            tuple(
                position_maps[degree - 1][face_row[source_index]]
                for source_index in degree_indices[degree]
            )
            for face_row in source.face_maps[degree - 1]
        )
        for degree in range(1, source.max_degree + 1)
    )
    subset_degeneracies = tuple(
        tuple(
            tuple(
                position_maps[degree + 1][degeneracy_row[source_index]]
                for source_index in degree_indices[degree]
            )
            for degeneracy_row in source.degeneracy_maps[degree]
        )
        for degree in range(source.max_degree)
    )
    subset = FiniteTruncatedSimplicialSet._from_kernel(
        max_degree=source.max_degree,
        sets=subset_sets,
        face_maps=subset_faces,
        degeneracy_maps=subset_degeneracies,
        total_simplices=sum(map(len, subset_sets)),
        checked_identities=source.checked_identities,
    )
    inclusion = TruncatedSimplicialMap(
        source=subset,
        target=source,
        maps=degree_indices,
    )
    return SimplicialSubsetPrefix(inclusion=inclusion)


__all__ = [
    "MAX_SUBSET_CHECK_WORK",
    "MAX_SUBSET_CLOSURE_INCIDENCE",
    "MAX_SUBSET_RESULT_CELLS",
    "simplicial_subset",
]
