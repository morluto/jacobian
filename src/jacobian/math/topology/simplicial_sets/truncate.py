"""Exact bounded truncation of finite simplicial-set prefixes."""

from __future__ import annotations

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.simplicial_sets._models import (
    MAX_SIMPLICIAL_SET_DEGREE,
    FiniteTruncatedSimplicialSet,
)
from jacobian.math.topology.simplicial_sets.operations import (
    _from_admitted_tables,
    admit_tables,
)
from jacobian.math.topology.simplicial_sets.truncate_models import (
    SimplicialSetTruncateRequest,
)

MAX_TRUNCATE_MAP_ENTRIES = 50_000
MAX_TRUNCATE_OUTPUT_BYTES = 1_000_000
MAX_JSON_ASCII_CHARS_PER_LABEL_CHAR = 12


def _estimate_output_bytes(
    sets: tuple[tuple[str, ...], ...], degree: int, map_entries: int
) -> int:
    labels = tuple(label for level in sets[: degree + 1] for label in level)
    # JSON ensure_ascii can encode one astral scalar as a 12-character surrogate
    # pair. Include per-label quoting/separators and map-list structure too.
    encoded_label_chars = sum(
        len(label) * MAX_JSON_ASCII_CHARS_PER_LABEL_CHAR + 3 for label in labels
    )
    map_count = sum(n + 1 for n in range(1, degree + 1)) + sum(
        n + 1 for n in range(degree)
    )
    # Each map index is below 32, so two digits plus a list separator suffice.
    # The fixed allowance covers all table/list/model keys, levels and scalars.
    return encoded_label_chars + map_entries * 3 + map_count * 3 + 1024


def truncate_simplicial_set(
    request: SimplicialSetTruncateRequest,
) -> FiniteTruncatedSimplicialSet:
    """Retain degrees 0..N and exactly the maps visible in that prefix."""
    source, degree = request.simplicial_set, request.max_degree
    if type(degree) is not int or not 0 <= degree <= MAX_SIMPLICIAL_SET_DEGREE:
        raise OperationDomainValidationError(
            location=("max_degree",),
            code="simplicial_set.degree_out_of_bounds",
            message=(
                f"max_degree must be an integer in 0..{MAX_SIMPLICIAL_SET_DEGREE}"
            ),
        )
    if type(source.max_degree) is not int or not 0 <= source.max_degree <= (
        MAX_SIMPLICIAL_SET_DEGREE
    ):
        raise OperationDomainValidationError(
            location=("simplicial_set", "max_degree"),
            code="simplicial_set.degree_out_of_bounds",
            message=(
                "source maximum degree must be an integer in "
                f"0..{MAX_SIMPLICIAL_SET_DEGREE}"
            ),
        )
    if degree > source.max_degree:
        raise OperationDomainValidationError(
            location=("max_degree",),
            code="simplicial_set.truncation_exceeds_source",
            message="max_degree must not exceed the source maximum degree",
        )

    if not all(
        isinstance(table, tuple)
        for table in (source.sets, source.face_maps, source.degeneracy_maps)
    ):
        raise OperationDomainValidationError(
            location=("simplicial_set",),
            code="simplicial_set.degree_coverage_invalid",
            message="source degree and map tables must be tuples",
        )
    sets = source.sets[: degree + 1]
    faces = source.face_maps[:degree]
    degeneracies = source.degeneracy_maps[:degree]
    sizes = admit_tables(degree, sets, faces, degeneracies)
    map_entries = sum(sizes[n] * (n + 1) for n in range(1, degree + 1))
    map_entries += sum(sizes[n] * (n + 1) for n in range(degree))
    estimated_bytes = _estimate_output_bytes(sets, degree, map_entries)
    if map_entries > MAX_TRUNCATE_MAP_ENTRIES:
        raise OperationResourceAdmissionError(
            location=("simplicial_set",),
            code="simplicial_set.truncation_map_entry_budget_exceeded",
            message=(
                f"truncation requires {map_entries} map entries, above "
                f"{MAX_TRUNCATE_MAP_ENTRIES}"
            ),
        )
    if estimated_bytes > MAX_TRUNCATE_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("simplicial_set",),
            code="simplicial_set.truncation_output_budget_exceeded",
            message=(
                f"estimated truncation output {estimated_bytes} bytes exceeds "
                f"{MAX_TRUNCATE_OUTPUT_BYTES}"
            ),
        )

    # Recheck identities only after the caller-supplied axes are admitted.
    checked = _from_admitted_tables(degree, sets, faces, degeneracies, sizes)
    if checked.status != "SIMPLICIAL_SET" or checked.simplicial_set is None:
        raise OperationDomainValidationError(
            location=("simplicial_set",),
            code="simplicial_set.truncation_source_invalid",
            message="retained source tables do not satisfy all visible simplicial identities",
        )
    return checked.simplicial_set


__all__ = ["truncate_simplicial_set"]
