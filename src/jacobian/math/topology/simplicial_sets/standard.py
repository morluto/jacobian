"""Finite standard simplices, boundaries, and horns.

The construction is deliberately finite: a request chooses a simplex dimension
and a degree prefix, and the returned tables contain every monotone map in that
prefix together with all in-range faces and degeneracies.
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from math import comb

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.simplicial_sets._models import (
    MAX_SIMPLICES_PER_DEGREE,
    MAX_SIMPLICIAL_SET_DEGREE,
    MAX_TOTAL_SIMPLICES,
    FiniteTruncatedSimplicialSet,
)


def _admit(n: int, max_degree: int) -> None:
    if (
        isinstance(n, bool)
        or not isinstance(n, int)
        or n < 0
        or n > MAX_SIMPLICIAL_SET_DEGREE
    ):
        raise OperationDomainValidationError(
            location=("dimension",),
            code="simplicial_set.dimension_invalid",
            message="dimension must be a nonnegative integer within the finite bound",
        )
    if (
        isinstance(max_degree, bool)
        or not isinstance(max_degree, int)
        or max_degree < 0
        or max_degree > MAX_SIMPLICIAL_SET_DEGREE
    ):
        raise OperationDomainValidationError(
            location=("max_degree",),
            code="simplicial_set.degree_invalid",
            message="max_degree must lie within the finite degree bound",
        )
    if max_degree + 1 > 5:
        raise OperationResourceAdmissionError(
            location=("max_degree",),
            code="simplicial_set.degree_budget",
            message="the finite simplicial-set degree prefix is bounded",
        )
    # Monotone maps [k] -> [n] are counted before any tables, labels, or
    # simplicial identities are materialized.  This coupled admission mirrors
    # the canonical carrier's per-degree and aggregate limits.
    total = 0
    for degree in range(max_degree + 1):
        count = comb(n + degree + 1, degree + 1)
        if count > MAX_SIMPLICES_PER_DEGREE:
            raise OperationResourceAdmissionError(
                location=("max_degree",),
                code="simplicial_set.degree_size_budget",
                message=(
                    "the requested dimension/degree prefix produces more than "
                    f"{MAX_SIMPLICES_PER_DEGREE} simplices in one degree"
                ),
            )
        total += count
        if total > MAX_TOTAL_SIMPLICES:
            raise OperationResourceAdmissionError(
                location=("max_degree",),
                code="simplicial_set.simplex_budget",
                message="the standard finite prefix exceeds the simplex output bound",
            )


def _allowed(kind: str, n: int, k: int | None, simplex: tuple[int, ...]) -> bool:
    image = set(simplex)
    if kind == "simplex":
        return True
    if kind == "boundary":
        return len(image) < n + 1
    # A simplex lies in Lambda^n_k iff it lies in a face opposite some i != k.
    return any(i != k and i not in image for i in range(n + 1))


def _tables(
    kind: str, n: int, max_degree: int, k: int | None = None
) -> FiniteTruncatedSimplicialSet:
    values: list[tuple[tuple[int, ...], ...]] = []
    for degree in range(max_degree + 1):
        level = tuple(
            simplex
            for simplex in combinations_with_replacement(range(n + 1), degree + 1)
            if _allowed(kind, n, k, simplex)
        )
        values.append(level)
    labels = tuple(
        tuple("(" + ",".join(map(str, simplex)) + ")" for simplex in level)
        for level in values
    )
    index = [{simplex: i for i, simplex in enumerate(level)} for level in values]
    faces: list[tuple[tuple[int, ...], ...]] = []
    for degree in range(1, max_degree + 1):
        maps = []
        for i in range(degree + 1):
            maps.append(
                tuple(
                    index[degree - 1][simplex[:i] + simplex[i + 1 :]]
                    for simplex in values[degree]
                )
            )
        faces.append(tuple(maps))
    degeneracies: list[tuple[tuple[int, ...], ...]] = []
    for degree in range(max_degree):
        maps = []
        for i in range(degree + 1):
            maps.append(
                tuple(
                    index[degree + 1][
                        (*simplex[: i + 1], simplex[i], *simplex[i + 1 :])
                    ]
                    for simplex in values[degree]
                )
            )
        degeneracies.append(tuple(maps))
    total = sum(map(len, labels))
    if total > MAX_TOTAL_SIMPLICES:
        raise OperationResourceAdmissionError(
            location=("max_degree",),
            code="simplicial_set.simplex_budget",
            message="the standard finite prefix exceeds the simplex output bound",
        )
    from jacobian.math.topology.simplicial_sets.operations import from_tables

    checked = from_tables(max_degree, labels, tuple(faces), tuple(degeneracies))
    if checked.status != "SIMPLICIAL_SET" or checked.simplicial_set is None:
        raise RuntimeError("standard simplex construction produced invalid identities")
    return checked.simplicial_set


def standard_simplex(dimension: int, max_degree: int) -> FiniteTruncatedSimplicialSet:
    _admit(dimension, max_degree)
    return _tables("simplex", dimension, max_degree)


def simplex_boundary(dimension: int, max_degree: int) -> FiniteTruncatedSimplicialSet:
    _admit(dimension, max_degree)
    return _tables("boundary", dimension, max_degree)


def simplex_horn(
    dimension: int, missing_face: int, max_degree: int
) -> FiniteTruncatedSimplicialSet:
    _admit(dimension, max_degree)
    if dimension < 1 or not 0 <= missing_face <= dimension:
        raise OperationDomainValidationError(
            location=("missing_face",),
            code="simplicial_set.horn_face_invalid",
            message="missing_face must index a face of a positive-dimensional simplex",
        )
    return _tables("horn", dimension, max_degree, missing_face)


__all__ = ["simplex_boundary", "simplex_horn", "standard_simplex"]
