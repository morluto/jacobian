"""Finite simplicial maps and normalized-chain prefixes."""

from __future__ import annotations

from jacobian._models import StrictModel
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.simplicial_sets._models import FiniteTruncatedSimplicialSet


class SimplicialMapRequest(StrictModel):
    source: FiniteTruncatedSimplicialSet
    target: FiniteTruncatedSimplicialSet
    maps: tuple[tuple[int, ...], ...]


class SimplicialMapResult(StrictModel):
    source: FiniteTruncatedSimplicialSet
    target: FiniteTruncatedSimplicialSet
    maps: tuple[tuple[int, ...], ...]
    identities_preserved: bool


class NormalizedChainsRequest(StrictModel):
    simplicial_set: FiniteTruncatedSimplicialSet


class NormalizedChainsResult(StrictModel):
    simplicial_set: FiniteTruncatedSimplicialSet
    nondegenerate_counts: tuple[int, ...]
    boundary_matrices: tuple[tuple[tuple[int, ...], ...], ...]
    differential_squared_zero: bool


def simplicial_map(request: SimplicialMapRequest) -> SimplicialMapResult:
    s, t = request.source, request.target
    if s.max_degree != t.max_degree or len(request.maps) != s.max_degree + 1:
        raise OperationDomainValidationError(
            location=("maps",),
            code="simplicial_map.degree_axis",
            message="map must cover every degree of the retained prefixes",
        )
    for n, row in enumerate(request.maps):
        if len(row) != len(s.sets[n]) or any(
            not isinstance(i, int) or i < 0 or i >= len(t.sets[n]) for i in row
        ):
            raise OperationDomainValidationError(
                location=("maps", n),
                code="simplicial_map.row_axis",
                message="map rows must cover source simplices with target indices",
            )
    ok = True
    for n in range(1, s.max_degree + 1):
        for i, face in enumerate(s.face_maps[n - 1]):
            for q, x in enumerate(face):
                if request.maps[n][q] >= len(t.face_maps[n - 1][i]) or (
                    t.face_maps[n - 1][i][request.maps[n][q]] != request.maps[n - 1][x]
                ):
                    ok = False
    for n in range(s.max_degree):
        for i, deg in enumerate(s.degeneracy_maps[n]):
            for q, x in enumerate(deg):
                if (
                    t.degeneracy_maps[n][i][request.maps[n][q]]
                    != request.maps[n + 1][x]
                ):
                    ok = False
    return SimplicialMapResult(
        source=s, target=t, maps=request.maps, identities_preserved=ok
    )


def normalized_chains(
    simplicial_set: FiniteTruncatedSimplicialSet,
) -> NormalizedChainsResult:
    s = simplicial_set
    nd = []
    for n, level in enumerate(s.sets):
        degenerate = {x for row in (s.degeneracy_maps[n - 1] if n else ()) for x in row}
        nd.append(tuple(i for i in range(len(level)) if i not in degenerate))
    matrices = []
    for n in range(1, s.max_degree + 1):
        lower = nd[n - 1]
        lower_index = {x: i for i, x in enumerate(lower)}
        matrix = []
        matrix = [[0] * len(nd[n]) for _ in lower]
        for col, q in enumerate(nd[n]):
            for i, face in enumerate(s.face_maps[n - 1]):
                target = face[q]
                if target in lower_index:
                    matrix[lower_index[target]][col] += (-1) ** i
        matrices.append(tuple(tuple(v for v in row) for row in matrix))
    square = True
    for i in range(len(matrices) - 1):
        a, b = matrices[i], matrices[i + 1]
        prod = [
            [
                sum(a[r][k] * b[k][c] for k in range(len(b)))
                for c in range(len(b[0]) if b else 0)
            ]
            for r in range(len(a))
        ]
        square = square and all(v == 0 for row in prod for v in row)
    return NormalizedChainsResult(
        simplicial_set=s,
        nondegenerate_counts=tuple(len(x) for x in nd),
        boundary_matrices=tuple(matrices),
        differential_squared_zero=square,
    )


__all__ = [
    "NormalizedChainsRequest",
    "NormalizedChainsResult",
    "SimplicialMapRequest",
    "SimplicialMapResult",
    "normalized_chains",
    "simplicial_map",
]
