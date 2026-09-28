"""Pure canonical construction for barycentric subdivision."""

from __future__ import annotations

from dataclasses import dataclass

type Face = tuple[str, ...]


@dataclass(frozen=True)
class BarycentricSubdivision:
    """The deterministic subdivision data for canonical nonempty faces."""

    vertices: tuple[str, ...]
    vertex_faces: tuple[Face, ...]
    facets: tuple[Face, ...]


class BarycentricSubdivisionLimitExceededError(ValueError):
    """The number of maximal chains exceeds a caller's output budget."""


def barycentric_subdivision(
    faces: list[Face], *, maximal_chain_limit: int
) -> BarycentricSubdivision:
    """Return the order-complex facets for faces in canonical order.

    Vertices use the compact ``bv{i}`` encoding indexed by ``faces``.  The
    caller owns validation of the source complex and construction of its
    typed result; this helper owns only the deterministic finite transform.
    """

    ordered_faces = tuple(sorted(set(faces), key=lambda face: (len(face), face)))
    face_frozens = [frozenset(face) for face in ordered_faces]
    covers = _cover_relations(face_frozens)
    minimal_indices = _minimal_face_indices(face_frozens)
    chain_count = _maximal_chain_count(covers, minimal_indices, maximal_chain_limit)
    if chain_count > maximal_chain_limit:
        raise BarycentricSubdivisionLimitExceededError
    maximal_chains = _maximal_chains_from_covers(covers, minimal_indices)
    vertices = tuple(f"bv{index}" for index in range(len(ordered_faces)))
    facets = tuple(
        sorted(
            {
                tuple(sorted(vertices[index] for index in chain))
                for chain in maximal_chains
            },
            key=lambda facet: (-len(facet), facet),
        )
    )
    return BarycentricSubdivision(
        vertices=vertices,
        vertex_faces=ordered_faces,
        facets=facets,
    )


def _cover_relations(face_frozens: list[frozenset[str]]) -> list[list[int]]:
    """Return the strict cover relation in a finite face poset."""

    covers: list[list[int]] = [[] for _ in face_frozens]
    face_index = {face: index for index, face in enumerate(face_frozens)}
    vertices = tuple(
        sorted(next(iter(face)) for face in face_frozens if len(face) == 1)
    )
    for lower, lower_face in enumerate(face_frozens):
        for vertex in vertices:
            if vertex in lower_face:
                continue
            upper = face_index.get(lower_face | {vertex})
            if upper is not None:
                covers[lower].append(upper)
    return covers


def _minimal_face_indices(face_frozens: list[frozenset[str]]) -> list[int]:
    return [index for index, face in enumerate(face_frozens) if len(face) == 1]


def _maximal_chain_count(
    covers: list[list[int]], minimal_indices: list[int], limit: int
) -> int:
    """Count maximal chains, saturating just above the output limit."""

    counts = [0] * len(covers)
    for index in range(len(covers) - 1, -1, -1):
        if not covers[index]:
            counts[index] = 1
        else:
            counts[index] = min(
                limit + 1, sum(counts[upper] for upper in covers[index])
            )
    return min(limit + 1, sum(counts[index] for index in minimal_indices))


def _maximal_chains_from_covers(
    covers: list[list[int]],
    minimal_indices: list[int],
) -> list[list[int]]:
    """Enumerate maximal chains from the Hasse diagram."""

    chains: list[list[int]] = []

    def visit(chain: list[int]) -> None:
        last = chain[-1]
        if not covers[last]:
            chains.append(list(chain))
            return
        for next_index in covers[last]:
            chain.append(next_index)
            visit(chain)
            chain.pop()

    for start in minimal_indices:
        visit([start])
    return chains
