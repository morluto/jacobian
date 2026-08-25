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


def barycentric_subdivision(faces: list[Face]) -> BarycentricSubdivision:
    """Return the order-complex facets for faces in canonical order.

    Vertices use the compact ``bv{i}`` encoding indexed by ``faces``.  The
    caller owns validation of the source complex and construction of its
    typed result; this helper owns only the deterministic finite transform.
    """

    face_frozens = [frozenset(face) for face in faces]
    covers = _cover_relations(face_frozens)
    minimal_indices = _minimal_face_indices(face_frozens)
    maximal_chains = _maximal_chains_from_covers(covers, minimal_indices, len(faces))
    vertices = tuple(f"bv{index}" for index in range(len(faces)))
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
        vertex_faces=tuple(faces),
        facets=facets,
    )


def _cover_relations(face_frozens: list[frozenset[str]]) -> list[list[int]]:
    """Return the strict cover relation in a finite face poset."""

    covers: list[list[int]] = [[] for _ in face_frozens]
    for lower, lower_face in enumerate(face_frozens):
        for upper, upper_face in enumerate(face_frozens):
            if lower_face < upper_face and not any(
                lower_face < candidate < upper_face for candidate in face_frozens
            ):
                covers[lower].append(upper)
    return covers


def _minimal_face_indices(face_frozens: list[frozenset[str]]) -> list[int]:
    return [
        index
        for index, face in enumerate(face_frozens)
        if not any(candidate < face for candidate in face_frozens)
    ]


def _maximal_chains_from_covers(
    covers: list[list[int]],
    minimal_indices: list[int],
    face_count: int,
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
    if not chains and face_count:
        chains.extend([[index] for index in range(face_count) if not covers[index]])
    return chains
