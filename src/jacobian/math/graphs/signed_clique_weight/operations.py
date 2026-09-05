"""Exact signed clique-weight maximization over nontrivial cliques."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.graphs.optimization._models import RationalWeightedGraph
from jacobian.math.graphs.signed_clique_weight._bounds import (
    admit_signed_clique_weight,
)
from jacobian.math.graphs.signed_clique_weight._models import (
    SignedCliqueWeightResult,
)

__all__ = ["signed_clique_weight_maximum"]


def _component_best(
    vertices: tuple[int, ...],
    adjacency: tuple[tuple[tuple[int, int], ...], ...],
    labels: tuple[str, ...],
) -> tuple[int, tuple[int, ...]] | None:
    """Return the best (scaled value, local witness) over nontrivial cliques.

    One Gray-code pass maintains the induced weight sum and the nonedge
    count incrementally: flipping x adjusts the sum over selected neighbors
    and the nonedge count over selected non-neighbors. A subset is eligible
    exactly with at least two vertices and zero nonedges. Ties resolve
    toward the lexicographically least label tuple.
    """

    size = len(vertices)
    neighbor_sets = tuple({neighbor for neighbor, _ in peers} for peers in adjacency)
    selected = [False] * size
    current_value = 0
    current_nonedges = 0
    selected_count = 0
    best_value: int | None = None
    best_set: tuple[int, ...] = ()
    best_key: tuple[str, ...] = ()
    previous_gray = 0
    for step in range(1, 1 << size):
        gray = step ^ (step >> 1)
        changed = (gray ^ previous_gray).bit_length() - 1
        if selected[changed]:
            selected[changed] = False
            selected_count -= 1
            current_value -= sum(
                weight for neighbor, weight in adjacency[changed] if selected[neighbor]
            )
            current_nonedges -= sum(
                1
                for peer in range(size)
                if peer != changed
                and selected[peer]
                and peer not in neighbor_sets[changed]
            )
        else:
            current_value += sum(
                weight for neighbor, weight in adjacency[changed] if selected[neighbor]
            )
            current_nonedges += sum(
                1
                for peer in range(size)
                if peer != changed
                and selected[peer]
                and peer not in neighbor_sets[changed]
            )
            selected[changed] = True
            selected_count += 1
        previous_gray = gray
        if selected_count >= 2 and current_nonedges == 0:
            witness = tuple(
                local for local, is_selected in enumerate(selected) if is_selected
            )
            key = tuple(labels[local] for local in witness)
            if (
                best_value is None
                or current_value > best_value
                or (current_value == best_value and key < best_key)
            ):
                best_value, best_set, best_key = current_value, witness, key
    if best_value is None:
        return None
    return best_value, best_set


def signed_clique_weight_maximum(
    graph: RationalWeightedGraph,
) -> SignedCliqueWeightResult:
    """Return the maximum signed edge-weight over cliques of order >= 2.

    Every nontrivial clique lies inside one biconnected block, so each
    block contributes its own optimum and the best (value, witness)
    pair wins globally. Graphs without edges report a missing optimum
    explicitly.
    """

    if not isinstance(graph, RationalWeightedGraph):
        raise TypeError("signed_clique_weight_maximum expects a RationalWeightedGraph")
    admission = admit_signed_clique_weight(graph)
    best_value: int | None = None
    best_clique: tuple[str, ...] = ()
    for component in admission.components:
        labels = tuple(graph.vertices[index] for index in component.vertices)
        found = _component_best(component.vertices, component.adjacency, labels)
        if found is None:
            continue
        value, local_set = found
        witness = tuple(
            graph.vertices[component.vertices[local]] for local in local_set
        )
        if (
            best_value is None
            or value > best_value
            or (value == best_value and witness < best_clique)
        ):
            best_value, best_clique = value, witness
    if best_value is None:
        return SignedCliqueWeightResult._from_kernel(
            graph=graph, optimum_value=None, clique=()
        )
    return SignedCliqueWeightResult._from_kernel(
        graph=graph,
        optimum_value=CanonicalRational.from_fraction(
            Fraction(best_value, admission.denominator)
        ),
        clique=best_clique,
    )
