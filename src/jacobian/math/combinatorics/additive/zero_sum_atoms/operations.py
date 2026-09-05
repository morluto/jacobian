"""Complete construction of finite-Abelian zero-sum atom hypergraphs."""

from __future__ import annotations

from itertools import combinations

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.additive.zero_sum_atoms._models import (
    MAX_ATOM_EDGES,
    MAX_ATOM_INCIDENCES,
    MAX_ATOM_SOURCE_ELEMENTS,
    MAX_ATOM_SUBSET_CHECKS,
    ZeroSumAtomHypergraphResult,
    ZeroSumAtomSource,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)

__all__ = ["construct_zero_sum_atom_hypergraph"]


def _reject(location: tuple[str | int, ...], code: str, message: str) -> None:
    raise OperationDomainValidationError(location=location, code=code, message=message)


def _add(
    left: tuple[int, ...],
    right: tuple[int, ...],
    moduli: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        (x + y) % modulus for x, y, modulus in zip(left, right, moduli, strict=True)
    )


def _admit_zero_sum_atom_source(source: ZeroSumAtomSource) -> None:
    if not isinstance(source, ZeroSumAtomSource):
        _reject(
            ("source",),
            "zero_sum_atom.source_domain",
            "source must be a ZeroSumAtomSource",
        )
    element_count = len(source.elements)
    if element_count > MAX_ATOM_SOURCE_ELEMENTS:
        _reject(
            ("source", "elements"),
            "zero_sum_atom.source_cardinality",
            "source exceeds the "
            f"{MAX_ATOM_SOURCE_ELEMENTS}-element zero-sum atom bound",
        )
    subset_checks = 1 << element_count
    if subset_checks > MAX_ATOM_SUBSET_CHECKS:
        _reject(
            ("source", "elements"),
            "zero_sum_atom.subset_checks",
            "complete zero-sum subset enumeration exceeds the "
            f"{MAX_ATOM_SUBSET_CHECKS:,}-subset bound",
        )


def construct_zero_sum_atom_hypergraph(
    source: ZeroSumAtomSource,
) -> ZeroSumAtomHypergraphResult:
    """Return every inclusion-minimal nonempty zero-sum subset of a source.

    The kernel first computes every zero-sum subset by increasing cardinality.
    A zero-sum subset is retained exactly when no previously retained atom is
    contained in it. Because all retained atoms have strictly smaller
    cardinality, that containment test is exactly the nonempty-proper-subset
    zero-sum test. Completeness is therefore established by one complete
    enumeration of nonempty source subsets; no timeout or partial search can
    produce this result.
    """

    _admit_zero_sum_atom_source(source)
    elements = source.elements
    moduli = source.group.moduli
    zero = tuple(0 for _ in moduli)
    atom_masks: list[int] = []
    subset_checks = 0
    minimality_checks = 0

    for size in range(1, len(elements) + 1):
        for positions in combinations(range(len(elements)), size):
            request_checkpoint("zero-sum atom subset enumeration")
            subset_checks += 1
            running = zero
            mask = 0
            for position in positions:
                running = _add(running, elements[position], moduli)
                mask |= 1 << position
            if running != zero:
                continue
            contains_atom = False
            for atom_mask in atom_masks:
                minimality_checks += 1
                if atom_mask & mask == atom_mask:
                    contains_atom = True
                    break
            if contains_atom:
                continue
            atom_masks.append(mask)

    atom_count = len(atom_masks)
    total_incidences = sum(mask.bit_count() for mask in atom_masks)
    if atom_count > MAX_ATOM_EDGES or total_incidences > MAX_ATOM_INCIDENCES:
        _reject(
            ("source", "elements"),
            "zero_sum_atom.result_size",
            "complete zero-sum atom family exceeds the hypergraph result bound",
        )

    label_width = len(str(max(0, len(elements) - 1)))

    def label(index: int) -> str:
        return str(index).zfill(label_width)

    vertices = tuple(label(index) for index in range(len(elements)))
    edges: list[tuple[str, tuple[str, ...]]] = []
    for mask in atom_masks:
        members = tuple(label(index) for index in range(len(elements)) if mask & (1 << index))
        edges.append((",".join(members), members))
    hypergraph = FiniteHypergraph(vertices=vertices, edges=tuple(edges))

    return ZeroSumAtomHypergraphResult(
        source=source,
        hypergraph=hypergraph,
        vertex_source_indices=tuple(range(len(elements))),
        atom_count=atom_count,
        total_incidences=total_incidences,
        subset_checks=subset_checks,
        minimality_checks=minimality_checks,
    )
