"""Domain-owned numerical semigroup operations."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.numerical_semigroups._algorithms import (
    betti_data,
    catenary_degree_from_factorizations,
    delta_periodicity_bound,
    factorization_length_extrema,
    factorization_lengths,
    factorizations,
    minimal_generating_system,
)
from jacobian.math.numerical_semigroups._models import (
    BettiCatenaryDegree,
    BettiElementsRequest,
    BettiElementsResult,
    CatenaryDegreeRequest,
    CatenaryDegreeResult,
    DeltaSetRequest,
    DeltaSetResult,
    ElasticityRequest,
    ElasticityResult,
    ElementCatenaryDegreeRequest,
    ElementCatenaryDegreeResult,
    ElementDeltaSetRequest,
    ElementDeltaSetResult,
    ElementElasticityRequest,
    ElementElasticityResult,
)


def _minimal_generators_list(gens: tuple[str, ...]) -> list[int]:
    """Return the sorted minimal generating set as a list of ints."""
    raw = tuple(sorted({parse_canonical_integer(g) for g in gens}))
    return list(minimal_generating_system(raw))


def _enumerate_factorizations(atoms: list[int], target: int) -> list[tuple[int, ...]]:
    """Enumerate all factorizations after request-level output-bound validation."""
    return list(factorizations(tuple(atoms), target))


def _factorizations(atoms: list[int], target: int) -> list[tuple[int, ...]]:
    """Wrapper for the enumeration routine."""
    return _enumerate_factorizations(atoms, target)


def _catenary_degree_of(atoms: list[int], target: int) -> int:
    """Catenary degree of one element.

    For every pair (z, z') of factorizations of *target*, the catenary degree is
    the minimum value *c* such that there exists a chain z -> z₁ -> ... -> z' with
    all consecutive distances ≤ *c*.  Equivalently, it is the maximum over all
    pairs (z, z') of the minimax path weight in the distance-weighted graph.
    """
    return catenary_degree_from_factorizations(tuple(_factorizations(atoms, target)))


# ---------------------------------------------------------------------------
# Public operation functions
# ---------------------------------------------------------------------------


def compute_element_delta_set(
    request: ElementDeltaSetRequest,
) -> ElementDeltaSetResult:
    atoms = _minimal_generators_list(request.generators)
    value = parse_canonical_integer(request.value)
    lengths = factorization_lengths(tuple(atoms), value)
    deltas = sorted({right - left for left, right in pairwise(lengths)})
    return ElementDeltaSetResult(
        value=request.value,
        minimal_generators=tuple(format_canonical_integer(atom) for atom in atoms),
        factorization_lengths=lengths,
        delta_set=tuple(deltas),
    )


def compute_element_elasticity(
    request: ElementElasticityRequest,
) -> ElementElasticityResult:
    atoms = _minimal_generators_list(request.generators)
    value = parse_canonical_integer(request.value)
    min_len, max_len = factorization_length_extrema(tuple(atoms), value)
    frac = Fraction(max_len, min_len)
    return ElementElasticityResult(
        value=request.value,
        minimal_generators=tuple(format_canonical_integer(atom) for atom in atoms),
        minimum_length=min_len,
        maximum_length=max_len,
        elasticity=f"{frac.numerator}/{frac.denominator}"
        if frac.denominator != 1
        else f"{frac.numerator}",
    )


def compute_element_catenary_degree(
    request: ElementCatenaryDegreeRequest,
) -> ElementCatenaryDegreeResult:
    atoms = _minimal_generators_list(request.generators)
    value = parse_canonical_integer(request.value)
    family = tuple(_factorizations(atoms, value))
    c = catenary_degree_from_factorizations(family)
    return ElementCatenaryDegreeResult(
        value=request.value,
        minimal_generators=tuple(format_canonical_integer(atom) for atom in atoms),
        factorization_count=len(family),
        catenary_degree=c,
    )


def compute_betti_elements(
    request: BettiElementsRequest,
) -> BettiElementsResult:
    atoms = tuple(_minimal_generators_list(request.generators))
    apery, candidates, disconnected = betti_data(atoms)
    return BettiElementsResult(
        minimal_generators=tuple(format_canonical_integer(a) for a in atoms),
        apery_set=tuple(format_canonical_integer(value) for value in apery),
        candidate_count=len(candidates),
        betti_elements=tuple(format_canonical_integer(b) for b in disconnected),
    )


def compute_delta_set(request: DeltaSetRequest) -> DeltaSetResult:
    atoms = tuple(_minimal_generators_list(request.generators))
    periodicity_bound = delta_periodicity_bound(atoms)
    checked_through = periodicity_bound + atoms[-1] - 1
    all_deltas: set[int] = set()
    length_sets: list[set[int]] = [set() for _ in range(atoms[-1])]
    length_sets[0].add(0)
    for value in range(1, checked_through + 1):
        lengths: set[int] = set()
        for atom in atoms:
            if value >= atom:
                lengths.update(
                    length + 1 for length in length_sets[(value - atom) % atoms[-1]]
                )
        ordered = sorted(lengths)
        all_deltas.update(right - left for left, right in pairwise(ordered))
        length_sets[value % atoms[-1]] = lengths
    return DeltaSetResult(
        minimal_generators=tuple(format_canonical_integer(atom) for atom in atoms),
        delta_set=tuple(sorted(all_deltas)),
        periodicity_bound=periodicity_bound,
        checked_through=checked_through,
    )


def compute_elasticity(request: ElasticityRequest) -> ElasticityResult:
    atoms = _minimal_generators_list(request.generators)
    max_atom = max(atoms)
    min_atom = min(atoms)
    frac = Fraction(max_atom, min_atom)
    return ElasticityResult(
        elasticity=f"{frac.numerator}/{frac.denominator}"
        if frac.denominator != 1
        else f"{frac.numerator}",
        smallest_generator=format_canonical_integer(min_atom),
        largest_generator=format_canonical_integer(max_atom),
    )


def compute_catenary_degree(
    request: CatenaryDegreeRequest,
) -> CatenaryDegreeResult:
    atoms = tuple(_minimal_generators_list(request.generators))
    _, _, disconnected = betti_data(atoms)
    degrees: list[BettiCatenaryDegree] = []
    for betti_value in disconnected:
        degrees.append(
            BettiCatenaryDegree(
                betti_element=format_canonical_integer(betti_value),
                catenary_degree=_catenary_degree_of(list(atoms), betti_value),
            )
        )
    maximum = max((record.catenary_degree for record in degrees), default=0)
    return CatenaryDegreeResult(
        minimal_generators=tuple(format_canonical_integer(atom) for atom in atoms),
        catenary_degree=maximum,
        betti_degrees=tuple(degrees),
        witness_betti_elements=tuple(
            record.betti_element
            for record in degrees
            if record.catenary_degree == maximum and maximum > 0
        ),
    )


__all__ = [
    "compute_betti_elements",
    "compute_catenary_degree",
    "compute_delta_set",
    "compute_elasticity",
    "compute_element_catenary_degree",
    "compute_element_delta_set",
    "compute_element_elasticity",
]
