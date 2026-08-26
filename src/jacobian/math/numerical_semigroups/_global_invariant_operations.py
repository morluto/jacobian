"""Global invariant operations for numerical semigroups."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise

from jacobian._exact import format_canonical_rational
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.numerical_semigroups._algorithms import (
    betti_data,
    catenary_degree_from_factorizations,
    delta_periodicity_bound,
    factorizations,
    minimal_generating_system,
)
from jacobian.math.numerical_semigroups._global_invariant_models import (
    BettiCatenaryDegree,
    BettiElementsRequest,
    BettiElementsResult,
    CatenaryDegreeRequest,
    CatenaryDegreeResult,
    DeltaSetRequest,
    DeltaSetResult,
    ElasticityRequest,
    ElasticityResult,
)


def _minimal_generators(generators: tuple[str, ...]) -> tuple[int, ...]:
    """Normalize an admitted presentation to its minimal atom axis."""

    return minimal_generating_system(
        tuple(sorted({parse_canonical_integer(generator) for generator in generators}))
    )


def _catenary_degree(atoms: tuple[int, ...], value: int) -> int:
    """Compute one Betti witness degree from its complete factorization family."""

    return catenary_degree_from_factorizations(factorizations(atoms, value))


def compute_betti_elements(
    request: BettiElementsRequest,
) -> BettiElementsResult:
    """Compute the complete Betti-element profile on the minimal atom axis."""

    atoms = _minimal_generators(request.generators)
    apery, candidates, disconnected = betti_data(atoms)
    return BettiElementsResult._from_kernel(
        minimal_generators=tuple(format_canonical_integer(atom) for atom in atoms),
        apery_set=tuple(format_canonical_integer(value) for value in apery),
        candidate_count=len(candidates),
        betti_elements=tuple(format_canonical_integer(value) for value in disconnected),
    )


def compute_delta_set(request: DeltaSetRequest) -> DeltaSetResult:
    """Compute the complete global delta set through its periodicity bound."""

    atoms = _minimal_generators(request.generators)
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
    return DeltaSetResult._from_kernel(
        minimal_generators=tuple(format_canonical_integer(atom) for atom in atoms),
        delta_set=tuple(sorted(all_deltas)),
        periodicity_bound=periodicity_bound,
        checked_through=checked_through,
    )


def compute_elasticity(request: ElasticityRequest) -> ElasticityResult:
    """Compute the exact global elasticity from the minimal atom extrema."""

    atoms = _minimal_generators(request.generators)
    return ElasticityResult(
        elasticity=format_canonical_rational(Fraction(atoms[-1], atoms[0])),
        smallest_generator=format_canonical_integer(atoms[0]),
        largest_generator=format_canonical_integer(atoms[-1]),
    )


def compute_catenary_degree(
    request: CatenaryDegreeRequest,
) -> CatenaryDegreeResult:
    """Compute the global catenary degree from its complete Betti witnesses."""

    atoms = _minimal_generators(request.generators)
    _, _, disconnected = betti_data(atoms)
    degrees = tuple(
        BettiCatenaryDegree(
            betti_element=format_canonical_integer(value),
            catenary_degree=_catenary_degree(atoms, value),
        )
        for value in disconnected
    )
    maximum = max((record.catenary_degree for record in degrees), default=0)
    return CatenaryDegreeResult._from_kernel(
        minimal_generators=tuple(format_canonical_integer(atom) for atom in atoms),
        catenary_degree=maximum,
        betti_degrees=degrees,
        witness_betti_elements=tuple(
            record.betti_element
            for record in degrees
            if record.catenary_degree == maximum and maximum > 0
        ),
    )


def verify_betti_elements_result(result: BettiElementsResult) -> bool:
    """Replay one bounded Betti-elements claim."""

    atoms = tuple(parse_canonical_integer(atom) for atom in result.minimal_generators)
    apery, candidates, disconnected = betti_data(atoms)
    return (
        tuple(result.apery_set)
        == tuple(format_canonical_integer(value) for value in apery)
        and result.candidate_count == len(candidates)
        and tuple(result.betti_elements)
        == tuple(format_canonical_integer(value) for value in disconnected)
    )


def verify_delta_set_result(result: DeltaSetResult) -> bool:
    """Replay one bounded complete global delta-set claim."""

    atoms = tuple(parse_canonical_integer(atom) for atom in result.minimal_generators)
    periodicity_bound = delta_periodicity_bound(atoms)
    checked_through = periodicity_bound + atoms[-1] - 1
    all_deltas: set[int] = set()
    length_sets: list[set[int]] = [set() for _ in range(atoms[-1])]
    length_sets[0].add(0)
    for value in range(1, checked_through + 1):
        lengths = {
            length + 1
            for atom in atoms
            if value >= atom
            for length in length_sets[(value - atom) % atoms[-1]]
        }
        ordered = sorted(lengths)
        all_deltas.update(right - left for left, right in pairwise(ordered))
        length_sets[value % atoms[-1]] = lengths
    return (
        result.delta_set == tuple(sorted(all_deltas))
        and result.periodicity_bound == periodicity_bound
        and result.checked_through == checked_through
    )


def verify_catenary_degree_result(result: CatenaryDegreeResult) -> bool:
    """Replay the bounded complete Betti catenary profile for a supplied claim."""

    atoms = tuple(parse_canonical_integer(atom) for atom in result.minimal_generators)
    _, _, disconnected = betti_data(atoms)
    degrees = tuple(
        BettiCatenaryDegree(
            betti_element=format_canonical_integer(value),
            catenary_degree=_catenary_degree(atoms, value),
        )
        for value in disconnected
    )
    maximum = max((record.catenary_degree for record in degrees), default=0)
    witnesses = tuple(
        record.betti_element
        for record in degrees
        if record.catenary_degree == maximum and maximum > 0
    )
    return (
        result.betti_degrees == degrees
        and result.catenary_degree == maximum
        and result.witness_betti_elements == witnesses
    )


__all__ = [
    "compute_betti_elements",
    "compute_catenary_degree",
    "compute_delta_set",
    "compute_elasticity",
    "verify_betti_elements_result",
    "verify_catenary_degree_result",
    "verify_delta_set_result",
]
