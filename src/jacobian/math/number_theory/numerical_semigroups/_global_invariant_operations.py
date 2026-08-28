"""Global invariant operations for numerical semigroups."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise

from jacobian._exact import format_canonical_rational
from jacobian.canonical import format_canonical_integer
from jacobian.math.number_theory.numerical_semigroups._algorithms import (
    betti_data,
    catenary_degree_from_factorizations,
    delta_periodicity_bound,
    factorizations,
)
from jacobian.math.number_theory.numerical_semigroups._global_invariant_models import (
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
from jacobian.math.number_theory.numerical_semigroups._models import (
    _require_global_betti_bound,
    _require_global_catenary_bound,
    _require_global_delta_bound,
    _require_minimal_generators,
    _run_admission,
)


def _catenary_degree(atoms: tuple[int, ...], value: int) -> int:
    """Compute one Betti witness degree from its complete factorization family."""

    return catenary_degree_from_factorizations(factorizations(atoms, value))


def compute_betti_elements(
    request: BettiElementsRequest,
) -> BettiElementsResult:
    """Compute the complete Betti-element profile on the minimal atom axis."""

    atoms = _run_admission(
        "betti_elements",
        ("generators",),
        lambda: _require_minimal_generators(request.generators),
    )
    _run_admission(
        "betti_elements", ("generators",), lambda: _require_global_betti_bound(atoms)
    )
    apery, candidates, disconnected = betti_data(atoms)
    return BettiElementsResult._from_kernel(
        minimal_generators=tuple(format_canonical_integer(atom) for atom in atoms),
        apery_set=tuple(format_canonical_integer(value) for value in apery),
        candidate_count=len(candidates),
        betti_elements=tuple(format_canonical_integer(value) for value in disconnected),
    )


def compute_delta_set(request: DeltaSetRequest) -> DeltaSetResult:
    """Compute the complete global delta set through its periodicity bound."""

    atoms = _run_admission(
        "delta_set",
        ("generators",),
        lambda: _require_minimal_generators(request.generators),
    )
    _run_admission(
        "delta_set", ("generators",), lambda: _require_global_delta_bound(atoms)
    )
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

    atoms = _run_admission(
        "elasticity",
        ("generators",),
        lambda: _require_minimal_generators(request.generators),
    )
    return ElasticityResult(
        elasticity=format_canonical_rational(Fraction(atoms[-1], atoms[0])),
        smallest_generator=format_canonical_integer(atoms[0]),
        largest_generator=format_canonical_integer(atoms[-1]),
    )


def compute_catenary_degree(
    request: CatenaryDegreeRequest,
) -> CatenaryDegreeResult:
    """Compute the global catenary degree from its complete Betti witnesses."""

    atoms = _run_admission(
        "catenary_degree",
        ("generators",),
        lambda: _require_minimal_generators(request.generators),
    )
    _run_admission(
        "catenary_degree",
        ("generators",),
        lambda: _require_global_catenary_bound(atoms),
    )
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


__all__ = [
    "compute_betti_elements",
    "compute_catenary_degree",
    "compute_delta_set",
    "compute_elasticity",
]
