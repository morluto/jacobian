"""Domain-owned numerical semigroup operations."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise

import networkx as _nx

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.numerical_semigroups._algorithms import (
    betti_data,
    catenary_degree_from_factorizations,
    delta_periodicity_bound,
    factorization_lengths,
    factorization_predecessors,
    factorizations,
    minimal_generating_system,
    reconstruct_factorization,
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
    FactorizationComputeRequest,
    FactorizationComputeResult,
    FactorizationDistanceRequest,
    FactorizationDistanceResult,
    FactorizationGraphComputeRequest,
    FactorizationGraphComputeResult,
    FactorizationLengthsComputeRequest,
    FactorizationLengthsComputeResult,
    MinimalPresentationRelation,
    MinimalPresentationRequest,
    MinimalPresentationResult,
    NumericalSemigroupSummaryRequest,
    NumericalSemigroupSummaryResult,
    PresentationBinomial,
    PresentationBinomialsRequest,
    PresentationBinomialsResult,
    SemigroupMembershipRequest,
    SemigroupMembershipResult,
)


def _normalize_generators(gens: tuple[str, ...]) -> list[int]:
    """Return sorted unique positive generators."""
    return sorted({parse_canonical_integer(generator) for generator in gens})


def _compute_summary(gens: list[int]) -> NumericalSemigroupSummaryResult:
    multiplicity = gens[0]
    if multiplicity == 1:
        return NumericalSemigroupSummaryResult(
            minimal_generators=("1",),
            multiplicity="1",
            embedding_dimension=1,
            frobenius_number="-1",
            conductor="0",
            genus=0,
            gaps=(),
        )

    limit = (multiplicity - 1) * max(gens)
    in_semigroup = [False] * (limit + 1)
    in_semigroup[0] = True
    run = 0
    conductor = limit + 1
    for value in range(1, limit + 1):
        in_semigroup[value] = any(
            value >= generator and in_semigroup[value - generator] for generator in gens
        )
        if in_semigroup[value]:
            run += 1
            if run == multiplicity:
                conductor = value - multiplicity + 1
                break
        else:
            run = 0

    gaps = [
        value
        for value in range(1, conductor)
        if value <= limit and not in_semigroup[value]
    ]
    frobenius = max(gaps) if gaps else -1

    min_gens = []
    for generator in gens:
        others = [other for other in gens if other != generator]
        if not others:
            min_gens.append(generator)
            continue
        can_reach = [False] * (generator + 1)
        can_reach[0] = True
        for value in range(1, generator + 1):
            can_reach[value] = any(
                value >= other and can_reach[value - other] for other in others
            )
        if not can_reach[generator]:
            min_gens.append(generator)

    return NumericalSemigroupSummaryResult(
        minimal_generators=tuple(
            format_canonical_integer(generator) for generator in min_gens
        ),
        multiplicity=format_canonical_integer(multiplicity),
        embedding_dimension=len(min_gens),
        frobenius_number=format_canonical_integer(frobenius),
        conductor=format_canonical_integer(conductor),
        genus=len(gaps),
        gaps=tuple(format_canonical_integer(gap) for gap in gaps),
    )


def compute_summary(
    request: NumericalSemigroupSummaryRequest,
) -> NumericalSemigroupSummaryResult:
    return _compute_summary(_normalize_generators(request.generators))


def compute_membership(
    request: SemigroupMembershipRequest,
) -> SemigroupMembershipResult:
    gens = _normalize_generators(request.generators)
    value = parse_canonical_integer(request.value)
    if value < 0:
        return SemigroupMembershipResult(value=request.value, in_semigroup=False)
    if value == 0:
        return SemigroupMembershipResult(value=request.value, in_semigroup=True)
    can_reach = [False] * (value + 1)
    can_reach[0] = True
    for index in range(1, value + 1):
        can_reach[index] = any(
            index >= generator and can_reach[index - generator] for generator in gens
        )
    return SemigroupMembershipResult(
        value=request.value,
        in_semigroup=can_reach[value],
    )


# __all__ defined at end


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


def _length(fact: tuple[int, ...]) -> int:
    """Length (norm) of a factorization = sum of coordinates."""
    return sum(fact)


def _gcd_factor(f1: tuple[int, ...], f2: tuple[int, ...]) -> tuple[int, ...]:
    """Coordinate-wise minimum of two factorizations."""
    return tuple(min(a, b) for a, b in zip(f1, f2, strict=True))


def _distance(f1: tuple[int, ...], f2: tuple[int, ...]) -> int:
    """Distance d(z, z') = max(|z - gcd|, |z' - gcd|) where gcd is coordinatewise min."""
    if not f1 or not f2:
        return 0
    g = _gcd_factor(f1, f2)
    l1 = sum(f1)
    l2 = sum(f2)
    lg = sum(g)
    return max(abs(l1 - lg), abs(l2 - lg))


def _build_factorization_graph(
    factorizations: list[tuple[int, ...]],
) -> tuple[list[tuple[int, int]], list[list[int]], bool]:
    """Build the standard factorization graph.

    Two factorizations are connected if their coordinatewise gcd is nonzero
    (they share a common atom).  Returns ``(edges, connected_components, is_connected)``.
    """
    n = len(factorizations)
    if n == 0:
        return [], [], True
    graph: _nx.Graph[int] = _nx.Graph()
    for i in range(n):
        graph.add_node(i)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if sum(_gcd_factor(factorizations[i], factorizations[j])) > 0:
                graph.add_edge(i, j)
                edges.append((i, j))
    components = [list(comp) for comp in _nx.connected_components(graph)]
    is_connected = len(components) <= 1
    return edges, components, is_connected


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


def compute_factorizations(
    request: FactorizationComputeRequest,
) -> FactorizationComputeResult:
    atoms = _minimal_generators_list(request.generators)
    value = parse_canonical_integer(request.value)
    if value < 0:
        return FactorizationComputeResult(
            value=request.value,
            minimal_generators=tuple(format_canonical_integer(a) for a in atoms),
            in_semigroup=False,
            factorizations=(),
        )
    facts = _factorizations(atoms, value)
    return FactorizationComputeResult(
        value=request.value,
        minimal_generators=tuple(format_canonical_integer(a) for a in atoms),
        in_semigroup=bool(facts),
        factorizations=tuple(facts),
    )


def compute_factorization_lengths(
    request: FactorizationLengthsComputeRequest,
) -> FactorizationLengthsComputeResult:
    atoms = _minimal_generators_list(request.generators)
    value = parse_canonical_integer(request.value)
    if value < 0:
        return FactorizationLengthsComputeResult(
            value=request.value,
            minimal_generators=tuple(format_canonical_integer(a) for a in atoms),
            in_semigroup=False,
            lengths=(),
        )
    lengths = factorization_lengths(tuple(atoms), value)
    return FactorizationLengthsComputeResult(
        value=request.value,
        minimal_generators=tuple(format_canonical_integer(a) for a in atoms),
        in_semigroup=bool(lengths),
        lengths=tuple(lengths),
    )


def compute_factorization_distance(
    request: FactorizationDistanceRequest,
) -> FactorizationDistanceResult:
    f1 = tuple(request.first)
    f2 = tuple(request.second)
    d = _distance(f1, f2)
    return FactorizationDistanceResult(
        value=request.value,
        distance=d,
        first_length=sum(f1),
        second_length=sum(f2),
    )


def compute_factorization_graph(
    request: FactorizationGraphComputeRequest,
) -> FactorizationGraphComputeResult:
    atoms = _minimal_generators_list(request.generators)
    value = parse_canonical_integer(request.value)
    if value < 0:
        return FactorizationGraphComputeResult(
            value=request.value,
            minimal_generators=tuple(format_canonical_integer(a) for a in atoms),
            in_semigroup=False,
            factorizations=(),
            edges=(),
            connected_components=(),
            is_connected=True,
        )
    facts = _factorizations(atoms, value)
    edges, components, connected = _build_factorization_graph(facts)
    return FactorizationGraphComputeResult(
        value=request.value,
        minimal_generators=tuple(format_canonical_integer(a) for a in atoms),
        in_semigroup=bool(facts),
        factorizations=tuple(facts),
        edges=tuple((i, j) for i, j in edges),
        connected_components=tuple(tuple(sorted(comp)) for comp in components),
        is_connected=connected,
    )


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
    lengths = factorization_lengths(tuple(atoms), value)
    min_len = min(lengths)
    max_len = max(lengths)
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


def compute_minimal_presentation(
    request: MinimalPresentationRequest,
) -> MinimalPresentationResult:
    atoms = tuple(_minimal_generators_list(request.generators))
    _, _, disconnected = betti_data(atoms)
    predecessors = factorization_predecessors(atoms, max(disconnected, default=0))
    relations: list[MinimalPresentationRelation] = []
    for betti_value, components in disconnected.items():
        representatives: list[tuple[int, ...]] = []
        for component in components:
            generator_index = component[0]
            residual = reconstruct_factorization(
                atoms, predecessors, betti_value - atoms[generator_index]
            )
            if residual is None:
                raise RuntimeError("Betti component has no factorization witness")
            coordinates = list(residual)
            coordinates[generator_index] += 1
            representatives.append(tuple(coordinates))
        for target_representative in representatives[1:]:
            relations.append(
                MinimalPresentationRelation(
                    first=representatives[0], second=target_representative
                )
            )
    return MinimalPresentationResult(
        minimal_generators=tuple(format_canonical_integer(a) for a in atoms),
        betti_elements=tuple(format_canonical_integer(b) for b in disconnected),
        relations=tuple(relations),
    )


def compute_presentation_binomials(
    request: PresentationBinomialsRequest,
) -> PresentationBinomialsResult:
    atoms = _minimal_generators_list(request.generators)
    binomials: list[PresentationBinomial] = []
    for relation in request.relations:
        binomials.append(
            PresentationBinomial(
                left_exponents=tuple(relation.first),
                right_exponents=tuple(relation.second),
            )
        )
    return PresentationBinomialsResult(
        minimal_generators=tuple(format_canonical_integer(a) for a in atoms),
        binomials=tuple(binomials),
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
    "compute_factorization_distance",
    "compute_factorization_graph",
    "compute_factorization_lengths",
    "compute_factorizations",
    "compute_membership",
    "compute_minimal_presentation",
    "compute_presentation_binomials",
    "compute_summary",
]
