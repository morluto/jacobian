"""Domain-owned numerical semigroup operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.numerical_semigroups._models import (
    MAX_ELEMENT,
    MAX_FACTOR_SEARCH,
    NumericalSemigroupSummaryRequest,
    NumericalSemigroupSummaryResult,
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


# ---------------------------------------------------------------------------
# Extended operations: factorization, elasticity, catenary degree, etc.
# ---------------------------------------------------------------------------

import networkx as _nx

from jacobian.math.numerical_semigroups._models import (
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
    MinimalPresentationRequest,
    MinimalPresentationRelation,
    MinimalPresentationResult,
    PresentationBinomial,
    PresentationBinomialsRequest,
    PresentationBinomialsResult,
)


def _minimal_generators_list(gens: tuple[str, ...]) -> list[int]:
    """Return the sorted minimal generating set as a list of ints."""
    raw = sorted({parse_canonical_integer(g) for g in gens})
    if not raw:
        return []
    if raw[0] == 1:
        return [1]
    result = []
    for generator in raw:
        others = [other for other in raw if other != generator]
        if not others:
            result.append(generator)
            continue
        can_reach = [False] * (generator + 1)
        can_reach[0] = True
        for value in range(1, generator + 1):
            can_reach[value] = any(
                value >= other and can_reach[value - other] for other in others
            )
        if not can_reach[generator]:
            result.append(generator)
    return result


def _enumerate_factorizations(
    atoms: list[int], target: int
) -> list[tuple[int, ...]]:
    """Enumerate all factorizations of *target* using *atoms* (minimal generators).

    Uses bounded dynamic programming.  Returns a list of tuples, each of length
    ``len(atoms)``, representing one factorization.
    """
    if target == 0:
        return [tuple([0] * len(atoms))]
    if not atoms:
        return []
    max_per_value = MAX_FACTOR_SEARCH * MAX_FACTOR_SEARCH
    dp: list[list[tuple[int, ...]]] = [[] for _ in range(target + 1)]
    dp[0] = [tuple([0] * len(atoms))]
    for v in range(1, target + 1):
        facts: list[tuple[int, ...]] = []
        for idx, atom in enumerate(atoms):
            if atom > v:
                continue
            for fact in dp[v - atom]:
                new_fact = list(fact)
                new_fact[idx] += 1
                facts.append(tuple(new_fact))
            if len(facts) > max_per_value:
                break
        seen: set[tuple[int, ...]] = set()
        unique: list[tuple[int, ...]] = []
        for f in facts:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        dp[v] = unique
    return dp[target]


def _factorizations(atoms: list[int], target: int) -> list[tuple[int, ...]]:
    """Wrapper for the enumeration routine."""
    return _enumerate_factorizations(atoms, target)


def _length(fact: tuple[int, ...]) -> int:
    """Length (norm) of a factorization = sum of coordinates."""
    return sum(fact)


def _gcd_factor(
    f1: tuple[int, ...], f2: tuple[int, ...]
) -> tuple[int, ...]:
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
    graph = _nx.Graph()
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
    facts = _factorizations(atoms, target)
    if len(facts) <= 1:
        return 0

    n = len(facts)
    _, components, _ = _build_factorization_graph(facts)
    if len(components) <= 1:
        return 0

    # Build a complete distance-weighted graph.
    dist_graph = _nx.Graph()
    for i in range(n):
        dist_graph.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            d = _distance(facts[i], facts[j])
            dist_graph.add_edge(i, j, weight=d)

    # Minimax path via MST.
    mst = _nx.minimum_spanning_tree(dist_graph)
    catenary = 0
    for i in range(n):
        for j in range(i + 1, n):
            try:
                path = _nx.shortest_path(mst, i, j, weight="weight")
            except _nx.NetworkXNoPath:
                continue
            if len(path) < 2:
                continue
            path_max = 0
            for k in range(len(path) - 1):
                edge_weight = mst[path[k]][path[k + 1]]["weight"]
                path_max = max(path_max, edge_weight)
            catenary = max(catenary, path_max)
    return catenary


def _delta_set_of(atoms: list[int], target: int) -> list[int]:
    """Delta set of one element = successive differences of the sorted length set."""
    facts = _factorizations(atoms, target)
    if len(facts) <= 1:
        return []
    lengths = sorted({sum(f) for f in facts})
    if len(lengths) <= 1:
        return []
    return [lengths[i + 1] - lengths[i] for i in range(len(lengths) - 1)]


def _betti_elements(atoms: list[int]) -> list[int]:
    """Betti elements = elements where the factorization graph is disconnected."""
    # The search space is bounded: Betti elements are at most lcm of pairs of
    # atoms times some small factor.  We search up to the Frobenius-like bound.
    if not atoms or atoms[0] == 1:
        return []
    max_atom = atoms[-1]
    # Betti elements are bounded by the Frobenius number + 1, but we use a
    # generous bound.  The conductor is at most (m-1)*max_atom where m is
    # the multiplicity (smallest generator).
    multiplicity = atoms[0]
    conductor_bound = (multiplicity - 1) * max_atom
    bound = conductor_bound * multiplicity
    if bound > MAX_ELEMENT:
        bound = MAX_ELEMENT
    betti: list[int] = []
    for n in range(1, bound + 1):
        facts = _factorizations(atoms, n)
        if len(facts) <= 1:
            continue
        _, _, connected = _build_factorization_graph(facts)
        if not connected:
            betti.append(n)
    return betti


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
            minimal_generators=tuple(
                format_canonical_integer(a) for a in atoms
            ),
            factorizations=(),
        )
    facts = _factorizations(atoms, value)
    return FactorizationComputeResult(
        value=request.value,
        minimal_generators=tuple(
            format_canonical_integer(a) for a in atoms
        ),
        factorizations=tuple(facts),
    )


def compute_factorization_lengths(
    request: FactorizationLengthsComputeRequest,
) -> FactorizationLengthsComputeResult:
    atoms = _minimal_generators_list(request.generators)
    value = parse_canonical_integer(request.value)
    if value < 0:
        return FactorizationLengthsComputeResult(value=request.value, lengths=())
    facts = _factorizations(atoms, value)
    lengths = sorted({sum(f) for f in facts})
    return FactorizationLengthsComputeResult(
        value=request.value,
        lengths=tuple(lengths),
    )


def compute_factorization_distance(
    request: FactorizationDistanceRequest,
) -> FactorizationDistanceResult:
    atoms = _minimal_generators_list(request.generators)
    value = parse_canonical_integer(request.value)
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
            minimal_generators=tuple(
                format_canonical_integer(a) for a in atoms
            ),
            factorizations=(),
            edges=(),
            connected_components=(),
            is_connected=True,
        )
    facts = _factorizations(atoms, value)
    edges, components, connected = _build_factorization_graph(facts)
    return FactorizationGraphComputeResult(
        value=request.value,
        minimal_generators=tuple(
            format_canonical_integer(a) for a in atoms
        ),
        factorizations=tuple(facts),
        edges=tuple((i, j) for i, j in edges),
        connected_components=tuple(
            tuple(sorted(comp)) for comp in components
        ),
        is_connected=connected,
    )


def compute_element_delta_set(
    request: ElementDeltaSetRequest,
) -> ElementDeltaSetResult:
    atoms = _minimal_generators_list(request.generators)
    value = parse_canonical_integer(request.value)
    if value < 0:
        return ElementDeltaSetResult(value=request.value, delta_set=())
    deltas = _delta_set_of(atoms, value)
    return ElementDeltaSetResult(
        value=request.value,
        delta_set=tuple(deltas),
    )


def compute_element_elasticity(
    request: ElementElasticityRequest,
) -> ElementElasticityResult:
    atoms = _minimal_generators_list(request.generators)
    value = parse_canonical_integer(request.value)
    if value < 0:
        return ElementElasticityResult(value=request.value, elasticity="0")
    facts = _factorizations(atoms, value)
    if not facts:
        return ElementElasticityResult(value=request.value, elasticity="0")
    lengths = [sum(f) for f in facts]
    min_len = min(lengths)
    max_len = max(lengths)
    if min_len == 0:
        return ElementElasticityResult(value=request.value, elasticity="0")
    frac = Fraction(max_len, min_len)
    return ElementElasticityResult(
        value=request.value,
        elasticity=f"{frac.numerator}/{frac.denominator}"
        if frac.denominator != 1
        else f"{frac.numerator}",
    )


def compute_element_catenary_degree(
    request: ElementCatenaryDegreeRequest,
) -> ElementCatenaryDegreeResult:
    atoms = _minimal_generators_list(request.generators)
    value = parse_canonical_integer(request.value)
    if value < 0:
        return ElementCatenaryDegreeResult(
            value=request.value, catenary_degree=0
        )
    c = _catenary_degree_of(atoms, value)
    return ElementCatenaryDegreeResult(
        value=request.value,
        catenary_degree=c,
    )


def compute_betti_elements(
    request: BettiElementsRequest,
) -> BettiElementsResult:
    atoms = _minimal_generators_list(request.generators)
    betti = _betti_elements(atoms)
    return BettiElementsResult(
        betti_elements=tuple(
            format_canonical_integer(b) for b in betti
        ),
    )


def compute_minimal_presentation(
    request: MinimalPresentationRequest,
) -> MinimalPresentationResult:
    atoms = _minimal_generators_list(request.generators)
    betti = _betti_elements(atoms)
    relations: list[MinimalPresentationRelation] = []
    for betti_val in betti:
        facts = _factorizations(atoms, betti_val)
        if len(facts) <= 1:
            continue
        edges, components, _ = _build_factorization_graph(facts)
        if len(components) <= 1:
            continue
        # For each pair of components, find one edge (relation) connecting them.
        # We pick the first factorization from each component and form a relation.
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                comp_i = list(components[i])
                comp_j = list(components[j])
                # Find the closest pair between the two components
                best_d = None
                best_pair = None
                for idx_i in comp_i:
                    for idx_j in comp_j:
                        d = _distance(facts[idx_i], facts[idx_j])
                        if best_d is None or d < best_d:
                            best_d = d
                            best_pair = (idx_i, idx_j)
                if best_pair is not None:
                    z, z2 = best_pair
                    relations.append(
                        MinimalPresentationRelation(
                            first=facts[z],
                            second=facts[z2],
                        )
                    )
    return MinimalPresentationResult(
        minimal_generators=tuple(
            format_canonical_integer(a) for a in atoms
        ),
        betti_elements=tuple(
            format_canonical_integer(b) for b in betti
        ),
        relations=tuple(relations),
    )


def compute_presentation_binomials(
    request: PresentationBinomialsRequest,
) -> PresentationBinomialsResult:
    atoms = _minimal_generators_list(request.generators)
    binomials: list[PresentationBinomial] = []
    for relation in request.relations:
        z1 = relation.first
        z2 = relation.second
        # left_coefficient = max of the two factorization coordinates (as strings)
        left_coef = format_canonical_integer(sum(z1))
        right_coef = format_canonical_integer(sum(z2))
        binomials.append(
            PresentationBinomial(
                left_coefficient=left_coef,
                left_exponents=tuple(z1),
                right_coefficient=right_coef,
                right_exponents=tuple(z2),
            )
        )
    return PresentationBinomialsResult(
        minimal_generators=tuple(
            format_canonical_integer(a) for a in atoms
        ),
        binomials=tuple(binomials),
    )


def compute_delta_set(request: DeltaSetRequest) -> DeltaSetResult:
    atoms = _minimal_generators_list(request.generators)
    betti = _betti_elements(atoms)
    all_deltas: set[int] = set()
    for betti_val in betti:
        for d in _delta_set_of(atoms, betti_val):
            all_deltas.add(d)
    return DeltaSetResult(delta_set=tuple(sorted(all_deltas)))


def compute_elasticity(request: ElasticityRequest) -> ElasticityResult:
    atoms = _minimal_generators_list(request.generators)
    if not atoms:
        return ElasticityResult(elasticity="0")
    if atoms[0] == 1:
        return ElasticityResult(elasticity="1")
    max_atom = max(atoms)
    min_atom = min(atoms)
    frac = Fraction(max_atom, min_atom)
    return ElasticityResult(
        elasticity=f"{frac.numerator}/{frac.denominator}"
        if frac.denominator != 1
        else f"{frac.numerator}"
    )


def compute_catenary_degree(
    request: CatenaryDegreeRequest,
) -> CatenaryDegreeResult:
    atoms = _minimal_generators_list(request.generators)
    betti = _betti_elements(atoms)
    max_cat = 0
    for betti_val in betti:
        c = _catenary_degree_of(atoms, betti_val)
        max_cat = max(max_cat, c)
    return CatenaryDegreeResult(catenary_degree=max_cat)

__all__ = [
    "compute_membership",
    "compute_summary",
    "compute_factorizations",
    "compute_factorization_lengths",
    "compute_factorization_distance",
    "compute_factorization_graph",
    "compute_element_delta_set",
    "compute_element_elasticity",
    "compute_element_catenary_degree",
    "compute_betti_elements",
    "compute_minimal_presentation",
    "compute_presentation_binomials",
    "compute_delta_set",
    "compute_elasticity",
    "compute_catenary_degree",
]
