from __future__ import annotations

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDescriptor,
    OperationDiscoveryMatch,
    OperationMatchRequest,
)
from jacobian.catalog.search import (
    discovery_terms,
    match_operations,
)


def _positions(need: str) -> dict[str, int]:
    catalog = Catalog.open()
    cursor: str | None = None
    matches: list[OperationDiscoveryMatch] = []
    while True:
        result = catalog.match(
            OperationMatchRequest(need=need, limit=20, cursor=cursor)
        )
        matches.extend(result.matches)
        if result.next_cursor is None:
            break
        cursor = result.next_cursor
    return {match.operation_id: index for index, match in enumerate(matches)}


def test_determinant_need_ranks_determinants_before_charpolys() -> None:
    catalog = Catalog.open()
    cursor: str | None = None
    matches: list[OperationDiscoveryMatch] = []
    while True:
        result = catalog.match(
            OperationMatchRequest(
                need="return the exact determinant of a rational matrix",
                namespace="matrix",
                limit=20,
                cursor=cursor,
            )
        )
        matches.extend(result.matches)
        if result.next_cursor is None:
            break
        cursor = result.next_cursor

    positions = {match.operation_id: index for index, match in enumerate(matches)}

    determinant_ids = (
        "matrix.determinant.compute",
        "matrix.symbolic.determinant.compute",
    )
    characteristic_polynomial_ids = (
        "matrix.characteristic_polynomial.compute",
        "matrix.symbolic.characteristic_polynomial.compute",
    )
    assert set(determinant_ids) <= positions.keys()
    assert set(characteristic_polynomial_ids) <= positions.keys()
    assert all(
        positions[determinant_id] < positions[characteristic_polynomial_id]
        for determinant_id in determinant_ids
        for characteristic_polynomial_id in characteristic_polynomial_ids
    )


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        ("sum", "sum"),
        ("sums", "sum"),
        ("representations", "representation"),
        ("points", "point"),
        ("distances", "distance"),
        ("counts", "count"),
        ("cardinalities", "cardinality"),
        ("trees", "tree"),
        ("cuts", "cut"),
        ("classes", "class"),
        ("complexes", "complex"),
        ("matches", "match"),
        ("alias", "alias"),
        ("aliases", "alias"),
        ("atlas", "atlas"),
        ("atlases", "atlas"),
        ("bias", "bias"),
        ("biases", "bias"),
        ("lens", "lens"),
        ("lenses", "lens"),
        ("basis", "basis"),
        ("bases", "bases"),
        ("class", "class"),
        ("grass", "grass"),
        ("series", "series"),
        ("dynamics", "dynamics"),
        ("chaos", "chaos"),
        ("guigues", "guigues"),
        ("sims", "sims"),
        ("sos", "sos"),
        ("does", "does"),
        ("lies", "lies"),
    ],
)
def test_discovery_terms_use_a_conservative_inflection_table(
    term: str,
    expected: str,
) -> None:
    assert discovery_terms(term) == frozenset({expected})


def test_subset_sum_singular_and_plural_queries_rank_the_profile_ahead_of_sidon() -> (
    None
):
    catalog = Catalog.open()
    for query in (
        "all subset sum and repeated representation of a finite integer set",
        "all subset sums and repeated representations of a finite integer set",
    ):
        result = catalog.match(OperationMatchRequest(need=query, limit=20))
        positions = {
            match.operation_id: index for index, match in enumerate(result.matches)
        }

        assert result.need == query
        assert (
            positions["additive.subset_sum.profile.compute"]
            < positions["combinatorics.integer_set.sidon.decide"]
        )


@pytest.mark.parametrize(
    ("queries", "expected_operation_id"),
    [
        (
            (
                "return the exact pairwise distance profile of rational points",
                "return exact pairwise distances profiles of rational point sets",
            ),
            "geometry.points.distance_profile.compute",
        ),
        (
            (
                "count independent vertex sets by cardinality in a tree",
                "counts independent vertex sets by cardinalities in trees",
            ),
            "graph.polynomial.independence.compute",
        ),
        (
            (
                "return the exact maximum cut and a partition witness",
                "return exact maximum cuts and partition witnesses",
            ),
            "graph.cut.maximum.compute",
        ),
    ],
)
def test_inflected_catalog_queries_retain_the_same_top_result(
    queries: tuple[str, str],
    expected_operation_id: str,
) -> None:
    catalog = Catalog.open()
    results = tuple(
        catalog.match(OperationMatchRequest(need=query, limit=10)) for query in queries
    )

    assert all(
        result.matches[0].operation_id == expected_operation_id for result in results
    )


def test_protected_mathematical_queries_retain_their_top_catalog_matches() -> None:
    catalog = Catalog.open()
    expected_top_matches = {
        "compute a canonical basis of an integer lattice": (
            "lattice.canonical_basis.compute"
        ),
        "return the communicating classes of a finite Markov chain": (
            "probability.markov_chain.communicating_classes.compute"
        ),
        "compose two truncated rational formal power series": (
            "formal_series.rational.compose.compute"
        ),
        "check a supplied rational sum-of-squares decomposition certificate": (
            "polynomial.sos.decomposition.check"
        ),
    }

    for query, expected_operation_id in expected_top_matches.items():
        result = catalog.match(OperationMatchRequest(need=query, limit=5))

        assert result.matches[0].operation_id == expected_operation_id


def test_inflected_discovery_ties_are_deterministic_and_operation_id_ordered() -> None:
    operations = tuple(
        OperationDescriptor(
            operation_id=operation_id,
            title="Inspect points",
            description="Examine objects.",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )
        for operation_id in ("fixture.z.inspect", "fixture.a.inspect")
    )
    request = OperationMatchRequest(need="point", limit=2)

    forward = match_operations(operations, request)
    reverse = match_operations(tuple(reversed(operations)), request)

    assert forward == reverse
    assert [match.operation_id for match in forward.matches] == [
        "fixture.a.inspect",
        "fixture.z.inspect",
    ]


def test_discovery_normalizes_only_audited_ordinary_plural_forms() -> None:
    assert discovery_terms("subset sums and repeated representations") == {
        "subset",
        "sum",
        "repeated",
        "representation",
    }
    assert discovery_terms("basis class series") == {
        "basis",
        "class",
        "series",
    }


def test_plural_queries_preserve_their_semantic_catalog_routing() -> None:
    subset_positions = _positions(
        "all subset sums and repeated representations of a finite integer set"
    )
    assert (
        subset_positions["additive.subset_sum.profile.compute"]
        < subset_positions["combinatorics.integer_set.sidon.decide"]
    )

    tree_positions = _positions(
        "counts independent vertex sets by cardinalities in trees"
    )
    assert (
        tree_positions["graph.polynomial.independence.compute"]
        < tree_positions["graph.independent_set.maximal.decide"]
    )


@pytest.mark.parametrize(
    ("need", "expected_operation_id"),
    (
        (
            "decide whether an integer set is Sidon and return a repeated-difference collision witness",
            "combinatorics.integer_set.sidon.decide",
        ),
        (
            "from a supplied edge-coloured complete graph construct the hypergraph whose edges are monochromatic cliques",
            "graph.edge_colored.monochromatic_clique_hypergraph.construct",
        ),
        (
            "find a generalized exact cover of a finite incidence system",
            "combinatorics.generalized_exact_cover.find",
        ),
        (
            "decide whether a supplied graph arrows another graph under every two-edge-colouring and return an avoiding colouring when it does not",
            "graph.edge_coloring_arrowing.decide",
        ),
    ),
)
def test_complete_local_needs_retain_problem_specific_intent(
    need: str,
    expected_operation_id: str,
) -> None:
    result = Catalog.open().match(OperationMatchRequest(need=need, limit=5))

    assert result.matches[0].operation_id == expected_operation_id


def test_requested_sidon_coverage_distinguishes_decision_from_extension_profile() -> (
    None
):
    catalog = Catalog.open()
    decision = catalog.match(
        OperationMatchRequest(
            need="decide whether this integer set is Sidon and return a collision",
            limit=5,
        )
    )
    extension = catalog.match(
        OperationMatchRequest(
            need="partition supplied candidate integers by whether each extends this Sidon set and return collision obstructions",
            limit=5,
        )
    )

    assert decision.matches[0].operation_id == (
        "combinatorics.integer_set.sidon.decide"
    )
    assert extension.matches[0].operation_id == (
        "combinatorics.integer_set.sidon.extension_profile.compute"
    )


@pytest.mark.parametrize(
    "need",
    (
        (
            "For A={0,1,4,10} and each integer x from 11 through 40, "
            "exhaustively decide whether all positive differences in A union {x} "
            "are distinct; for every invalid x return two distinct unordered pairs "
            "with the same positive difference."
        ),
        (
            "Exhaustively test, for A={0,1,4,10} and each integer x from 11 "
            "through 40, whether A union {x} has all positive pairwise differences "
            "distinct; return every rejected x with two distinct unordered "
            "element-pairs witnessing one repeated positive difference, and the "
            "full admissible/rejected partition."
        ),
    ),
)
def test_observed_exhaustive_sidon_needs_surface_the_aggregate_profile(
    need: str,
) -> None:
    result = Catalog.open().match(OperationMatchRequest(need=need, limit=5))

    assert "combinatorics.integer_set.sidon.extension_profile.compute" in {
        match.operation_id for match in result.matches
    }


def test_observed_ramsey_need_retains_arrowing_over_proper_edge_coloring() -> None:
    result = Catalog.open().match(
        OperationMatchRequest(
            need=(
                "Exhaustively decide whether every red/blue edge-coloring of the "
                "complete graph K5 has a monochromatic triangle; if not, return a "
                "complete coloring witness avoiding monochromatic triangles."
            ),
            limit=20,
        )
    )
    positions = {
        match.operation_id: index for index, match in enumerate(result.matches)
    }

    assert (
        positions["graph.edge_coloring_arrowing.decide"]
        < positions["graph.edge_coloring.k_decide"]
    )
    assert (
        positions["graph.edge_coloring_arrowing.decide"]
        < positions["graph.edge_coloring.check"]
    )


def test_euler_phi_discovery_terms_outrank_generic_inverse_and_solver_operations() -> (
    None
):
    for query, displaced in (
        ("inverse totient preimages", "arithmetic.dirichlet_inverse.compute"),
        ("totient inverse image", "matrix.inverse.compute"),
        ("solve phi(n)=m", "matrix.symbolic.linear_system.solve"),
    ):
        positions = _positions(query)
        assert (
            positions["number_theory.euler_phi.preimages.compute"]
            < positions[displaced]
        )


def test_t_codegree_discovery_terms_route_to_incidence_containment_profiles() -> None:
    for query in (
        "compute t-codegrees of a finite hypergraph",
        "uniform codegree profile",
    ):
        positions = _positions(query)
        assert (
            positions["incidence.containment_profiles.compute"]
            < positions["hypergraph.parameters.compute"]
        )


def test_containment_profile_example_names_complete_pair_codegrees() -> None:
    operation = Catalog.open().operation("incidence.containment_profiles.compute")

    assert operation is not None
    assert operation.examples[0].name == "triangle_pair_codegrees"
    assert operation.examples[0].input["t"] == 2
    assert "all pairs" in operation.examples[0].description
    assert "zero codegree" in operation.examples[0].description
