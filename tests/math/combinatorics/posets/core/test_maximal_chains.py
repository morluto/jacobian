"""Complete maximal-chain enumeration from canonical cover relations."""

from jacobian.math.combinatorics.posets.core._maximal_chains import (
    enumerate_maximal_chains,
)
from jacobian.math.combinatorics.posets.core._models import (
    PosetRequest,
    PresentationPair,
    ReflexivePairPolicy,
    RelationInterpretation,
)
from jacobian.math.combinatorics.posets.core.operations import materialize_finite_poset


def test_diamond_chains_endpoints_and_histogram() -> None:
    poset = materialize_finite_poset(
        ("0", "a", "b", "1"),
        tuple(
            PresentationPair(lower=lower, upper=upper)
            for lower, upper in (("0", "a"), ("0", "b"), ("a", "1"), ("b", "1"))
        ),
        RelationInterpretation.COVER_EDGES,
        ReflexivePairPolicy.FORBIDDEN,
    )
    result = enumerate_maximal_chains(PosetRequest(poset=poset))
    assert tuple(row.elements for row in result.chains) == (
        ("0", "a", "1"),
        ("0", "b", "1"),
    )
    assert all(
        row.lower_endpoint == "0" and row.upper_endpoint == "1" for row in result.chains
    )
    assert [(cell.length, cell.count) for cell in result.length_histogram] == [(3, 2)]


def test_empty_poset_has_single_empty_maximal_chain() -> None:
    poset = materialize_finite_poset(
        (), (), RelationInterpretation.COVER_EDGES, ReflexivePairPolicy.FORBIDDEN
    )
    result = enumerate_maximal_chains(PosetRequest(poset=poset))
    assert result.chains[0].elements == ()
    assert result.chains[0].lower_endpoint is None
    assert [(cell.length, cell.count) for cell in result.length_histogram] == [(0, 1)]
