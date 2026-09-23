"""Independent defining-invariant tests for the finite-poset Möbius function."""

from itertools import pairwise

from jacobian.math.combinatorics.posets.core._models import (
    FinitePosetRequest,
    MobiusScope,
    PresentationPair,
    RelationInterpretation,
)
from jacobian.math.combinatorics.posets.core.operations import (
    materialize_finite_poset,
    mobius_function,
)


def test_chain_mobius_values_match_the_interval_formula() -> None:
    elements = ("a", "b", "c", "d", "e")
    request = FinitePosetRequest(
        elements=elements,
        relation=tuple(
            PresentationPair(lower=lower, upper=upper)
            for lower, upper in pairwise(elements)
        ),
        interpretation=RelationInterpretation.COVER_EDGES,
    )
    poset = materialize_finite_poset(
        request.elements,
        request.relation,
        request.interpretation,
        request.reflexive_pairs,
    )

    result = mobius_function(poset, MobiusScope.COMPLETE_MATRIX, ())

    values = {(row.lower, row.upper): row.value for row in result.values}
    expected = {
        (lower, upper): 1
        if upper_index == lower_index
        else -1
        if upper_index == lower_index + 1
        else 0
        for lower_index, lower in enumerate(elements)
        for upper_index, upper in enumerate(elements[lower_index:], start=lower_index)
    }
    assert values == expected
