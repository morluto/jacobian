"""Independent small-universe oracle for homomorphism count and search."""

from itertools import product

from jacobian.math.logic.relational_structures.operations import (
    count_homomorphisms,
    search_homomorphism,
)
from jacobian.math.logic.relational_structures.values import (
    FiniteRelationalStructure,
    FiniteRelationSymbol,
)


def _structures(size: int) -> tuple[FiniteRelationalStructure, ...]:
    symbol = FiniteRelationSymbol(symbol_id="R", arity=2)
    cells = tuple(product(range(size), repeat=2))
    return tuple(
        FiniteRelationalStructure(
            carrier_size=size,
            signature=(symbol,),
            relation_tables=(
                tuple(row for bit, row in enumerate(cells) if mask & (1 << bit)),
            ),
        )
        for mask in range(1 << len(cells))
    )


def _oracle_maps(source: FiniteRelationalStructure, target: FiniteRelationalStructure):
    maps = product(range(target.carrier_size), repeat=source.carrier_size)
    for candidate in maps:
        if all(
            tuple(candidate[index] for index in row) in target.relation_tables[0]
            for row in source.relation_tables[0]
        ):
            yield candidate


def test_all_binary_relations_on_carriers_up_to_two_match_independent_oracle():
    structures = tuple(s for size in range(3) for s in _structures(size))
    for source in structures:
        for target in structures:
            expected = tuple(_oracle_maps(source, target))
            counted = count_homomorphisms(source, target)
            searched = search_homomorphism(source, target)
            assert counted.count == len(expected)
            assert (searched.status.value == "FOUND") == bool(expected)
            if expected:
                assert searched.check is not None
                assert searched.check.carrier_map == expected[0]
            else:
                assert searched.candidates_examined == searched.total_candidates
