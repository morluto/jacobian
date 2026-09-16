"""Exact bounded native kernel for relational homomorphism checking."""

from __future__ import annotations

from collections.abc import Sequence

from jacobian.math.logic.relational_structures._admission import (
    admit_homomorphism_check,
)
from jacobian.math.logic.relational_structures._models import (
    HomomorphismCheckResult,
    HomomorphismStatus,
    HomomorphismViolationWitness,
    SymbolTransportProfile,
)
from jacobian.math.logic.relational_structures.values import (
    FiniteRelationalStructure,
)


def check_homomorphism(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
    carrier_map: Sequence[int],
) -> HomomorphismCheckResult:
    """Decide one candidate carrier map by exhaustive invariant replay.

    For every relation symbol ``R`` and every tuple ``t`` in the complete
    source table ``R^A``, the coordinatewise image ``h(t)`` is tested for
    membership in the complete target table ``R^B``. The replay is exhaustive
    (never sampled or short-circuited), records complete per-symbol transport
    counts, and retains the first violating ``(symbol, source tuple, image
    tuple)`` witness in deterministic signature/row order. The identity map on
    any structure is a homomorphism, and the composition of two checked
    homomorphisms is again checked as a homomorphism by this same replay.
    """

    admit_homomorphism_check(source, target, carrier_map)
    checked_map = tuple(carrier_map)
    target_tables = tuple(set(table) for table in target.relation_tables)

    witness: HomomorphismViolationWitness | None = None
    profiles: list[SymbolTransportProfile] = []
    for symbol_index, symbol in enumerate(source.signature):
        source_table = source.relation_tables[symbol_index]
        target_table = target_tables[symbol_index]
        preserved = 0
        for source_tuple in source_table:
            image_tuple = tuple(checked_map[coordinate] for coordinate in source_tuple)
            if image_tuple in target_table:
                preserved += 1
            elif witness is None:
                witness = HomomorphismViolationWitness(
                    symbol_id=symbol.symbol_id,
                    arity=symbol.arity,
                    source_tuple=source_tuple,
                    image_tuple=image_tuple,
                )
        profiles.append(
            SymbolTransportProfile(
                symbol_id=symbol.symbol_id,
                arity=symbol.arity,
                source_tuples=len(source_table),
                preserved_tuples=preserved,
            )
        )

    status = (
        HomomorphismStatus.NOT_HOMOMORPHISM
        if witness is not None
        else HomomorphismStatus.HOMOMORPHISM
    )
    return HomomorphismCheckResult._from_kernel(
        status=status,
        source=source,
        target=target,
        carrier_map=checked_map,
        witness=witness,
        symbol_profiles=tuple(profiles),
    )


__all__ = ["check_homomorphism"]
