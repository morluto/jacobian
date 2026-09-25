"""State-transition algebras of complete deterministic tree automata."""

from __future__ import annotations

from itertools import product

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.automata.tree.values import (
    CompleteDeterministicBottomUpTreeAutomaton,
)
from jacobian.math.universal_algebra.values import (
    MAX_ARITY,
    MAX_CARRIER_SIZE,
    MAX_SIGNATURE_SIZE,
    MAX_TABLE_CELLS,
    FiniteAlgebra,
    OperationSymbol,
)


def deterministic_tree_automaton_state_algebra(
    automaton: CompleteDeterministicBottomUpTreeAutomaton,
) -> FiniteAlgebra:
    """Return the finite algebra whose operations are the automaton transitions.

    Carrier position ``i`` is automaton state ``i``. Ranked symbol ``j`` is
    represented by operation ``tree_symbol_j`` with arity ``automaton.arity[j]``.
    Each operation table uses the universal-algebra row-major Cartesian order,
    which agrees with lexicographic tuples of child-state positions.
    """

    carrier_size = automaton.state_count
    symbol_count = len(automaton.arity)
    if carrier_size > MAX_CARRIER_SIZE:
        _reject("state count exceeds the finite-algebra carrier bound")
    if symbol_count > MAX_SIGNATURE_SIZE:
        _reject("ranked alphabet exceeds the finite-algebra signature bound")
    if any(rank > MAX_ARITY for rank in automaton.arity):
        _reject("symbol arity exceeds the finite-algebra operation bound")

    table_cell_count = sum(carrier_size**rank for rank in automaton.arity)
    if table_cell_count > MAX_TABLE_CELLS:
        _reject("transition tables exceed the finite-algebra cell bound")

    transition_targets = {
        (row.symbol, row.child_states): row.target_state
        for row in automaton.transitions
    }
    operations = tuple(
        OperationSymbol(operation_id=f"tree_symbol_{symbol}", arity=rank)
        for symbol, rank in enumerate(automaton.arity)
    )
    tables = tuple(
        tuple(
            transition_targets[(symbol, children)]
            for children in product(range(carrier_size), repeat=rank)
        )
        for symbol, rank in enumerate(automaton.arity)
    )
    return FiniteAlgebra(
        carrier=tuple(f"state_{state}" for state in range(carrier_size)),
        operations=operations,
        tables=tables,
    )


def _reject(message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("automaton",),
        code="tree_automata.state_algebra.resource_bound",
        message=message,
    )


__all__ = ["deterministic_tree_automaton_state_algebra"]
