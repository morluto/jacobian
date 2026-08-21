"""Exact native kernels over finite algebras."""

from __future__ import annotations

from itertools import product as iproduct

from .values import (
    ApplicationTerm,
    FiniteAlgebra,
    FlatTerm,
    OperationSymbol,
    VariableTerm,
    require_term_for_algebra,
)

__all__ = [
    "congruence_check",
    "equation_profile",
    "evaluate_term",
    "generated_subalgebra",
    "quotient",
]


def _evaluate_node(
    algebra: FiniteAlgebra,
    term: FlatTerm,
    assignment: dict[int, int],
    n: int,
    index: int,
) -> int:
    node = term.nodes[index]
    if isinstance(node, VariableTerm):
        if node.variable_id not in assignment:
            raise ValueError("incomplete assignment")
        return assignment[node.variable_id]
    if isinstance(node, ApplicationTerm):
        args = [_evaluate_node(algebra, term, assignment, n, c) for c in node.children]
        cell_index = 0
        for arg in args:
            cell_index = cell_index * n + arg
        return algebra.tables[node.operation][cell_index]
    raise AssertionError("closed term union admitted an unknown node")


def evaluate_term(
    algebra: FiniteAlgebra, term: FlatTerm, assignment: dict[int, int]
) -> int:
    """Evaluate a source-bound term under a complete variable assignment.

    Return the exact carrier value ``t^A(alpha)``.
    """
    n = len(algebra.carrier)
    require_term_for_algebra(term, algebra)
    if any(not 0 <= v < n for v in assignment.values()):
        raise ValueError("assignment value out of carrier range")
    return _evaluate_node(algebra, term, assignment, n, term.root)


def equation_profile(
    algebra: FiniteAlgebra, left: FlatTerm, right: FlatTerm, variable_count: int
) -> dict[str, object]:
    """Evaluate ``s = t`` over all assignments.

    Return ``HOLDS`` with the satisfying assignment count, or ``FAILS`` with
    the first counterassignment and exact left/right values.
    """
    from itertools import product as iproduct

    n = len(algebra.carrier)
    satisfying = 0
    first_counterassignment = None
    for values in iproduct(range(n), repeat=variable_count):
        assignment = dict(enumerate(values))
        lv = evaluate_term(algebra, left, assignment)
        rv = evaluate_term(algebra, right, assignment)
        if lv == rv:
            satisfying += 1
        else:
            if first_counterassignment is None:
                first_counterassignment = {
                    "assignment": tuple(values),
                    "left_value": lv,
                    "right_value": rv,
                }
    if satisfying == n**variable_count:
        return {"status": "HOLDS", "satisfying_count": satisfying}
    return {
        "status": "FAILS",
        "satisfying_count": satisfying,
        "first_counterassignment": first_counterassignment,
    }


def generated_subalgebra(
    algebra: FiniteAlgebra, generators: tuple[int, ...]
) -> dict[str, object]:
    """Return the least subalgebra containing the generating set by finite
    closure under all basic operations and nullary constants."""
    n = len(algebra.carrier)
    carrier_set = set(generators)
    for op_idx, symbol in enumerate(algebra.operations):
        if symbol.arity == 0:
            for output in algebra.tables[op_idx]:
                carrier_set.add(output)
    changed = True
    rounds = 0
    while changed:
        changed = False
        rounds += 1
        for op_idx, symbol in enumerate(algebra.operations):
            if symbol.arity == 0:
                continue
            from itertools import product as iproduct

            for args in iproduct(carrier_set, repeat=symbol.arity):
                cell_index = 0
                for arg in args:
                    cell_index = cell_index * n + arg
                output = algebra.tables[op_idx][cell_index]
                if output not in carrier_set:
                    carrier_set.add(output)
                    changed = True
    sorted_carrier = sorted(carrier_set)
    return {
        "generated_carrier": tuple(sorted_carrier),
        "rounds": rounds,
        "is_closed": set(generators) == carrier_set if generators else True,
    }


def _compatibility_violation(
    algebra: FiniteAlgebra,
    block_of: dict[int, int],
    n: int,
    op_idx: int,
    symbol: OperationSymbol,
    x: tuple[int, ...],
    y: tuple[int, ...],
) -> dict[str, object] | None:
    if not all(block_of[x[j]] == block_of[y[j]] for j in range(symbol.arity)):
        return None
    cell_x = 0
    cell_y = 0
    for j in range(symbol.arity):
        cell_x = cell_x * n + x[j]
        cell_y = cell_y * n + y[j]
    fx = algebra.tables[op_idx][cell_x]
    fy = algebra.tables[op_idx][cell_y]
    if block_of[fx] == block_of[fy]:
        return None
    return {
        "is_congruence": False,
        "obstruction": "compatibility_violation",
        "operation": op_idx,
        "x": x,
        "y": y,
    }


def _check_compatibility(
    algebra: FiniteAlgebra,
    block_of: dict[int, int],
    n: int,
) -> dict[str, object] | None:
    """Check congruence compatibility, returning an obstruction or None."""
    from itertools import product as iproduct

    for op_idx, symbol in enumerate(algebra.operations):
        if symbol.arity == 0:
            continue
        for x in iproduct(range(n), repeat=symbol.arity):
            for j in range(symbol.arity):
                for y_elem in range(n):
                    if y_elem == x[j]:
                        continue
                    if block_of[x[j]] != block_of[y_elem]:
                        continue
                    y_list = list(x)
                    y_list[j] = y_elem
                    y: tuple[int, ...] = tuple(y_list)
                    violation = _compatibility_violation(
                        algebra, block_of, n, op_idx, symbol, x, y
                    )
                    if violation is not None:
                        return violation
    return None


def congruence_check(
    algebra: FiniteAlgebra, partition: tuple[tuple[int, ...], ...]
) -> dict[str, object]:
    """Check whether a carrier partition is a compatible equivalence
    relation (congruence).

    A congruence theta satisfies: if x_j theta y_j for every argument j, then
    f(x_1,...,x_r) theta f(y_1,...,y_r) for every basic operation.
    """
    n = len(algebra.carrier)
    block_of: dict[int, int] = {}
    for block_idx, block in enumerate(partition):
        for elem in block:
            block_of[elem] = block_idx
    if len(block_of) != n:
        return {
            "is_congruence": False,
            "obstruction": "partition does not cover carrier",
        }
    result = _check_compatibility(algebra, block_of, n)
    return result if result is not None else {"is_congruence": True}


def quotient(
    algebra: FiniteAlgebra, partition: tuple[tuple[int, ...], ...]
) -> tuple[FiniteAlgebra, tuple[int, ...]]:
    """Return the quotient algebra ``A/theta`` induced by a congruence."""
    check = congruence_check(algebra, partition)
    if not check["is_congruence"]:
        raise ValueError("partition is not a congruence")
    n = len(algebra.carrier)
    block_of: dict[int, int] = {}
    for block_idx, block in enumerate(partition):
        for elem in block:
            block_of[elem] = block_idx
    quotient_carrier = tuple(f"B{i}" for i in range(len(partition)))
    quotient_tables: list[tuple[int, ...]] = []
    for op_idx, symbol in enumerate(algebra.operations):
        if symbol.arity == 0:
            original_output = algebra.tables[op_idx][0]
            quotient_tables.append((block_of[original_output],))
        else:
            block_count = len(partition)
            # Use a representative for each block (the minimum element)
            representatives = [min(block) for block in partition]
            table = []
            for args in iproduct(range(block_count), repeat=symbol.arity):
                # Compute the operation on the representatives
                cell_index = 0
                for arg in args:
                    cell_index = cell_index * n + representatives[arg]
                output = algebra.tables[op_idx][cell_index]
                table.append(block_of[output])
            quotient_tables.append(tuple(table))
    quotient_algebra = FiniteAlgebra(
        carrier=quotient_carrier,
        operations=algebra.operations,
        tables=tuple(quotient_tables),
    )
    quotient_map = tuple(block_of[element] for element in range(n))
    return quotient_algebra, quotient_map
