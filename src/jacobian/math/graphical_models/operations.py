"""Domain-owned graphical model kernels."""

from __future__ import annotations

from fractions import Fraction

from jacobian.math.graphical_models.values import Factor

__all__ = [
    "d_separation",
    "factor_marginalize",
    "factor_multiply",
    "variable_elimination",
]


def _str_to_fraction(s: str) -> Fraction:
    return Fraction(s)


def _fraction_to_str(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator)


def _index_to_assignment(
    index: int, variables: tuple[int, ...], domain_sizes: tuple[int, ...],
) -> tuple[int, ...]:
    assignment: list[int] = []
    remaining = index
    for var in reversed(variables):
        assignment.append(remaining % domain_sizes[var])
        remaining //= domain_sizes[var]
    return tuple(reversed(assignment))


def _assignment_to_index(
    assignment: tuple[int, ...], variables: tuple[int, ...],
    domain_sizes: tuple[int, ...],
) -> int:
    index = 0
    for var, val in zip(variables, assignment, strict=False):
        index = index * domain_sizes[var] + val
    return index


def factor_multiply(left: Factor, right: Factor) -> Factor:
    """Multiply two factors over the union of their variables."""
    all_vars = list(left.variables)
    for v in right.variables:
        if v not in all_vars:
            all_vars.append(v)
    all_vars = tuple(all_vars)
    left_idx = [all_vars.index(v) for v in left.variables]
    right_idx = [all_vars.index(v) for v in right.variables]

    new_table: list[str] = []
    # Use the larger domain_sizes (factors may have partial domain_sizes)
    domain_sizes = left.domain_sizes if len(left.domain_sizes) >= len(right.domain_sizes) else right.domain_sizes
    total = 1
    for v in all_vars:
        if v >= len(domain_sizes):
            raise ValueError(f"variable {v} out of range for domain_sizes")
        total *= domain_sizes[v]
    for combined_idx in range(total):
        assignment = _index_to_assignment(combined_idx, all_vars, domain_sizes)
        left_assignment = tuple(assignment[i] for i in left_idx)
        left_table_idx = _assignment_to_index(
            left_assignment, left.variables, domain_sizes,
        )
        right_assignment = tuple(assignment[i] for i in right_idx)
        right_table_idx = _assignment_to_index(
            right_assignment, right.variables, domain_sizes,
        )
        val = _str_to_fraction(left.table[left_table_idx]) * _str_to_fraction(
            right.table[right_table_idx]
        )
        new_table.append(_fraction_to_str(val))

    return Factor(
        variables=all_vars,
        domain_sizes=domain_sizes,
        table=tuple(new_table),
    )


def factor_marginalize(factor: Factor, variable: int) -> Factor:
    """Sum out a variable from a factor."""
    remaining_vars = tuple(v for v in factor.variables if v != variable)
    if not remaining_vars:
        total = Fraction(0)
        for v in factor.table:
            total += _str_to_fraction(v)
        return Factor(
            variables=(0,),
            domain_sizes=factor.domain_sizes,
            table=(_fraction_to_str(total),) * max(factor.domain_sizes[0], 1),
        )
    domain_sizes = factor.domain_sizes
    remaining_size = 1
    for v in remaining_vars:
        remaining_size *= domain_sizes[v]
    new_table: list[str] = []
    var_idx = factor.variables.index(variable)
    for combined_idx in range(remaining_size):
        assignment = _index_to_assignment(
            combined_idx, remaining_vars, domain_sizes,
        )
        total = Fraction(0)
        for val in range(domain_sizes[variable]):
            full_assignment = list(assignment)
            full_assignment.insert(var_idx, val)
            table_idx = _assignment_to_index(
                tuple(full_assignment), factor.variables, domain_sizes,
            )
            total += _str_to_fraction(factor.table[table_idx])
        new_table.append(_fraction_to_str(total))
    return Factor(
        variables=remaining_vars,
        domain_sizes=domain_sizes,
        table=tuple(new_table),
    )


def variable_elimination(
    factors: list[Factor],
    domain_sizes: tuple[int, ...],
    elimination_order: tuple[int, ...],
    query_variables: tuple[int, ...],
) -> Factor:
    """Compute a marginal via variable elimination."""
    result: list[Factor] = list(factors)
    for var in elimination_order:
        relevant: list[Factor] = []
        remaining: list[Factor] = []
        for f in result:
            if var in f.variables:
                relevant.append(f)
            else:
                remaining.append(f)
        if relevant:
            product_factor = relevant[0]
            for f in relevant[1:]:
                product_factor = factor_multiply(product_factor, f)
            marginalized = factor_marginalize(product_factor, var)
            remaining.append(marginalized)
        result = remaining
    if not result:
        return factors[0]
    final = result[0]
    for f in result[1:]:
        final = factor_multiply(final, f)
    return final


def d_separation(  # noqa: C901
    variable_count: int,
    edges: tuple[tuple[int, int], ...],
    set_a: tuple[int, ...],
    set_b: tuple[int, ...],
    set_c: tuple[int, ...],
) -> bool:
    """Check d-separation of set_a and set_b given set_c.

    Uses the Bayes-ball algorithm: finds all nodes reachable from set_a
    via active trails given evidence set_c. If any node in set_b is
    reachable, the sets are NOT d-separated.
    """
    set_b_set = set(set_b)
    set_c_set = set(set_c)
    parents: dict[int, set[int]] = {i: set() for i in range(variable_count)}
    children: dict[int, set[int]] = {i: set() for i in range(variable_count)}
    for parent, child in edges:
        parents[child].add(parent)
        children[parent].add(child)

    # Bayes-ball: reach[i] = True if node i is reachable from set_a
    # via an active trail given set_c
    # Start from all nodes in set_a, visiting via (node, direction)
    # direction: True = coming via child (arriving from below), False = coming via parent
    reachable: set[tuple[int, bool]] = set()
    queue: list[tuple[int, bool]] = []
    for node in set_a:
        queue.append((node, True))
        queue.append((node, False))
    while queue:
        node, from_child = queue.pop()
        if (node, from_child) in reachable:
            continue
        reachable.add((node, from_child))
        if from_child:
            # Arriving from below: can go to parents if not observed
            if node not in set_c_set:
                for parent in parents[node]:
                    if (parent, False) not in reachable:
                        queue.append((parent, False))
            # Can go to children if not observed
            if node not in set_c_set:
                for child in children[node]:
                    if (child, True) not in reachable:
                        queue.append((child, True))
        else:
            # Arriving from above: can go to children if not observed
            if node not in set_c_set:
                for child in children[node]:
                    if (child, True) not in reachable:
                        queue.append((child, True))
            # Can go to parents if observed (collider)
            if node in set_c_set:
                for parent in parents[node]:
                    if (parent, False) not in reachable:
                        queue.append((parent, False))
    reachable_nodes = {node for node, _ in reachable}
    return not (reachable_nodes & set_b_set)
