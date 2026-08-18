"""Exact bounded native kernels for finite graphical models."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations

from jacobian.math.graphical_models.values import (
    MAX_FACTOR_COUNT,
    MAX_MODEL_VARS,
    Factor,
    parse_canonical_rational,
    scope_size,
)


def factor_multiply(left: Factor, right: Factor) -> Factor:
    """Multiply two exact factors over their canonical union scope."""

    _require_compatible_domains((left, right), left.domain_sizes)
    variables = tuple(sorted(set(left.variables) | set(right.variables)))
    total = scope_size(variables, left.domain_sizes)
    table: list[str] = []
    for index in range(total):
        assignment = _index_to_assignment(index, variables, left.domain_sizes)
        left_index = _projected_index(
            assignment, variables, left.variables, left.domain_sizes
        )
        right_index = _projected_index(
            assignment, variables, right.variables, right.domain_sizes
        )
        value = parse_canonical_rational(
            left.table[left_index]
        ) * parse_canonical_rational(right.table[right_index])
        table.append(str(value))
    return Factor(
        variables=variables,
        domain_sizes=left.domain_sizes,
        table=tuple(table),
    )


def factor_marginalize(factor: Factor, variable: int) -> Factor:
    """Sum one variable out of an exact factor, possibly yielding a scalar."""

    if variable not in factor.variables:
        raise ValueError("variable is not in factor")
    variables = tuple(item for item in factor.variables if item != variable)
    table: list[str] = []
    for index in range(scope_size(variables, factor.domain_sizes)):
        assignment = _index_to_assignment(index, variables, factor.domain_sizes)
        total = Fraction(0)
        for value in range(factor.domain_sizes[variable]):
            full_assignment = dict(zip(variables, assignment, strict=True))
            full_assignment[variable] = value
            source_index = _assignment_to_index(
                tuple(full_assignment[item] for item in factor.variables),
                factor.variables,
                factor.domain_sizes,
            )
            total += parse_canonical_rational(factor.table[source_index])
        table.append(str(total))
    return Factor(
        variables=variables,
        domain_sizes=factor.domain_sizes,
        table=tuple(table),
    )


def variable_elimination(
    factors: Sequence[Factor],
    domain_sizes: tuple[int, ...],
    elimination_order: tuple[int, ...],
    query_variables: tuple[int, ...],
) -> Factor:
    """Return the exact unnormalized marginal factor for a complete order."""

    _require_elimination_contract(
        factors, domain_sizes, elimination_order, query_variables
    )
    working = list(factors)
    for variable in elimination_order:
        relevant = [factor for factor in working if variable in factor.variables]
        working = [factor for factor in working if variable not in factor.variables]
        product = _multiply_all(relevant)
        working.append(factor_marginalize(product, variable))
    result = _multiply_all(working)
    if result.variables != query_variables:
        raise RuntimeError("variable elimination did not produce the bound query scope")
    return result


def d_separation(
    variable_count: int,
    edges: tuple[tuple[int, int], ...],
    set_a: tuple[int, ...],
    set_b: tuple[int, ...],
    set_c: tuple[int, ...],
) -> bool:
    """Decide d-separation by ancestral restriction and moralization."""

    parents = _validate_dag(variable_count, edges)
    _validate_node_sets(variable_count, set_a, set_b, set_c)
    ancestral = _ancestors(set(set_a) | set(set_b) | set(set_c), parents)
    adjacency: dict[int, set[int]] = {node: set() for node in ancestral}
    for child in ancestral:
        relevant_parents = sorted(parents[child] & ancestral)
        for parent in relevant_parents:
            adjacency[parent].add(child)
            adjacency[child].add(parent)
        for left, right in combinations(relevant_parents, 2):
            adjacency[left].add(right)
            adjacency[right].add(left)
    blocked = set(set_c)
    targets = set(set_b)
    queue = deque(node for node in set_a if node not in blocked)
    reachable = set(queue)
    while queue:
        node = queue.popleft()
        if node in targets:
            return False
        for neighbor in adjacency[node] - blocked - reachable:
            reachable.add(neighbor)
            queue.append(neighbor)
    return True


def validate_variable_elimination_input(
    factors: Sequence[Factor],
    domain_sizes: tuple[int, ...],
    elimination_order: tuple[int, ...],
    query_variables: tuple[int, ...],
) -> None:
    """Validate the complete bounded elimination contract without arithmetic."""

    _require_elimination_contract(
        factors, domain_sizes, elimination_order, query_variables
    )


def validate_d_separation_input(
    variable_count: int,
    edges: tuple[tuple[int, int], ...],
    set_a: tuple[int, ...],
    set_b: tuple[int, ...],
    set_c: tuple[int, ...],
) -> None:
    """Validate one bounded DAG and its pairwise-disjoint node sets."""

    _validate_dag(variable_count, edges)
    _validate_node_sets(variable_count, set_a, set_b, set_c)


def _multiply_all(factors: Sequence[Factor]) -> Factor:
    if not factors:
        raise ValueError("at least one factor is required")
    result = factors[0]
    for factor in factors[1:]:
        result = factor_multiply(result, factor)
    return result


def _require_compatible_domains(
    factors: Sequence[Factor], domain_sizes: tuple[int, ...]
) -> None:
    if not 1 <= len(domain_sizes) <= MAX_MODEL_VARS:
        raise ValueError("domain_sizes must describe between 1 and 16 variables")
    if any(factor.domain_sizes != domain_sizes for factor in factors):
        raise ValueError("all factors must share the exact model domain_sizes")


def _require_elimination_contract(
    factors: Sequence[Factor],
    domain_sizes: tuple[int, ...],
    elimination_order: tuple[int, ...],
    query_variables: tuple[int, ...],
) -> None:
    if not 1 <= len(factors) <= MAX_FACTOR_COUNT:
        raise ValueError("factor family must contain between 1 and 64 factors")
    _require_compatible_domains(factors, domain_sizes)
    model_variables = {variable for factor in factors for variable in factor.variables}
    if query_variables != tuple(sorted(set(query_variables))):
        raise ValueError("query variables must be distinct and sorted")
    if not set(query_variables) <= model_variables:
        raise ValueError("query variables must occur in the factor family")
    if len(set(elimination_order)) != len(elimination_order):
        raise ValueError("elimination order cannot repeat a variable")
    if set(elimination_order) != model_variables - set(query_variables):
        raise ValueError("elimination order must contain every non-query variable once")
    _require_bounded_intermediate_scopes(
        tuple(factor.variables for factor in factors),
        domain_sizes,
        elimination_order,
    )


def _require_bounded_intermediate_scopes(
    scopes: tuple[tuple[int, ...], ...],
    domain_sizes: tuple[int, ...],
    elimination_order: tuple[int, ...],
) -> None:
    working = list(scopes)
    for variable in elimination_order:
        relevant = [scope for scope in working if variable in scope]
        working = [scope for scope in working if variable not in scope]
        union = tuple(sorted({item for scope in relevant for item in scope}))
        scope_size(union, domain_sizes)
        working.append(tuple(item for item in union if item != variable))
    scope_size(
        tuple(sorted({item for scope in working for item in scope})), domain_sizes
    )


def _validate_dag(
    variable_count: int, edges: tuple[tuple[int, int], ...]
) -> dict[int, set[int]]:
    if not 1 <= variable_count <= MAX_MODEL_VARS:
        raise ValueError("variable_count must be between 1 and 16")
    if len(set(edges)) != len(edges):
        raise ValueError("directed edges must be distinct")
    parents: dict[int, set[int]] = {node: set() for node in range(variable_count)}
    children: dict[int, set[int]] = {node: set() for node in range(variable_count)}
    for parent, child in edges:
        if not 0 <= parent < variable_count or not 0 <= child < variable_count:
            raise ValueError("edge endpoint is outside the graph")
        if parent == child:
            raise ValueError("directed graph cannot contain a self-loop")
        parents[child].add(parent)
        children[parent].add(child)
    indegree = {node: len(parents[node]) for node in parents}
    queue = deque(node for node, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if visited != variable_count:
        raise ValueError("d-separation requires a directed acyclic graph")
    return parents


def _validate_node_sets(
    variable_count: int,
    set_a: tuple[int, ...],
    set_b: tuple[int, ...],
    set_c: tuple[int, ...],
) -> None:
    node_sets = (set_a, set_b, set_c)
    if not set_a or not set_b:
        raise ValueError("sets A and B must be nonempty")
    if any(len(values) != len(set(values)) for values in node_sets):
        raise ValueError("d-separation node sets cannot contain duplicates")
    if any(not 0 <= node < variable_count for values in node_sets for node in values):
        raise ValueError("d-separation node is outside the graph")
    if set(set_a) & set(set_b) or set(set_a) & set(set_c) or set(set_b) & set(set_c):
        raise ValueError("d-separation node sets must be pairwise disjoint")


def _ancestors(nodes: set[int], parents: dict[int, set[int]]) -> set[int]:
    result = set(nodes)
    queue = list(nodes)
    while queue:
        node = queue.pop()
        for parent in parents[node] - result:
            result.add(parent)
            queue.append(parent)
    return result


def _index_to_assignment(
    index: int, variables: tuple[int, ...], domain_sizes: tuple[int, ...]
) -> tuple[int, ...]:
    assignment: list[int] = []
    for variable in reversed(variables):
        assignment.append(index % domain_sizes[variable])
        index //= domain_sizes[variable]
    return tuple(reversed(assignment))


def _assignment_to_index(
    assignment: tuple[int, ...],
    variables: tuple[int, ...],
    domain_sizes: tuple[int, ...],
) -> int:
    index = 0
    for variable, value in zip(variables, assignment, strict=True):
        index = index * domain_sizes[variable] + value
    return index


def _projected_index(
    assignment: tuple[int, ...],
    variables: tuple[int, ...],
    projected_variables: tuple[int, ...],
    domain_sizes: tuple[int, ...],
) -> int:
    positions = {variable: index for index, variable in enumerate(variables)}
    projected = tuple(
        assignment[positions[variable]] for variable in projected_variables
    )
    return _assignment_to_index(projected, projected_variables, domain_sizes)


__all__ = [
    "d_separation",
    "factor_marginalize",
    "factor_multiply",
    "variable_elimination",
]
