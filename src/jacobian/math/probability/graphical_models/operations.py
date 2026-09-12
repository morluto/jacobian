"""Exact bounded native kernels for finite graphical models."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations
from math import gcd

from jacobian._exact import (
    CanonicalRational,
    require_bounded_rational,
)
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math._rational_height import RationalHeight
from jacobian.math.probability.graphical_models._models import DSeparationResult
from jacobian.math.probability.graphical_models._validation import (
    validate_d_separation_input,
)
from jacobian.math.probability.graphical_models.values import (
    MAX_FACTOR_COUNT,
    MAX_MODEL_VARS,
    MAX_RATIONAL_DIGITS,
    Factor,
    scope_size,
)


def _reject(code: str, message: str, *location: str) -> None:
    raise OperationDomainValidationError(
        location=location,
        code=f"graphical_model.{code}",
        message=message,
    )


def _reject_rational_growth(operation: str, phase: str) -> None:
    raise OperationResourceAdmissionError(
        location=("factor", "table"),
        code=f"graphical_model.factor_{operation}_rational_bound",
        message=(
            f"{phase} rational growth exceeds the "
            f"{MAX_RATIONAL_DIGITS}-digit factor bound"
        ),
    )


def _integer_digits(value: int) -> int:
    return len(format_canonical_integer(abs(value)))


def _factor_height(value: CanonicalRational) -> RationalHeight:
    return RationalHeight.from_canonical(value)


def _admit_factor_source(factor: Factor, operation: str) -> tuple[RationalHeight, ...]:
    """Check source components once before deriving an arithmetic profile."""

    heights: list[RationalHeight] = []
    for value in factor.table:
        try:
            require_bounded_rational(
                value,
                max_digits=MAX_RATIONAL_DIGITS,
                label="factor source",
            )
        except ValueError:
            _reject_rational_growth(operation, "source")
        heights.append(_factor_height(value))
    return tuple(heights)


def _sum_height_bound(values: Sequence[CanonicalRational]) -> RationalHeight:
    """Bound a nonnegative rational sum without constructing a common denominator."""

    nonzero = tuple(value for value in values if value.num != 0)
    if not nonzero:
        return RationalHeight(1, 1)
    heights = tuple(_factor_height(value) for value in nonzero)
    # A product of one copy of every distinct denominator is a valid common
    # denominator.  Repeated denominators must not pay for the same factor
    # repeatedly: this retains useful near-limit sums such as 32/d.
    denominator_digits = sum(
        _integer_digits(den) for den in {value.den for value in nonzero}
    )
    numerator_digits = max(
        height.numerator_digits + denominator_digits - height.denominator_digits
        for height in heights
    )
    if len(nonzero) > 1:
        numerator_digits += _integer_digits(len(nonzero))
    return RationalHeight(numerator_digits, denominator_digits)


def _product_height(
    left: CanonicalRational, right: CanonicalRational
) -> RationalHeight:
    """Return the exact reduced component height of one rational product."""

    if left.num == 0 or right.num == 0:
        return RationalHeight(1, 1)
    # Cross-cancel before multiplying.  Since both operands are canonical,
    # the two cross-cancellations leave coprime numerator and denominator
    # products, so this is the exact product height without Fraction replay.
    left_num = left.num
    right_den = right.den
    cancellation = gcd(abs(left_num), right_den)
    left_num //= cancellation
    right_den //= cancellation
    right_num = right.num
    left_den = left.den
    cancellation = gcd(abs(right_num), left_den)
    right_num //= cancellation
    left_den //= cancellation
    return RationalHeight(
        _integer_digits(left_num * right_num),
        _integer_digits(left_den * right_den),
    )


def _bounded_integer_product(
    left: int,
    right: int,
    *,
    max_digits: int,
    operation: str,
    phase: str,
) -> int:
    """Multiply integers only when their product can fit the admitted height."""

    if left == 0 or right == 0:
        return 0
    # A product of a-digit and b-digit positive integers has at most a+b
    # digits and at least a+b-1.  The one-digit slack lets us inspect a
    # boundary product exactly while avoiding any over-height allocation.
    if _integer_digits(left) + _integer_digits(right) > max_digits + 1:
        _reject_rational_growth(operation, phase)
    product = left * right
    if _integer_digits(product) > max_digits:
        _reject_rational_growth(operation, phase)
    return product


def _admit_bounded_sum(values: Sequence[CanonicalRational], operation: str) -> None:
    """Check a risky sum with capped integer arithmetic before kernel expansion."""

    numerator = 0
    denominator = 1
    for value in values:
        if value.num == 0:
            continue
        if numerator == 0:
            numerator, denominator = value.num, value.den
            continue
        # Match Fraction._add: form the lifted numerator, cancel against the
        # shared denominator gcd, then multiply only the reduced denominator.
        shared = gcd(denominator, value.den)
        left_scale = value.den // shared
        right_scale = denominator // shared
        left_term = _bounded_integer_product(
            numerator,
            left_scale,
            max_digits=MAX_RATIONAL_DIGITS + 1,
            operation=operation,
            phase="intermediate numerator",
        )
        right_term = _bounded_integer_product(
            value.num,
            right_scale,
            max_digits=MAX_RATIONAL_DIGITS + 1,
            operation=operation,
            phase="intermediate numerator",
        )
        lifted_numerator = left_term + right_term
        if _integer_digits(lifted_numerator) > MAX_RATIONAL_DIGITS + 1:
            _reject_rational_growth(operation, "intermediate numerator")
        cancelled = gcd(lifted_numerator, shared)
        numerator = lifted_numerator // cancelled
        denominator = _bounded_integer_product(
            right_scale,
            value.den // cancelled,
            max_digits=MAX_RATIONAL_DIGITS,
            operation=operation,
            phase="intermediate denominator",
        )
        if max(_integer_digits(numerator), _integer_digits(denominator)) > (
            MAX_RATIONAL_DIGITS
        ):
            _reject_rational_growth(operation, "output")


def _admit_factor_multiply_growth(
    left: Factor,
    right: Factor,
    variables: tuple[int, ...],
    total: int,
) -> None:
    """Admit source, product, and output rational heights before table expansion."""

    left_heights = _admit_factor_source(left, "multiply")
    right_heights = _admit_factor_source(right, "multiply")
    coarse = RationalHeight(
        max(height.numerator_digits for height in left_heights)
        + max(height.numerator_digits for height in right_heights),
        max(height.denominator_digits for height in left_heights)
        + max(height.denominator_digits for height in right_heights),
    )
    if not coarse.exceeds(MAX_RATIONAL_DIGITS):
        return

    # The coarse profile is intentionally source-only and may overestimate
    # because its maxima can come from different cells or cancel in a product.
    # Inspect only this risky path, using cross-cancelled components capped at
    # twice the source height, before allocating the output table.
    for index in range(total):
        assignment = _index_to_assignment(index, variables, left.domain_sizes)
        left_index = _projected_index(
            assignment, variables, left.variables, left.domain_sizes
        )
        right_index = _projected_index(
            assignment, variables, right.variables, right.domain_sizes
        )
        if _product_height(left.table[left_index], right.table[right_index]).exceeds(
            MAX_RATIONAL_DIGITS
        ):
            _reject_rational_growth("multiply", "intermediate/output")


def _admit_factor_marginalize_growth(
    factor: Factor,
    variable: int,
    variables: tuple[int, ...],
) -> None:
    """Admit source, summation, and output heights before table expansion."""

    _admit_factor_source(factor, "marginalize")
    for index in range(scope_size(variables, factor.domain_sizes)):
        assignment = _index_to_assignment(index, variables, factor.domain_sizes)
        source_values: list[CanonicalRational] = []
        for value in range(factor.domain_sizes[variable]):
            full_assignment = dict(zip(variables, assignment, strict=True))
            full_assignment[variable] = value
            source_index = _assignment_to_index(
                tuple(full_assignment[item] for item in factor.variables),
                factor.variables,
                factor.domain_sizes,
            )
            source_values.append(factor.table[source_index])
        bound = _sum_height_bound(source_values)
        if bound.exceeds(MAX_RATIONAL_DIGITS):
            # The fallback preserves accepted boundary sums when the common
            # denominator profile is conservative, while refusing before any
            # over-height common-denominator integer is formed.
            _admit_bounded_sum(source_values, "marginalize")


def factor_multiply(left: Factor, right: Factor) -> Factor:
    """Multiply two exact factors over their canonical union scope."""

    _require_compatible_domains((left, right), left.domain_sizes)
    variables = tuple(sorted(set(left.variables) | set(right.variables)))
    total = scope_size(variables, left.domain_sizes)
    _admit_factor_multiply_growth(left, right, variables, total)
    table: list[CanonicalRational] = []
    for index in range(total):
        assignment = _index_to_assignment(index, variables, left.domain_sizes)
        left_index = _projected_index(
            assignment, variables, left.variables, left.domain_sizes
        )
        right_index = _projected_index(
            assignment, variables, right.variables, right.domain_sizes
        )
        value = (
            left.table[left_index].as_fraction()
            * right.table[right_index].as_fraction()
        )
        table.append(CanonicalRational.from_fraction(value))
    return _factor_from_kernel_table(variables, left.domain_sizes, table, "multiply")


def factor_marginalize(factor: Factor, variable: int) -> Factor:
    """Sum one variable out of an exact factor, possibly yielding a scalar."""

    if variable not in factor.variables:
        _reject("factor_variable_missing", "variable is not in factor", "variable")
    variables = tuple(item for item in factor.variables if item != variable)
    _admit_factor_marginalize_growth(factor, variable, variables)
    table: list[CanonicalRational] = []
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
            total += factor.table[source_index].as_fraction()
        table.append(CanonicalRational.from_fraction(total))
    return _factor_from_kernel_table(
        variables, factor.domain_sizes, table, "marginalize"
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
    if set(result.variables) != set(query_variables):
        raise RuntimeError("variable elimination did not produce the bound query scope")
    return _reindex_factor(result, query_variables)


def _reindex_factor(factor: Factor, target: tuple[int, ...]) -> Factor:
    """Transport a factor into an exact target variable order.

    Factor scopes are ordered; the table is lexicographic in that order.
    A singleton product preserves its source order, so the final scope can
    carry the right axes in a different valid order than the sorted query.
    Reindexing permutes the table exactly instead of relabeling it.
    """

    if factor.variables == target:
        return factor
    assignment_for = _index_to_assignment
    table: list[CanonicalRational] = []
    for index in range(scope_size(target, factor.domain_sizes)):
        assignment = assignment_for(index, target, factor.domain_sizes)
        value_for = dict(zip(target, assignment, strict=True))
        source_assignment = tuple(value_for[variable] for variable in factor.variables)
        source_index = _assignment_to_index(
            source_assignment, factor.variables, factor.domain_sizes
        )
        table.append(factor.table[source_index])
    return _factor_from_kernel_table(target, factor.domain_sizes, table, "reindex")


def _factor_from_kernel_table(
    variables: tuple[int, ...],
    domain_sizes: tuple[int, ...],
    table: Sequence[CanonicalRational],
    operation: str,
) -> Factor:
    """Bind one already-computed table after checking its exact result height.

    Factor construction validates caller-owned values, but a kernel result is
    not caller input.  Check the result height at this boundary so an exact
    product or marginal that exceeds the owner envelope becomes a typed
    admission failure instead of leaking a Pydantic ``ValidationError``.
    The table is produced once by the kernel and then trusted by the result
    model; no validator recomputes the mathematical operation.
    """

    for value in table:
        try:
            require_bounded_rational(
                value,
                max_digits=MAX_RATIONAL_DIGITS,
                label="factor result",
            )
        except ValueError as error:
            raise OperationResourceAdmissionError(
                location=("factor", "table"),
                code=f"graphical_model.factor_{operation}_rational_bound",
                message="exact factor result exceeds the rational digit bound",
            ) from error
    return Factor(variables=variables, domain_sizes=domain_sizes, table=tuple(table))


def d_separation(
    variable_count: int,
    edges: tuple[tuple[int, int], ...],
    set_a: tuple[int, ...],
    set_b: tuple[int, ...],
    set_c: tuple[int, ...],
) -> bool:
    """Decide d-separation by ancestral restriction and moralization."""

    try:
        validate_d_separation_input(variable_count, edges, set_a, set_b, set_c)
    except ValueError as error:
        raise OperationDomainValidationError(
            location=("edges", "set_a", "set_b", "set_c"),
            code="graphical_model.d_separation_invalid",
            message=str(error),
        ) from error
    parents = _parents(variable_count, edges)
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


def verify_d_separation(claim: DSeparationResult) -> bool:
    """Verify a serialized d-separation conclusion against its source query."""

    query = claim.query
    try:
        expected = d_separation(
            query.dag.variable_count,
            query.dag.edges,
            query.set_a,
            query.set_b,
            query.set_c,
        )
    except (OperationDomainValidationError, TypeError, ValueError):
        return False
    return claim.d_separated == expected


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
        _reject(
            "domain_size_count",
            f"domain_sizes must describe between 1 and {MAX_MODEL_VARS} variables",
            "domain_sizes",
        )
    if any(factor.domain_sizes != domain_sizes for factor in factors):
        _reject(
            "factor_domains_mismatch",
            "all factors must share the exact model domain_sizes",
            "factors",
        )


def _require_elimination_contract(
    factors: Sequence[Factor],
    domain_sizes: tuple[int, ...],
    elimination_order: tuple[int, ...],
    query_variables: tuple[int, ...],
) -> None:
    if not 1 <= len(factors) <= MAX_FACTOR_COUNT:
        _reject(
            "factor_count",
            f"factor family must contain between 1 and {MAX_FACTOR_COUNT} factors",
            "factors",
        )
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


def _parents(
    variable_count: int, edges: tuple[tuple[int, int], ...]
) -> dict[int, set[int]]:
    parents: dict[int, set[int]] = {node: set() for node in range(variable_count)}
    for parent, child in edges:
        parents[child].add(parent)
    return parents


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
    "verify_d_separation",
]
