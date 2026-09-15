"""Exact bounded native kernels for finite graphical models."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations, pairwise
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
from jacobian.math.probability.graphical_models._models import (
    DSeparationResult,
    EliminationStep,
)
from jacobian.math.probability.graphical_models._validation import (
    validate_d_separation_input,
)
from jacobian.math.probability.graphical_models.values import (
    MAX_FACTOR_COUNT,
    MAX_MODEL_VARS,
    MAX_RATIONAL_DIGITS,
    BayesianNetwork,
    ConditionalProbabilityTable,
    Factor,
    scope_size,
)


def _reject(code: str, message: str, *location: str) -> None:
    raise OperationDomainValidationError(
        location=location,
        code=f"graphical_model.{code}",
        message=message,
    )


def _growth_location(operation: str) -> tuple[str, ...]:
    if operation == "multiply":
        return ("left", "right")
    return ("factor", "table")


def _reject_rational_growth(operation: str, phase: str) -> None:
    raise OperationResourceAdmissionError(
        location=_growth_location(operation),
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
        # Form the lifted numerator without a premature digit cap so a shared
        # denominator gcd can cancel before the reduced height is enforced.
        left_term = numerator * left_scale
        right_term = value.num * right_scale
        lifted_numerator = left_term + right_term
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
    """Bind one already-admitted table into a factor result.

    Every caller admits the exact result height before expansion:
    ``factor_multiply`` and ``factor_marginalize`` bound the product or sum with
    ``_product_height``/``_sum_height_bound`` (sound upper bounds), and
    ``_reindex_factor`` only permutes values from an already-admitted factor.
    The kernel result is trusted here; re-scanning the table would replay the
    admission the preflight already established.
    """

    del operation
    return Factor.model_construct(
        variables=variables,
        domain_sizes=domain_sizes,
        table=tuple(table),
    )


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


def variable_elimination_trace(
    factors: Sequence[Factor],
    domain_sizes: tuple[int, ...],
    elimination_order: tuple[int, ...],
    query_variables: tuple[int, ...],
) -> tuple[tuple[EliminationStep, ...], Factor]:
    """Return one trace step per eliminated variable plus the final factor.

    Each step retains input scopes, the product scope/factor, the marginalized
    output scope/factor, and fill edges completing the product scope beyond
    previously co-occurring pairs. No order-optimality claim is made.
    """

    try:
        _require_elimination_contract(
            factors, domain_sizes, elimination_order, query_variables
        )
    except OperationDomainValidationError:
        raise
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("factors", "elimination_order", "query_variables"),
            code="graphical_model.elimination_contract",
            message=str(exc),
        ) from exc
    co_occurring: set[tuple[int, int]] = set()
    for factor in factors:
        for left_index in range(len(factor.variables)):
            for right_index in range(left_index + 1, len(factor.variables)):
                pair = tuple(
                    sorted(
                        (
                            factor.variables[left_index],
                            factor.variables[right_index],
                        )
                    )
                )
                co_occurring.add((pair[0], pair[1]))
    working = list(factors)
    steps: list[EliminationStep] = []
    for variable in elimination_order:
        relevant = [factor for factor in working if variable in factor.variables]
        working = [factor for factor in working if variable not in factor.variables]
        input_scopes = tuple(factor.variables for factor in relevant)
        union = tuple(sorted({item for scope in input_scopes for item in scope}))
        product = _multiply_all(relevant)
        output = factor_marginalize(product, variable)
        output_scope = tuple(item for item in union if item != variable)
        fill: list[tuple[int, int]] = []
        for left_index in range(len(union)):
            for right_index in range(left_index + 1, len(union)):
                pair = (union[left_index], union[right_index])
                if pair not in co_occurring:
                    fill.append(pair)
                    co_occurring.add(pair)
        steps.append(
            EliminationStep._from_kernel(
                eliminated=variable,
                input_scopes=input_scopes,
                product_scope=union,
                product=product,
                output_scope=output_scope,
                output=output,
                fill_edges=tuple(fill),
            )
        )
        working.append(output)
    final = _reindex_factor(_multiply_all(working), query_variables)
    return tuple(steps), final


def junction_tree_calibrate(
    network: BayesianNetwork,
    elimination_order: tuple[int, ...],
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[Factor, ...],
    tuple[Factor, ...],
    Factor,
]:
    """Calibrate clique/separator marginals of a Bayes-net joint.

    Cliques are the elimination product scopes plus the final query scope;
    separators are consecutive clique intersections. Marginals marginalize the
    exact joint onto each scope, so adjacent cliques agree on separators and
    the partition (joint sum) equals one.
    """

    all_vars = tuple(sorted({table.variable for table in network.tables}))
    if len(set(elimination_order)) != len(elimination_order):
        _reject("elimination_order", "elimination order cannot repeat", "elimination_order")
    if not set(elimination_order) <= set(all_vars):
        _reject("elimination_order", "elimination order must use model variables", "elimination_order")
    query = tuple(sorted(set(all_vars) - set(elimination_order)))
    if not query:
        query = all_vars[:1]
    factors = [table.as_factor() for table in network.tables]
    steps, _ = variable_elimination_trace(
        factors, network.domain_sizes, elimination_order, query
    )
    cliques = tuple(step.product_scope for step in steps)
    if not cliques or set(cliques[-1]) != set(query):
        cliques = (*cliques, query)
    separators = tuple(
        tuple(sorted(set(left) & set(right)))
        for left, right in pairwise(cliques)
        if set(left) & set(right)
    )
    joint = bayes_net_joint(network)

    def _marginal(scope: tuple[int, ...]) -> Factor:
        result = joint
        for variable in [v for v in result.variables if v not in scope]:
            result = factor_marginalize(result, variable)
        return _reindex_factor(result, scope)

    clique_marginals = tuple(_marginal(clique) for clique in cliques)
    separator_marginals = tuple(_marginal(separator) for separator in separators)
    partition = Factor.model_construct(
        variables=(),
        domain_sizes=network.domain_sizes,
        table=(CanonicalRational.from_fraction(Fraction(1)),),
    )
    return cliques, separators, clique_marginals, separator_marginals, partition


def _multiply_all(factors: Sequence[Factor]) -> Factor:
    if not factors:
        raise ValueError("at least one factor is required")
    result = factors[0]
    for factor in factors[1:]:
        result = factor_multiply(result, factor)
    return result


def _check_cpt_rows(table: ConditionalProbabilityTable) -> None:
    """Establish exact row normalization (each parent row sums to one)."""

    variables = table.variables
    domain_sizes = table.domain_sizes
    parents = table.parents
    if not parents:
        total = sum((value.as_fraction() for value in table.table), Fraction(0))
        if total != 1:
            _reject(
                "cpt_row_not_normalized",
                "cpt rows must sum exactly to one",
                "table",
            )
        return
    parent_size = scope_size(parents, domain_sizes)
    for parent_index in range(parent_size):
        parent_assignment = _index_to_assignment(
            parent_index, parents, domain_sizes
        )
        positions = {
            variable: index for index, variable in enumerate(variables)
        }
        total = Fraction(0)
        for value in range(domain_sizes[table.variable]):
            full = [0] * len(variables)
            for variable, position in positions.items():
                if variable == table.variable:
                    full[position] = value
                else:
                    full[position] = parent_assignment[
                        parents.index(variable)
                    ]
            flat = _assignment_to_index(tuple(full), variables, domain_sizes)
            total += table.table[flat].as_fraction()
        if total != 1:
            _reject(
                "cpt_row_not_normalized",
                "cpt rows must sum exactly to one",
                "table",
            )


def _check_acyclic(variable_count: int, edges: tuple[tuple[int, int], ...]) -> None:
    children: dict[int, set[int]] = {node: set() for node in range(variable_count)}
    indegree = [0] * variable_count
    for parent, child in edges:
        if child not in children[parent]:
            children[parent].add(child)
            indegree[child] += 1
    queue = deque(node for node in range(variable_count) if indegree[node] == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if visited != variable_count:
        _reject("network_cycle", "network edges must be acyclic", "edges")


def construct_bayes_net(
    variable_count: int,
    edges: tuple[tuple[int, int], ...],
    domain_sizes: tuple[int, ...],
    tables: tuple[ConditionalProbabilityTable, ...],
) -> BayesianNetwork:
    """Bind every CPT exactly to its DAG parent set with normalized rows."""

    if type(variable_count) is not int or not 1 <= variable_count <= MAX_MODEL_VARS:
        _reject("variable_count", "variable count is outside its bound", "variable_count")
    normalized: list[tuple[int, int]] = []
    for edge in edges:
        if type(edge) is not tuple or len(edge) != 2:
            _reject("edge_shape", "edges must be integer pairs", "edges")
        parent, child = int(edge[0]), int(edge[1])
        if not 0 <= parent < variable_count or not 0 <= child < variable_count:
            _reject(
                "edge_endpoints",
                "network edges must join declared model variables",
                "edges",
            )
        normalized.append((parent, child))
    normalized_edges = tuple(normalized)
    _check_acyclic(variable_count, normalized_edges)
    network = BayesianNetwork.model_construct(
        variable_count=variable_count,
        edges=tuple(sorted(set(normalized_edges))),
        domain_sizes=domain_sizes,
        tables=tables,
    )
    # Structural binding is re-established here because model_construct skips it;
    # malformed caller values raise typed domain errors, not pydantic errors.
    try:
        network = BayesianNetwork.model_validate(network.model_dump())
    except Exception as error:
        _reject("network_binding", f"network binding failed: {error}", "tables")
    for table in network.tables:
        _admit_factor_source(table.as_factor(), "cpt")
        _check_cpt_rows(table)
    # Joint-size preflight: the induced joint must fit the factor bound.
    scope_size(tuple(range(variable_count)), domain_sizes)
    return network


def bayes_net_joint(network: BayesianNetwork) -> Factor:
    """Return the induced joint distribution, checked to sum to one."""

    for table in network.tables:
        _check_cpt_rows(table)
    factors = [table.as_factor() for table in sorted(network.tables, key=lambda t: t.variable)]
    joint = _multiply_all(factors)
    total = sum((value.as_fraction() for value in joint.table), Fraction(0))
    if total != 1:
        _reject("joint_not_normalized", "induced joint must sum to one", "tables")
    if set(joint.variables) != set(range(network.variable_count)):
        raise RuntimeError("bayes-net joint did not cover every variable")
    return _reindex_factor(joint, tuple(range(network.variable_count)))


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
    "bayes_net_joint",
    "construct_bayes_net",
    "d_separation",
    "factor_marginalize",
    "factor_multiply",
    "junction_tree_calibrate",
    "variable_elimination",
    "variable_elimination_trace",
    "verify_d_separation",
]
