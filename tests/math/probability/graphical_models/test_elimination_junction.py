"""Variable-elimination traces and junction-tree calibration (#3715)."""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.probability.graphical_models import (
    bayes_net_joint,
    construct_bayes_net,
    factor_marginalize,
    junction_tree_calibrate,
    variable_elimination,
    variable_elimination_trace,
)
from jacobian.math.probability.graphical_models.values import (
    BayesianNetwork,
    ConditionalProbabilityTable,
    Factor,
)


def _q(num: int, den: int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(num, den))


def _factors() -> tuple[tuple[Factor, ...], tuple[int, ...]]:
    prior = Factor(variables=(0,), domain_sizes=(2, 2), table=(_q(3, 5), _q(2, 5)))
    likelihood = Factor(
        variables=(0, 1),
        domain_sizes=(2, 2),
        table=(_q(1, 4), _q(1, 2), _q(3, 4), _q(1, 2)),
    )
    return (prior, likelihood), (2, 2)


def test_trace_returns_scopes_factors_and_fill() -> None:
    factors, domain = _factors()
    steps, final = variable_elimination_trace(factors, domain, (0,), (1,))
    assert len(steps) == 1
    step = steps[0]
    assert step.eliminated == 0
    assert step.product_scope == (0, 1)
    assert step.output_scope == (1,)
    assert tuple(v.as_fraction() for v in final.table) == (
        Fraction(9, 20),
        Fraction(1, 2),
    )
    # Final agrees with the plain elimination kernel.
    assert final == variable_elimination(factors, domain, (0,), (1,))


def test_junction_calibration_eliminating_every_variable() -> None:
    """A complete elimination order leaves the empty scalar partition."""

    root = ConditionalProbabilityTable(
        variable=0, parents=(), domain_sizes=(2, 2), table=(_q(3, 5), _q(2, 5))
    )
    child = ConditionalProbabilityTable(
        variable=1,
        parents=(0,),
        domain_sizes=(2, 2),
        table=(_q(1, 4), _q(1, 2), _q(3, 4), _q(1, 2)),
    )
    network = construct_bayes_net(2, ((0, 1),), (2, 2), (root, child))
    cliques, separators, _clique_marginals, _separator_marginals, partition = (
        junction_tree_calibrate(network, (0, 1))
    )
    assert cliques == ((0, 1), (1,), ())
    assert separators == ((1,),)
    assert partition.table[0].as_fraction() == 1


def test_junction_calibration_is_consistent() -> None:
    root = ConditionalProbabilityTable(
        variable=0, parents=(), domain_sizes=(2, 2), table=(_q(3, 5), _q(2, 5))
    )
    child = ConditionalProbabilityTable(
        variable=1,
        parents=(0,),
        domain_sizes=(2, 2),
        table=(_q(1, 4), _q(1, 2), _q(3, 4), _q(1, 2)),
    )
    network = construct_bayes_net(2, ((0, 1),), (2, 2), (root, child))
    cliques, separators, clique_marginals, separator_marginals, partition = (
        junction_tree_calibrate(network, (0,))
    )
    assert cliques[0] == (0, 1)
    assert partition.table[0].as_fraction() == 1
    # Adjacent cliques agree on their separator (denominator-cleared identity).
    assert separators == ((1,),)
    sep = separator_marginals[0]
    for marginal in clique_marginals:
        projected = marginal
        for variable in [v for v in projected.variables if v not in sep.variables]:
            projected = factor_marginalize(projected, variable)
        assert tuple(v.as_fraction() for v in projected.table) == tuple(
            v.as_fraction() for v in sep.table
        )
    # Clique marginals marginalize the exact joint.
    joint = bayes_net_joint(network)
    assert tuple(v.as_fraction() for v in joint.table) == (
        Fraction(3, 20),
        Fraction(9, 20),
        Fraction(1, 5),
        Fraction(1, 5),
    )


def _fraction_factor(
    variables: tuple[int, ...],
    table: tuple[Fraction, ...],
    domain_sizes: tuple[int, ...],
) -> Factor:
    return Factor(
        variables=variables,
        domain_sizes=domain_sizes,
        table=tuple(CanonicalRational.from_fraction(value) for value in table),
    )


def _brute_force_marginal_table(
    factors: tuple[Factor, ...],
    domain_sizes: tuple[int, ...],
    query: tuple[int, ...],
) -> tuple[Fraction, ...]:
    """Enumerate the exact joint directly, independently of elimination kernels."""
    model = tuple(sorted({v for factor in factors for v in factor.variables}))
    totals: dict[tuple[int, ...], Fraction] = {}
    for index in range(_joint_size(model, domain_sizes)):
        assignment = _assignment(index, model, domain_sizes)
        values = dict(zip(model, assignment, strict=True))
        product = Fraction(1)
        for factor in factors:
            position = _flat_index(
                tuple(values[v] for v in factor.variables),
                factor.variables,
                domain_sizes,
            )
            product *= factor.table[position].as_fraction()
        key = tuple(values[v] for v in query)
        totals[key] = totals.get(key, Fraction(0)) + product
    return tuple(
        totals[_assignment(index, query, domain_sizes)]
        for index in range(_joint_size(query, domain_sizes))
    )


def _joint_size(variables: tuple[int, ...], domain_sizes: tuple[int, ...]) -> int:
    size = 1
    for variable in variables:
        size *= domain_sizes[variable]
    return size


def _assignment(
    index: int, variables: tuple[int, ...], domain_sizes: tuple[int, ...]
) -> tuple[int, ...]:
    values: list[int] = []
    for variable in reversed(variables):
        values.append(index % domain_sizes[variable])
        index //= domain_sizes[variable]
    return tuple(reversed(values))


def _flat_index(
    assignment: tuple[int, ...],
    variables: tuple[int, ...],
    domain_sizes: tuple[int, ...],
) -> int:
    index = 0
    for variable, value in zip(variables, assignment, strict=True):
        index = index * domain_sizes[variable] + value
    return index


def _chain_factors() -> tuple[tuple[Factor, ...], tuple[int, ...]]:
    prior = _fraction_factor((0,), (Fraction(1, 4), Fraction(3, 4)), (2, 2, 2))
    middle = _fraction_factor(
        (0, 1),
        (Fraction(1, 2), Fraction(1, 2), Fraction(1, 3), Fraction(2, 3)),
        (2, 2, 2),
    )
    last = _fraction_factor(
        (1, 2),
        (Fraction(1, 5), Fraction(4, 5), Fraction(3, 5), Fraction(2, 5)),
        (2, 2, 2),
    )
    return (prior, middle, last), (2, 2, 2)


def _diamond_network() -> BayesianNetwork:
    root = ConditionalProbabilityTable(
        variable=0, parents=(), domain_sizes=(2, 2, 2), table=(_q(1, 3), _q(2, 3))
    )
    left = ConditionalProbabilityTable(
        variable=1,
        parents=(0,),
        domain_sizes=(2, 2, 2),
        table=(_q(1, 4), _q(1, 2), _q(3, 4), _q(1, 2)),
    )
    right = ConditionalProbabilityTable(
        variable=2,
        parents=(0,),
        domain_sizes=(2, 2, 2),
        table=(_q(2, 5), _q(1, 5), _q(3, 5), _q(4, 5)),
    )
    return construct_bayes_net(3, ((0, 1), (0, 2)), (2, 2, 2), (root, left, right))


def _project_joint(joint: Factor, scope: tuple[int, ...]) -> tuple[Fraction, ...]:
    """Project an exact joint table onto a scope without elimination kernels."""
    assert set(scope) <= set(joint.variables)
    totals: dict[tuple[int, ...], Fraction] = {}
    for index in range(_joint_size(joint.variables, joint.domain_sizes)):
        assignment = _assignment(index, joint.variables, joint.domain_sizes)
        values = dict(zip(joint.variables, assignment, strict=True))
        key = tuple(values[v] for v in scope)
        position = _flat_index(assignment, joint.variables, joint.domain_sizes)
        totals[key] = totals.get(key, Fraction(0)) + joint.table[position].as_fraction()
    return tuple(
        totals[_assignment(index, scope, joint.domain_sizes)]
        for index in range(_joint_size(scope, joint.domain_sizes))
    )


def test_elimination_orders_agree_with_brute_force_chain() -> None:
    factors, domain_sizes = _chain_factors()
    expected = _brute_force_marginal_table(factors, domain_sizes, (2,))
    for order in ((0, 1), (1, 0)):
        result = variable_elimination(factors, domain_sizes, order, (2,))
        assert result.variables == (2,)
        assert tuple(v.as_fraction() for v in result.table) == expected
    steps, final = variable_elimination_trace(factors, domain_sizes, (0, 1), (2,))
    assert tuple(v.as_fraction() for v in final.table) == expected
    assert [step.eliminated for step in steps] == [0, 1]
    assert steps[0].product_scope == (0, 1)
    assert steps[0].output_scope == (1,)
    assert steps[1].product_scope == (1, 2)
    assert steps[1].output_scope == (2,)


def test_diamond_marginal_matches_brute_force_and_joint_projection() -> None:
    network = _diamond_network()
    factors = tuple(table.as_factor() for table in network.tables)
    expected = _brute_force_marginal_table(factors, network.domain_sizes, (1, 2))
    for order in ((0,),):
        result = variable_elimination(factors, network.domain_sizes, order, (1, 2))
        assert tuple(v.as_fraction() for v in result.table) == expected
    joint = bayes_net_joint(network)
    assert _project_joint(joint, (1, 2)) == expected
    assert sum((v.as_fraction() for v in joint.table), Fraction(0)) == 1


def test_junction_marginals_match_joint_projections_and_partition() -> None:
    network = _diamond_network()
    cliques, separators, clique_marginals, separator_marginals, partition = (
        junction_tree_calibrate(network, (0,))
    )
    joint = bayes_net_joint(network)
    assert partition.variables == ()
    assert partition.table[0].as_fraction() == sum(
        (v.as_fraction() for v in joint.table), Fraction(0)
    )
    assert partition.table[0].as_fraction() == 1
    for clique, marginal in zip(cliques, clique_marginals, strict=True):
        assert marginal.variables == clique
        assert tuple(v.as_fraction() for v in marginal.table) == _project_joint(
            joint, clique
        )
    for separator, marginal in zip(separators, separator_marginals, strict=True):
        assert tuple(v.as_fraction() for v in marginal.table) == _project_joint(
            joint, separator
        )
    for clique, marginal in zip(cliques, clique_marginals, strict=True):
        for separator, separator_marginal in zip(
            separators, separator_marginals, strict=True
        ):
            if set(separator) <= set(clique):
                assert _project_joint(marginal, separator) == tuple(
                    v.as_fraction() for v in separator_marginal.table
                )


def test_full_elimination_matches_partition_scalar() -> None:
    network = _diamond_network()
    factors = tuple(table.as_factor() for table in network.tables)
    eliminated = variable_elimination(factors, network.domain_sizes, (0, 1, 2), ())
    _, _, _, _, partition = junction_tree_calibrate(network, (0, 1, 2))
    assert eliminated.variables == ()
    assert eliminated.table[0].as_fraction() == 1
    assert partition == eliminated


def test_scalar_factor_eliminates_to_itself() -> None:
    scalar = _fraction_factor((), (Fraction(5),), (2,))
    result = variable_elimination((scalar,), (2,), (), ())
    assert result.variables == ()
    assert result.table[0].as_fraction() == 5


def test_duplicate_table_network_has_nonnormalized_joint() -> None:
    root = ConditionalProbabilityTable(
        variable=0, parents=(), domain_sizes=(2, 2), table=(_q(1, 3), _q(2, 3))
    )
    forged = BayesianNetwork.model_construct(
        variable_count=2,
        edges=((0, 1),),
        domain_sizes=(2, 2),
        tables=(root, root),
    )
    with pytest.raises(OperationDomainValidationError, match="joint must sum"):
        bayes_net_joint(forged)
