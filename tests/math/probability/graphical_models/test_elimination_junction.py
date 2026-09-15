"""Variable-elimination traces and junction-tree calibration (#3715)."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.probability.graphical_models import (
    bayes_net_joint,
    construct_bayes_net,
    factor_marginalize,
    junction_tree_calibrate,
    variable_elimination,
    variable_elimination_trace,
)
from jacobian.math.probability.graphical_models.values import (
    ConditionalProbabilityTable,
    Factor,
)


def _q(num: int, den: int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(num, den))


def _factors() -> tuple[tuple[Factor, ...], tuple[int, ...]]:
    prior = Factor(
        variables=(0,), domain_sizes=(2, 2), table=(_q(3, 5), _q(2, 5))
    )
    likelihood = Factor(
        variables=(0, 1), domain_sizes=(2, 2),
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
        Fraction(9, 20), Fraction(1, 2),
    )
    # Final agrees with the plain elimination kernel.
    assert final == variable_elimination(factors, domain, (0,), (1,))


def test_junction_calibration_is_consistent() -> None:
    root = ConditionalProbabilityTable(
        variable=0, parents=(), domain_sizes=(2, 2), table=(_q(3, 5), _q(2, 5))
    )
    child = ConditionalProbabilityTable(
        variable=1, parents=(0,), domain_sizes=(2, 2),
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
        Fraction(3, 20), Fraction(9, 20), Fraction(1, 5), Fraction(1, 5),
    )
