"""Bayesian-network/CPT slice (#3714)."""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.probability.graphical_models import (
    bayes_net_joint,
    construct_bayes_net,
)
from jacobian.math.probability.graphical_models._models import (
    BayesNetConstructRequest,
    BayesNetJointRequest,
)
from jacobian.math.probability.graphical_models._tools import (
    _bayes_net_construct,
    _bayes_net_joint,
)
from jacobian.math.probability.graphical_models.values import (
    ConditionalProbabilityTable,
)


def _q(num: int, den: int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(num, den))


def _chain_tables() -> tuple[ConditionalProbabilityTable, ConditionalProbabilityTable]:
    root = ConditionalProbabilityTable(
        variable=0, parents=(), domain_sizes=(2, 2),
        table=(_q(3, 5), _q(2, 5)),
    )
    child = ConditionalProbabilityTable(
        variable=1, parents=(0,), domain_sizes=(2, 2),
        table=(_q(1, 4), _q(1, 2), _q(3, 4), _q(1, 2)),
    )
    return root, child


def test_joint_known_answer() -> None:
    root, child = _chain_tables()
    network = construct_bayes_net(2, ((0, 1),), (2, 2), (root, child))
    joint = bayes_net_joint(network)
    assert tuple(v.as_fraction() for v in joint.table) == (
        Fraction(3, 20), Fraction(9, 20), Fraction(1, 5), Fraction(1, 5),
    )
    assert sum((v.as_fraction() for v in joint.table), Fraction(0)) == 1
    result = _bayes_net_joint(BayesNetJointRequest(network=network))
    assert result.joint == joint


def test_single_root_network() -> None:
    table = ConditionalProbabilityTable(
        variable=0, parents=(), domain_sizes=(2,),
        table=(_q(1, 3), _q(2, 3)),
    )
    network = construct_bayes_net(1, (), (2,), (table,))
    joint = bayes_net_joint(network)
    assert tuple(v.as_fraction() for v in joint.table) == (Fraction(1, 3), Fraction(2, 3))


def test_rejects_non_normalized_row() -> None:
    bad = ConditionalProbabilityTable(
        variable=0, parents=(), domain_sizes=(2,),
        table=(_q(1, 2), _q(1, 3)),
    )
    with pytest.raises(OperationDomainValidationError, match="sum exactly"):
        construct_bayes_net(1, (), (2,), (bad,))


def test_rejects_parent_mismatch_and_cycle() -> None:
    root, _ = _chain_tables()
    wrong = ConditionalProbabilityTable(
        variable=1, parents=(), domain_sizes=(2, 2),
        table=(_q(1, 2), _q(1, 2)),
    )
    with pytest.raises(OperationDomainValidationError, match="parent"):
        construct_bayes_net(2, ((0, 1),), (2, 2), (root, wrong))
    with pytest.raises(OperationDomainValidationError, match="acyclic"):
        construct_bayes_net(2, ((0, 1), (1, 0)), (2, 2), _chain_tables())


def test_construct_tool_native_request() -> None:
    root, child = _chain_tables()
    request = BayesNetConstructRequest(
        variable_count=2, edges=((0, 1),), domain_sizes=(2, 2), tables=(root, child)
    )
    result = _bayes_net_construct(request)
    assert result.network.variable_count == 2
