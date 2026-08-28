from fractions import Fraction

import pytest

from jacobian.math.number_theory import numerical_semigroups
from jacobian.math.number_theory import numerical_semigroups as ns
from jacobian.math.number_theory.numerical_semigroups._tools import TOOLS


def test_catalog_contains_only_audited_agent_outcomes() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "number_theory.numerical_semigroup.summary.compute",
        "number_theory.numerical_semigroup.membership.compute",
        "number_theory.numerical_semigroup.factorizations.compute",
        "number_theory.numerical_semigroup.factorization_graph.compute",
        "number_theory.numerical_semigroup.betti_elements.compute",
        "number_theory.numerical_semigroup.minimal_presentation.compute",
        "number_theory.numerical_semigroup.presentation_binomials.compute",
        "number_theory.numerical_semigroup.delta_set.compute",
        "number_theory.numerical_semigroup.catenary_degree.compute",
        "number_theory.numerical_semigroup.elasticity.compute",
        "number_theory.numerical_semigroup.elasticity.global_compute",
    }


def test_exploratory_factorization_operations_remain_native() -> None:
    generators = (3, 5)
    family = ns.factorizations(generators, 15)
    assert family == ((0, 3), (5, 0))
    assert ns.factorization_lengths(generators, 15) == (3, 5)
    assert ns.factorization_distance(family[0], family[1]) == 5
    assert ns.factorization_graph(family).components == ((0,), (1,))
    assert ns.element_delta_set(generators, 15) == (2,)
    assert ns.element_elasticity(generators, 15) == Fraction(5, 3)
    assert ns.element_catenary_degree(generators, 15) == 5
    assert ns.elasticity(generators) == Fraction(5, 3)


def test_native_operations_reject_invalid_or_undefined_inputs() -> None:
    with pytest.raises(ValueError, match="minimal generating system"):
        ns.elasticity((3, 5, 6))
    with pytest.raises(ValueError, match="undefined for zero"):
        ns.element_elasticity((3, 5), 0)
    with pytest.raises(ValueError, match="same positive dimension"):
        ns.factorization_distance((1,), (1, 0))


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the numerical_semigroups public API."""
    expected = (
        "FactorizationGraph",
        "apery_set",
        "belongs",
        "elasticity",
        "element_catenary_degree",
        "element_delta_set",
        "element_elasticity",
        "factorization_count",
        "factorization_distance",
        "factorization_graph",
        "factorization_lengths",
        "factorizations",
        "minimal_generating_system",
    )
    assert tuple(numerical_semigroups.__all__) == expected
    assert len(numerical_semigroups.__all__) == len(set(numerical_semigroups.__all__))
    assert all(not name.startswith("_") for name in numerical_semigroups.__all__)
    assert all(
        hasattr(numerical_semigroups, name) for name in numerical_semigroups.__all__
    )
