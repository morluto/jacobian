"""Tests for graphical model operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.graphical_models._models import (
    FactorMarginalizeRequest,
)
from jacobian.math.graphical_models.operations import (
    d_separation,
    factor_marginalize,
    factor_multiply,
    variable_elimination,
)
from jacobian.math.graphical_models.values import Factor


class TestFactorMultiply:
    def test_simple_multiply(self):
        left = Factor(
            variables=(0,),
            domain_sizes=(2,),
            table=("1/2", "1/2"),
        )
        right = Factor(
            variables=(1,),
            domain_sizes=(2, 2),
            table=("1/3", "2/3"),
        )
        result = factor_multiply(left, right)
        assert result.variables == (0, 1)
        assert result.table == ("1/6", "1/3", "1/6", "1/3")

    def test_shared_variable(self):
        left = Factor(
            variables=(0, 1),
            domain_sizes=(2, 2, 2),
            table=("1/2", "1/2", "1/2", "1/2"),
        )
        right = Factor(
            variables=(1, 2),
            domain_sizes=(2, 2, 2),
            table=("1/4", "3/4", "1/4", "3/4"),
        )
        result = factor_multiply(left, right)
        assert 0 in result.variables
        assert 1 in result.variables
        assert 2 in result.variables
        assert len(result.table) == 8


class TestFactorMarginalize:
    def test_marginalize_single_var(self):
        factor = Factor(
            variables=(0,),
            domain_sizes=(2,),
            table=("1/3", "2/3"),
        )
        result = factor_marginalize(factor, 0)
        # result is a 1-element factor with the sum
        assert result.table[0] in ("1", "3/3")

    def test_marginalize_one_of_two(self):
        factor = Factor(
            variables=(0, 1),
            domain_sizes=(2, 2),
            table=("1/4", "1/4", "1/4", "1/4"),
        )
        result = factor_marginalize(factor, 0)
        assert result.variables == (1,)
        assert result.table == ("1/2", "1/2")


class TestVariableElimination:
    def test_simple_elimination(self):
        f1 = Factor(
            variables=(0,),
            domain_sizes=(2, 2),
            table=("1/2", "1/2"),
        )
        f2 = Factor(
            variables=(1,),
            domain_sizes=(2, 2),
            table=("1/3", "2/3"),
        )
        result = variable_elimination(
            [f1, f2],
            (2, 2),
            elimination_order=(0,),
            query_variables=(1,),
        )
        assert 1 in result.variables


    def test_marginalize_out_one(self):
        f1 = Factor(
            variables=(0, 1),
            domain_sizes=(2, 2),
            table=("1/4", "1/4", "1/4", "1/4"),
        )
        result = factor_marginalize(f1, 1)
        assert result.variables == (0,)
        assert result.table == ("1/2", "1/2")


class TestDSeparation:
    def test_connected_not_separated(self):
        # Chain: 0 -> 1 -> 2
        result = d_separation(
            variable_count=3,
            edges=((0, 1), (1, 2)),
            set_a=(0,),
            set_b=(2,),
            set_c=(),
        )
        assert result is False

    def test_separated_by_conditioning(self):
        # Chain: 0 -> 1 -> 2
        result = d_separation(
            variable_count=3,
            edges=((0, 1), (1, 2)),
            set_a=(0,),
            set_b=(2,),
            set_c=(1,),
        )
        assert result is True

    def test_independent(self):
        # Two disconnected nodes
        result = d_separation(
            variable_count=3,
            edges=(),
            set_a=(0,),
            set_b=(1,),
            set_c=(),
        )
        assert result is True


class TestValidation:
    def test_empty_factor_rejected(self):
        with pytest.raises(ValidationError):
            Factor(
                variables=(),
                domain_sizes=(2,),
                table=("1",),
            )

    def test_wrong_table_size_rejected(self):
        with pytest.raises(ValidationError):
            Factor(
                variables=(0,),
                domain_sizes=(2,),
                table=("1/2",),
            )

    def test_marginalize_missing_variable_rejected(self):
        with pytest.raises(ValidationError):
            FactorMarginalizeRequest(
                factor=Factor(
                    variables=(0,),
                    domain_sizes=(2,),
                    table=("1/2", "1/2"),
                ),
                variable=1,
            )
