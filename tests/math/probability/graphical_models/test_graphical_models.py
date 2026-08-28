"""Known-answer and adversarial tests for finite graphical models."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.probability.graphical_models import (
    Factor,
    d_separation,
    factor_marginalize,
    factor_multiply,
    variable_elimination,
)
from jacobian.math.probability.graphical_models._models import (
    DSeparationRequest,
    FactorMarginalizeRequest,
    FactorMultiplyRequest,
)
from jacobian.math.probability.graphical_models._tools import (
    _d_separation,
    _factor_marginalize,
    _factor_multiply,
)


def _factor(
    variables: tuple[int, ...],
    table: tuple[str, ...],
    domain_sizes: tuple[int, ...] = (2, 2, 2),
) -> Factor:
    return Factor(
        variables=variables,
        domain_sizes=domain_sizes,
        table=_table(*table),
    )


def _table(*values: str) -> tuple[CanonicalRational, ...]:
    return tuple(CanonicalRational.from_fraction(Fraction(value)) for value in values)


def _strings(values: tuple[CanonicalRational, ...]) -> tuple[str, ...]:
    return tuple(str(value.as_fraction()) for value in values)


class TestFactorValuesAndOperations:
    def test_multiply_disjoint_factors(self) -> None:
        left = _factor((0,), ("1/2", "1/2"))
        right = _factor((1,), ("1/3", "2/3"))

        result = factor_multiply(left, right)

        assert result.variables == (0, 1)
        assert _strings(result.table) == ("1/6", "1/3", "1/6", "1/3")

    def test_multiply_projects_noncanonical_input_scope_orders(self) -> None:
        left = _factor((1, 0), ("1", "2", "3", "4"))
        right = _factor((2, 1), ("5", "6", "7", "8"))

        result = factor_multiply(left, right)

        assert result.variables == (0, 1, 2)
        assert _strings(result.table) == (
            "5",
            "7",
            "18",
            "24",
            "10",
            "14",
            "24",
            "32",
        )

    def test_marginalize_one_variable(self) -> None:
        result = factor_marginalize(_factor((0, 1), ("1/4", "1/4", "1/4", "1/4")), 0)

        assert result.variables == (1,)
        assert _strings(result.table) == ("1/2", "1/2")

    def test_marginalize_last_variable_returns_exact_scalar(self) -> None:
        result = factor_marginalize(_factor((0,), ("1/3", "2/3")), 0)

        assert result.variables == ()
        assert _strings(result.table) == ("1",)

    def test_marginalize_respects_an_ordered_nonascending_scope(self) -> None:
        factor = _factor(
            (1, 0),
            ("1", "2", "10", "20", "100", "200"),
            domain_sizes=(2, 3),
        )

        result = factor_marginalize(factor, 1)

        assert result.variables == (0,)
        assert _strings(result.table) == ("111", "222")

    @pytest.mark.parametrize("value", ["01", "+1", "2/4", "1.0"])
    def test_factor_entries_must_be_canonical_rationals(self, value: str) -> None:
        with pytest.raises(ValidationError) as error:
            Factor.model_validate(
                {"variables": (), "domain_sizes": (2,), "table": (value,)}
            )
        assert error.value.errors()[0]["type"] == "model_type"

    def test_duplicate_factor_variables_are_rejected(self) -> None:
        with pytest.raises(ValidationError) as error:
            _factor((0, 0), ("1", "1", "1", "1"))
        assert (
            error.value.errors()[0]["type"]
            == "graphical_model.factor_variables_not_unique"
        )

    def test_zero_domain_size_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Factor(variables=(0,), domain_sizes=(0,), table=_table("1"))

    def test_wrong_table_size_is_rejected(self) -> None:
        with pytest.raises(ValidationError) as error:
            _factor((0,), ("1",))
        assert error.value.errors()[0]["type"] == "graphical_model.factor_table_size"

    def test_native_multiply_rejects_different_model_domains(self) -> None:
        left = _factor((0,), ("1", "1"))
        right = _factor((0,), ("1", "1", "1"), domain_sizes=(3,))

        with pytest.raises(ValueError, match="exact model"):
            factor_multiply(left, right)


class TestBoundResultContracts:
    def test_factor_multiply_contract_version_tracks_wire_shape(self) -> None:
        from jacobian.math.probability.graphical_models._tools import TOOLS

        next(
            tool
            for tool in TOOLS
            if tool.operation_id == "graphical_model.factor.multiply"
        )

    def test_multiply_adapter_binds_operands(self) -> None:
        request = FactorMultiplyRequest(
            left=_factor((0,), ("1", "2")),
            right=_factor((0,), ("3", "4")),
        )

        result = _factor_multiply(request)

        assert result.left == request.left
        assert result.right == request.right
        assert _strings(result.factor.table) == ("3", "8")

    def test_marginal_adapter_binds_source_and_variable(self) -> None:
        source = _factor((0,), ("1", "2"))
        result = _factor_marginalize(
            FactorMarginalizeRequest(factor=source, variable=0)
        )

        assert result.source_factor == source
        assert result.variable == 0
        assert _strings(result.factor.table) == ("3",)


class TestVariableElimination:
    def test_exact_chain_marginal(self) -> None:
        factor_x = _factor((0,), ("1/4", "3/4"))
        factor_y_given_x = _factor((0, 1), ("1/2", "1/2", "1/3", "2/3"))
        result = variable_elimination(
            (factor_x, factor_y_given_x),
            (2, 2, 2),
            elimination_order=(0,),
            query_variables=(1,),
        )

        assert result.variables == (1,)
        assert _strings(result.table) == ("3/8", "5/8")

    def test_eliminating_all_variables_returns_partition_scalar(self) -> None:
        result = variable_elimination(
            (_factor((0,), ("2", "3")),),
            (2, 2, 2),
            elimination_order=(0,),
            query_variables=(),
        )

        assert result.variables == ()
        assert _strings(result.table) == ("5",)

    def test_incomplete_elimination_order_is_rejected_before_computation(self) -> None:
        with pytest.raises(ValueError, match="every non-query"):
            variable_elimination(
                (_factor((0, 1), ("1", "1", "1", "1")),),
                (2, 2, 2),
                elimination_order=(),
                query_variables=(1,),
            )

    def test_query_must_occur_and_be_canonical(self) -> None:
        with pytest.raises(ValueError, match="occur"):
            variable_elimination(
                (_factor((0,), ("1", "1")),),
                (2, 2, 2),
                elimination_order=(0,),
                query_variables=(1,),
            )

    def test_oversized_intermediate_scope_is_rejected_before_computation(self) -> None:
        domain_sizes = (2,) * 16
        left = Factor(
            variables=tuple(range(8)),
            domain_sizes=domain_sizes,
            table=_table(*(("1",) * 256)),
        )
        right = Factor(
            variables=tuple(range(8, 16)),
            domain_sizes=domain_sizes,
            table=_table(*(("1",) * 256)),
        )

        with pytest.raises(ValueError, match="size bound"):
            variable_elimination(
                (left, right),
                domain_sizes,
                elimination_order=(),
                query_variables=tuple(range(16)),
            )

    def test_variable_elimination_remains_native_only(self) -> None:
        from jacobian.math.probability.graphical_models._tools import TOOLS

        assert "graphical_model.variable_elimination.compute" not in {
            tool.operation_id for tool in TOOLS
        }


class TestDSeparation:
    @pytest.mark.parametrize(
        ("edges", "set_c", "expected"),
        [
            (((0, 1), (1, 2)), (), False),
            (((0, 1), (1, 2)), (1,), True),
            (((0, 1), (0, 2)), (), False),
            (((0, 2), (1, 2)), (), True),
            (((0, 2), (1, 2)), (2,), False),
        ],
    )
    def test_chain_fork_and_collider_cases(
        self,
        edges: tuple[tuple[int, int], ...],
        set_c: tuple[int, ...],
        expected: bool,
    ) -> None:
        assert (
            d_separation(3, edges, (0,), (1 if edges[0][1] == 2 else 2,), set_c)
            is expected
        )

    def test_conditioned_descendant_activates_collider(self) -> None:
        edges = ((0, 2), (1, 2), (2, 3))

        assert d_separation(4, edges, (0,), (1,), (3,)) is False

    def test_adapter_binds_graph_and_decision(self) -> None:
        request = DSeparationRequest(
            variable_count=3,
            edges=((0, 1), (1, 2)),
            set_a=(0,),
            set_b=(2,),
            set_c=(1,),
        )

        result = _d_separation(request)

        assert result.d_separated is True
        assert result.edges == request.edges

    @pytest.mark.parametrize(
        "edges",
        [
            ((0, 1), (1, 0)),
            ((0, 0),),
            ((0, 3),),
        ],
    )
    def test_invalid_dag_is_rejected(self, edges: tuple[tuple[int, int], ...]) -> None:
        request = DSeparationRequest(
            variable_count=3,
            edges=edges,
            set_a=(0,),
            set_b=(1,),
            set_c=(),
        )
        with pytest.raises(OperationDomainValidationError) as error:
            _d_separation(request)
        assert error.value.errors()[0]["type"] == "graphical_model.d_separation_invalid"

    def test_node_sets_must_be_pairwise_disjoint(self) -> None:
        request = DSeparationRequest(
            variable_count=2,
            set_a=(0,),
            set_b=(1,),
            set_c=(1,),
        )
        with pytest.raises(OperationDomainValidationError) as error:
            _d_separation(request)
        assert error.value.errors()[0]["type"] == "graphical_model.d_separation_invalid"
