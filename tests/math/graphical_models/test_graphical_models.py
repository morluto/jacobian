"""Known-answer and adversarial tests for finite graphical models."""

import pytest
from pydantic import ValidationError

from jacobian.math.graphical_models import (
    Factor,
    d_separation,
    factor_marginalize,
    factor_multiply,
    variable_elimination,
)
from jacobian.math.graphical_models._models import (
    DSeparationRequest,
    DSeparationResult,
    FactorMarginalizeRequest,
    FactorMultiplyRequest,
    FactorMultiplyResult,
    VariableEliminationRequest,
)
from jacobian.math.graphical_models._operations import (
    compute_d_separation,
    compute_factor_marginalize,
    compute_factor_multiply,
    compute_variable_elimination,
)


def _factor(
    variables: tuple[int, ...],
    table: tuple[str, ...],
    domain_sizes: tuple[int, ...] = (2, 2, 2),
) -> Factor:
    return Factor(variables=variables, domain_sizes=domain_sizes, table=table)


class TestFactorValuesAndOperations:
    def test_multiply_disjoint_factors(self) -> None:
        left = _factor((0,), ("1/2", "1/2"))
        right = _factor((1,), ("1/3", "2/3"))

        result = factor_multiply(left, right)

        assert result.variables == (0, 1)
        assert result.table == ("1/6", "1/3", "1/6", "1/3")

    def test_multiply_projects_noncanonical_input_scope_orders(self) -> None:
        left = _factor((1, 0), ("1", "2", "3", "4"))
        right = _factor((2, 1), ("5", "6", "7", "8"))

        result = factor_multiply(left, right)

        assert result.variables == (0, 1, 2)
        assert result.table == ("5", "7", "18", "24", "10", "14", "24", "32")

    def test_marginalize_one_variable(self) -> None:
        result = factor_marginalize(_factor((0, 1), ("1/4", "1/4", "1/4", "1/4")), 0)

        assert result.variables == (1,)
        assert result.table == ("1/2", "1/2")

    def test_marginalize_last_variable_returns_exact_scalar(self) -> None:
        result = factor_marginalize(_factor((0,), ("1/3", "2/3")), 0)

        assert result.variables == ()
        assert result.table == ("1",)

    @pytest.mark.parametrize("value", ["01", "+1", "2/4", "1.0"])
    def test_factor_entries_must_be_canonical_rationals(self, value: str) -> None:
        with pytest.raises(ValidationError, match="canonical"):
            _factor((), (value,))

    def test_duplicate_factor_variables_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="distinct"):
            _factor((0, 0), ("1", "1", "1", "1"))

    def test_zero_domain_size_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Factor(variables=(0,), domain_sizes=(0,), table=("1",))

    def test_wrong_table_size_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="table size"):
            _factor((0,), ("1",))

    def test_native_multiply_rejects_different_model_domains(self) -> None:
        left = _factor((0,), ("1", "1"))
        right = _factor((0,), ("1", "1", "1"), domain_sizes=(3,))

        with pytest.raises(ValueError, match="exact model"):
            factor_multiply(left, right)


class TestBoundResultContracts:
    def test_multiply_adapter_binds_operands(self) -> None:
        request = FactorMultiplyRequest(
            left=_factor((0,), ("1", "2")),
            right=_factor((0,), ("3", "4")),
        )

        result = compute_factor_multiply(request)

        assert result.left == request.left
        assert result.right == request.right
        assert result.factor.table == ("3", "8")

    def test_false_product_is_rejected(self) -> None:
        left = _factor((0,), ("1", "2"))
        right = _factor((0,), ("3", "4"))

        with pytest.raises(ValidationError, match="exact product"):
            FactorMultiplyResult(left=left, right=right, factor=left)

    def test_marginal_adapter_binds_source_and_variable(self) -> None:
        source = _factor((0,), ("1", "2"))
        result = compute_factor_marginalize(
            FactorMarginalizeRequest(factor=source, variable=0)
        )

        assert result.source_factor == source
        assert result.variable == 0
        assert result.factor.table == ("3",)


class TestVariableElimination:
    def test_exact_chain_marginal(self) -> None:
        factor_x = _factor((0,), ("1/4", "3/4"))
        factor_y_given_x = _factor((0, 1), ("1/2", "1/2", "1/3", "2/3"))
        request = VariableEliminationRequest(
            factors=(factor_x, factor_y_given_x),
            domain_sizes=(2, 2, 2),
            elimination_order=(0,),
            query_variables=(1,),
        )

        result = compute_variable_elimination(request)

        assert result.factor.variables == (1,)
        assert result.factor.table == ("3/8", "5/8")
        assert result.factors == request.factors

    def test_eliminating_all_variables_returns_partition_scalar(self) -> None:
        result = variable_elimination(
            (_factor((0,), ("2", "3")),),
            (2, 2, 2),
            elimination_order=(0,),
            query_variables=(),
        )

        assert result.variables == ()
        assert result.table == ("5",)

    def test_incomplete_elimination_order_is_rejected_before_computation(self) -> None:
        with pytest.raises(ValidationError, match="every non-query"):
            VariableEliminationRequest(
                factors=(_factor((0, 1), ("1", "1", "1", "1")),),
                domain_sizes=(2, 2, 2),
                elimination_order=(),
                query_variables=(1,),
            )

    def test_query_must_occur_and_be_canonical(self) -> None:
        with pytest.raises(ValidationError, match="occur"):
            VariableEliminationRequest(
                factors=(_factor((0,), ("1", "1")),),
                domain_sizes=(2, 2, 2),
                elimination_order=(0,),
                query_variables=(1,),
            )

    def test_oversized_intermediate_scope_is_rejected_before_computation(self) -> None:
        domain_sizes = (2,) * 16
        left = Factor(
            variables=tuple(range(8)),
            domain_sizes=domain_sizes,
            table=("1",) * 256,
        )
        right = Factor(
            variables=tuple(range(8, 16)),
            domain_sizes=domain_sizes,
            table=("1",) * 256,
        )

        with pytest.raises(ValidationError, match="size bound"):
            VariableEliminationRequest(
                factors=(left, right),
                domain_sizes=domain_sizes,
                elimination_order=(),
                query_variables=tuple(range(16)),
            )


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

        result = compute_d_separation(request)

        assert result.d_separated is True
        assert result.edges == request.edges

    def test_false_decision_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="bound d-separation"):
            DSeparationResult(
                variable_count=2,
                edges=((0, 1),),
                set_a=(0,),
                set_b=(1,),
                set_c=(),
                d_separated=True,
            )

    @pytest.mark.parametrize(
        ("edges", "match"),
        [
            (((0, 1), (1, 0)), "acyclic"),
            (((0, 0),), "self-loop"),
            (((0, 3),), "outside"),
        ],
    )
    def test_invalid_dag_is_rejected(
        self, edges: tuple[tuple[int, int], ...], match: str
    ) -> None:
        with pytest.raises(ValidationError, match=match):
            DSeparationRequest(
                variable_count=3,
                edges=edges,
                set_a=(0,),
                set_b=(1,),
                set_c=(),
            )

    def test_node_sets_must_be_pairwise_disjoint(self) -> None:
        with pytest.raises(ValidationError, match="pairwise disjoint"):
            DSeparationRequest(
                variable_count=2,
                set_a=(0,),
                set_b=(1,),
                set_c=(1,),
            )
