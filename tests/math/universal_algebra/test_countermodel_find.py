"""Bounded finite-magma countermodel search (#1686).

Covers the FOUND/EXHAUSTED_UP_TO_BOUND/UNKNOWN contract: the held-out
benchmark implication, exact exhaustion receipts, budget truncation that
never yields a negative conclusion, and admission.
"""

from __future__ import annotations

import json
from itertools import product
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import MathTool, OperationDomainValidationError
from jacobian.math.universal_algebra import (
    countermodel_find,
    implication_countermodel_check,
)
from jacobian.math.universal_algebra._models import (
    CountermodelFindRequest,
    CountermodelFindResult,
    MagmaEquation,
)
from jacobian.math.universal_algebra._tools import TOOLS
from jacobian.math.universal_algebra.values import FlatTerm


def _tool(operation_id: str) -> MathTool[Any, Any]:
    return next(tool for tool in TOOLS if tool.operation_id == operation_id)


def _variable(variable_id: int) -> dict[str, object]:
    return {"kind": "variable", "variable_id": variable_id}


def _apply(children: tuple[int, ...]) -> dict[str, object]:
    return {"kind": "application", "operation": 0, "children": list(children)}


def _term(nodes: list[dict[str, object]], root: int) -> FlatTerm:
    return FlatTerm.model_validate({"nodes": nodes, "root": root})


def _benchmark_premise() -> MagmaEquation:
    # (x*y) = (((y*x)*x)*y).
    left = _term([_variable(0), _variable(1), _apply((0, 1))], 2)
    right = _term(
        [
            _variable(0),
            _variable(1),
            _apply((1, 0)),
            _apply((2, 0)),
            _apply((3, 1)),
        ],
        4,
    )
    return MagmaEquation(left=left, right=right)


def _benchmark_target() -> MagmaEquation:
    # ((x*y)*y) = ((y*x)*x).
    left = _term(
        [
            _variable(0),
            _variable(1),
            _apply((0, 1)),
            _apply((2, 1)),
        ],
        3,
    )
    right = _term(
        [
            _variable(0),
            _variable(1),
            _apply((1, 0)),
            _apply((2, 0)),
        ],
        3,
    )
    return MagmaEquation(left=left, right=right)


def _idempotent_law() -> MagmaEquation:
    left = _term([_variable(0), _apply((0, 0))], 1)
    right = _term([_variable(0)], 0)
    return MagmaEquation(left=left, right=right)


class TestFound:
    def test_benchmark_implication_fails_in_orders_one_and_two(self) -> None:
        # The held-out benchmark verdict is FALSE with search_orders [1, 2].
        result = countermodel_find(
            (_benchmark_premise(),), _benchmark_target(), 1, 2, 1000
        )

        assert result.status == "FOUND"
        assert result.order in (1, 2)
        assert result.orders_complete == tuple(range(1, result.order or 1))
        assert result.certificate is not None
        assert result.certificate.is_countermodel is True
        assert len(result.certificate.algebra.carrier) == result.order

    def test_found_certificate_replays_through_the_checker(self) -> None:
        result = countermodel_find(
            (_benchmark_premise(),), _benchmark_target(), 1, 2, 1000
        )

        assert result.status == "FOUND"
        assert result.certificate is not None
        replayed = implication_countermodel_check(
            result.certificate.algebra, (_benchmark_premise(),), _benchmark_target()
        )

        assert replayed == result.certificate
        assert replayed.is_countermodel is True

    def test_idempotent_free_countermodel_is_reachable(self) -> None:
        # Regression: skipping tables with a nonzero 0-diamond-0 (as an unsound
        # "symmetry break" once did) drops every idempotent-free magma and can
        # report a false exhaustion.  This witness has no idempotent at all, so
        # a complete order-2 search must still find it.
        premise = MagmaEquation(
            left=_term([_variable(0), _variable(1), _apply((0, 1)), _apply((2, 0))], 3),
            right=_term(
                [_variable(0), _variable(1), _apply((0, 1)), _apply((0, 2))], 3
            ),
        )
        target = MagmaEquation(
            left=_term([_variable(0), _variable(1), _apply((0, 1)), _apply((2, 0))], 3),
            right=_term(
                [_variable(0), _variable(1), _apply((0, 0)), _apply((2, 1))], 3
            ),
        )

        result = countermodel_find((premise,), target, 1, 2, 1000)

        assert result.status == "FOUND"
        assert result.order == 2
        assert result.certificate is not None
        assert result.certificate.algebra.tables[0][0] != 0

    def test_search_is_deterministic(self) -> None:
        first = countermodel_find(
            (_benchmark_premise(),), _benchmark_target(), 1, 3, 50_000
        )
        second = countermodel_find(
            (_benchmark_premise(),), _benchmark_target(), 1, 3, 50_000
        )

        assert first == second


class TestExhausted:
    def test_idempotence_implies_itself(self) -> None:
        law = _idempotent_law()
        result = countermodel_find((law,), law, 1, 2, 1000)

        assert result.status == "EXHAUSTED_UP_TO_BOUND"
        assert result.total_tables == 1 + 16
        assert result.tables_examined == 17
        assert result.orders_complete == (1, 2)

    def test_exhaustion_matches_independent_finite_oracle(self) -> None:
        """Replay every table and valuation without calling Jacobian kernels."""
        # The left-zero law x*y=x implies associativity, but the oracle checks
        # that implication directly across every order-1 and order-2 table.
        left_zero = MagmaEquation(
            left=_term([_variable(0), _variable(1), _apply((0, 1))], 2),
            right=_term([_variable(0)], 0),
        )
        associative = MagmaEquation(
            left=_term(
                [
                    _variable(0),
                    _variable(1),
                    _variable(2),
                    _apply((0, 1)),
                    _apply((3, 2)),
                ],
                4,
            ),
            right=_term(
                [
                    _variable(0),
                    _variable(1),
                    _variable(2),
                    _apply((1, 2)),
                    _apply((0, 3)),
                ],
                4,
            ),
        )
        result = countermodel_find((left_zero,), associative, 1, 2, 1000)

        def evaluate(
            term: FlatTerm, assignment: tuple[int, ...], table: tuple[int, ...], n: int
        ) -> int:
            values: list[int] = []
            for node in term.nodes:
                if node.kind == "variable":
                    values.append(assignment[node.variable_id])
                else:
                    left, right = (values[index] for index in node.children)
                    values.append(table[left * n + right])
            return values[term.root]

        examined = 0
        countermodels = 0
        for n in (1, 2):
            for table in product(range(n), repeat=n * n):
                examined += 1
                premise_holds = all(
                    evaluate(left_zero.left, assignment, table, n)
                    == evaluate(left_zero.right, assignment, table, n)
                    for assignment in product(range(n), repeat=2)
                )
                target_fails = any(
                    evaluate(associative.left, assignment, table, n)
                    != evaluate(associative.right, assignment, table, n)
                    for assignment in product(range(n), repeat=3)
                )
                if premise_holds and target_fails:
                    countermodels += 1

        assert (examined, countermodels) == (17, 0)
        assert result.status == "EXHAUSTED_UP_TO_BOUND"
        assert result.tables_examined == examined
        assert result.total_tables == examined
        assert result.orders_complete == (1, 2)


class TestUnknown:
    def test_truncated_search_never_claims_exhaustion(self) -> None:
        law = _idempotent_law()
        result = countermodel_find((law,), law, 1, 3, 100)

        assert result.status == "UNKNOWN"
        assert result.stop_reason == "TABLE_BUDGET_EXHAUSTED"
        assert result.tables_examined == 100
        assert result.total_tables == 1 + 16 + 19_683
        assert result.current_order == 3
        assert result.orders_complete == (1, 2)


class TestInvalidRequests:
    def test_reversed_orders_are_rejected(self) -> None:
        law = _idempotent_law()

        with pytest.raises(OperationDomainValidationError) as exc_info:
            countermodel_find((law,), law, 2, 1, 100)
        assert exc_info.value.errors()[0]["type"] == "universal_algebra.order_range"

    def test_zero_budget_is_rejected(self) -> None:
        law = _idempotent_law()

        with pytest.raises(OperationDomainValidationError) as exc_info:
            countermodel_find((law,), law, 1, 2, 0)
        assert (
            exc_info.value.errors()[0]["type"]
            == "universal_algebra.table_budget_positive"
        )

    def test_overlarge_budget_is_rejected(self) -> None:
        law = _idempotent_law()

        with pytest.raises(OperationDomainValidationError) as exc_info:
            countermodel_find((law,), law, 1, 2, 500_001)
        assert (
            exc_info.value.errors()[0]["type"] == "universal_algebra.table_budget_bound"
        )

    def test_order_above_native_range_is_rejected(self) -> None:
        law = _idempotent_law()

        with pytest.raises(OperationDomainValidationError) as exc_info:
            countermodel_find((law,), law, 1, 5, 100)
        assert exc_info.value.errors()[0]["type"] == "universal_algebra.order_range"

    def test_request_validation_rejects_bad_range(self) -> None:
        law = _idempotent_law()

        with pytest.raises(ValidationError):
            CountermodelFindRequest(
                premises=(law,),
                target=law,
                min_order=2,
                max_order=1,
                table_budget=100,
            )

    def test_aggregate_search_work_is_admitted_before_enumeration(self) -> None:
        nodes: list[dict[str, object]] = [_variable(0)]
        for _ in range(50):
            nodes.append(_apply((len(nodes) - 1, 0)))
        deep_term = _term(nodes, len(nodes) - 1)
        law = MagmaEquation(left=deep_term, right=_term([_variable(0)], 0))
        with pytest.raises(OperationDomainValidationError) as exc_info:
            countermodel_find((law,), law, 1, 4, 500_000)
        assert exc_info.value.errors()[0]["type"] == (
            "universal_algebra.countermodel_search_work_bound"
        )


class TestAdmissionAndParity:
    def test_native_and_catalog_paths_agree(self) -> None:
        law = _idempotent_law()
        native = countermodel_find((law,), law, 1, 2, 1000)
        tool = _tool("universal_algebra.magma_implication.countermodel.find")

        assert (
            tool.run(
                CountermodelFindRequest(
                    premises=(law,),
                    target=law,
                    min_order=1,
                    max_order=2,
                    table_budget=1000,
                )
            )
            == native
        )

    def test_round_trip(self) -> None:
        for premises, target, min_order, max_order, budget in (
            ((_benchmark_premise(),), _benchmark_target(), 1, 2, 1000),
            ((_idempotent_law(),), _idempotent_law(), 1, 2, 1000),
            ((_idempotent_law(),), _idempotent_law(), 1, 3, 100),
        ):
            result = countermodel_find(premises, target, min_order, max_order, budget)
            restored = CountermodelFindResult.model_validate_json(
                result.model_dump_json()
            )

            assert restored == result

    def test_exhausted_count_forgery_is_rejected(self) -> None:
        law = _idempotent_law()
        result = countermodel_find((law,), law, 1, 2, 1000)
        assert result.status == "EXHAUSTED_UP_TO_BOUND"
        forged = json.loads(result.model_dump_json())
        forged["tables_examined"] = 16
        with pytest.raises(ValidationError):
            CountermodelFindResult.model_validate_json(json.dumps(forged))

    def test_exhaustion_total_must_match_declared_orders(self) -> None:
        law = _idempotent_law()
        result = countermodel_find((law,), law, 1, 2, 1000)
        forged = json.loads(result.model_dump_json())
        forged["total_tables"] = forged["tables_examined"] = 1
        with pytest.raises(ValidationError):
            CountermodelFindResult.model_validate_json(json.dumps(forged))

    def test_result_rejects_reversed_order_range(self) -> None:
        law = _idempotent_law()
        result = countermodel_find((law,), law, 1, 2, 1000)
        forged = json.loads(result.model_dump_json())
        forged["min_order"], forged["max_order"] = 2, 1
        forged["total_tables"] = 0
        forged["tables_examined"] = 0
        forged["orders_complete"] = []
        with pytest.raises(ValidationError):
            CountermodelFindResult.model_validate_json(json.dumps(forged))

    def test_found_certificate_must_match_declared_equations_and_range(self) -> None:
        premise = _benchmark_premise()
        target = _benchmark_target()
        result = countermodel_find((premise,), target, 1, 2, 1000)
        assert result.status == "FOUND"
        assert result.certificate is not None

        wrong_equation = json.loads(result.model_dump_json())
        wrong_equation["target"] = premise.model_dump(mode="json")
        with pytest.raises(ValidationError):
            CountermodelFindResult.model_validate_json(json.dumps(wrong_equation))

        wrong_range = json.loads(result.model_dump_json())
        wrong_range["min_order"] = wrong_range["max_order"] = 1
        wrong_range["total_tables"] = 1
        with pytest.raises(ValidationError):
            CountermodelFindResult.model_validate_json(json.dumps(wrong_range))
