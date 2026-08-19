"""Tests for universal-algebra operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.universal_algebra import (
    FiniteAlgebra,
    FlatTerm,
    OperationSymbol,
    Term,
)
from jacobian.math.universal_algebra._models import (
    CongruenceRequest,
    EquationProfileRequest,
    EvaluateRequest,
    QuotientRequest,
    SubalgebraRequest,
)
from jacobian.math.universal_algebra._operations import (
    compute_congruence,
    compute_equation_profile,
    compute_evaluate,
    compute_generated_subalgebra,
    compute_quotient,
)
from jacobian.math.universal_algebra._tools import TOOLS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _boolean_algebra() -> FiniteAlgebra:
    return FiniteAlgebra(
        carrier=("0", "1"),
        operations=(
            OperationSymbol(operation_id="and", arity=2),
            OperationSymbol(operation_id="or", arity=2),
        ),
        tables=((0, 0, 0, 1), (0, 1, 1, 1)),
    )


def _variable_term(variable_id: int) -> FlatTerm:
    return FlatTerm(nodes=(Term(kind="variable", variable_id=variable_id),), root=0)


def _and_term() -> FlatTerm:
    return FlatTerm(
        nodes=(
            Term(kind="variable", variable_id=0),
            Term(kind="variable", variable_id=1),
            Term(kind="application", operation=0, children=(0, 1)),
        ),
        root=2,
    )


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def test_catalog_contains_only_audited_agent_outcomes() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "universal_algebra.term.evaluate.compute",
        "universal_algebra.equation.profile.compute",
        "universal_algebra.subalgebra.generated.compute",
        "universal_algebra.congruence.check.compute",
        "universal_algebra.quotient.compute",
    }


# ---------------------------------------------------------------------------
# Term evaluation
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_and_00(self) -> None:
        result = compute_evaluate(
            EvaluateRequest(
                algebra=_boolean_algebra(), term=_and_term(), assignment=(0, 0)
            )
        )
        assert result.value == 0

    def test_and_11(self) -> None:
        result = compute_evaluate(
            EvaluateRequest(
                algebra=_boolean_algebra(), term=_and_term(), assignment=(1, 1)
            )
        )
        assert result.value == 1


# ---------------------------------------------------------------------------
# Equation profile
# ---------------------------------------------------------------------------


class TestEquationProfile:
    def test_holds(self) -> None:
        # AND(x, x) = x: this should hold in the Boolean algebra.
        # Term: AND(x0, x0) — application of operation 0 (and) with children (0, 0).
        left = FlatTerm(
            nodes=(
                Term(kind="variable", variable_id=0),
                Term(kind="variable", variable_id=0),
                Term(kind="application", operation=0, children=(0, 0)),
            ),
            root=2,
        )
        right = _variable_term(0)
        result = compute_equation_profile(
            EquationProfileRequest(
                algebra=_boolean_algebra(), left=left, right=right, variable_count=1
            )
        )
        assert result.status == "HOLDS"
        assert result.satisfying_count == 2

    def test_fails(self) -> None:
        # AND(x, y) = x: this does NOT hold in general (AND(0, 1) = 0, but
        # AND(1, 0) = 0 != 1).
        left = _and_term()
        right = _variable_term(0)
        result = compute_equation_profile(
            EquationProfileRequest(
                algebra=_boolean_algebra(), left=left, right=right, variable_count=2
            )
        )
        assert result.status == "FAILS"
        assert result.satisfying_count < 4
        assert result.first_counterassignment is not None


# ---------------------------------------------------------------------------
# Generated subalgebra
# ---------------------------------------------------------------------------


class TestGeneratedSubalgebra:
    def test_generated_by_0(self) -> None:
        result = compute_generated_subalgebra(
            SubalgebraRequest(algebra=_boolean_algebra(), generators=(0,))
        )
        assert result.generated_carrier == (0,)
        assert result.is_closed is True

    def test_generated_by_both(self) -> None:
        result = compute_generated_subalgebra(
            SubalgebraRequest(algebra=_boolean_algebra(), generators=(0, 1))
        )
        assert result.generated_carrier == (0, 1)


# ---------------------------------------------------------------------------
# Congruence
# ---------------------------------------------------------------------------


class TestCongruence:
    def test_universal_partition_is_congruence(self) -> None:
        result = compute_congruence(
            CongruenceRequest(algebra=_boolean_algebra(), partition=((0, 1),))
        )
        assert result.is_congruence is True

    def test_equality_partition_is_congruence(self) -> None:
        result = compute_congruence(
            CongruenceRequest(algebra=_boolean_algebra(), partition=((0,), (1,)))
        )
        assert result.is_congruence is True


# ---------------------------------------------------------------------------
# Quotient
# ---------------------------------------------------------------------------


class TestQuotient:
    def test_trivial_quotient(self) -> None:
        result = compute_quotient(
            QuotientRequest(algebra=_boolean_algebra(), partition=((0, 1),))
        )
        assert result.carrier == ("B0",)
        assert len(result.operations) == 2


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_duplicate_carrier_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unique"):
            FiniteAlgebra(
                carrier=("a", "a"),
                operations=(OperationSymbol(operation_id="f", arity=1),),
                tables=((0, 0),),
            )

    def test_wrong_table_size_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cell count"):
            FiniteAlgebra(
                carrier=("a", "b"),
                operations=(OperationSymbol(operation_id="f", arity=2),),
                tables=((0, 0, 0),),  # Should be 4 cells, not 3
            )

    def test_out_of_range_output_rejected(self) -> None:
        with pytest.raises(ValidationError, match="carrier range"):
            FiniteAlgebra(
                carrier=("a", "b"),
                operations=(OperationSymbol(operation_id="f", arity=1),),
                tables=((0, 2),),  # 2 is out of range
            )
