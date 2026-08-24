"""Tests for universal-algebra operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.canonical import CanonicalLimits
from jacobian.math.universal_algebra import (
    ApplicationTerm,
    FiniteAlgebra,
    FiniteAlgebraCarrierMap,
    FiniteAlgebraHomomorphism,
    FlatTerm,
    OperationSymbol,
    VariableTerm,
)
from jacobian.math.universal_algebra._models import (
    CongruenceRequest,
    EquationProfileRequest,
    EvaluateRequest,
    HomomorphismProfileRequest,
    HomomorphismProfileResult,
    QuotientRequest,
    SubalgebraRequest,
)
from jacobian.math.universal_algebra._operations import (
    compute_congruence,
    compute_equation_profile,
    compute_evaluate,
    compute_generated_subalgebra,
    compute_homomorphism_profile,
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
    return FlatTerm(
        nodes=(VariableTerm(kind="variable", variable_id=variable_id),), root=0
    )


def _and_term() -> FlatTerm:
    return FlatTerm(
        nodes=(
            VariableTerm(kind="variable", variable_id=0),
            VariableTerm(kind="variable", variable_id=1),
            ApplicationTerm(kind="application", operation=0, children=(0, 1)),
        ),
        root=2,
    )


def _cyclic_addition_algebra(order: int) -> FiniteAlgebra:
    return FiniteAlgebra(
        carrier=tuple(str(index) for index in range(order)),
        operations=(OperationSymbol(operation_id="add", arity=2),),
        tables=(
            tuple(
                (left + right) % order
                for left in range(order)
                for right in range(order)
            ),
        ),
    )


def _empty_signature_algebra(size: int, prefix: str) -> FiniteAlgebra:
    return FiniteAlgebra(
        carrier=tuple(f"{prefix}{index}" for index in range(size)),
        operations=(),
        tables=(),
    )


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def test_catalog_contains_only_audited_agent_outcomes() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "universal_algebra.term.evaluate.compute",
        "universal_algebra.equation.profile.compute",
        "universal_algebra.subalgebra.generated.compute",
        "universal_algebra.map.homomorphism_profile.compute",
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
                VariableTerm(kind="variable", variable_id=0),
                ApplicationTerm(kind="application", operation=0, children=(0, 0)),
            ),
            root=1,
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
# Supplied-map homomorphism profile
# ---------------------------------------------------------------------------


class TestHomomorphismProfile:
    def test_identity_is_a_reusable_checked_isomorphism(self) -> None:
        algebra = _boolean_algebra()
        result = compute_homomorphism_profile(
            HomomorphismProfileRequest(
                carrier_map=FiniteAlgebraCarrierMap(
                    source=algebra,
                    target=algebra,
                    mapping=(0, 1),
                )
            )
        )

        assert result.status == "HOMOMORPHISM"
        assert isinstance(result.homomorphism, FiniteAlgebraHomomorphism)
        assert result.kernel_partition == ((0,), (1,))
        assert result.image == (0, 1)
        assert result.injective is True
        assert result.surjective is True
        assert result.isomorphism is True

    def test_quotient_map_has_canonical_kernel_and_image(self) -> None:
        result = compute_homomorphism_profile(
            HomomorphismProfileRequest(
                carrier_map=FiniteAlgebraCarrierMap(
                    source=_cyclic_addition_algebra(4),
                    target=_cyclic_addition_algebra(2),
                    mapping=(0, 1, 0, 1),
                )
            )
        )

        assert result.status == "HOMOMORPHISM"
        assert result.kernel_partition == ((0, 2), (1, 3))
        assert result.image == (0, 1)
        assert result.injective is False
        assert result.surjective is True
        assert result.isomorphism is False

    def test_injective_nonsurjective_map(self) -> None:
        result = compute_homomorphism_profile(
            HomomorphismProfileRequest(
                carrier_map=FiniteAlgebraCarrierMap(
                    source=_empty_signature_algebra(2, "s"),
                    target=_empty_signature_algebra(3, "t"),
                    mapping=(0, 2),
                )
            )
        )

        assert result.status == "HOMOMORPHISM"
        assert result.image == (0, 2)
        assert result.injective is True
        assert result.surjective is False
        assert result.isomorphism is False

    def test_first_unary_obstruction_is_complete_and_deterministic(self) -> None:
        symbol = (OperationSymbol(operation_id="flip", arity=1),)
        carrier_map = FiniteAlgebraCarrierMap(
            source=FiniteAlgebra(
                carrier=("0", "1"), operations=symbol, tables=((1, 0),)
            ),
            target=FiniteAlgebra(
                carrier=("a", "b"), operations=symbol, tables=((0, 1),)
            ),
            mapping=(0, 1),
        )
        result = compute_homomorphism_profile(
            HomomorphismProfileRequest(carrier_map=carrier_map)
        )

        assert result.status == "NOT_A_HOMOMORPHISM"
        assert result.carrier_map == carrier_map
        assert result.obstruction is not None
        assert result.obstruction.operation == 0
        assert result.obstruction.operation_id == "flip"
        assert result.obstruction.source_arguments == (0,)
        assert result.obstruction.target_arguments == (0,)
        assert result.obstruction.source_output == 1
        assert result.obstruction.mapped_source_output == 1
        assert result.obstruction.target_output == 0

    def test_higher_arity_obstruction_uses_lexicographic_tuple_order(self) -> None:
        result = compute_homomorphism_profile(
            HomomorphismProfileRequest(
                carrier_map=FiniteAlgebraCarrierMap(
                    source=_boolean_algebra(),
                    target=_boolean_algebra(),
                    mapping=(1, 0),
                )
            )
        )

        assert result.status == "NOT_A_HOMOMORPHISM"
        assert result.obstruction is not None
        assert result.obstruction.operation_id == "and"
        assert result.obstruction.source_arguments == (0, 1)
        assert result.obstruction.target_arguments == (1, 0)

    def test_nullary_constants_are_checked(self) -> None:
        symbol = (OperationSymbol(operation_id="zero", arity=0),)
        result = compute_homomorphism_profile(
            HomomorphismProfileRequest(
                carrier_map=FiniteAlgebraCarrierMap(
                    source=FiniteAlgebra(
                        carrier=("0", "1"), operations=symbol, tables=((0,),)
                    ),
                    target=FiniteAlgebra(
                        carrier=("a", "b"), operations=symbol, tables=((1,),)
                    ),
                    mapping=(0, 1),
                )
            )
        )

        assert result.status == "NOT_A_HOMOMORPHISM"
        assert result.obstruction is not None
        assert result.obstruction.source_arguments == ()
        assert result.obstruction.target_arguments == ()
        assert result.obstruction.source_output == 0
        assert result.obstruction.mapped_source_output == 0
        assert result.obstruction.target_output == 1

    def test_result_validation_rejects_forged_positive_and_negative_data(
        self,
    ) -> None:
        positive = compute_homomorphism_profile(
            HomomorphismProfileRequest(
                carrier_map=FiniteAlgebraCarrierMap(
                    source=_cyclic_addition_algebra(4),
                    target=_cyclic_addition_algebra(2),
                    mapping=(0, 1, 0, 1),
                )
            )
        ).model_dump(mode="json")
        positive["kernel_partition"] = [[0, 1], [2, 3]]
        with pytest.raises(ValidationError, match="canonical fibers"):
            HomomorphismProfileResult.model_validate(positive)

        symbol = (OperationSymbol(operation_id="flip", arity=1),)
        negative = compute_homomorphism_profile(
            HomomorphismProfileRequest(
                carrier_map=FiniteAlgebraCarrierMap(
                    source=FiniteAlgebra(
                        carrier=("0", "1"), operations=symbol, tables=((1, 0),)
                    ),
                    target=FiniteAlgebra(
                        carrier=("a", "b"), operations=symbol, tables=((0, 1),)
                    ),
                    mapping=(0, 1),
                )
            )
        ).model_dump(mode="json")
        negative["obstruction"]["target_output"] = 1
        with pytest.raises(ValidationError, match="first exact"):
            HomomorphismProfileResult.model_validate(negative)

    def test_source_mutation_invalidates_negative_conclusion(self) -> None:
        symbol = (OperationSymbol(operation_id="flip", arity=1),)
        payload = compute_homomorphism_profile(
            HomomorphismProfileRequest(
                carrier_map=FiniteAlgebraCarrierMap(
                    source=FiniteAlgebra(
                        carrier=("0", "1"), operations=symbol, tables=((1, 0),)
                    ),
                    target=FiniteAlgebra(
                        carrier=("a", "b"), operations=symbol, tables=((0, 1),)
                    ),
                    mapping=(0, 1),
                )
            )
        ).model_dump(mode="json")
        payload["carrier_map"]["target"]["tables"] = [[1, 0]]
        with pytest.raises(ValidationError, match="preserves every"):
            HomomorphismProfileResult.model_validate(payload)

    def test_source_mutation_invalidates_positive_homomorphism(self) -> None:
        payload = compute_homomorphism_profile(
            HomomorphismProfileRequest(
                carrier_map=FiniteAlgebraCarrierMap(
                    source=_cyclic_addition_algebra(4),
                    target=_cyclic_addition_algebra(2),
                    mapping=(0, 1, 0, 1),
                )
            )
        ).model_dump(mode="json")
        payload["homomorphism"]["target"]["tables"][0][0] = 1
        with pytest.raises(ValidationError, match="does not preserve"):
            HomomorphismProfileResult.model_validate(payload)

    def test_maximum_source_table_budget_is_scanned_completely(self) -> None:
        size = 16
        source = FiniteAlgebra(
            carrier=tuple(str(index) for index in range(size)),
            operations=(OperationSymbol(operation_id="f", arity=4),),
            tables=((0,) * (size**4),),
        )
        target = FiniteAlgebra(
            carrier=tuple(str(index) for index in range(size)),
            operations=(OperationSymbol(operation_id="f", arity=4),),
            tables=((0,) * (size**4 - 1) + (1,),),
        )
        result = compute_homomorphism_profile(
            HomomorphismProfileRequest(
                carrier_map=FiniteAlgebraCarrierMap(
                    source=source,
                    target=target,
                    mapping=tuple(range(size)),
                )
            )
        )
        assert result.status == "NOT_A_HOMOMORPHISM"
        assert result.obstruction is not None
        assert result.obstruction.source_arguments == (15, 15, 15, 15)

    def test_retained_source_must_fit_the_canonical_output(self) -> None:
        oversized_label = "x" * (5 * 1024 * 1024)
        source = FiniteAlgebra(carrier=(oversized_label,), operations=(), tables=())
        target = FiniteAlgebra(
            carrier=(oversized_label + "y",), operations=(), tables=()
        )
        with pytest.raises(ValidationError, match="canonical output limit"):
            HomomorphismProfileRequest(
                carrier_map=FiniteAlgebraCarrierMap(
                    source=source,
                    target=target,
                    mapping=(0,),
                )
            )


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
        source = _boolean_algebra()
        result = compute_quotient(QuotientRequest(algebra=source, partition=((0, 1),)))
        assert isinstance(result, FiniteAlgebraHomomorphism)
        assert result.source == source
        assert result.target.carrier == ("B0",)
        assert len(result.target.operations) == 2
        assert result.mapping == (0, 0)
        SubalgebraRequest(algebra=result.target, generators=(0,))

    def test_quotient_homomorphism_composes_with_profile_unchanged(self) -> None:
        quotient_map = compute_quotient(
            QuotientRequest(algebra=_boolean_algebra(), partition=((0, 1),))
        )

        native_profile = compute_homomorphism_profile(
            HomomorphismProfileRequest(carrier_map=quotient_map)
        )
        wire_profile = compute_homomorphism_profile(
            HomomorphismProfileRequest.model_validate(
                {"carrier_map": quotient_map.model_dump(mode="json")}
            )
        )

        assert native_profile.status == "HOMOMORPHISM"
        assert native_profile.homomorphism == quotient_map
        assert wire_profile == native_profile

    def test_quotient_rejects_when_retained_source_has_no_output_headroom(
        self,
    ) -> None:
        source = FiniteAlgebra(
            carrier=("x" * (CanonicalLimits().max_output_bytes - 1_024),),
            operations=(),
            tables=(),
        )

        with pytest.raises(ValidationError, match="canonical quotient homomorphism"):
            QuotientRequest(algebra=source, partition=((0,),))

    def test_quotient_charges_construction_and_map_replay_work(self) -> None:
        def constant_ternary_algebra(size: int) -> FiniteAlgebra:
            return FiniteAlgebra(
                carrier=tuple(f"a{index}" for index in range(size)),
                operations=(OperationSymbol(operation_id="f", arity=3),),
                tables=((0,) * size**3,),
            )

        accepted_size = 23
        accepted = QuotientRequest(
            algebra=constant_ternary_algebra(accepted_size),
            partition=tuple((index,) for index in range(accepted_size)),
        )
        assert len(accepted.partition) == accepted_size

        rejected_size = 24
        with pytest.raises(ValidationError, match="construction and homomorphism"):
            QuotientRequest(
                algebra=constant_ternary_algebra(rejected_size),
                partition=tuple((index,) for index in range(rejected_size)),
            )


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

    @pytest.mark.parametrize("kind", ["", "VARIABLE", "arbitrary"])
    def test_term_node_kind_is_closed(self, kind: str) -> None:
        with pytest.raises(ValidationError):
            FlatTerm.model_validate(
                {"nodes": [{"kind": kind, "variable_id": 0}], "root": 0}
            )

    def test_term_rejects_cycles_forward_edges_and_unreachable_nodes(self) -> None:
        with pytest.raises(ValidationError, match="earlier nodes"):
            FlatTerm.model_validate(
                {
                    "nodes": [{"kind": "application", "operation": 0, "children": [0]}],
                    "root": 0,
                }
            )
        with pytest.raises(ValidationError, match="reachable"):
            FlatTerm(
                nodes=(
                    VariableTerm(kind="variable", variable_id=0),
                    VariableTerm(kind="variable", variable_id=1),
                ),
                root=1,
            )

    def test_term_signature_and_assignment_are_checked_in_request(self) -> None:
        wrong_arity = FlatTerm(
            nodes=(
                VariableTerm(kind="variable", variable_id=0),
                ApplicationTerm(kind="application", operation=0, children=(0,)),
            ),
            root=1,
        )
        with pytest.raises(ValidationError, match="arity"):
            EvaluateRequest(
                algebra=_boolean_algebra(), term=wrong_arity, assignment=(0,)
            )
        with pytest.raises(ValidationError, match="cover exactly"):
            EvaluateRequest(
                algebra=_boolean_algebra(), term=_and_term(), assignment=(0,)
            )

    @pytest.mark.parametrize("partition", [((0,),), ((), (0, 1)), ((0, 1), (1,))])
    def test_partition_must_be_nonempty_disjoint_exact_cover(
        self, partition: tuple[tuple[int, ...], ...]
    ) -> None:
        with pytest.raises(ValidationError):
            CongruenceRequest(algebra=_boolean_algebra(), partition=partition)

    def test_equation_profile_rejects_work_before_enumeration(self) -> None:
        algebra = FiniteAlgebra(
            carrier=tuple(str(index) for index in range(10)),
            operations=(),
            tables=(),
        )
        term = _variable_term(0)
        with pytest.raises(ValidationError, match="work budget"):
            EquationProfileRequest(
                algebra=algebra,
                left=term,
                right=term,
                variable_count=7,
            )

    def test_carrier_map_rejects_incomplete_or_wrong_signature_input(self) -> None:
        source = _boolean_algebra()
        with pytest.raises(ValidationError, match="one target index"):
            FiniteAlgebraCarrierMap(
                source=source,
                target=source,
                mapping=(0,),
            )
        wrong_signature = FiniteAlgebra(
            carrier=("0", "1"),
            operations=(OperationSymbol(operation_id="and", arity=1),),
            tables=((0, 1),),
        )
        with pytest.raises(ValidationError, match="must match exactly"):
            FiniteAlgebraCarrierMap(
                source=source,
                target=wrong_signature,
                mapping=(0, 1),
            )

    def test_table_budget_rejects_immediately_above_boundary(self) -> None:
        size = 16
        with pytest.raises(ValidationError, match="cell budget"):
            FiniteAlgebra(
                carrier=tuple(str(index) for index in range(size)),
                operations=(
                    OperationSymbol(operation_id="f", arity=4),
                    OperationSymbol(operation_id="c", arity=0),
                ),
                tables=((0,) * (size**4), (0,)),
            )
