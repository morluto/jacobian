"""Tests for the exact Lie-algebra Killing form."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.lie_algebras._models import (
    FiniteDimensionalLieAlgebra,
    LieAlgebraRequest,
    LieKillingRadicalResult,
    LieKillingResult,
    StructureConstant,
)
from jacobian.math.lie_algebras.operations import (
    lie_killing_form,
    lie_killing_form_radical,
)


def _algebra(
    basis: tuple[str, ...],
    constants: tuple[tuple[int, int, int, int], ...],
) -> FiniteDimensionalLieAlgebra:
    return FiniteDimensionalLieAlgebra.model_validate(
        {
            "basis": list(basis),
            "structure_constants": [
                {"i": i, "j": j, "k": k, "coefficient": {"num": value, "den": 1}}
                for i, j, k, value in constants
            ],
        }
    )


def _unchecked_algebra(
    basis: tuple[str, ...], constants: tuple[tuple[int, int, int, int], ...]
) -> FiniteDimensionalLieAlgebra:
    return FiniteDimensionalLieAlgebra.model_construct(
        basis=basis,
        structure_constants=tuple(
            StructureConstant.model_construct(
                i=i,
                j=j,
                k=k,
                coefficient=CanonicalRational.model_construct(num=value, den=1),
            )
            for i, j, k, value in constants
        ),
    )


def _entries(result: LieKillingResult) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(entry.as_fraction() for entry in row)
        for row in result.killing_form.entries
    )


SL2 = _algebra(("e", "f", "h"), ((0, 1, 2, 1), (0, 2, 0, -2), (1, 2, 1, 2)))
SO3 = _algebra(("x", "y", "z"), ((0, 1, 2, 1), (0, 2, 1, -1), (1, 2, 0, 1)))
HEISENBERG = _algebra(("x", "y", "z"), ((0, 1, 2, 1),))
ABELIAN_2 = _algebra(("a", "b"), ())
AFFINE = _algebra(("a", "b"), ((0, 1, 1, 1),))
JACOBI_VIOLATOR = _unchecked_algebra(
    ("e", "f", "h"), ((0, 1, 2, 1), (0, 2, 0, 1), (1, 2, 0, 1))
)


class TestKillingKnownAnswers:
    def test_sl2_killing_form(self) -> None:
        assert _entries(lie_killing_form(SL2)) == (
            (Fraction(0), Fraction(4), Fraction(0)),
            (Fraction(4), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(8)),
        )

    def test_so3_killing_form_is_negative_scalar(self) -> None:
        assert _entries(lie_killing_form(SO3)) == (
            (Fraction(-2), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(-2), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(-2)),
        )

    def test_heisenberg_killing_form_vanishes(self) -> None:
        assert _entries(lie_killing_form(HEISENBERG)) == ((Fraction(0),) * 3,) * 3

    def test_abelian_killing_form_vanishes(self) -> None:
        assert _entries(lie_killing_form(ABELIAN_2)) == (
            (Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(0)),
        )

    def test_affine_killing_form_is_degenerate_but_nonzero(self) -> None:
        """The solvable non-nilpotent ax+b algebra has rank-one Killing form:
        degenerate yet not identically zero."""
        assert _entries(lie_killing_form(AFFINE)) == (
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(0)),
        )


class TestKillingDefiningInvariant:
    def test_sl2_ad_invariance(self) -> None:
        """κ([e,f],h) = κ(h,h) = 8 equals κ(e,[f,h]) = κ(e,2f) = 2κ(e,f)."""
        entries = _entries(lie_killing_form(SL2))
        assert entries[2][2] == Fraction(8)
        assert 2 * entries[0][1] == Fraction(8)

    def test_result_binds_source_algebra(self) -> None:
        result = lie_killing_form(SL2)
        assert result.algebra == SL2
        assert (result.killing_form.row_count, result.killing_form.column_count) == (
            3,
            3,
        )

    def test_so3_determinant_is_nonzero(self) -> None:
        """Compact so(3) is semisimple: Cartan's criterion reads -8 off
        the diagonal matrix without a determinant operation."""
        entries = _entries(lie_killing_form(SO3))
        assert entries[0][0] * entries[1][1] * entries[2][2] == Fraction(-8)


class TestKillingRejections:
    def test_jacobi_violator_rejected_before_traces(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            lie_killing_form(JACOBI_VIOLATOR)
        assert exc_info.value.errors()[0]["type"] == "lie_algebra.jacobi_identity"

    def test_native_mapping_payload_uses_shared_admission(self) -> None:
        payload = {
            "basis": ["e", "f", "h"],
            "structure_constants": [
                {"i": 0, "j": 1, "k": 2, "coefficient": {"num": 1, "den": 1}},
                {"i": 0, "j": 2, "k": 0, "coefficient": {"num": -2, "den": 1}},
                {"i": 1, "j": 2, "k": 1, "coefficient": {"num": 2, "den": 1}},
            ],
        }
        assert _entries(lie_killing_form(payload)) == _entries(lie_killing_form(SL2))

    def test_forged_asymmetric_matrix_rejected(self) -> None:
        import json

        result = lie_killing_form(SL2)
        forged = json.loads(result.model_dump_json())
        forged["killing_form"]["entries"][0][1] = {"num": "5", "den": "1"}
        with pytest.raises(ValidationError) as exc_info:
            LieKillingResult.model_validate_json(json.dumps(forged))
        assert (
            exc_info.value.errors(include_url=False)[0]["type"]
            == "lie_algebra.killing_symmetry"
        )


class TestKillingComposition:
    def test_request_model_and_native_paths_agree(self) -> None:
        request = LieAlgebraRequest(algebra=SL2)
        assert lie_killing_form(request.algebra) == lie_killing_form(SL2)

    def test_serialized_result_round_trips(self) -> None:
        result = lie_killing_form(SL2)
        assert LieKillingResult.model_validate_json(result.model_dump_json()) == result

    def test_catalog_declares_the_operation_with_a_valid_example(self) -> None:
        from jacobian.canonical import encode_strict_json
        from jacobian.math.lie_algebras._tools import TOOLS

        tools = {
            tool.operation_id: tool
            for tool in TOOLS
            if tool.operation_id == "lie_algebra.killing_form.compute"
        }
        assert set(tools) == {"lie_algebra.killing_form.compute"}
        tool = tools["lie_algebra.killing_form.compute"]
        assert tool.examples
        payload = tool.request_type.model_validate_json(
            encode_strict_json(tool.examples[0].input), strict=True
        )
        assert tool.run(payload) == lie_killing_form(payload.algebra)


class TestKillingFormRadical:
    def test_sl2_has_zero_killing_form_radical(self) -> None:
        result = lie_killing_form_radical(SL2)
        expected_killing = (
            (Fraction(0), Fraction(4), Fraction(0)),
            (Fraction(4), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(8)),
        )

        assert _entries(result.killing_result) == expected_killing
        assert result.killing_result.algebra == SL2
        assert result.radical.basis == SL2.basis
        assert result.radical.generators.row_count == 0
        assert result.radical.generators.column_count == 3

    def test_heisenberg_has_full_killing_form_radical(self) -> None:
        result = lie_killing_form_radical(HEISENBERG)
        expected_killing = ((Fraction(0),) * 3,) * 3
        expected_radical = (
            (Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(1)),
        )

        assert _entries(result.killing_result) == expected_killing
        assert (
            tuple(
                tuple(value.as_fraction() for value in row)
                for row in result.radical.generators.entries
            )
            == expected_radical
        )
        assert result.radical.basis == HEISENBERG.basis

    def test_result_round_trips_and_retains_killing_result(self) -> None:
        result = lie_killing_form_radical(SL2)
        restored = LieKillingRadicalResult.model_validate_json(result.model_dump_json())

        assert restored == result
        assert restored.killing_result == lie_killing_form(SL2)

    def test_catalog_example_returns_the_declared_operation_value(self) -> None:
        from jacobian.canonical import encode_strict_json
        from jacobian.math.lie_algebras._tools import TOOLS

        tool = next(
            item
            for item in TOOLS
            if item.operation_id == "lie_algebra.killing_form.radical.compute"
        )
        payload = tool.request_type.model_validate_json(
            encode_strict_json(tool.examples[0].input), strict=True
        )

        assert tool.run(payload) == lie_killing_form_radical(payload.algebra)
