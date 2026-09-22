"""Tests for Lie-algebra ideals, quotients, and direct sums."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.lie_algebras._models import (
    FiniteDimensionalLieAlgebra,
    LieAlgebraElement,
    LieIdealCheckResult,
    LieIdealRequest,
    LieQuotientRequest,
    LieQuotientResult,
    LieSubspace,
)
from jacobian.math.lie_algebras.operations import (
    check_ideal,
    lie_center,
    lie_direct_sum,
    lie_killing_form,
    lie_quotient,
)

F = Fraction


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


def _subspace(basis: tuple[str, ...], rows: tuple[tuple[int, ...], ...]) -> LieSubspace:
    return LieSubspace.model_validate(
        {
            "basis": list(basis),
            "generators": {
                "domain": "QQ",
                "row_count": len(rows),
                "column_count": len(basis),
                "entries": [
                    [{"num": value, "den": 1} for value in row] for row in rows
                ],
            },
        }
    )


def _zero_subspace(basis: tuple[str, ...]) -> LieSubspace:
    return LieSubspace.model_validate(
        {
            "basis": list(basis),
            "generators": {
                "domain": "QQ",
                "row_count": 0,
                "column_count": len(basis),
                "entries": [],
            },
        }
    )


def _fracs(element: LieAlgebraElement) -> tuple[Fraction, ...]:
    return tuple(coordinate.as_fraction() for coordinate in element.coordinates)


SL2 = _algebra(("e", "f", "h"), ((0, 1, 2, 1), (0, 2, 0, -2), (1, 2, 1, 2)))
HEISENBERG = _algebra(("x", "y", "z"), ((0, 1, 2, 1),))
ABELIAN_2 = _algebra(("a", "b"), ())
AFFINE = _algebra(("a", "b"), ((0, 1, 1, 1),))
GL2 = _algebra(
    ("a", "b", "c", "d"),
    (
        (0, 1, 1, 1),
        (0, 2, 2, -1),
        (1, 2, 0, 1),
        (1, 2, 3, -1),
        (1, 3, 1, 1),
        (2, 3, 2, -1),
    ),
)
JACOBI_VIOLATOR = _algebra(("e", "f", "h"), ((0, 1, 2, 1), (0, 2, 0, 1), (1, 2, 0, 1)))

HEISENBERG_CENTER = _subspace(("x", "y", "z"), ((0, 0, 1),))
HEISENBERG_X = _subspace(("x", "y", "z"), ((1, 0, 0),))
SL2_BOREL = _subspace(("e", "f", "h"), ((1, 0, 0), (0, 0, 1)))
AFFINE_B = _subspace(("a", "b"), ((0, 1),))
GL2_SL2 = _subspace(("a", "b", "c", "d"), ((1, 0, 0, -1), (0, 1, 0, 0), (0, 0, 1, 0)))


class TestIdealCheck:
    @pytest.mark.parametrize(
        ("algebra", "candidate", "is_ideal"),
        (
            ("sl2", "whole", True),
            ("sl2", "zero", True),
            ("sl2", "borel", False),
            ("heisenberg", "center", True),
            ("heisenberg", "x", False),
            ("affine", "b", True),
            ("gl2", "sl2", True),
            ("abelian", "whole", True),
        ),
    )
    def test_ideal_decisions(
        self, algebra: str, candidate: str, is_ideal: bool
    ) -> None:
        algebras = {
            "sl2": (SL2, ("e", "f", "h")),
            "heisenberg": (HEISENBERG, ("x", "y", "z")),
            "affine": (AFFINE, ("a", "b")),
            "gl2": (GL2, ("a", "b", "c", "d")),
            "abelian": (ABELIAN_2, ("a", "b")),
        }
        basis_algebra, basis = algebras[algebra]
        dimension = len(basis)
        candidates = {
            "whole": _subspace(
                basis,
                tuple(
                    tuple(int(i == j) for j in range(dimension))
                    for i in range(dimension)
                ),
            ),
            "zero": _zero_subspace(basis),
            "borel": SL2_BOREL,
            "center": HEISENBERG_CENTER,
            "x": HEISENBERG_X,
            "b": AFFINE_B,
            "sl2": GL2_SL2,
        }
        result = check_ideal(basis_algebra, candidates[candidate])
        assert result.is_ideal is is_ideal
        assert (result.witness is None) is is_ideal

    def test_heisenberg_x_witness(self) -> None:
        result = check_ideal(HEISENBERG, HEISENBERG_X)
        assert result.is_ideal is False
        assert result.witness is not None
        assert result.witness.basis_index == 1
        assert result.witness.subspace_row == 0
        assert _fracs(result.witness.bracket) == (F(0), F(0), F(-1))

    def test_sl2_borel_witness_is_first_escape(self) -> None:
        result = check_ideal(SL2, SL2_BOREL)
        assert result.is_ideal is False
        assert result.witness is not None
        assert result.witness.basis_index == 1
        assert result.witness.subspace_row == 1
        assert _fracs(result.witness.bracket) == (F(0), F(2), F(0))

    def test_center_is_always_ideal(self) -> None:
        """Cross-operation theorem: the computed center absorbs brackets."""
        for algebra in (SL2, HEISENBERG, AFFINE, GL2, ABELIAN_2):
            center = lie_center(algebra).center
            assert check_ideal(algebra, center).is_ideal is True

    def test_jacobi_violator_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            check_ideal(JACOBI_VIOLATOR, HEISENBERG_CENTER)
        assert exc_info.value.errors()[0]["type"] == "lie_algebra.jacobi_identity"

    def test_wrong_basis_candidate_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            check_ideal(SL2, HEISENBERG_CENTER)
        assert exc_info.value.errors()[0]["type"] == "lie_algebra.candidate_basis"
        with pytest.raises(ValidationError) as wire_info:
            LieIdealRequest(algebra=SL2, candidate=HEISENBERG_CENTER)
        assert (
            wire_info.value.errors(include_url=False)[0]["type"]
            == "lie_algebra.candidate_basis"
        )


class TestQuotient:
    def test_heisenberg_mod_center_is_abelian_plane(self) -> None:
        result = lie_quotient(HEISENBERG, HEISENBERG_CENTER, ("u", "v"))
        assert result.quotient.basis == ("u", "v")
        assert result.quotient.structure_constants == ()

    def test_gl2_mod_sl2_is_line(self) -> None:
        result = lie_quotient(GL2, GL2_SL2, ("t",))
        assert result.quotient.basis == ("t",)
        assert result.quotient.structure_constants == ()

    def test_sl2_mod_zero_is_sl2(self) -> None:
        zero = _zero_subspace(("e", "f", "h"))
        result = lie_quotient(SL2, zero, ("p", "q", "r"))
        assert result.quotient.basis == ("p", "q", "r")
        assert len(result.quotient.structure_constants) == len(SL2.structure_constants)

    def test_affine_mod_b_is_line(self) -> None:
        result = lie_quotient(AFFINE, AFFINE_B, ("s",))
        assert result.quotient.structure_constants == ()

    def test_non_ideal_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            lie_quotient(SL2, SL2_BOREL, ("u", "v"))
        assert exc_info.value.errors()[0]["type"] == "lie_algebra.not_an_ideal"

    def test_wrong_label_count_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            lie_quotient(HEISENBERG, HEISENBERG_CENTER, ("u",))
        assert exc_info.value.errors()[0]["type"] == "lie_algebra.quotient_dimension"

    def test_duplicate_labels_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            lie_quotient(HEISENBERG, HEISENBERG_CENTER, ("u", "u"))
        assert exc_info.value.errors()[0]["type"] == "lie_algebra.quotient_labels"

    def test_non_string_labels_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            lie_quotient(HEISENBERG, HEISENBERG_CENTER, ("u", 1))  # type: ignore[arg-type]
        assert exc_info.value.errors()[0]["type"] == "lie_algebra.quotient_labels"

    def test_jacobi_violator_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            lie_quotient(JACOBI_VIOLATOR, HEISENBERG_CENTER, ("u", "v"))
        assert exc_info.value.errors()[0]["type"] == "lie_algebra.jacobi_identity"

    def test_nonabelian_quotient_recovery(self) -> None:
        """(sl2 ⊕ line) / line recovers sl2 constants exactly: the
        quotient bracket is well defined and computed, not merely
        abelian by dimension."""
        line = _algebra(("t",), ())
        total = lie_direct_sum(SL2, line, ("e", "f", "h", "t"))
        axis = _subspace(("e", "f", "h", "t"), ((0, 0, 0, 1),))
        result = lie_quotient(total, axis, ("p", "q", "r"))
        recovered = tuple(
            (constant.i, constant.j, constant.k, constant.coefficient.as_fraction())
            for constant in result.quotient.structure_constants
        )
        assert recovered == (
            (0, 1, 2, F(1)),
            (0, 2, 0, F(-2)),
            (1, 2, 1, F(2)),
        )


class TestPublishedDirectSum:
    def test_catalog_operation_round_trips_exact_algebra(self) -> None:
        from jacobian.math.lie_algebras._models import LieDirectSumRequest
        from jacobian.math.lie_algebras._tools import TOOLS

        tool = next(
            tool
            for tool in TOOLS
            if tool.operation_id == "lie_algebra.direct_sum.compute"
        )
        line = _algebra(("t",), ())
        request = LieDirectSumRequest(left=SL2, right=line, basis=("e", "f", "h", "t"))
        result = tool.run(request)

        assert result == lie_direct_sum(SL2, line, request.basis)
        assert (
            FiniteDimensionalLieAlgebra.model_validate_json(result.model_dump_json())
            == result
        )


class TestDirectSum:
    def test_sl2_plus_line(self) -> None:
        line = _algebra(("t",), ())
        total = lie_direct_sum(SL2, line, ("e", "f", "h", "t"))
        assert total.basis == ("e", "f", "h", "t")
        assert len(total.structure_constants) == 3
        assert all(constant.i < 3 for constant in total.structure_constants)

    def test_killing_form_is_block_diagonal(self) -> None:
        """Cross-operation composition: the sum Killing form blocks."""
        line = _algebra(("t",), ())
        total = lie_direct_sum(SL2, line, ("e", "f", "h", "t"))
        entries = tuple(
            tuple(entry.as_fraction() for entry in row)
            for row in lie_killing_form(total).killing_form.entries
        )
        assert entries[3] == (F(0), F(0), F(0), F(0))
        assert tuple(row[3] for row in entries) == (F(0), F(0), F(0), F(0))
        assert entries[0][:3] == (F(0), F(4), F(0))

    def test_center_dimensions_add(self) -> None:
        line = _algebra(("t",), ())
        total = lie_direct_sum(SL2, line, ("e", "f", "h", "t"))
        assert lie_center(total).center.generators.row_count == 1

    def test_over_dimension_rejected(self) -> None:
        big = _algebra(tuple(f"x{i}" for i in range(8)), ())
        line = _algebra(("t",), ())
        with pytest.raises(OperationDomainValidationError) as exc_info:
            lie_direct_sum(big, line, tuple(f"y{i}" for i in range(9)))
        assert exc_info.value.errors()[0]["type"] == "lie_algebra.direct_sum_basis"

    def test_duplicate_labels_rejected(self) -> None:
        line = _algebra(("e",), ())
        with pytest.raises(OperationDomainValidationError) as exc_info:
            lie_direct_sum(SL2, line, ("e", "f", "h", "e"))
        assert exc_info.value.errors()[0]["type"] == "lie_algebra.direct_sum_basis"

    @pytest.mark.parametrize("basis", (None, 4, 1.5, object()))
    def test_non_sequence_basis_is_typed_rejected(self, basis: object) -> None:
        line = _algebra(("t",), ())
        with pytest.raises(OperationDomainValidationError) as exc_info:
            lie_direct_sum(SL2, line, basis)  # type: ignore[arg-type]
        assert exc_info.value.errors()[0]["type"] == "lie_algebra.direct_sum_basis"


class TestIdealQuotientComposition:
    def test_request_paths_agree(self) -> None:
        from jacobian.math.lie_algebras._tools import _run_lie_ideal_check

        request = LieIdealRequest(algebra=HEISENBERG, candidate=HEISENBERG_CENTER)
        assert check_ideal(request.algebra, request.candidate) == _run_lie_ideal_check(
            request
        )

    def test_serialized_results_round_trip(self) -> None:

        ideal = check_ideal(HEISENBERG, HEISENBERG_CENTER)
        assert LieIdealCheckResult.model_validate_json(ideal.model_dump_json()) == ideal
        quotient = lie_quotient(HEISENBERG, HEISENBERG_CENTER, ("u", "v"))
        assert (
            LieQuotientResult.model_validate_json(quotient.model_dump_json())
            == quotient
        )

    def test_catalog_declares_both_operations_with_valid_examples(self) -> None:
        from jacobian.canonical import encode_strict_json
        from jacobian.math.lie_algebras._tools import TOOLS

        ideal_tools = {
            tool.operation_id: tool
            for tool in TOOLS
            if tool.operation_id == "lie_algebra.ideal.check"
        }
        assert set(ideal_tools) == {"lie_algebra.ideal.check"}
        ideal_tool = ideal_tools["lie_algebra.ideal.check"]
        assert ideal_tool.examples
        ideal_payload = LieIdealRequest.model_validate_json(
            encode_strict_json(ideal_tool.examples[0].input), strict=True
        )
        assert ideal_tool.run(ideal_payload) == check_ideal(
            ideal_payload.algebra, ideal_payload.candidate
        )

        quotient_tools = {
            tool.operation_id: tool
            for tool in TOOLS
            if tool.operation_id == "lie_algebra.quotient.compute"
        }
        assert set(quotient_tools) == {"lie_algebra.quotient.compute"}
        quotient_tool = quotient_tools["lie_algebra.quotient.compute"]
        assert quotient_tool.examples
        quotient_payload = LieQuotientRequest.model_validate_json(
            encode_strict_json(quotient_tool.examples[0].input), strict=True
        )
        assert quotient_tool.run(quotient_payload).quotient.basis == ("u", "v")
