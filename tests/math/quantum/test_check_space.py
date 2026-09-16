"""Tests for exact stabilizer check-space canonicalization."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.quantum import (
    BinaryPauliRow,
    CheckSpaceCanonicalizeRequest,
    canonicalize_check_space,
)
from jacobian.math.quantum._tools import TOOLS


def _row(row_id: str, x: list[int], z: list[int]) -> BinaryPauliRow:
    return BinaryPauliRow(row_id=row_id, x_bits=tuple(x), z_bits=tuple(z))


class TestKnownAnswer:
    def test_bell_pair_checks_are_isotropic_rank_two(self) -> None:
        result = canonicalize_check_space(
            ("q0", "q1"),
            (_row("xx", [1, 1], [0, 0]), _row("zz", [0, 0], [1, 1])),
        )
        assert result.status == "ISOTROPIC_CHECK_SPACE"
        assert result.rank == 2
        assert [row.pivot for row in result.basis] == [0, 2]
        assert result.witness is None

    def test_single_z_check_rank_one(self) -> None:
        result = canonicalize_check_space(("q0",), (_row("z", [0], [1]),))
        assert result.status == "ISOTROPIC_CHECK_SPACE"
        assert result.rank == 1
        assert result.basis[0].pivot == 1

    def test_anticommuting_pair_witnessed(self) -> None:
        result = canonicalize_check_space(
            ("q0",),
            (_row("x", [1], [0]), _row("z", [0], [1])),
        )
        assert result.status == "NOT_ISOTROPIC"
        assert result.rank == 0
        assert result.basis == ()
        assert result.witness is not None
        assert {result.witness.first_row_id, result.witness.second_row_id} == {"x", "z"}
        assert result.witness.pairing == 1


class TestBoundary:
    def test_dependent_rows_reduce_rank(self) -> None:
        result = canonicalize_check_space(
            ("q0", "q1"),
            (
                _row("a", [1, 0], [0, 0]),
                _row("b", [1, 0], [0, 0]),
                _row("c", [0, 0], [0, 1]),
            ),
        )
        assert result.status == "ISOTROPIC_CHECK_SPACE"
        assert result.rank == 2

    def test_zero_row_contributes_no_pivot(self) -> None:
        result = canonicalize_check_space(
            ("q0",),
            (_row("i", [0], [0]), _row("z", [0], [1])),
        )
        assert result.status == "ISOTROPIC_CHECK_SPACE"
        assert result.rank == 1

    def test_three_qubit_ghz_checks(self) -> None:
        result = canonicalize_check_space(
            ("q0", "q1", "q2"),
            (
                _row("xxi", [1, 1, 0], [0, 0, 0]),
                _row("ixx", [0, 1, 1], [0, 0, 0]),
                _row("zzz", [0, 0, 0], [1, 1, 1]),
            ),
        )
        assert result.status == "ISOTROPIC_CHECK_SPACE"
        assert result.rank == 3


class TestAdversarial:
    def test_register_length_mismatch_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            canonicalize_check_space(("q0", "q1"), (_row("x", [1], [0]),))

    def test_duplicate_qubits_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            canonicalize_check_space(("q0", "q0"), (_row("x", [1, 0], [0, 0]),))

    def test_nonbinary_bits_rejected_at_value_boundary(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BinaryPauliRow(row_id="bad", x_bits=(2,), z_bits=(0,))

    def test_native_rejects_non_rows(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            canonicalize_check_space(("q0",), ("not-a-row",))  # type: ignore[arg-type]


class TestDefiningInvariant:
    def test_pairing_zero_iff_commute(self) -> None:
        # Y ~ (1,1) anticommutes with X ~ (1,0): pairing 1*0+1*1 = 1.
        result = canonicalize_check_space(
            ("q0",),
            (_row("x", [1], [0]), _row("y", [1], [1])),
        )
        assert result.status == "NOT_ISOTROPIC"

    def test_row_reordering_preserves_canonical_basis(self) -> None:
        rows = (
            _row("xx", [1, 1], [0, 0]),
            _row("zz", [0, 0], [1, 1]),
        )
        first = canonicalize_check_space(("q0", "q1"), rows)
        second = canonicalize_check_space(("q0", "q1"), rows[::-1])
        assert first.status == second.status == "ISOTROPIC_CHECK_SPACE"
        assert [(r.pivot, r.x_bits, r.z_bits) for r in first.basis] == [
            (r.pivot, r.x_bits, r.z_bits) for r in second.basis
        ]
        assert first.rank == second.rank == 2

    def test_rref_spans_input_rows(self) -> None:
        rows = (
            _row("a", [1, 1], [0, 1]),
            _row("b", [0, 1], [1, 0]),
        )
        result = canonicalize_check_space(("q0", "q1"), rows)
        assert result.status == "ISOTROPIC_CHECK_SPACE"
        # Every input row is a GF(2) combination of the basis rows.

        span = set()
        for mask in range(1 << len(result.basis)):
            combo = [0] * 4
            for i, brow in enumerate(result.basis):
                if mask >> i & 1:
                    bits = (*brow.x_bits, *brow.z_bits)
                    combo = [(a + b) % 2 for a, b in zip(combo, bits, strict=True)]
            span.add(tuple(combo))
        for row in rows:
            assert (*row.x_bits, *row.z_bits) in span


class TestNativeVsCatalogParity:
    def test_catalog_entry_matches_native(self) -> None:
        request = CheckSpaceCanonicalizeRequest(
            qubit_ids=("q0", "q1"),
            generators=(
                _row("xx", [1, 1], [0, 0]),
                _row("zz", [0, 0], [1, 1]),
            ),
        )
        tool = next(
            tool
            for tool in TOOLS
            if tool.operation_id == "stabilizer.check_space.canonicalize"
        )
        assert tool.run(request) == canonicalize_check_space(
            ("q0", "q1"),
            (
                _row("xx", [1, 1], [0, 0]),
                _row("zz", [0, 0], [1, 1]),
            ),
        )

    def test_operation_is_published(self) -> None:
        assert "stabilizer.check_space.canonicalize" in {
            tool.operation_id for tool in TOOLS
        }
