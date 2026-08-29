from __future__ import annotations

from jacobian.math.combinatorics.additive.cyclic_prefix_sum.operations import (
    compute_cyclic_prefix_sum_residue_profile,
)
from jacobian.math.combinatorics.additive.values import IndexedIntegerSequence


def _sequence(*values: int) -> IndexedIntegerSequence:
    return IndexedIntegerSequence(items=tuple(str(value) for value in values))


def test_fixture_z5_113() -> None:
    """In Z/5Z, sequence (1,1,3) has prefix residues 1,2,0."""
    result = compute_cyclic_prefix_sum_residue_profile(_sequence(1, 1, 3), "5")
    residue_map = {r.residue: r.positions for r in result.rows}
    assert residue_map["1"] == (1,)
    assert residue_map["2"] == (2,)
    assert residue_map["0"] == (3,)


def test_replay_residues() -> None:
    """Replay: each position's prefix sum matches its residue."""
    seq = (3, 7, 2, 5, 1)
    m = 4
    result = compute_cyclic_prefix_sum_residue_profile(_sequence(*seq), str(m))
    for row in result.rows:
        for pos in row.positions:
            prefix_sum = sum(seq[:pos]) % m
            assert str(prefix_sum) == row.residue


def test_collision_classes() -> None:
    """Positions with equal prefix residues are grouped."""
    result = compute_cyclic_prefix_sum_residue_profile(_sequence(1, 2, 3), "6")
    for row in result.rows:
        positions = list(row.positions)
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                assert positions[i] < positions[j]


def test_empty_sequence() -> None:
    """Empty sequence has no rows."""
    result = compute_cyclic_prefix_sum_residue_profile(_sequence(), "5")
    assert len(result.rows) == 0


def test_single_element() -> None:
    """Single element has one row."""
    result = compute_cyclic_prefix_sum_residue_profile(_sequence(2), "5")
    assert len(result.rows) == 1
    assert result.rows[0].residue == "2"
    assert result.rows[0].positions == (1,)


def test_total_positions() -> None:
    """Total positions across all rows equals sequence length."""
    seq = (1, 2, 3, 4, 5)
    result = compute_cyclic_prefix_sum_residue_profile(_sequence(*seq), "7")
    total = sum(len(r.positions) for r in result.rows)
    assert total == len(seq)


def test_large_modulus() -> None:
    """Large modulus with small sequence: only few occupied residues."""
    result = compute_cyclic_prefix_sum_residue_profile(_sequence(1, 2, 3), "1000")
    assert len(result.rows) == 3  # all distinct residues
    residues = {r.residue for r in result.rows}
    assert residues == {"1", "3", "6"}


def test_zero_modulus_not_applicable() -> None:
    """Modulus 1 collapses everything to residue 0."""
    result = compute_cyclic_prefix_sum_residue_profile(_sequence(5, 7, 3), "1")
    assert len(result.rows) == 1
    assert result.rows[0].residue == "0"
    assert result.rows[0].positions == (1, 2, 3)


def test_result_preserves_modulus() -> None:
    """Result retains the modulus."""
    result = compute_cyclic_prefix_sum_residue_profile(_sequence(1, 2, 3), "7")
    assert result.modulus == "7"


def test_native_admission_rejects_nonpositive_modulus() -> None:
    """Native execution rejects modulus zero through the domain channel."""
    import pytest

    with pytest.raises(ValueError, match="modulus must be positive"):
        compute_cyclic_prefix_sum_residue_profile(_sequence(1), "0")


def test_native_admission_rejects_noncanonical_sequence() -> None:
    """Native execution uses the shared canonical indexed sequence value."""
    import pytest

    with pytest.raises(ValueError, match="indexed integer sequence"):
        compute_cyclic_prefix_sum_residue_profile((1, 2), "5")  # type: ignore[arg-type]
