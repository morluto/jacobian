from __future__ import annotations

import pytest

from jacobian.math.finite_fields.paley_tournament.operations import (
    construct_paley_tournament,
)


def test_f3() -> None:
    """F_3: residues are {1}. Tournament: 0->1, 1->2, 2->0."""
    result = construct_paley_tournament(3)
    assert result.field_order == 3
    assert len(result.vertices) == 3
    assert result.edge_count == 3


def test_f7() -> None:
    """F_7: residues are {1,2,4}. Tournament has 7*6/2 = 21 edges."""
    result = construct_paley_tournament(7)
    assert result.edge_count == 21  # C(7,2) = 21


def test_f11() -> None:
    """F_11: residues are {1,3,4,5,9}. Tournament has C(11,2) = 55 edges."""
    result = construct_paley_tournament(11)
    assert result.edge_count == 55


def test_invalid_order() -> None:
    with pytest.raises(ValueError):
        construct_paley_tournament(5)  # 5 ≡ 1 (mod 4), not 3


def test_result_preserves_order() -> None:
    result = construct_paley_tournament(3)
    assert result.field_order == 3
