from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.coding_theory_extended import (
    DualCodeRequest,
    GeneratorMatrix,
    PunctureRequest,
    ShortenRequest,
)
from jacobian.domains.coding_theory_extended.operations import (
    compute_dual_code,
    compute_puncture,
    compute_shorten,
)


def test_dual_code_repetition() -> None:
    """Dual of the [3,1] repetition code is the [3,2] even-weight code."""
    code = GeneratorMatrix(field_order=2, generator_matrix=((1, 1, 1),))
    result = compute_dual_code(DualCodeRequest(code=code))
    assert result.code_length == 3
    assert result.code_dimension == 1
    # The parity-check matrix should have 2 rows (dimension of dual)
    assert len(result.parity_check_matrix) == 2
    # Each row should be orthogonal to the generator row
    for row in result.parity_check_matrix:
        dot = sum(a * b for a, b in zip((1, 1, 1), row, strict=False)) % 2
        assert dot == 0


def test_dual_code_identity_matrix() -> None:
    """Dual of the identity code is itself (as parity-check)."""
    code = GeneratorMatrix(field_order=2, generator_matrix=((1, 0), (0, 1)))
    result = compute_dual_code(DualCodeRequest(code=code))
    assert result.code_length == 2
    # The nullspace of I_2 is empty (rank 2, no free columns)
    assert len(result.parity_check_matrix) == 0


def test_puncture_deletes_column() -> None:
    """Puncturing the repetition code at position 1 gives [[1, 1]]."""
    code = GeneratorMatrix(field_order=2, generator_matrix=((1, 1, 1),))
    result = compute_puncture(PunctureRequest(code=code, position=1))
    assert result.code_length == 2
    assert result.generator_matrix == ((1, 1),)
    assert result.method == "COLUMN_DELETION"


def test_puncture_preserves_multiple_rows() -> None:
    """Puncturing a 2-row code at position 0 removes the first column."""
    code = GeneratorMatrix(
        field_order=2,
        generator_matrix=((1, 0, 1), (0, 1, 1)),
    )
    result = compute_puncture(PunctureRequest(code=code, position=0))
    assert result.code_length == 2
    assert result.generator_matrix == ((0, 1), (1, 1))


def test_shorten_value_zero() -> None:
    """Shortening the repetition code at position 0 with value 0."""
    code = GeneratorMatrix(field_order=2, generator_matrix=((1, 1, 1),))
    result = compute_shorten(ShortenRequest(code=code, position=0, value=0))
    # The shortened code has no generator rows (the repetition code has no
    # codewords with 0 at position 0 that aren't the zero word)
    assert result.method == "COORDINATE_FIX_AND_DELETE"


def test_contract_rejects_nonprime_field() -> None:
    with pytest.raises(ValidationError, match="prime"):
        GeneratorMatrix(field_order=4, generator_matrix=((1, 0),))


def test_contract_rejects_non_rectangular_matrix() -> None:
    with pytest.raises(ValidationError, match="equal length"):
        GeneratorMatrix(field_order=2, generator_matrix=((1, 0), (1,)))


def test_contract_rejects_out_of_range_entries() -> None:
    with pytest.raises(ValidationError, match="residues"):
        GeneratorMatrix(field_order=2, generator_matrix=((2, 0),))


def test_contract_rejects_invalid_puncture_position() -> None:
    code = GeneratorMatrix(field_order=2, generator_matrix=((1, 1),))
    with pytest.raises(ValidationError, match="position"):
        PunctureRequest(code=code, position=5)
