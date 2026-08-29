"""Exact public API contract for jacobian.math.matrices.combinatorial."""

from __future__ import annotations

from jacobian.math.matrices import combinatorial as combinatorial_matrices


def test_exact_public_api_symbols() -> None:
    expected = (
        "HadamardMatrix",
        "SignMatrix",
        "determinant_profile",
        "gram_profile",
        "kronecker",
        "normalize",
        "recognize_hadamard",
        "sign_profile",
        "sylvester",
    )
    assert tuple(combinatorial_matrices.__all__) == expected
    assert len(combinatorial_matrices.__all__) == len(
        set(combinatorial_matrices.__all__)
    )
    assert all(not name.startswith("_") for name in combinatorial_matrices.__all__)
    assert all(
        hasattr(combinatorial_matrices, name) for name in combinatorial_matrices.__all__
    )
