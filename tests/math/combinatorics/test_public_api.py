"""Exact public API contract for jacobian.math.combinatorics."""

from __future__ import annotations

from jacobian.math import combinatorics


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the combinatorics public API."""
    expected = (
        "IndexedRecurrenceResidual",
        "PolynomialCoefficientRecurrenceTableRequest",
        "PolynomialCoefficientRecurrenceTableResult",
        "bell_number",
        "bernoulli_number",
        "catalan_number",
        "derangement_number",
        "double_factorial",
        "fibonacci_number",
        "integer_partitions",
        "lucas_number",
        "motzkin_number",
        "partition_number",
        "recurrence_table_residuals",
        "stirling_first",
        "stirling_second",
    )
    assert tuple(combinatorics.__all__) == expected
    assert len(combinatorics.__all__) == len(set(combinatorics.__all__))
    assert all(not name.startswith("_") for name in combinatorics.__all__)
    assert all(hasattr(combinatorics, name) for name in combinatorics.__all__)
