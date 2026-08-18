"""Owner-local exact public API contract for combinatorics."""

from __future__ import annotations

import importlib


def test_public_manifest_is_exact() -> None:
    module = importlib.import_module("jacobian.math.combinatorics")
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
    assert tuple(module.__all__) == expected
    assert len(expected) == len(set(expected))
    assert all(not name.startswith("_") for name in expected)
    assert all(hasattr(module, name) for name in expected)
