"""Tests for divisibility edge profiles."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._divisibility_edge_profile import (
    compute_divisibility_edge_profile,
    divisibility_edge_profile,
)
from jacobian.math.number_theory._divisibility_edge_profile_models import (
    DivisibilityEdgeProfileRequest,
    DivisibilityEdgeProfileResult,
)


def _edges(values: list[str]) -> dict:
    request = DivisibilityEdgeProfileRequest(values=tuple(values))
    result = compute_divisibility_edge_profile(request)
    return {(e.source, e.target): e for e in result.edges}


def test_fixture_24612() -> None:
    """For (2,4,6,12), the complete proper-divisibility rows are correct."""
    edges = _edges(["2", "4", "6", "12"])
    # 2 -> 4: quotient 2, LPF 2
    assert edges[("2", "4")].quotient == "2"
    assert edges[("2", "4")].least_prime_factor == "2"
    # 2 -> 6: quotient 3, LPF 3
    assert edges[("2", "6")].quotient == "3"
    assert edges[("2", "6")].least_prime_factor == "3"
    # 2 -> 12: quotient 6, LPF 2
    assert edges[("2", "12")].quotient == "6"
    assert edges[("2", "12")].least_prime_factor == "2"
    # 4 -> 12: quotient 3, LPF 3
    assert edges[("4", "12")].quotient == "3"
    assert edges[("4", "12")].least_prime_factor == "3"
    # 6 -> 12: quotient 2, LPF 2
    assert edges[("6", "12")].quotient == "2"
    assert edges[("6", "12")].least_prime_factor == "2"


def test_non_edge_absent() -> None:
    """4 does not divide 6, so edge 4->6 is absent."""
    edges = _edges(["2", "4", "6"])
    assert ("4", "6") not in edges
    assert ("6", "4") not in edges


def test_no_reflexive_edges() -> None:
    """No edge connects a value to itself."""
    edges = _edges(["2", "4", "8"])
    for a, b in edges:
        assert a != b


def test_lpf_is_prime() -> None:
    """Every least_prime_factor is prime."""
    from sympy import isprime

    edges = _edges(["1", "2", "3", "6", "12", "24"])
    for edge in edges.values():
        assert isprime(int(edge.least_prime_factor))


def test_lpf_divides_quotient() -> None:
    """The LPF divides the quotient."""
    edges = _edges(["1", "2", "4", "6", "12", "24"])
    for edge in edges.values():
        assert int(edge.quotient) % int(edge.least_prime_factor) == 0


def test_quotient_reconstructs() -> None:
    """Every quotient reconstructs b = a * quotient."""
    edges = _edges(["2", "4", "6", "12"])
    for (a, b), edge in edges.items():
        assert int(a) * int(edge.quotient) == int(b)


def test_native_rejects_empty_source_set() -> None:
    """Native and wire callers share the non-empty source-set admission."""
    with pytest.raises(ValueError, match="at least one"):
        divisibility_edge_profile(())


@pytest.mark.parametrize("values", [(True, "2"), ("02",)])
def test_native_rejects_noncanonical_values(values: tuple[object, ...]) -> None:
    """Native inputs use the same strict canonical domain as wire requests."""
    with pytest.raises(ValueError):
        divisibility_edge_profile(values)  # type: ignore[arg-type]


def test_native_rejects_oversized_value_before_parsing() -> None:
    """The representation bound rejects huge strings during preflight."""
    with pytest.raises(ValueError, match="digit bound"):
        divisibility_edge_profile(("1" * (256 + 1),))


def test_native_rejects_values_beyond_worker_factorization_envelope() -> None:
    """The edge profile shares the direct factorization worker's 20-digit bound."""
    with pytest.raises(ValueError, match="digit bound"):
        divisibility_edge_profile(("1" * 21,))


def test_native_rejects_oversized_integer_before_formatting() -> None:
    """Huge Python integers are bounded before decimal conversion."""
    with pytest.raises(ValueError, match="digit bound"):
        divisibility_edge_profile((1 << 10_000_000,))


def test_resource_admission_belongs_to_operation_execution() -> None:
    """Wire parsing accepts shape-valid input; execution owns work rejection."""
    request = DivisibilityEdgeProfileRequest(
        values=tuple(str(value) for value in range(1, 501))
    )
    with pytest.raises(OperationDomainValidationError, match="factorization"):
        compute_divisibility_edge_profile(request)


@pytest.mark.parametrize(
    "values",
    [("0",), ("-1",), ("2", "2")],
)
def test_result_rejects_invalid_source_set(values: tuple[str, ...]) -> None:
    """Deserialized results retain positive, distinct source semantics."""
    with pytest.raises(ValueError):
        DivisibilityEdgeProfileResult(values=values, edges=())
