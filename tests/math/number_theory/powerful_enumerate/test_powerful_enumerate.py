"""Tests for bounded powerful-number enumeration."""

from __future__ import annotations

from sympy import factorint

from jacobian.math.number_theory._powerful_enumerate import (
    enumerate_powerful_numbers,
)
from jacobian.math.number_theory._powerful_enumerate_kernels import enumerate_powerful
from jacobian.math.number_theory._powerful_enumerate_models import (
    PowerfulEnumerateRequest,
)


def test_powerful_to_10() -> None:
    """On [1,10], the powerful integers are exactly 1, 4, 8, 9."""
    result = enumerate_powerful(10)
    assert result == [1, 4, 8, 9]


def test_12_not_powerful() -> None:
    """12 = 2^2 * 3 is not powerful because exponent of 3 is 1."""
    result = enumerate_powerful(100)
    assert 12 not in result


def test_1_is_powerful() -> None:
    """1 is powerful by convention (no prime factors)."""
    result = enumerate_powerful(1)
    assert result == [1]


def test_all_results_are_powerful() -> None:
    """Every returned integer has all prime exponents >= 2."""
    result = enumerate_powerful(1000)
    for n in result:
        factors = factorint(n)
        assert all(e >= 2 for e in factors.values()), f"{n} is not powerful"


def test_sorted_and_unique() -> None:
    """The family is sorted in increasing order with no duplicates."""
    result = enumerate_powerful(500)
    assert result == sorted(result)
    assert len(result) == len(set(result))


def test_completeness_via_factorization() -> None:
    """Cross-check by independently scanning the interval."""
    cutoff = 200
    result = enumerate_powerful(cutoff)
    expected = [
        n for n in range(1, cutoff + 1) if all(e >= 2 for e in factorint(n).values())
    ]
    assert result == expected


def test_operation_round_trip() -> None:
    """The operation model round-trips through the kernel."""
    request = PowerfulEnumerateRequest(cutoff=100)
    result = enumerate_powerful_numbers(request)
    assert result.cutoff == 100
    assert result.count == len(result.family)
    assert int(result.family[0]) == 1
    assert "4" in result.family
    assert "12" not in result.family


def test_square_cube_representation() -> None:
    """Each powerful integer n > 1 has a canonical n = a^2 * b^3 with b squarefree."""
    result = enumerate_powerful(500)
    for n in result:
        if n == 1:
            continue
        found = False
        for b in range(1, n + 1):
            b3 = b**3
            if b3 > n:
                break
            if n % b3 != 0:
                continue
            a2 = n // b3
            a = int(a2**0.5 + 0.5)
            if a * a == a2:
                # Verify b is squarefree
                b_factors = factorint(b)
                assert all(e == 1 for e in b_factors.values()), f"b={b} not squarefree"
                found = True
                break
        assert found, f"No square-cube representation for {n}"


def test_operation_is_discoverable_with_one_executable_example() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    operation = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "integer.powerful.enumerate"
    )
    assert len(operation.examples) == 1
    example = operation.examples[0]
    request = operation.request_type.model_validate(example.input)
    result = operation.run(request)
    assert result.cutoff == 100
    assert "1" in result.family
    assert "4" in result.family
    assert "8" in result.family
    assert "9" in result.family
    assert "12" not in result.family
    assert result.count == len(result.family)
