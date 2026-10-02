"""Exact ordered wheel rows with one final-model buffer and one CRT setup."""

from collections.abc import Iterator
from math import prod
from typing import Any

import pytest
from sympy import primerange
from sympy.ntheory.modular import crt1, crt2

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.prime_affine_forms import _kernel
from jacobian.math.number_theory.prime_affine_forms._residue_wheel import (
    PrimeTupleResidueWheelEnumeration,
)
from jacobian.math.number_theory.prime_affine_forms.operations import (
    enumerate_residue_wheel,
    residue_wheel,
)
from jacobian.math.number_theory.prime_affine_forms.values import (
    PrimeAffineTuple,
    PrimitiveIntegerAffineForm,
)


def source(*pairs: tuple[int, int]) -> PrimeAffineTuple:
    return PrimeAffineTuple(
        forms=tuple(
            PrimitiveIntegerAffineForm(form_id=f"f{i}", coefficient=a, constant=b)
            for i, (a, b) in enumerate(pairs)
        )
    )


def brute_rows(
    s: PrimeAffineTuple, primes: tuple[int, ...]
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    return tuple(
        (r, tuple(r % p for p in primes))
        for r in range(prod(primes))
        if all(
            (form.coefficient * r + form.constant) % p
            for p in primes
            for form in s.forms
        )
    )


@pytest.mark.parametrize(
    "pairs,primes",
    [
        (((1, 0),), ()),
        (((1, 0),), (2,)),
        (((3, 1),), (3,)),  # Every local residue survives.
        (((1, 0), (1, 1)), (2, 3, 5)),  # Obstructed.
        (((1, 0), (1, 2)), (2, 3, 5, 7)),
        (((-3, 2), (5, -7)), (2, 3, 5, 7)),
        (((15, 1),), (3, 5, 7)),  # Some moduli divide the coefficient.
        (((1, 0),), (2, 8191)),  # Non-interned large component values.
        (((1, 0),), (3, 17, 257)),  # Exactly 8192 rows.
    ],
)
def test_streamed_rows_equal_independent_ordered_enumeration(
    pairs: tuple[tuple[int, int], ...], primes: tuple[int, ...]
) -> None:
    s = source(*pairs)
    expected = brute_rows(s, primes)
    iterator = _kernel.iter_wheel_rows(s, primes)
    assert isinstance(iterator, Iterator)
    assert tuple(sorted(iterator)) == expected
    wheel = residue_wheel(s, primes)
    result = enumerate_residue_wheel(wheel)
    assert tuple((row.residue, row.components) for row in result.residues) == expected
    assert len(result.residues) == wheel.valid_count
    assert (
        PrimeTupleResidueWheelEnumeration.model_validate_json(result.model_dump_json())
        == result
    )


def test_one_maintained_setup_and_complete_non_tautological_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup, combine = crt1, crt2
    setups, rows = [], []

    def counted_setup(*args: Any, **kwargs: Any) -> Any:
        setups.append(args[0])
        return setup(*args, **kwargs)

    def counted_combine(*args: Any, **kwargs: Any) -> Any:
        result = combine(*args, **kwargs)
        rows.append(tuple(args[1]))
        assert all(
            int(result[0]) % p == c for p, c in zip(args[0], args[1], strict=True)
        )
        return result

    monkeypatch.setattr(_kernel, "crt1", counted_setup)
    monkeypatch.setattr(_kernel, "crt2", counted_combine)
    result = enumerate_residue_wheel(residue_wheel(source((1, 0)), (3, 17, 257)))
    assert len(setups) == 1
    assert len(rows) == len(result.residues) == 8192
    assert len(set(rows)) == 8192


@pytest.mark.parametrize("primes", [(), (8191,), (2, 3, 5)])
def test_empty_single_prime_and_obstruction_need_no_crt_setup(
    primes: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = crt1
    calls = []

    def counted_setup(*args: Any, **kwargs: Any) -> Any:
        calls.append(args[0])
        return setup(*args, **kwargs)

    monkeypatch.setattr(_kernel, "crt1", counted_setup)
    s = source((1, 0), (1, 1)) if len(primes) > 1 else source((1, 0))
    enumerate_residue_wheel(residue_wheel(s, primes))
    assert calls == []


def test_sparse_huge_modulus_never_requires_a_modulus_scan() -> None:
    primes = tuple(int(p) for p in primerange(2, 48))
    modulus = prod(primes)
    forms = []
    # Each form excludes one nonzero residue at one prime. At every other
    # prime its slope vanishes and its constant is 1, so only residue zero
    # survives at each prime. All coefficients/constants remain coprime.
    for p in primes:
        a = modulus // p
        for excluded in range(1, p):
            b = 1 + a * ((-a * excluded - 1) * pow(a, -1, p) % p)
            forms.append((a, b))
    s = source(*forms)
    wheel = residue_wheel(s, primes)
    assert modulus > 10**17
    assert wheel.valid_count == 1
    result = enumerate_residue_wheel(wheel)
    assert tuple((row.residue, row.components) for row in result.residues) == (
        (0, (0,) * len(primes)),
    )


@pytest.mark.parametrize("primes", [(8209,), (32771,), (2, 8209), (2, 32771)])
def test_admission_stays_before_lazy_backend_setup(
    primes: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    expand = _kernel.valid_residues
    calls = []

    def counted_expansion(*args: Any, **kwargs: Any) -> Any:
        calls.append(args[0])
        return expand(*args, **kwargs)

    monkeypatch.setattr(_kernel, "valid_residues", counted_expansion)
    with pytest.raises(OperationDomainValidationError):
        enumerate_residue_wheel(residue_wheel(source((1, 0)), primes))
    assert calls == []


def test_exact_cell_cap_preserves_shared_large_component_integers() -> None:
    from sympy.ntheory.modular import crt

    primes = (263, 269, 271, 277, 281, 283, 293)
    counts = (4, 4, 4, 4, 4, 4, 2)
    forms = []
    for i in range(291):
        combined = crt(
            primes, tuple(-(i % (p - c)) for p, c in zip(primes, counts, strict=True))
        )
        assert combined is not None
        forms.append((1, int(combined[0])))
    s = source(*forms)
    wheel = residue_wheel(s, primes)
    assert wheel.valid_count == 8192
    result = enumerate_residue_wheel(wheel)
    assert len(result.residues) * (len(primes) + 1) == 65536
    assert all(
        a.residue < b.residue
        for a, b in zip(result.residues, result.residues[1:], strict=False)
    )
    for axis, (p, c) in enumerate(zip(primes, counts, strict=True)):
        expected = set(range(p - c, p))
        assert min(expected) > 256
        assert {row.components[axis] for row in result.residues} == expected
        # This internal allocation property prevents regenerating one Python
        # integer per output cell instead of sharing the admitted local table.
        assert len({id(row.components[axis]) for row in result.residues}) == c
        assert all(row.residue % p == row.components[axis] for row in result.residues)
    assert (
        PrimeTupleResidueWheelEnumeration.model_validate_json(result.model_dump_json())
        == result
    )
