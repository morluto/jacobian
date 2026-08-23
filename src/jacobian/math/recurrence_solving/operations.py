"""Recurrence solving backed by SymPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from jacobian._exact import CanonicalRational
from jacobian.math.recurrence_solving._models import PrimeFieldRecurrence

__all__ = [
    "ClosedForm",
    "PrimeFieldRecurrence",
    "Recurrence",
    "berlekamp_massey",
    "closed_form",
    "find_recurrence",
]


@dataclass(frozen=True, slots=True)
class Recurrence:
    coefficients: tuple[CanonicalRational, ...]
    order: int
    status: Literal["FOUND", "NO_FITTING_RECURRENCE"]


_MAX_FIELD_PRIME = 10_000
_MAX_FIELD_SEQUENCE_LENGTH = 256


def _validate_berlekamp_inputs(sequence: list[int], prime: int) -> None:
    if type(prime) is not int or not 2 <= prime <= _MAX_FIELD_PRIME:
        raise ValueError(
            f"prime must be a prime number between 2 and {_MAX_FIELD_PRIME}"
        )
    from sympy import isprime

    if not isprime(prime):
        raise ValueError("prime must be a prime integer")
    if not 2 <= len(sequence) <= _MAX_FIELD_SEQUENCE_LENGTH:
        raise ValueError(
            f"sequence must have length between 2 and {_MAX_FIELD_SEQUENCE_LENGTH}"
        )
    for value in sequence:
        if type(value) is not int or not 0 <= value < prime:
            raise ValueError(
                "sequence values must be canonical residues modulo the prime"
            )


def _berlekamp_discrepancy(
    s: list[int], coeffs: list[int], prime: int, index: int
) -> int:
    discrepancy = s[index] % prime
    for j in range(1, len(coeffs)):
        discrepancy = (discrepancy + coeffs[j] * s[index - j]) % prime
    return discrepancy


def _berlekamp_combine(coeffs: list[int], shifted: list[int], prime: int) -> list[int]:
    if len(shifted) > len(coeffs):
        coeffs = coeffs + [0] * (len(shifted) - len(coeffs))
    elif len(shifted) < len(coeffs):
        shifted = shifted + [0] * (len(coeffs) - len(shifted))
    return [(c + s_coeff) % prime for c, s_coeff in zip(coeffs, shifted, strict=True)]


@dataclass(frozen=True, slots=True)
class ClosedForm:
    expression: str


def find_recurrence(sequence: tuple[CanonicalRational, ...]) -> Recurrence:
    import sympy

    from jacobian.math.recurrence_solving._models import RecurrenceFindRequest

    request = RecurrenceFindRequest(sequence=sequence)

    values = [sympy.Rational(*value.as_integer_ratio()) for value in request.sequence]
    for order in range(1, len(values)):
        coefficient_matrix = sympy.Matrix(
            [
                [values[index - offset] for offset in range(1, order + 1)]
                for index in range(order, len(values))
            ]
        )
        targets = sympy.Matrix(values[order:])
        try:
            solution, parameters = coefficient_matrix.gauss_jordan_solve(targets)
        except ValueError:
            continue
        if parameters:
            solution = solution.subs(dict.fromkeys(parameters, 0))
        return Recurrence(
            coefficients=tuple(
                CanonicalRational.from_integer_ratio(int(value.p), int(value.q))
                for value in solution
            ),
            order=order,
            status="FOUND",
        )
    return Recurrence(coefficients=(), order=0, status="NO_FITTING_RECURRENCE")


def closed_form(
    char_coeffs: tuple[CanonicalRational, ...],
    initial_values: tuple[CanonicalRational, ...],
) -> ClosedForm:
    import sympy

    from jacobian.math.recurrence_solving._models import ClosedFormRequest

    request = ClosedFormRequest(
        characteristic_coefficients=char_coeffs,
        initial_values=initial_values,
    )

    x = sympy.Symbol("x")
    n = sympy.Symbol("n", integer=True, nonnegative=True)
    char_poly_coeffs = [
        sympy.Rational(*value.as_integer_ratio())
        for value in request.characteristic_coefficients
    ]
    char_poly = sum(
        c * x ** (len(char_poly_coeffs) - 1 - i) for i, c in enumerate(char_poly_coeffs)
    )
    roots = sympy.Poly(char_poly, x).all_roots()
    zero_root_multiplicity = sum(root == 0 for root in roots)
    nonzero_roots = list(dict.fromkeys(root for root in roots if root != 0))
    basis = [sympy.KroneckerDelta(index, n) for index in range(zero_root_multiplicity)]
    basis.extend(
        n**power * root**n
        for root in nonzero_roots
        for power in range(roots.count(root))
    )
    init = [
        sympy.Rational(*value.as_integer_ratio()) for value in request.initial_values
    ]
    a = sympy.Matrix([[term.subs(n, i) for term in basis] for i in range(len(basis))])
    b = sympy.Matrix(init)
    consts = a.solve(b)
    expr = sum(c * term for c, term in zip(consts, basis, strict=True))
    return ClosedForm(expression=str(sympy.simplify(expr)))


def berlekamp_massey(sequence: list[int], prime: int) -> PrimeFieldRecurrence:
    """Return the minimal LFSR over ``GF(p)`` via Berlekamp-Massey.

    Returns a :class:`PrimeFieldRecurrence` containing the field ``prime``
    and the coefficient tuple ``(c_1, ..., c_L)`` such that the equation
    ``s_n = c_1 s_{n-1} + ... + c_L s_{n-L}`` (mod p) holds for every index
    of the supplied prefix with ``L <= n < len(sequence)``, with ``L``
    minimal. The algorithm observes only this finite prefix, so the value
    establishes nothing about terms at or beyond ``len(sequence)``. The
    returned value retains its field; ``[1, 1]`` over ``GF(2)`` and
    ``GF(7)`` are distinct values. An order-zero ``FOUND`` result
    represents the all-zero sequence.
    """
    _validate_berlekamp_inputs(sequence, prime)
    s = [int(x) for x in sequence]
    n = len(s)
    coeffs = [1]  # C(x) = 1
    b = [1]  # B(x) = 1
    length = 0
    m = 1
    last_discrepancy = 1

    for i in range(n):
        discrepancy = _berlekamp_discrepancy(s, coeffs, prime, i)
        if discrepancy == 0:
            m += 1
            continue
        factor = (discrepancy * pow(last_discrepancy, prime - 2, prime)) % prime
        shifted = [0] * m + [(-factor * x) % prime for x in b]
        if 2 * length <= i:
            temp = list(coeffs)
            coeffs = _berlekamp_combine(coeffs, shifted, prime)
            length = i + 1 - length
            b = temp
            last_discrepancy = discrepancy
            m = 1
        else:
            coeffs = _berlekamp_combine(coeffs, shifted, prime)
            m += 1

    # The connection polynomial is C(x) = 1 + c_1 x + ... + c_L x^L.
    # The recurrence is s_n = -c_1 s_{n-1} - ... - c_L s_{n-L}, so the
    # recurrence coefficients are [-c_1, ..., -c_L].
    if length == 0:
        return PrimeFieldRecurrence(
            prime=prime, coefficients=(), order=0, status="FOUND"
        )
    recurrence = tuple((-coeffs[j]) % prime for j in range(1, length + 1))
    return PrimeFieldRecurrence(
        prime=prime, coefficients=recurrence, order=length, status="FOUND"
    )
