"""Exact Gamma0 trivial-character Sturm and coefficient transforms."""

from __future__ import annotations

from fractions import Fraction
from math import gcd

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormSpace,
    ModularQExpansion,
)
from jacobian.math.polynomials.series._models import TruncatedSeries

from .transform_models import SturmBoundResult


def _index(level: int) -> int:
    n = level
    result = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            result = result // p * (p + 1)
            while n % p == 0:
                n //= p
        p += 1
    if n > 1:
        result = result // n * (n + 1)
    return result


def _space(space):
    if (
        not isinstance(space, ModularFormSpace)
        or space.group != "GAMMA0"
        or space.character != "TRIVIAL"
        or space.coefficient_domain != "QQ"
    ):
        raise OperationDomainValidationError(
            location=("space",),
            code="modular_form.unsupported_space",
            message="only trivial-character Gamma0(QQ) spaces are supported",
        )
    if space.level > 10_000:
        raise OperationResourceAdmissionError(
            location=("space", "level"),
            code="modular_form.level_bound",
            message="Gamma0 level exceeds the bounded exact space envelope",
        )


def _euler_counts(n):
    import sympy as sp

    primes = []
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            primes.append(p)
            while m % p == 0:
                m //= p
        p += 1
    if m > 1:
        primes.append(m)
    e2 = 0 if n % 4 == 0 else 1
    e3 = 0 if n % 9 == 0 else 1
    for p in primes:
        if p == 2:
            e2 *= 0
        elif p % 4 == 1:
            e2 *= 2
        else:
            e2 *= 0
        if p == 3:
            e3 *= 0
        elif p % 3 == 1:
            e3 *= 2
        else:
            e3 *= 0
    cusps = sum(int(sp.totient(gcd(d, n // d))) for d in range(1, n + 1) if n % d == 0)
    return e2, e3, cusps


def sturm_bound(space):
    _space(space)
    return SturmBoundResult(
        space=space,
        index=_index(space.level),
        bound=(space.weight * _index(space.level)) // 12,
    )


def _result(exp, coeffs, weight=None, space=None):
    series = TruncatedSeries(
        variable="q",
        truncation_order=len(coeffs),
        coefficients=tuple(CanonicalRational.from_fraction(c) for c in coeffs),
    )
    return ModularQExpansion(
        space=space or exp.space,
        weight=exp.weight if weight is None else weight,
        q_expansion=series,
        basis_id=exp.basis_id,
    )


def _require_precision(exp, p, scale):
    if scale * p > exp.q_expansion.truncation_order:
        raise OperationDomainValidationError(
            location=("expansion", "q_expansion"),
            code="modular_form.insufficient_precision",
            message=f"source precision must be at least {scale} times the requested precision",
        )


def hecke(expansion, index, output_precision):
    if expansion.space is None:
        raise OperationDomainValidationError(
            location=("expansion", "space"),
            code="modular_form.missing_space_parent",
            message="a Hecke transform requires a bound modular-form space",
        )
    _space(expansion.space)
    if expansion.space.level != 1:
        raise OperationDomainValidationError(
            location=("expansion", "space", "level"),
            code="modular_form.hecke_level",
            message="the published Hecke formula is admitted only at level one",
        )
    if index < 1 or output_precision < 1:
        raise OperationDomainValidationError(
            location=(),
            code="modular_form.transform_bounds",
            message="operator index and precision must be positive",
        )
    _require_precision(expansion, output_precision, index)
    a = [c.as_fraction() for c in expansion.q_expansion.coefficients]
    k = expansion.weight
    out = []
    for m in range(output_precision):
        value = Fraction(0)
        for d in range(1, gcd(index, m) + 1):
            if index % d == 0 and m % d == 0:
                value += Fraction(d ** (k - 1)) * a[(index * m) // (d * d)]
        out.append(value)
    return _result(expansion, out)


def u_operator(expansion, prime, output_precision):
    if expansion.space is None:
        raise OperationDomainValidationError(
            location=("expansion", "space"),
            code="modular_form.missing_space_parent",
            message="a U transform requires a bound modular-form space",
        )
    _space(expansion.space)
    if prime < 2 or any(prime % d == 0 for d in range(2, int(prime**0.5) + 1)):
        raise OperationDomainValidationError(
            location=("prime",),
            code="modular_form.operator_prime",
            message="the U operator index must be prime",
        )
    _require_precision(expansion, output_precision * prime, 1)
    a = expansion.q_expansion.coefficients
    return _result(
        expansion, [a[prime * n].as_fraction() for n in range(output_precision)]
    )


def v_operator(expansion, prime, output_precision):
    if expansion.space is None:
        raise OperationDomainValidationError(
            location=("expansion", "space"),
            code="modular_form.missing_space_parent",
            message="a V transform requires a bound modular-form space",
        )
    _space(expansion.space)
    if prime < 2 or any(prime % d == 0 for d in range(2, int(prime**0.5) + 1)):
        raise OperationDomainValidationError(
            location=("prime",),
            code="modular_form.operator_prime",
            message="the V operator index must be prime",
        )
    if output_precision > expansion.q_expansion.truncation_order * prime:
        raise OperationDomainValidationError(
            location=("output_precision",),
            code="modular_form.precision_not_supported",
            message="V operator output exceeds represented source precision",
        )
    a = expansion.q_expansion.coefficients
    return _result(
        expansion,
        [
            a[n // prime].as_fraction() if n % prime == 0 else Fraction(0)
            for n in range(output_precision)
        ],
    )


def named_q_expansion(space, form, precision):
    from .operations import level_one_named_q_expansion

    if space.level != 1 or form not in {"E4", "E6", "DELTA"}:
        raise OperationDomainValidationError(
            location=("space", "form"),
            code="modular_form.named_unsupported",
            message="only E4, E6, and Delta are admitted in the level-one named slice",
        )
    named = level_one_named_q_expansion(form, precision)
    return ModularQExpansion(
        space=space,
        weight=named.weight,
        q_expansion=named.q_expansion,
        basis_id=f"named:{form}",
    )


__all__ = ["hecke", "named_q_expansion", "sturm_bound", "u_operator", "v_operator"]
