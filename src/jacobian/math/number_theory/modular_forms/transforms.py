"""Exact Gamma0 trivial-character Sturm and coefficient transforms."""

from __future__ import annotations

from fractions import Fraction
from math import gcd

from jacobian._exact import CanonicalRational, require_bounded_rational
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

MAX_MODULAR_TRANSFORM_PRECISION = 4_096
MAX_MODULAR_OPERATOR_INDEX = 256


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
    if not isinstance(space, ModularFormSpace):
        raise OperationDomainValidationError(location=("space",), code="modular_form.space_type", message="space must be a modular-form space value")
    try:
        valid = (space.group == "GAMMA0" and space.character == "TRIVIAL" and space.coefficient_domain == "QQ" and type(space.level) is int and type(space.weight) is int and space.level >= 1 and space.weight >= 0)
    except (AttributeError, TypeError):
        valid = False
    if not valid:
        raise OperationDomainValidationError(location=("space",), code="modular_form.unsupported_space", message="only trivial-character Gamma0(QQ) spaces are supported")
    if space.level > 10_000:
        raise OperationResourceAdmissionError(
            location=("space", "level"),
            code="modular_form.level_bound",
            message="Gamma0 level exceeds the bounded exact space envelope",
        )


def _expansion(expansion):
    if not isinstance(expansion, ModularQExpansion):
        raise OperationDomainValidationError(location=("expansion",), code="modular_form.expansion_type", message="expansion must be a modular q-expansion value")
    try:
        series = expansion.q_expansion
        if type(expansion.weight) is not int or expansion.weight < 0 or not isinstance(series, TruncatedSeries) or series.variable != "q" or type(series.truncation_order) is not int or series.truncation_order < 1 or series.truncation_order > MAX_MODULAR_TRANSFORM_PRECISION * 8 or len(series.coefficients) != series.truncation_order:
            raise ValueError
        if any(not isinstance(value, CanonicalRational) for value in series.coefficients):
            raise ValueError
        for value in series.coefficients:
            require_bounded_rational(value, max_digits=4_096, label="q coefficient")
        _space(expansion.space)
        if expansion.space.weight != expansion.weight:
            raise ValueError
    except (AttributeError, TypeError, ValueError):
        raise OperationDomainValidationError(location=("expansion",), code="modular_form.expansion_structure", message="expansion must retain a canonical q-axis, precision, coefficients, and parent") from None
    return expansion


def _plain_int(value, location, code, message):
    if type(value) is not int:
        raise OperationDomainValidationError(location=location, code=code, message=message)


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
    if space.level != 1:
        raise OperationDomainValidationError(location=("space", "level"), code="modular_form.sturm_level", message="the published Sturm metadata is admitted only at level one")
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
    _expansion(expansion)
    _plain_int(index, ("index",), "modular_form.transform_bounds", "operator index must be an integer")
    _plain_int(output_precision, ("output_precision",), "modular_form.transform_bounds", "precision must be an integer")
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
    if index > MAX_MODULAR_OPERATOR_INDEX or output_precision > MAX_MODULAR_TRANSFORM_PRECISION:
        raise OperationResourceAdmissionError(location=(), code="modular_form.transform_envelope", message="Hecke index or output precision exceeds the admitted transform envelope")
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
    _expansion(expansion)
    _plain_int(prime, ("prime",), "modular_form.operator_prime", "the U operator index must be an integer")
    _plain_int(output_precision, ("output_precision",), "modular_form.transform_bounds", "precision must be an integer")
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
    if prime > MAX_MODULAR_OPERATOR_INDEX or output_precision > MAX_MODULAR_TRANSFORM_PRECISION:
        raise OperationResourceAdmissionError(location=(), code="modular_form.transform_envelope", message="U operator index or output precision exceeds the admitted transform envelope")
    _require_precision(expansion, output_precision * prime, 1)
    a = expansion.q_expansion.coefficients
    return _result(
        expansion, [a[prime * n].as_fraction() for n in range(output_precision)]
    )


def v_operator(expansion, prime, output_precision):
    _expansion(expansion)
    _plain_int(prime, ("prime",), "modular_form.operator_prime", "the V operator index must be an integer")
    _plain_int(output_precision, ("output_precision",), "modular_form.transform_bounds", "precision must be an integer")
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
    if prime > MAX_MODULAR_OPERATOR_INDEX or output_precision > MAX_MODULAR_TRANSFORM_PRECISION:
        raise OperationResourceAdmissionError(location=(), code="modular_form.transform_envelope", message="V operator index or output precision exceeds the admitted transform envelope")
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

    _space(space)
    _plain_int(precision, ("precision",), "modular_form.transform_bounds", "precision must be an integer")
    if space.level != 1 or type(form) is not str or form not in {"E4", "E6", "DELTA"}:
        raise OperationDomainValidationError(
            location=("space", "form"),
            code="modular_form.named_unsupported",
            message="only E4, E6, and Delta are admitted in the level-one named slice",
        )
    expected = {"E4": 4, "E6": 6, "DELTA": 12}[form]
    if space.weight != expected or space.kind != ("S" if form == "DELTA" else "M"):
        raise OperationDomainValidationError(location=("space",), code="modular_form.named_parent_mismatch", message="the space weight and kind must match the named form")
    if precision < 1:
        raise OperationDomainValidationError(location=("precision",), code="modular_form.transform_bounds", message="precision must be positive")
    named = level_one_named_q_expansion(form, precision)
    return ModularQExpansion(
        space=space,
        weight=named.weight,
        q_expansion=named.q_expansion,
        basis_id=f"named:{form}",
    )


__all__ = ["hecke", "named_q_expansion", "sturm_bound", "u_operator", "v_operator"]
