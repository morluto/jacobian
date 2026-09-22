"""Exact Gamma0 trivial-character Sturm and coefficient transforms."""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import cast

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

from .kernel import NamedLevelOneModularForm
from .transform_models import SturmBoundResult

_MAX_TRANSFORM_PRECISION = 4_096
_MAX_HECKE_INDEX = 4_096


def _positive_int(value: object, *, location: tuple[str, ...], code: str) -> int:
    if type(value) is not int or value < 1:
        raise OperationDomainValidationError(
            location=location,
            code=code,
            message="value must be a positive integer",
        )
    return value


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


def _space(space: object) -> ModularFormSpace:
    if not isinstance(space, ModularFormSpace):
        raise OperationDomainValidationError(
            location=("space",),
            code="modular_form.space_type",
            message="space must be a modular-form space value",
        )
    try:
        group = space.group
        character = space.character
        domain = space.coefficient_domain
        kind = space.kind
        level = space.level
        weight = space.weight
    except AttributeError as exc:
        raise OperationDomainValidationError(
            location=("space",),
            code="modular_form.space_shape",
            message="space must carry a canonical Gamma0 parent",
        ) from exc
    if (
        group != "GAMMA0"
        or character != "TRIVIAL"
        or domain != "QQ"
        or kind not in {"M", "S"}
        or type(level) is not int
        or type(weight) is not int
    ):
        raise OperationDomainValidationError(
            location=("space",),
            code="modular_form.unsupported_space",
            message="only trivial-character Gamma0(QQ) spaces are supported",
        )
    if not 1 <= level <= 10_000:
        raise OperationResourceAdmissionError(
            location=("space", "level"),
            code="modular_form.level_bound",
            message="Gamma0 level exceeds the bounded exact space envelope",
        )
    if not 0 <= weight <= 1_000_000:
        raise OperationResourceAdmissionError(
            location=("space", "weight"),
            code="modular_form.weight_bound",
            message="modular-form weight exceeds the bounded exact space envelope",
        )
    return space


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
    # For Gamma0(N), the p=2 and p=3 factors are neutral at exponent one;
    # only 4 | N and 9 | N force the corresponding elliptic count to zero.
    e2 = 0 if n % 4 == 0 else 1
    e3 = 0 if n % 9 == 0 else 1
    for p in primes:
        if p != 2:
            e2 *= 2 if p % 4 == 1 else 0
        if p != 3:
            e3 *= 2 if p % 3 == 1 else 0
    cusps = sum(int(sp.totient(gcd(d, n // d))) for d in range(1, n + 1) if n % d == 0)
    return e2, e3, cusps


def sturm_bound(space):
    _space(space)
    return SturmBoundResult(
        space=space,
        index=_index(space.level),
        bound=(space.weight * _index(space.level)) // 12,
    )


def _expansion(value: object) -> ModularQExpansion:
    if not isinstance(value, ModularQExpansion):
        raise OperationDomainValidationError(
            location=("expansion",),
            code="modular_form.expansion_type",
            message="expansion must be a modular q-expansion value",
        )
    try:
        series = value.q_expansion
        weight = value.weight
        parent = value.space
    except AttributeError as exc:
        raise OperationDomainValidationError(
            location=("expansion",),
            code="modular_form.expansion_shape",
            message="expansion must carry a canonical q-prefix",
        ) from exc
    if (
        not isinstance(series, TruncatedSeries)
        or series.variable != "q"
        or type(series.truncation_order) is not int
        or series.truncation_order < 1
        or not isinstance(series.coefficients, tuple)
        or len(series.coefficients) != series.truncation_order
        or type(weight) is not int
    ):
        raise OperationDomainValidationError(
            location=("expansion",),
            code="modular_form.expansion_shape",
            message="expansion must carry a canonical q-prefix",
        )
    space = _space(parent)
    if space.weight != weight:
        raise OperationDomainValidationError(
            location=("expansion", "space"),
            code="modular_form.expansion_parent",
            message="expansion weight must match its space parent",
        )
    return value


def _result(expansion: ModularQExpansion, coeffs, weight=None, space=None):
    expansion = _expansion(expansion)
    if not isinstance(coeffs, list) or len(coeffs) > _MAX_TRANSFORM_PRECISION:
        raise OperationResourceAdmissionError(
            location=("output_precision",),
            code="modular_form.output_bound",
            message="transform output exceeds the bounded precision envelope",
        )
    series = TruncatedSeries(
        variable="q",
        truncation_order=len(coeffs),
        coefficients=tuple(CanonicalRational.from_fraction(c) for c in coeffs),
    )
    return ModularQExpansion(
        space=space if space is not None else expansion.space,
        weight=expansion.weight if weight is None else weight,
        q_expansion=series,
        basis_id=expansion.basis_id,
    )


def _require_precision(expansion: ModularQExpansion, p: object, scale: int) -> None:
    p = _positive_int(
        p, location=("output_precision",), code="modular_form.transform_bounds"
    )
    if p > _MAX_TRANSFORM_PRECISION:
        raise OperationResourceAdmissionError(
            location=("output_precision",),
            code="modular_form.output_bound",
            message="transform output exceeds the bounded precision envelope",
        )
    if scale * p > expansion.q_expansion.truncation_order:
        raise OperationDomainValidationError(
            location=("expansion", "q_expansion"),
            code="modular_form.insufficient_precision",
            message=f"source precision must be at least {scale} times the requested precision",
        )


def hecke(expansion, index, output_precision):
    expansion = _expansion(expansion)
    if not isinstance(expansion, ModularQExpansion):
        raise OperationDomainValidationError(
            location=("expansion",),
            code="modular_form.expansion_type",
            message="expansion must be a modular q-expansion value",
        )
    index = _positive_int(
        index, location=("index",), code="modular_form.transform_bounds"
    )
    output_precision = _positive_int(
        output_precision,
        location=("output_precision",),
        code="modular_form.transform_bounds",
    )
    if index > _MAX_HECKE_INDEX:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="modular_form.operator_bound",
            message="Hecke index exceeds the bounded transform envelope",
        )
    _space(expansion.space)
    if expansion.space.level != 1:
        raise OperationDomainValidationError(
            location=("expansion", "space", "level"),
            code="modular_form.hecke_level",
            message="the published Hecke formula is admitted only at level one",
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
    """Reject U_p until a formal q-series carrier can carry its codomain."""

    _expansion(expansion)
    raise OperationDomainValidationError(
        location=("expansion",),
        code="modular_form.u_requires_formal_series",
        message=(
            "U_p is not published for parent-bound modular forms; it requires "
            "a formal q-series carrier with an explicit codomain"
        ),
    )


def v_operator(expansion, prime, output_precision):
    expansion = _expansion(expansion)
    if not isinstance(expansion, ModularQExpansion):
        raise OperationDomainValidationError(
            location=("expansion",),
            code="modular_form.expansion_type",
            message="expansion must be a modular q-expansion value",
        )
    prime = _positive_int(
        prime, location=("prime",), code="modular_form.operator_prime"
    )
    output_precision = _positive_int(
        output_precision,
        location=("output_precision",),
        code="modular_form.transform_bounds",
    )
    _space(expansion.space)
    if prime > _MAX_HECKE_INDEX:
        raise OperationResourceAdmissionError(
            location=("prime",),
            code="modular_form.operator_bound",
            message="operator prime exceeds the bounded transform envelope",
        )
    if prime < 2 or any(prime % d == 0 for d in range(2, int(prime**0.5) + 1)):
        raise OperationDomainValidationError(
            location=("prime",),
            code="modular_form.operator_prime",
            message="the V operator index must be prime",
        )
    if output_precision > _MAX_TRANSFORM_PRECISION:
        raise OperationResourceAdmissionError(
            location=("output_precision",),
            code="modular_form.output_bound",
            message="transform output exceeds the bounded precision envelope",
        )
    if output_precision > expansion.q_expansion.truncation_order * prime:
        raise OperationDomainValidationError(
            location=("output_precision",),
            code="modular_form.precision_not_supported",
            message="V operator output exceeds represented source precision",
        )
    a = expansion.q_expansion.coefficients
    target_level = expansion.space.level * prime
    if target_level > 10_000:
        raise OperationResourceAdmissionError(
            location=("expansion", "space", "level"),
            code="modular_form.level_bound",
            message="V_p would exceed the bounded modular-form level envelope",
        )
    target_space = expansion.space.model_copy(update={"level": target_level})
    return _result(
        expansion,
        [
            a[n // prime].as_fraction() if n % prime == 0 else Fraction(0)
            for n in range(output_precision)
        ],
        space=target_space,
    )


def named_q_expansion(space, form, precision):
    from .operations import level_one_named_q_expansion

    space = _space(space)
    if type(form) is not str or form not in {"E4", "E6", "DELTA"}:
        raise OperationDomainValidationError(
            location=("form",),
            code="modular_form.named_unsupported",
            message="only E4, E6, and Delta are admitted in the level-one named slice",
        )
    precision = _positive_int(
        precision,
        location=("precision",),
        code="modular_form.precision_bounds",
    )
    if space.level != 1:
        raise OperationDomainValidationError(
            location=("space", "level"),
            code="modular_form.named_level",
            message="named forms are admitted only in the level-one slice",
        )
    expected_weight = {"E4": 4, "E6": 6, "DELTA": 12}[form]
    expected_kind = "S" if form == "DELTA" else "M"
    if space.weight != expected_weight:
        raise OperationDomainValidationError(
            location=("space", "weight"),
            code="modular_form.named_weight_mismatch",
            message="space weight must match the named form",
        )
    if space.kind != expected_kind:
        raise OperationDomainValidationError(
            location=("space", "kind"),
            code="modular_form.named_kind_mismatch",
            message="the named form is not a member of the requested space kind",
        )
    named = level_one_named_q_expansion(cast(NamedLevelOneModularForm, form), precision)
    return ModularQExpansion(
        space=space,
        weight=named.weight,
        q_expansion=named.q_expansion,
        basis_id=f"named:{form}",
    )


__all__ = ["hecke", "named_q_expansion", "sturm_bound", "u_operator", "v_operator"]
