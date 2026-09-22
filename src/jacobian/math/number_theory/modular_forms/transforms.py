"""Exact Gamma0 trivial-character Sturm and coefficient transforms."""

from __future__ import annotations

from fractions import Fraction
from math import gcd, isqrt
from typing import Literal, cast

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.modular_forms.values import (
    MAX_GAMMA0_OPERATION_LEVEL,
    MAX_MODULAR_FORM_WEIGHT,
    MAX_Q_TRANSFORM_COEFFICIENT_DIGITS,
    MAX_Q_TRANSFORM_OUTPUT_PRECISION,
    MAX_Q_TRANSFORM_SOURCE_ORDER,
    ModularFormSpace,
    ModularQExpansion,
)
from jacobian.math.polynomials.series._models import TruncatedSeries

from .transform_models import SturmBoundResult


def _strict_positive_int(value: object, location: tuple[str, ...], code: str) -> int:
    if type(value) is not int or value < 1:
        raise OperationDomainValidationError(
            location=location,
            code=code,
            message="value must be a positive integer",
        )
    return value


_MISSING = object()


def _canonical_source_coefficient(value: object) -> CanonicalRational:
    """Revalidate a scalar carrier, including native ``model_construct`` values."""

    if not isinstance(value, CanonicalRational):
        raise OperationDomainValidationError(
            location=("expansion", "q_expansion", "coefficients"),
            code="modular_form.source_coefficient_type",
            message="q-expansion coefficients must be canonical rationals",
        )
    num = getattr(value, "num", _MISSING)
    den = getattr(value, "den", _MISSING)
    if type(num) is not int or type(den) is not int or den <= 0:
        raise OperationDomainValidationError(
            location=("expansion", "q_expansion", "coefficients"),
            code="modular_form.source_coefficient_value",
            message="q-expansion coefficients must have a positive denominator",
        )
    try:
        rational = Fraction(num, den)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise OperationDomainValidationError(
            location=("expansion", "q_expansion", "coefficients"),
            code="modular_form.source_coefficient_value",
            message="q-expansion coefficients must be valid rationals",
        ) from error
    if (num, den) != (rational.numerator, rational.denominator):
        raise OperationDomainValidationError(
            location=("expansion", "q_expansion", "coefficients"),
            code="modular_form.source_coefficient_value",
            message="q-expansion coefficients must be reduced canonical rationals",
        )
    try:
        require_bounded_rational(
            value,
            max_digits=MAX_Q_TRANSFORM_COEFFICIENT_DIGITS,
            label="q-transform source coefficient",
        )
    except ValueError as error:
        raise OperationResourceAdmissionError(
            location=("expansion", "q_expansion", "coefficients"),
            code="modular_form.source_coefficient_bound",
            message=str(error),
        ) from error
    return value


def _expansion(expansion: object) -> ModularQExpansion:
    """Re-admit a q-prefix before any transform indexes or copies it."""

    if not isinstance(expansion, ModularQExpansion):
        raise OperationDomainValidationError(
            location=("expansion",),
            code="modular_form.expansion_type",
            message="expansion must be a modular q-expansion value",
        )
    weight = getattr(expansion, "weight", _MISSING)
    basis_id = getattr(expansion, "basis_id", _MISSING)
    space = getattr(expansion, "space", _MISSING)
    series = getattr(expansion, "q_expansion", _MISSING)
    if any(value is _MISSING for value in (weight, basis_id, space, series)):
        raise OperationDomainValidationError(
            location=("expansion",),
            code="modular_form.expansion_structure",
            message="q-expansion is missing required structural fields",
        )
    if type(weight) is not int or weight < 0:
        raise OperationDomainValidationError(
            location=("expansion", "weight"),
            code="modular_form.expansion_weight",
            message="expansion weight must be a nonnegative integer",
        )
    if weight > MAX_MODULAR_FORM_WEIGHT:
        raise OperationResourceAdmissionError(
            location=("expansion", "weight"),
            code="modular_form.weight_bound",
            message="q-expansion weight exceeds the supported exact envelope",
        )
    if type(basis_id) is not str or not 1 <= len(basis_id) <= 96:
        raise OperationDomainValidationError(
            location=("expansion", "basis_id"),
            code="modular_form.expansion_basis_id",
            message="q-expansion basis_id must be a bounded nonempty string",
        )
    if space is not None:
        _space(space)
        space_weight = getattr(space, "weight", _MISSING)
        if space_weight != weight:
            raise OperationDomainValidationError(
                location=("expansion", "weight"),
                code="modular_form.expansion_weight_parent",
                message="q-expansion weight must agree with its space parent",
            )
    if not isinstance(series, TruncatedSeries):
        raise OperationDomainValidationError(
            location=("expansion", "q_expansion"),
            code="modular_form.expansion_series",
            message="expansion must carry a truncated q-series",
        )
    variable = getattr(series, "variable", _MISSING)
    order = getattr(series, "truncation_order", _MISSING)
    coefficients = getattr(series, "coefficients", _MISSING)
    if variable != "q":
        raise OperationDomainValidationError(
            location=("expansion", "q_expansion", "variable"),
            code="modular_form.expansion_variable",
            message="modular q-expansions must use variable q",
        )
    if type(order) is not int or order < 1:
        raise OperationDomainValidationError(
            location=("expansion", "q_expansion", "truncation_order"),
            code="modular_form.source_precision",
            message="source truncation order must be a positive integer",
        )
    if order > MAX_Q_TRANSFORM_SOURCE_ORDER:
        raise OperationResourceAdmissionError(
            location=("expansion", "q_expansion", "truncation_order"),
            code="modular_form.source_precision_bound",
            message=(
                "q-transform source precision exceeds the "
                f"{MAX_Q_TRANSFORM_SOURCE_ORDER}-term envelope"
            ),
        )
    if not isinstance(coefficients, tuple) or len(coefficients) != order:
        raise OperationDomainValidationError(
            location=("expansion", "q_expansion", "coefficients"),
            code="modular_form.source_shape",
            message="q-expansion coefficients must match source truncation order",
        )
    for coefficient in coefficients:
        _canonical_source_coefficient(coefficient)
    return expansion


def _admit_output_precision(value: object) -> int:
    precision = _strict_positive_int(
        value, ("output_precision",), "modular_form.transform_bounds"
    )
    if precision > MAX_Q_TRANSFORM_OUTPUT_PRECISION:
        raise OperationResourceAdmissionError(
            location=("output_precision",),
            code="modular_form.output_precision_bound",
            message=(
                "q-transform output precision exceeds the "
                f"{MAX_Q_TRANSFORM_OUTPUT_PRECISION}-term envelope"
            ),
        )
    return precision


def _admit_prime(value: object) -> int:
    prime = _strict_positive_int(value, ("prime",), "modular_form.operator_prime")
    if prime < 2 or prime > MAX_GAMMA0_OPERATION_LEVEL:
        raise OperationResourceAdmissionError(
            location=("prime",),
            code="modular_form.operator_prime_bound",
            message=(
                "operator prime must lie in the bounded Gamma0 level envelope "
                f"[2, {MAX_GAMMA0_OPERATION_LEVEL}]"
            ),
        )
    return prime


def _is_prime(value: int) -> bool:
    # The numeric envelope is admitted before this finite trial division, so
    # primality testing cannot become an unbounded native or wire workload.
    if value == 2:
        return True
    if value % 2 == 0:
        return False
    return all(value % divisor for divisor in range(3, isqrt(value) + 1, 2))


def _admit_hecke_work(
    expansion: ModularQExpansion, index: int, output_precision: int
) -> None:
    source_order = expansion.q_expansion.truncation_order
    if index > source_order // output_precision:
        raise OperationDomainValidationError(
            location=("expansion", "q_expansion"),
            code="modular_form.insufficient_precision",
            message="source precision must cover every Hecke coefficient requested",
        )
    # The kernel scans every d through gcd(index, m), including index choices
    # that do not divide. Charge that complete loop before expansion.
    work = output_precision * index
    if work > 4_000_000:
        raise OperationResourceAdmissionError(
            location=("output_precision",),
            code="modular_form.transform_work_bound",
            message="Hecke coefficient work exceeds the bounded exact envelope",
        )
    max_source_digits = max(
        (
            max(len(str(abs(c.num))), len(str(c.den)))
            for c in expansion.q_expansion.coefficients
        ),
        default=1,
    )
    # Bound d^(k-1) without constructing it.  The binary-to-decimal estimate
    # is deliberately upward biased; it protects the Fraction temporary as
    # well as the exact returned coefficient.
    power_digits = (
        0
        if expansion.weight <= 1 or index <= 1
        else ((expansion.weight - 1) * max(1, index.bit_length()) * 30_103) // 100_000
        + 2
    )
    # At most 2*sqrt(index)+1 divisors can contribute to one coefficient.
    # Adding exact rationals may multiply all participating denominators, so
    # bound the unreduced common denominator and numerator rather than merely
    # adding the decimal width of the term count. For weight zero, the 1/d
    # factor contributes one further index-sized denominator factor per term.
    term_count = 2 * isqrt(index) + 1
    denominator_digits = max_source_digits + (
        len(str(index)) if expansion.weight == 0 else 0
    )
    numerator_digits = max_source_digits + power_digits
    output_digits = max(
        term_count * denominator_digits,
        numerator_digits
        + (term_count - 1) * denominator_digits
        + len(str(term_count))
        + 1,
    )
    if output_digits > MAX_Q_TRANSFORM_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("expansion", "weight"),
            code="modular_form.coefficient_growth_bound",
            message="Hecke coefficient growth exceeds the bounded exact envelope",
        )


def _admit_transform_space(space: ModularFormSpace) -> None:
    _space(space)
    if space.weight > MAX_MODULAR_FORM_WEIGHT:
        raise OperationResourceAdmissionError(
            location=("space", "weight"),
            code="modular_form.weight_bound",
            message="transform weight exceeds the supported exact envelope",
        )


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


def _space(space: object) -> None:
    if not isinstance(space, ModularFormSpace):
        raise OperationDomainValidationError(
            location=("space",),
            code="modular_form.unsupported_space",
            message="only trivial-character Gamma0(QQ) spaces are supported",
        )
    level = getattr(space, "level", _MISSING)
    weight = getattr(space, "weight", _MISSING)
    group = getattr(space, "group", _MISSING)
    kind = getattr(space, "kind", _MISSING)
    character = getattr(space, "character", _MISSING)
    coefficient_domain = getattr(space, "coefficient_domain", _MISSING)
    if (
        type(level) is not int
        or type(weight) is not int
        or level < 1
        or weight < 0
        or group != "GAMMA0"
        or type(kind) is not str
        or kind not in {"M", "S"}
        or character != "TRIVIAL"
        or coefficient_domain != "QQ"
    ):
        raise OperationDomainValidationError(
            location=("space",),
            code="modular_form.unsupported_space",
            message="only trivial-character Gamma0(QQ) spaces are supported",
        )
    if level > MAX_GAMMA0_OPERATION_LEVEL:
        raise OperationResourceAdmissionError(
            location=("space", "level"),
            code="modular_form.level_bound",
            message="Gamma0 level exceeds the bounded exact space envelope",
        )
    if weight > MAX_MODULAR_FORM_WEIGHT:
        raise OperationResourceAdmissionError(
            location=("space", "weight"),
            code="modular_form.weight_bound",
            message="modular-form weight exceeds the bounded exact space envelope",
        )


def _euler_counts(n: int) -> tuple[int, int, int]:
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


def sturm_bound(space: ModularFormSpace) -> SturmBoundResult:
    _space(space)
    return SturmBoundResult(
        space=space,
        index=_index(space.level),
        bound=(space.weight * _index(space.level)) // 12,
    )


def _result(
    exp: ModularQExpansion,
    coeffs: list[Fraction],
    weight: int | None = None,
    space: ModularFormSpace | None = None,
) -> ModularQExpansion:
    if len(coeffs) > MAX_Q_TRANSFORM_OUTPUT_PRECISION:
        raise OperationResourceAdmissionError(
            location=("output_precision",),
            code="modular_form.output_precision_bound",
            message="q-transform result exceeds the bounded output envelope",
        )
    for coefficient in coeffs:
        if (
            max(len(str(abs(coefficient.numerator))), len(str(coefficient.denominator)))
            > MAX_Q_TRANSFORM_COEFFICIENT_DIGITS
        ):
            raise OperationResourceAdmissionError(
                location=("coefficients",),
                code="modular_form.coefficient_growth_bound",
                message="q-transform coefficient exceeds the bounded exact envelope",
            )
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


def _require_source_precision(
    exp: ModularQExpansion, required: int, scale: int
) -> None:
    _strict_positive_int(required, ("output_precision",), "modular_form.precision")
    _strict_positive_int(scale, ("scale",), "modular_form.precision_scale")
    if (
        required > MAX_Q_TRANSFORM_SOURCE_ORDER
        or scale * required > exp.q_expansion.truncation_order
    ):
        raise OperationDomainValidationError(
            location=("expansion", "q_expansion"),
            code="modular_form.insufficient_precision",
            message="source precision is insufficient for the requested transform",
        )


def hecke(
    expansion: object, index: object, output_precision: object
) -> ModularQExpansion:
    index = _strict_positive_int(index, ("index",), "modular_form.transform_bounds")
    if index > MAX_Q_TRANSFORM_SOURCE_ORDER:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="modular_form.index_bound",
            message="Hecke index exceeds the bounded exact source envelope",
        )
    output_precision = _admit_output_precision(output_precision)
    expansion = _expansion(expansion)
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
    _require_source_precision(expansion, output_precision, index)
    _admit_hecke_work(expansion, index, output_precision)
    a = [c.as_fraction() for c in expansion.q_expansion.coefficients]
    k = expansion.weight
    out = []
    for m in range(output_precision):
        value = Fraction(0)
        for d in range(1, gcd(index, m) + 1):
            if index % d == 0 and m % d == 0:
                factor = Fraction(1, d) if k == 0 else Fraction(d ** (k - 1))
                value += factor * a[(index * m) // (d * d)]
        out.append(value)
    return _result(expansion, out)


def _operator_target_space(
    expansion: ModularQExpansion, prime: int
) -> ModularFormSpace:
    """Admit and construct the codomain of the published level-one operators.

    For a level-one form, both coefficient extraction U_p and dilation V_p
    have level Gamma0(p), not Gamma0(1).  Keeping this rule here makes the
    parent change part of the operation's admitted mathematical result rather
    than an incidental backend choice.
    """

    if expansion.space is None:
        raise OperationDomainValidationError(
            location=("expansion", "space"),
            code="modular_form.missing_space_parent",
            message="a q-expansion transform requires a bound modular-form space",
        )
    _admit_transform_space(expansion.space)
    if prime < 2 or prime > MAX_GAMMA0_OPERATION_LEVEL:
        raise OperationResourceAdmissionError(
            location=("prime",),
            code="modular_form.operator_prime_bound",
            message="operator prime exceeds the bounded Gamma0 level envelope",
        )
    if expansion.space.level != 1:
        raise OperationDomainValidationError(
            location=("expansion", "space", "level"),
            code="modular_form.operator_level",
            message="the published U and V formulas are admitted only from level one",
        )
    target = ModularFormSpace(
        level=prime,
        weight=expansion.weight,
        kind=expansion.space.kind,
        group=expansion.space.group,
        character=expansion.space.character,
        coefficient_domain=expansion.space.coefficient_domain,
    )
    _space(target)
    return target


def u_operator(
    expansion: object, prime: object, output_precision: object
) -> ModularQExpansion:
    prime = _admit_prime(prime)
    output_precision = _admit_output_precision(output_precision)
    expansion = _expansion(expansion)
    if not _is_prime(prime):
        raise OperationDomainValidationError(
            location=("prime",),
            code="modular_form.operator_prime",
            message="the U operator index must be prime",
        )
    target = _operator_target_space(expansion, prime)
    _require_source_precision(expansion, output_precision, prime)
    a = expansion.q_expansion.coefficients
    return _result(
        expansion,
        [a[prime * n].as_fraction() for n in range(output_precision)],
        space=target,
    )


def v_operator(
    expansion: object, prime: object, output_precision: object
) -> ModularQExpansion:
    prime = _admit_prime(prime)
    output_precision = _admit_output_precision(output_precision)
    expansion = _expansion(expansion)
    if not _is_prime(prime):
        raise OperationDomainValidationError(
            location=("prime",),
            code="modular_form.operator_prime",
            message="the V operator index must be prime",
        )
    target = _operator_target_space(expansion, prime)
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
        space=target,
    )


def named_q_expansion(
    space: ModularFormSpace, form: object, precision: object
) -> ModularQExpansion:
    from .operations import level_one_named_q_expansion

    _space(space)
    precision = _strict_positive_int(
        precision, ("precision",), "modular_form.precision"
    )
    if type(form) is not str:
        raise OperationDomainValidationError(
            location=("form",),
            code="modular_form.named_form_type",
            message="form must be a named modular-form identifier",
        )
    expected = {"E4": (4, "M"), "E6": (6, "M"), "DELTA": (12, "S")}
    if space.level != 1 or form not in expected:
        raise OperationDomainValidationError(
            location=("space", "form"),
            code="modular_form.named_unsupported",
            message="only E4, E6, and Delta are admitted in the level-one named slice",
        )
    expected_weight, expected_kind = expected[form]
    if space.weight != expected_weight or (space.kind == "M") != (expected_kind == "M"):
        raise OperationDomainValidationError(
            location=("space", "weight"),
            code="modular_form.named_parent_mismatch",
            message="named form does not belong to the supplied space",
        )
    named = level_one_named_q_expansion(
        cast(Literal["E4", "E6", "DELTA"], form), precision
    )
    return ModularQExpansion(
        space=space,
        weight=named.weight,
        q_expansion=named.q_expansion,
        basis_id=f"named:{form}",
    )


__all__ = ["hecke", "named_q_expansion", "sturm_bound", "u_operator", "v_operator"]
