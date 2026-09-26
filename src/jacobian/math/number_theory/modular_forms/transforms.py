"""Exact Gamma0 trivial-character Sturm and coefficient transforms."""

from __future__ import annotations

from fractions import Fraction
from math import gcd, isqrt
from typing import Literal, cast

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._execution import request_checkpoint
from jacobian.canonical import format_canonical_integer, strict_json_object_size
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.characters.operations import (
    require_complete_character_group,
)
from jacobian.math.number_theory.characters.values import (
    MAX_CHARACTER_GROUP_MODULUS,
    DirichletCharacter,
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

MAX_FORMAL_Q_SERIES_OPERATOR_ALLOCATION_BYTES = 10 * 1024 * 1024
MAX_FORMAL_Q_SERIES_OPERATOR_WORK = 8_192


def _strict_positive_int(value: object, location: tuple[str, ...], code: str) -> int:
    if type(value) is not int or value < 1:
        raise OperationDomainValidationError(
            location=location,
            code=code,
            message="value must be a positive integer",
        )
    return value


_MISSING = object()


def _canonical_source_coefficient(
    value: object,
    *,
    location: tuple[str | int, ...] = ("expansion", "q_expansion", "coefficients"),
    error_namespace: str = "modular_form",
) -> CanonicalRational:
    """Revalidate a scalar carrier, including native ``model_construct`` values."""

    if not isinstance(value, CanonicalRational):
        raise OperationDomainValidationError(
            location=location,
            code=f"{error_namespace}.source_coefficient_type",
            message="q-expansion coefficients must be canonical rationals",
        )
    num = getattr(value, "num", _MISSING)
    den = getattr(value, "den", _MISSING)
    if type(num) is not int or type(den) is not int or den <= 0:
        raise OperationDomainValidationError(
            location=location,
            code=f"{error_namespace}.source_coefficient_value",
            message="q-expansion coefficients must have a positive denominator",
        )
    try:
        rational = Fraction(num, den)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise OperationDomainValidationError(
            location=location,
            code=f"{error_namespace}.source_coefficient_value",
            message="q-expansion coefficients must be valid rationals",
        ) from error
    if (num, den) != (rational.numerator, rational.denominator):
        raise OperationDomainValidationError(
            location=location,
            code=f"{error_namespace}.source_coefficient_value",
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
            location=location,
            code=f"{error_namespace}.source_coefficient_bound",
            message=str(error),
        ) from error
    return value


def _formal_q_series_operator_admission(
    series: object,
    prime: object,
    output_precision: object,
    *,
    operator: Literal["U", "V"],
) -> tuple[TruncatedSeries, int, int, tuple[CanonicalRational, ...]]:
    """Admit a formal q-prefix transform without claiming modularity."""

    prime = _admit_prime(prime, error_namespace="formal_q_series")
    output_precision = _admit_output_precision(
        output_precision, error_namespace="formal_q_series"
    )
    if not _is_prime(prime):
        raise OperationDomainValidationError(
            location=("prime",),
            code="formal_q_series.operator_prime",
            message="formal U/V operator index must be prime",
        )
    if not isinstance(series, TruncatedSeries):
        raise OperationDomainValidationError(
            location=("series",),
            code="formal_q_series.series_type",
            message="series must be an exact truncated-series value",
        )
    variable = getattr(series, "variable", None)
    source_order = getattr(series, "truncation_order", None)
    coefficients = getattr(series, "coefficients", None)
    if variable != "q":
        raise OperationDomainValidationError(
            location=("series", "variable"),
            code="formal_q_series.variable",
            message="formal U/V index transforms require the variable q",
        )
    if type(source_order) is not int or source_order < 1:
        raise OperationDomainValidationError(
            location=("series", "truncation_order"),
            code="formal_q_series.source_precision",
            message="source truncation order must be a positive integer",
        )
    if source_order > MAX_Q_TRANSFORM_SOURCE_ORDER:
        raise OperationResourceAdmissionError(
            location=("series", "truncation_order"),
            code="formal_q_series.source_precision_bound",
            message="formal q-prefix exceeds the bounded source-order envelope",
        )
    if not isinstance(coefficients, tuple) or len(coefficients) != source_order:
        raise OperationDomainValidationError(
            location=("series", "coefficients"),
            code="formal_q_series.source_shape",
            message="coefficient count must equal the source truncation order",
        )

    required_source_order = (
        prime * (output_precision - 1) + 1
        if operator == "U"
        else (output_precision - 1) // prime + 1
    )
    if required_source_order > MAX_Q_TRANSFORM_SOURCE_ORDER:
        raise OperationResourceAdmissionError(
            location=("output_precision",),
            code="formal_q_series.required_source_precision_bound",
            message=(
                f"{operator}_p output would require source order "
                f"{required_source_order}, above the admitted source-order bound"
            ),
        )
    if required_source_order > source_order:
        raise OperationDomainValidationError(
            location=("series", "truncation_order"),
            code="formal_q_series.insufficient_precision",
            message=(
                f"{operator}_p output requires source precision "
                f"{required_source_order}, got {source_order}"
            ),
        )
    selected_count = (
        output_precision if operator == "U" else (output_precision - 1) // prime + 1
    )
    work = output_precision + selected_count
    if work > MAX_FORMAL_Q_SERIES_OPERATOR_WORK:
        raise OperationResourceAdmissionError(
            location=("output_precision",),
            code="formal_q_series.work_bound",
            message="formal q-series index transform exceeds its work envelope",
        )

    source_indices = (
        tuple(prime * index for index in range(output_precision))
        if operator == "U"
        else tuple(
            index // prime for index in range(output_precision) if index % prime == 0
        )
    )
    admitted = []
    maximum_digits = 1
    for offset, index in enumerate(source_indices):
        if offset % 128 == 0:
            request_checkpoint(f"admitting formal q-series {operator}_p source")
        coefficient = _canonical_source_coefficient(
            coefficients[index],
            location=("series", "coefficients", index),
            error_namespace="formal_q_series",
        )
        admitted.append(coefficient)
        maximum_digits = max(
            maximum_digits,
            len(format_canonical_integer(abs(coefficient.num))),
            len(format_canonical_integer(coefficient.den)),
        )

    coefficient_json_size = strict_json_object_size(
        (
            ("num", maximum_digits + 3),
            ("den", maximum_digits + 2),
        )
    )
    coefficients_json_size = (
        2 + max(output_precision - 1, 0) + output_precision * coefficient_json_size
    )
    allocation_bytes = strict_json_object_size(
        (
            ("variable", 3),
            ("truncation_order", len(str(output_precision))),
            ("coefficients", coefficients_json_size),
        )
    )
    if allocation_bytes > MAX_FORMAL_Q_SERIES_OPERATOR_ALLOCATION_BYTES:
        raise OperationResourceAdmissionError(
            location=("output_precision",),
            code="formal_q_series.output_bound",
            message="formal q-series result exceeds its admitted allocation envelope",
        )
    return series, prime, output_precision, tuple(admitted)


def formal_q_series_u_operator(
    series: object, prime: object, output_precision: object
) -> TruncatedSeries:
    """Return the formal prefix U_p(sum a_n q^n) = sum a_(pn) q^n."""

    source, prime, precision, selected = _formal_q_series_operator_admission(
        series, prime, output_precision, operator="U"
    )
    del source
    return TruncatedSeries(
        variable="q",
        truncation_order=precision,
        coefficients=selected,
    )


def formal_q_series_v_operator(
    series: object, prime: object, output_precision: object
) -> TruncatedSeries:
    """Return the formal prefix V_p(sum a_n q^n) = sum a_n q^(pn)."""

    _, prime, precision, selected = _formal_q_series_operator_admission(
        series, prime, output_precision, operator="V"
    )
    selected_by_index = iter(selected)
    zero = CanonicalRational(num=0, den=1)
    coefficients = tuple(
        next(selected_by_index) if index % prime == 0 else zero
        for index in range(precision)
    )
    return TruncatedSeries(
        variable="q",
        truncation_order=precision,
        coefficients=coefficients,
    )


def _admit_output_precision(
    value: object, *, error_namespace: str = "modular_form"
) -> int:
    precision = _strict_positive_int(
        value, ("output_precision",), f"{error_namespace}.transform_bounds"
    )
    if precision > MAX_Q_TRANSFORM_OUTPUT_PRECISION:
        raise OperationResourceAdmissionError(
            location=("output_precision",),
            code=f"{error_namespace}.output_precision_bound",
            message=(
                "q-transform output precision exceeds the "
                f"{MAX_Q_TRANSFORM_OUTPUT_PRECISION}-term envelope"
            ),
        )
    return precision


def _admit_prime(value: object, *, error_namespace: str = "modular_form") -> int:
    prime = _strict_positive_int(value, ("prime",), f"{error_namespace}.operator_prime")
    if prime < 2 or prime > MAX_GAMMA0_OPERATION_LEVEL:
        raise OperationResourceAdmissionError(
            location=("prime",),
            code=f"{error_namespace}.operator_prime_bound",
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
    supported_character = character == "TRIVIAL"
    if isinstance(character, DirichletCharacter):
        group_value = require_complete_character_group(character.group)
        supported_character = (
            level == 4
            and weight in (1, 3)
            and group_value.modulus == 4
            and group_value.generator_orders == (2,)
            and character.coordinates == (1,)
        )
    if (
        type(level) is not int
        or type(weight) is not int
        or level < 1
        or weight < 0
        or group != "GAMMA0"
        or type(kind) is not str
        or kind not in {"M", "S"}
        or not supported_character
        or coefficient_domain != "QQ"
    ):
        raise OperationDomainValidationError(
            location=("space",),
            code="modular_form.unsupported_space",
            message=(
                "only trivial-character Gamma0(QQ) spaces and the exact "
                "M/S_1 or M/S_3 Gamma0(4), chi_{-4} parents are supported"
            ),
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


def _sturm_space(space: object) -> ModularFormSpace:
    if type(space) is not ModularFormSpace:
        raise OperationDomainValidationError(
            location=("space",),
            code="modular_form.unsupported_space",
            message="Sturm bounds require a canonical Gamma0 modular-form space",
        )
    raw_level = getattr(space, "level", None)
    if type(raw_level) is int and raw_level > MAX_GAMMA0_OPERATION_LEVEL:
        raise OperationResourceAdmissionError(
            location=("space", "level"),
            code="modular_form.level_bound",
            message="modular-form level exceeds the exact Sturm-index envelope",
        )
    raw_character = getattr(space, "character", None)
    if isinstance(raw_character, DirichletCharacter):
        raw_group = getattr(raw_character, "group", None)
        raw_units = getattr(raw_group, "unit_residues", None)
        raw_coordinates = getattr(raw_group, "unit_coordinates", None)
        if type(raw_units) is not tuple or type(raw_coordinates) is not tuple:
            raise OperationDomainValidationError(
                location=("space", "character", "group"),
                code="modular_form.invalid_character_group",
                message="Sturm bounds require bounded canonical character group tables",
            )
        if (
            len(raw_units) > MAX_CHARACTER_GROUP_MODULUS
            or len(raw_coordinates) > MAX_CHARACTER_GROUP_MODULUS
        ):
            raise OperationResourceAdmissionError(
                location=("space", "character", "group"),
                code="modular_form.character_group_bound",
                message="character group tables exceed the bounded Sturm parent envelope",
            )
        if any(type(row) is not tuple or len(row) > 32 for row in raw_coordinates):
            raise OperationResourceAdmissionError(
                location=("space", "character", "group", "unit_coordinates"),
                code="modular_form.character_group_bound",
                message="character coordinate rows exceed the bounded Sturm parent envelope",
            )
    try:
        # Re-run the parent validators because a caller can construct a model
        # without validation and later rely on its character/field claims.
        space = ModularFormSpace.model_validate(space.model_dump(mode="python"))
    except (TypeError, ValueError, AttributeError) as error:
        raise OperationDomainValidationError(
            location=("space",),
            code="modular_form.unsupported_space",
            message="Sturm bounds require a valid exact modular-form parent",
        ) from error
    if isinstance(space.character, DirichletCharacter):
        try:
            group = require_complete_character_group(space.character.group)
        except (
            OperationDomainValidationError,
            OperationResourceAdmissionError,
        ) as error:
            raise OperationDomainValidationError(
                location=("space", "character", "group"),
                code="modular_form.invalid_character_group",
                message="Sturm bounds require a complete canonical character group",
            ) from error
        coordinates = space.character.coordinates
        if (
            type(coordinates) is not tuple
            or len(coordinates) != len(group.generator_orders)
            or any(
                type(value) is not int or value < 0 or value >= order
                for value, order in zip(
                    coordinates, group.generator_orders, strict=True
                )
            )
        ):
            raise OperationDomainValidationError(
                location=("space", "character", "coordinates"),
                code="modular_form.invalid_character_coordinates",
                message="Sturm bounds require canonical character coordinates",
            )
    if (
        space.group != "GAMMA0"
        or type(space.level) is not int
        or type(space.weight) is not int
        or space.kind not in {"M", "S"}
    ):
        raise OperationDomainValidationError(
            location=("space",),
            code="modular_form.unsupported_space",
            message="Sturm bounds require a bounded exact Gamma0 space",
        )
    return space


def sturm_bound(space: ModularFormSpace) -> SturmBoundResult:
    space = _sturm_space(space)
    return SturmBoundResult(
        space=space,
        index=_index(space.level),
        bound=(space.weight * _index(space.level)) // 12,
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


__all__ = [
    "formal_q_series_u_operator",
    "formal_q_series_v_operator",
    "named_q_expansion",
    "sturm_bound",
]
