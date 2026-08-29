"""Value-based native operations on canonical prime-affine tuple values."""

from __future__ import annotations

from fractions import Fraction
from math import prod

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.number_theory.affine_forms.values import (
    MAX_AFFINE_COMPONENT_DIGITS,
    MAX_FORM_ID_LENGTH,
)
from jacobian.math.number_theory.prime_affine_forms._admissibility import (
    PrimeTupleAdmissibilityResult,
    compute_local_admissibility,
)
from jacobian.math.number_theory.prime_affine_forms._interval import (
    MAX_INTERVAL_EVALUATIONS,
    PrimePatternIntervalCountResult,
    PrimePatternIntervalEnumerateResult,
    _admit_interval_count,
    _admit_interval_enumerate,
    _parse_interval,
    require_bounded_affine_endpoints,
)
from jacobian.math.number_theory.prime_affine_forms._kernel import (
    interval_match_summary,
    interval_matches,
    local_bad_residues,
    local_factor_from_bad_count,
    wheel_modulus,
    wheel_rows,
)
from jacobian.math.number_theory.prime_affine_forms._local_factors import (
    FinitePrimeTupleFactorProduct,
    PrimeTupleLocalFactorResult,
    PrimeTupleLocalFactorRow,
    _admit_local_factor,
    _admit_local_factors,
    local_summary,
)
from jacobian.math.number_theory.prime_affine_forms._models import (
    MAX_BATCH_PRIME,
    MAX_RESULT_CHARACTER_BUDGET,
    _digits,
    _require_prime_set,
    _run_admission,
    _source_character_upper_bound,
    _summary_character_upper_bound,
    _validation_error,
)
from jacobian.math.number_theory.prime_affine_forms._residue_wheel import (
    MAX_COMPACT_WHEEL_ROOT_WORK,
    MAX_COMPACT_WHEEL_SCALAR_DIGITS,
    MAX_WHEEL_ENUMERATION_WORK,
    MAX_WHEEL_INTERVAL_LENGTH,
    MAX_WHEEL_LOCAL_RESIDUES,
    MAX_WHEEL_RESIDUES,
    MAX_WHEEL_RESULT_CELLS,
    PrimeTupleIntervalResidueProfileResult,
    PrimeTupleResidueWheel,
    PrimeTupleResidueWheelEnumeration,
    PrimeTupleWheelMembershipResult,
    PrimeTupleWheelResidueRow,
)
from jacobian.math.number_theory.prime_affine_forms._translation import (
    PrimeAffineTranslationResult,
    compute_translation,
)
from jacobian.math.number_theory.prime_affine_forms.values import PrimeAffineTuple


def local_factor(source: PrimeAffineTuple, prime: int) -> PrimeTupleLocalFactorResult:
    """Return the complete modulo-``prime`` residue partition and local factor."""

    _run_admission(lambda: _admit_local_factor(source, prime))
    return PrimeTupleLocalFactorResult._from_kernel(
        source=source, prime=prime, bad=local_bad_residues(source, prime)
    )


def local_factors(
    source: PrimeAffineTuple, primes: tuple[int, ...]
) -> FinitePrimeTupleFactorProduct:
    """Return exact compact local factors over one finite prime set."""

    _run_admission(lambda: _admit_local_factors(source, primes))
    product = Fraction(1, 1)
    rows: list[PrimeTupleLocalFactorRow] = []
    first_obstruction: int | None = None
    for prime in primes:
        summary = local_summary(source, prime)
        factor = local_factor_from_bad_count(
            source.form_count, prime, summary.bad_count
        )
        rows.append(
            PrimeTupleLocalFactorRow(
                summary=summary,
                factor=CanonicalRational.from_fraction(factor),
            )
        )
        product *= factor
        if first_obstruction is None and summary.valid_count == 0:
            first_obstruction = prime
    return FinitePrimeTupleFactorProduct._from_kernel(
        source=source,
        primes=primes,
        rows=tuple(rows),
        product=product,
        first_obstruction=first_obstruction,
    )


def local_admissibility(source: PrimeAffineTuple) -> PrimeTupleAdmissibilityResult:
    """Decide local admissibility by checking exactly every prime at most k."""

    return compute_local_admissibility(source)


def interval_count(
    source: PrimeAffineTuple, lower: str | int, upper: str | int
) -> PrimePatternIntervalCountResult:
    """Count every admitted positive-prime affine tuple in the interval."""

    lower_text = lower if isinstance(lower, str) else format_canonical_integer(lower)
    upper_text = upper if isinstance(upper, str) else format_canonical_integer(upper)
    lower, upper, _, _ = _run_admission(
        lambda: _admit_interval_count(source, lower_text, upper_text)
    )
    count, first, last = interval_match_summary(source, lower, upper)
    return PrimePatternIntervalCountResult._from_kernel(
        source=source,
        lower=lower_text,
        upper=upper_text,
        count=count,
        first=first,
        last=last,
    )


def interval_enumerate(
    source: PrimeAffineTuple, lower: str | int, upper: str | int
) -> PrimePatternIntervalEnumerateResult:
    """Materialize every admitted positive-prime affine tuple in the interval."""

    lower_text = lower if isinstance(lower, str) else format_canonical_integer(lower)
    upper_text = upper if isinstance(upper, str) else format_canonical_integer(upper)
    lower, upper, _ = _run_admission(
        lambda: _admit_interval_enumerate(source, lower_text, upper_text)
    )
    return PrimePatternIntervalEnumerateResult._from_kernel(
        source=source,
        lower=lower_text,
        upper=upper_text,
        matches=interval_matches(source, lower, upper),
    )


def _admit_residue_wheel(source: PrimeAffineTuple, primes: tuple[int, ...]) -> None:
    _require_prime_set(primes, maximum=MAX_BATCH_PRIME)
    root_cells = source.form_count * len(primes)
    root_work = 6 * root_cells
    if root_work > MAX_COMPACT_WHEEL_ROOT_WORK:
        raise _validation_error(
            f"compact wheel computation needs {root_work} root steps, exceeding "
            f"{MAX_COMPACT_WHEEL_ROOT_WORK}"
        )
    modulus_digits = sum(_digits(prime) for prime in primes) or 1
    if modulus_digits > MAX_COMPACT_WHEEL_SCALAR_DIGITS:
        raise _validation_error(
            "compact wheel modulus exceeds the conservative exact scalar "
            f"digit bound {MAX_COMPACT_WHEEL_SCALAR_DIGITS}"
        )
    estimated_characters = (
        _source_character_upper_bound(source)
        + sum(_summary_character_upper_bound(source, prime) for prime in primes)
        + 2 * modulus_digits
        + 128
    )
    if estimated_characters > MAX_RESULT_CHARACTER_BUDGET:
        raise _validation_error(
            "compact wheel exceeds the conservative serialized bound"
        )


def _admit_verified_wheel(wheel: PrimeTupleResidueWheel) -> None:
    _require_prime_set(wheel.primes, maximum=MAX_BATCH_PRIME)
    expected_rows = tuple(local_summary(wheel.source, prime) for prime in wheel.primes)
    expected_modulus = wheel_modulus(wheel.primes)
    expected_valid_count = prod((row.valid_count for row in expected_rows), start=1)
    if wheel.local_rows != expected_rows:
        raise _validation_error(
            "wheel must equal the compact wheel for its source and primes"
        )
    if parse_canonical_integer(wheel.modulus) != expected_modulus:
        raise _validation_error("wheel modulus does not match its canonical prime set")
    if parse_canonical_integer(wheel.valid_count) != expected_valid_count:
        raise _validation_error("wheel valid count does not match its local rows")


def _admit_wheel_enumeration(wheel: PrimeTupleResidueWheel) -> None:
    _admit_verified_wheel(wheel)
    local_residue_rows = sum(wheel.primes)
    if local_residue_rows > MAX_WHEEL_LOCAL_RESIDUES:
        raise _validation_error(
            f"wheel local residue enumeration {local_residue_rows} exceeds "
            f"{MAX_WHEEL_LOCAL_RESIDUES}"
        )
    result_count = parse_canonical_integer(wheel.valid_count)
    if result_count > MAX_WHEEL_RESIDUES:
        raise _validation_error(
            f"wheel has {result_count} valid residues, exceeding {MAX_WHEEL_RESIDUES}"
        )
    result_cells = result_count * (len(wheel.primes) + 1)
    if result_cells > MAX_WHEEL_RESULT_CELLS:
        raise _validation_error(
            f"wheel result needs {result_cells} cells, exceeding {MAX_WHEEL_RESULT_CELLS}"
        )
    root_cells = wheel.source.form_count * len(wheel.primes)
    enumeration_work = (
        result_count * len(wheel.primes) + local_residue_rows + root_cells
    )
    if enumeration_work > MAX_WHEEL_ENUMERATION_WORK:
        raise _validation_error(
            f"wheel enumeration needs {enumeration_work} bounded steps, exceeding "
            f"{MAX_WHEEL_ENUMERATION_WORK}"
        )
    modulus_digits = _digits(wheel.modulus)
    component_digits = sum(_digits(prime) for prime in wheel.primes)
    serialized_characters = (
        len(wheel.model_dump_json())
        + result_count
        * (modulus_digits + component_digits + 4 * len(wheel.primes) + 64)
        + 128
    )
    if serialized_characters > MAX_RESULT_CHARACTER_BUDGET:
        raise _validation_error(
            "wheel enumeration exceeds the conservative serialized bound"
        )


def _admit_wheel_membership(wheel: PrimeTupleResidueWheel, value: int) -> None:
    _admit_verified_wheel(wheel)
    if _digits(value) > MAX_AFFINE_COMPONENT_DIGITS:
        raise _validation_error(
            f"membership value must have at most {MAX_AFFINE_COMPONENT_DIGITS} digits"
        )
    result_characters = (
        len(wheel.model_dump_json())
        + len(str(value))
        + _digits(wheel.modulus)
        + sum(_digits(prime) for prime in wheel.primes)
        + MAX_FORM_ID_LENGTH * wheel.source.form_count
        + 256
    )
    if result_characters > MAX_RESULT_CHARACTER_BUDGET:
        raise _validation_error(
            "wheel membership result exceeds the conservative serialized bound"
        )


def _admit_interval_residue_profile(
    wheel: PrimeTupleResidueWheel, lower: int, upper: int
) -> tuple[int, int]:
    _admit_verified_wheel(wheel)
    lower_text = format_canonical_integer(lower)
    upper_text = format_canonical_integer(upper)
    require_bounded_affine_endpoints(
        wheel.source, lower_text, upper_text, label="interval"
    )
    lower, upper, interval_size = _parse_interval(lower_text, upper_text)
    if interval_size > MAX_WHEEL_INTERVAL_LENGTH:
        raise _validation_error(
            f"wheel interval length {interval_size} exceeds {MAX_WHEEL_INTERVAL_LENGTH}"
        )
    membership_checks = interval_size * max(1, len(wheel.primes))
    if membership_checks > MAX_INTERVAL_EVALUATIONS:
        raise _validation_error(
            f"wheel interval profile needs {membership_checks} modular checks, "
            f"exceeding {MAX_INTERVAL_EVALUATIONS}"
        )
    endpoint_digits = max(_digits(lower), _digits(upper))
    result_characters = (
        len(wheel.model_dump_json()) + interval_size * (endpoint_digits + 4) + 192
    )
    if result_characters > MAX_RESULT_CHARACTER_BUDGET:
        raise _validation_error(
            "wheel interval profile exceeds the conservative serialized bound"
        )
    return lower, upper


def residue_wheel(
    source: PrimeAffineTuple, primes: tuple[int, ...]
) -> PrimeTupleResidueWheel:
    """Construct the compact source-bound CRT wheel of one finite prime set."""

    _run_admission(lambda: _admit_residue_wheel(source, primes))
    local_rows = tuple(local_summary(source, prime) for prime in primes)
    return PrimeTupleResidueWheel.model_construct(
        source=source,
        primes=primes,
        local_rows=local_rows,
        modulus=format_canonical_integer(wheel_modulus(primes)),
        valid_count=format_canonical_integer(
            prod((row.valid_count for row in local_rows), start=1)
        ),
    )


def enumerate_residue_wheel(
    wheel: PrimeTupleResidueWheel,
) -> PrimeTupleResidueWheelEnumeration:
    """Materialize every permitted CRT residue of a supplied compact wheel."""

    _run_admission(lambda: _admit_wheel_enumeration(wheel))
    rows = wheel_rows(wheel.source, wheel.primes)
    return PrimeTupleResidueWheelEnumeration.model_construct(
        wheel=wheel,
        residues=tuple(
            PrimeTupleWheelResidueRow(
                residue=format_canonical_integer(residue),
                components=components,
            )
            for residue, components in rows
        ),
    )


def wheel_membership(
    wheel: PrimeTupleResidueWheel, value: int
) -> PrimeTupleWheelMembershipResult:
    """Reduce one exact integer through a source-bound residue wheel."""

    _run_admission(lambda: _admit_wheel_membership(wheel, value))
    modulus = parse_canonical_integer(wheel.modulus)
    residue = value % modulus
    components = tuple(value % prime for prime in wheel.primes)
    first_prime: int | None = None
    form_ids: tuple[str, ...] = ()
    for summary, component in zip(wheel.local_rows, components, strict=True):
        bad = {row.residue: row.form_ids for row in summary.bad_residues}
        if component in bad:
            first_prime = summary.prime
            form_ids = bad[component]
            break
    return PrimeTupleWheelMembershipResult.model_construct(
        wheel=wheel,
        value=format_canonical_integer(value),
        canonical_residue=format_canonical_integer(residue),
        components=components,
        is_permitted=first_prime is None,
        first_excluded_prime=first_prime,
        vanishing_form_ids=form_ids,
    )


def interval_residue_profile(
    wheel: PrimeTupleResidueWheel, lower: int, upper: int
) -> PrimeTupleIntervalResidueProfileResult:
    """Enumerate wheel-permitted integers on one bounded closed interval."""

    lower, upper = _run_admission(
        lambda: _admit_interval_residue_profile(wheel, lower, upper)
    )
    bad_by_prime = tuple(
        {row.residue for row in summary.bad_residues} for summary in wheel.local_rows
    )
    survivors = tuple(
        value
        for value in range(lower, upper + 1)
        if all(
            value % prime not in bad
            for prime, bad in zip(wheel.primes, bad_by_prime, strict=True)
        )
    )
    return PrimeTupleIntervalResidueProfileResult.model_construct(
        wheel=wheel,
        lower=format_canonical_integer(lower),
        upper=format_canonical_integer(upper),
        interval_size=upper - lower + 1,
        survivors=tuple(format_canonical_integer(value) for value in survivors),
    )


def translate_tuple(
    source: PrimeAffineTuple, shift: int
) -> PrimeAffineTranslationResult:
    """Apply the substitution n -> n+shift to every labelled form."""

    return compute_translation(source, shift)


__all__ = [
    "enumerate_residue_wheel",
    "interval_count",
    "interval_enumerate",
    "interval_residue_profile",
    "local_admissibility",
    "local_factor",
    "local_factors",
    "residue_wheel",
    "translate_tuple",
    "wheel_membership",
]
