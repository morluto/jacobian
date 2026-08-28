"""Domain operations for exact prime-affine local arithmetic."""

from __future__ import annotations

from math import prod

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.number_theory.affine_forms.values import (
    MAX_AFFINE_COMPONENT_DIGITS,
    MAX_FORM_ID_LENGTH,
)
from jacobian.math.number_theory.prime_affine_forms._interval import (
    MAX_INTERVAL_EVALUATIONS,
    _parse_interval,
    require_bounded_affine_endpoints,
)
from jacobian.math.number_theory.prime_affine_forms._kernel import (
    wheel_modulus,
    wheel_rows,
)
from jacobian.math.number_theory.prime_affine_forms._local_factors import local_summary
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
    PrimeTupleIntervalResidueProfileRequest,
    PrimeTupleIntervalResidueProfileResult,
    PrimeTupleResidueWheel,
    PrimeTupleResidueWheelEnumeration,
    PrimeTupleResidueWheelEnumerationRequest,
    PrimeTupleResidueWheelRequest,
    PrimeTupleWheelMembershipRequest,
    PrimeTupleWheelMembershipResult,
    PrimeTupleWheelResidueRow,
)


def _admit_residue_wheel(request: PrimeTupleResidueWheelRequest) -> None:
    _require_prime_set(request.primes, maximum=MAX_BATCH_PRIME)
    root_cells = request.source.form_count * len(request.primes)
    root_work = 6 * root_cells
    if root_work > MAX_COMPACT_WHEEL_ROOT_WORK:
        raise _validation_error(
            f"compact wheel computation needs {root_work} root steps, exceeding "
            f"{MAX_COMPACT_WHEEL_ROOT_WORK}"
        )
    modulus_digits = sum(_digits(prime) for prime in request.primes) or 1
    if modulus_digits > MAX_COMPACT_WHEEL_SCALAR_DIGITS:
        raise _validation_error(
            "compact wheel modulus exceeds the conservative exact scalar "
            f"digit bound {MAX_COMPACT_WHEEL_SCALAR_DIGITS}"
        )
    estimated_characters = (
        _source_character_upper_bound(request.source)
        + sum(
            _summary_character_upper_bound(request.source, prime)
            for prime in request.primes
        )
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


def _admit_wheel_enumeration(request: PrimeTupleResidueWheelEnumerationRequest) -> None:
    _admit_verified_wheel(request.wheel)
    local_residue_rows = sum(request.wheel.primes)
    if local_residue_rows > MAX_WHEEL_LOCAL_RESIDUES:
        raise _validation_error(
            f"wheel local residue enumeration {local_residue_rows} exceeds "
            f"{MAX_WHEEL_LOCAL_RESIDUES}"
        )
    result_count = parse_canonical_integer(request.wheel.valid_count)
    if result_count > MAX_WHEEL_RESIDUES:
        raise _validation_error(
            f"wheel has {result_count} valid residues, exceeding {MAX_WHEEL_RESIDUES}"
        )
    result_cells = result_count * (len(request.wheel.primes) + 1)
    if result_cells > MAX_WHEEL_RESULT_CELLS:
        raise _validation_error(
            f"wheel result needs {result_cells} cells, exceeding {MAX_WHEEL_RESULT_CELLS}"
        )
    root_cells = request.wheel.source.form_count * len(request.wheel.primes)
    enumeration_work = (
        result_count * len(request.wheel.primes) + local_residue_rows + root_cells
    )
    if enumeration_work > MAX_WHEEL_ENUMERATION_WORK:
        raise _validation_error(
            f"wheel enumeration needs {enumeration_work} bounded steps, exceeding "
            f"{MAX_WHEEL_ENUMERATION_WORK}"
        )
    modulus_digits = _digits(request.wheel.modulus)
    component_digits = sum(_digits(prime) for prime in request.wheel.primes)
    serialized_characters = (
        len(request.wheel.model_dump_json())
        + result_count
        * (modulus_digits + component_digits + 4 * len(request.wheel.primes) + 64)
        + 128
    )
    if serialized_characters > MAX_RESULT_CHARACTER_BUDGET:
        raise _validation_error(
            "wheel enumeration exceeds the conservative serialized bound"
        )


def _admit_wheel_membership(request: PrimeTupleWheelMembershipRequest) -> None:
    _admit_verified_wheel(request.wheel)
    if _digits(request.value) > MAX_AFFINE_COMPONENT_DIGITS:
        raise _validation_error(
            f"membership value must have at most {MAX_AFFINE_COMPONENT_DIGITS} digits"
        )
    result_characters = (
        len(request.wheel.model_dump_json())
        + len(request.value)
        + _digits(request.wheel.modulus)
        + sum(_digits(prime) for prime in request.wheel.primes)
        + MAX_FORM_ID_LENGTH * request.wheel.source.form_count
        + 256
    )
    if result_characters > MAX_RESULT_CHARACTER_BUDGET:
        raise _validation_error(
            "wheel membership result exceeds the conservative serialized bound"
        )


def _admit_interval_residue_profile(
    request: PrimeTupleIntervalResidueProfileRequest,
) -> tuple[int, int]:
    _admit_verified_wheel(request.wheel)
    require_bounded_affine_endpoints(
        request.wheel.source, request.lower, request.upper, label="interval"
    )
    lower, upper, interval_size = _parse_interval(request.lower, request.upper)
    if interval_size > MAX_WHEEL_INTERVAL_LENGTH:
        raise _validation_error(
            f"wheel interval length {interval_size} exceeds {MAX_WHEEL_INTERVAL_LENGTH}"
        )
    membership_checks = interval_size * max(1, len(request.wheel.primes))
    if membership_checks > MAX_INTERVAL_EVALUATIONS:
        raise _validation_error(
            f"wheel interval profile needs {membership_checks} modular checks, "
            f"exceeding {MAX_INTERVAL_EVALUATIONS}"
        )
    endpoint_digits = max(_digits(request.lower), _digits(request.upper))
    result_characters = (
        len(request.wheel.model_dump_json())
        + interval_size * (endpoint_digits + 4)
        + 192
    )
    if result_characters > MAX_RESULT_CHARACTER_BUDGET:
        raise _validation_error(
            "wheel interval profile exceeds the conservative serialized bound"
        )
    return lower, upper


def compute_residue_wheel(
    request: PrimeTupleResidueWheelRequest,
) -> PrimeTupleResidueWheel:
    _run_admission(lambda: _admit_residue_wheel(request))
    local_rows = tuple(local_summary(request.source, prime) for prime in request.primes)
    return PrimeTupleResidueWheel._from_kernel(
        request,
        local_rows=local_rows,
        modulus=format_canonical_integer(wheel_modulus(request.primes)),
        valid_count=format_canonical_integer(
            prod((row.valid_count for row in local_rows), start=1)
        ),
    )


def compute_residue_wheel_enumeration(
    request: PrimeTupleResidueWheelEnumerationRequest,
) -> PrimeTupleResidueWheelEnumeration:
    _run_admission(lambda: _admit_wheel_enumeration(request))
    rows = wheel_rows(request.wheel.source, request.wheel.primes)
    return PrimeTupleResidueWheelEnumeration._from_kernel(
        request,
        residues=tuple(
            PrimeTupleWheelResidueRow(
                residue=format_canonical_integer(residue),
                components=components,
            )
            for residue, components in rows
        ),
    )


def compute_wheel_membership(
    request: PrimeTupleWheelMembershipRequest,
) -> PrimeTupleWheelMembershipResult:
    _run_admission(lambda: _admit_wheel_membership(request))
    value = parse_canonical_integer(request.value)
    modulus = parse_canonical_integer(request.wheel.modulus)
    residue = value % modulus
    components = tuple(value % prime for prime in request.wheel.primes)
    first_prime: int | None = None
    form_ids: tuple[str, ...] = ()
    for summary, component in zip(request.wheel.local_rows, components, strict=True):
        bad = {row.residue: row.form_ids for row in summary.bad_residues}
        if component in bad:
            first_prime = summary.prime
            form_ids = bad[component]
            break
    return PrimeTupleWheelMembershipResult._from_kernel(
        request,
        canonical_residue=format_canonical_integer(residue),
        components=components,
        is_permitted=first_prime is None,
        first_excluded_prime=first_prime,
        vanishing_form_ids=form_ids,
    )


def compute_interval_residue_profile(
    request: PrimeTupleIntervalResidueProfileRequest,
) -> PrimeTupleIntervalResidueProfileResult:
    lower, upper = _run_admission(lambda: _admit_interval_residue_profile(request))
    bad_by_prime = tuple(
        {row.residue for row in summary.bad_residues}
        for summary in request.wheel.local_rows
    )
    survivors = tuple(
        value
        for value in range(lower, upper + 1)
        if all(
            value % prime not in bad
            for prime, bad in zip(request.wheel.primes, bad_by_prime, strict=True)
        )
    )
    return PrimeTupleIntervalResidueProfileResult._from_kernel(
        request,
        survivors=tuple(format_canonical_integer(value) for value in survivors),
    )


__all__ = [
    "compute_interval_residue_profile",
    "compute_residue_wheel",
    "compute_residue_wheel_enumeration",
    "compute_wheel_membership",
]
