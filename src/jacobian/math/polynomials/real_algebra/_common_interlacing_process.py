"""Killable request-scoped process boundary for common interlacing."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from pydantic import TypeAdapter, ValidationError
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.algebraic_numbers.real import RealAlgebraicValue
from jacobian.math.polynomials.real_algebra._common_interlacing_models import (
    MAX_COMMON_INTERLACING_FAMILY_SIZE,
    MAX_COMMON_INTERLACING_INPUT_DIGITS,
    MAX_COMMON_INTERLACING_SOURCE_TERMS,
    CommonInterlacingOutcome,
    CommonInterlacingProfile,
    LabelledRationalPolynomial,
    PolynomialRealRoot,
    SourceRootProfile,
    _require_family_shape,
)

_WORKER = Path(__file__).resolve().with_name("_common_interlacing_worker.py")
_WALL_SECONDS = 3_600.0
_ADDRESS_SPACE_BYTES = 2 * 1024 * 1024 * 1024
_STDOUT_LIMIT = 11 * 1024 * 1024
_STDERR_LIMIT = 64 * 1024
_CANONICAL_POLYNOMIAL = TypeAdapter(tuple[CanonicalInteger, ...])
_OUTCOME: TypeAdapter[CommonInterlacingOutcome] = TypeAdapter(CommonInterlacingOutcome)


def _preflight_family_size(
    family: tuple[LabelledRationalPolynomial, ...],
) -> None:
    """Reject oversized families before serializing them for the worker."""

    if not 2 <= len(family) <= MAX_COMMON_INTERLACING_FAMILY_SIZE:
        raise OperationDomainValidationError(
            location=("family",),
            code="polynomial.common_interlacing_family_size",
            message=(
                "common interlacing requires between 2 and "
                f"{MAX_COMMON_INTERLACING_FAMILY_SIZE} family members"
            ),
        )
    for source_index, source in enumerate(family):
        terms = source.polynomial.polynomial.terms
        if len(terms) > MAX_COMMON_INTERLACING_SOURCE_TERMS:
            raise OperationDomainValidationError(
                location=("family", source_index, "polynomial", "terms"),
                code="polynomial.common_interlacing_term_count",
                message=(
                    "a common interlacing source exceeds the "
                    f"{MAX_COMMON_INTERLACING_SOURCE_TERMS}-term bound"
                ),
            )
        for term in terms:
            coeff = term.coefficient
            if (
                max(len(coeff.num.lstrip("-")), len(coeff.den))
                > MAX_COMMON_INTERLACING_INPUT_DIGITS
            ):
                raise OperationDomainValidationError(
                    location=("family", source_index, "polynomial", "terms"),
                    code="polynomial.common_interlacing_coefficient_digits",
                    message=(
                        "a common interlacing source coefficient exceeds the "
                        f"{MAX_COMMON_INTERLACING_INPUT_DIGITS}-digit input bound"
                    ),
                )


def _root_profile_from_worker(
    value: object,
    declared_factors: list[list[tuple[str, ...]]],
    factor_root_counts: list[int],
) -> SourceRootProfile:
    """Parse bounded worker structure without replaying exact factorization.

    The killable worker has already established all mathematical invariants
    (factorization, irreducibility, real-root isolation, source reconstruction).
    This parent performs only cheap structural checks: type, canonical shape,
    array bounds, membership in declared factors, and uniqueness of root
    identities.  No SymPy kernel work (irreducibility recognition, root
    isolation, divisibility, source reconstruction) is replayed.
    """

    if not isinstance(value, dict) or not isinstance(value.get("roots"), list):
        raise ValueError("malformed root profile")
    source_index = value.get("source_index")
    if type(source_index) is not int:
        raise ValueError("malformed root profile source index")
    declared_factor_set = {
        tuple(int(c) for c in f) for f in declared_factors
    }
    roots: list[PolynomialRealRoot] = []
    seen_identities: set[tuple[tuple[str, ...], int]] = set()
    factor_multiplicities: dict[tuple[int, ...], int] = {}
    for raw_root in value["roots"]:
        if not isinstance(raw_root, dict):
            raise ValueError("malformed root row")
        raw_value = raw_root.get("value")
        if not isinstance(raw_value, dict):
            raise ValueError("malformed algebraic value")
        polynomial = _CANONICAL_POLYNOMIAL.validate_python(raw_value.get("polynomial"))
        root_index = raw_value.get("real_root_index")
        if type(root_index) is not int or root_index < 0:
            raise ValueError("malformed algebraic root index")
        coefficients = [int(coefficient) for coefficient in polynomial]
        if len(coefficients) < 2:
            raise ValueError("worker root polynomial is constant")
        if coefficients[0] <= 0:
            raise ValueError("worker root polynomial has non-positive leading coefficient")
        content = 0
        for coefficient in coefficients:
            content = math.gcd(content, abs(coefficient))
        if content != 1:
            raise ValueError("worker root polynomial is not primitive canonical form")
        if root_index >= len(coefficients) - 1:
            raise ValueError("worker root index is outside its canonical factor")
        poly_key = tuple(int(c) for c in polynomial)
        if poly_key not in declared_factor_set:
            raise ValueError(
                "worker root polynomial is not a declared irreducible factor"
            )
        multiplicity = raw_root.get("multiplicity")
        if type(multiplicity) is not int or multiplicity < 1:
            raise ValueError("malformed root multiplicity")
        if poly_key in factor_multiplicities:
            if factor_multiplicities[poly_key] != multiplicity:
                raise ValueError("worker rows disagree on factor multiplicity")
        else:
            factor_multiplicities[poly_key] = multiplicity

        # Reject duplicate root identities: the same (polynomial, real_root_index)
        # must not appear in more than one row.
        identity = (polynomial, root_index)
        if identity in seen_identities:
            raise ValueError("worker profile contains duplicate root identity")
        seen_identities.add(identity)

        algebraic_value = RealAlgebraicValue._from_admitted_polynomial(
            polynomial=polynomial,
            real_root_index=root_index,
        )
        root_payload = dict(raw_root)
        root_payload["value"] = algebraic_value
        root_payload["multiplicity"] = multiplicity
        roots.append(
            PolynomialRealRoot.model_validate(root_payload)
        )
    # Verify that every real root of each declared factor is reported
    factor_row_counts: dict[tuple[int, ...], int] = {}
    for raw_root in value["roots"]:
        if not isinstance(raw_root, dict):
            continue
        rv = raw_root.get("value")
        if not isinstance(rv, dict):
            continue
        pk = tuple(int(c) for c in _CANONICAL_POLYNOMIAL.validate_python(rv.get("polynomial")))
        factor_row_counts[pk] = factor_row_counts.get(pk, 0) + 1
    if factor_root_counts:
        declared_set = {tuple(int(c) for c in f) for f in declared_factors}
        for idx, (factor, count) in enumerate(zip(declared_factors, factor_root_counts)):
            fk = tuple(int(c) for c in factor)
            if fk in factor_row_counts:
                if factor_row_counts[fk] != count:
                    raise ValueError("worker omitted real roots of a source factor")
            elif count > 0:
                raise ValueError("worker omitted real roots of a source factor")

    if roots and not declared_factor_set:
        raise ValueError("worker omitted factor declarations despite reporting roots")
    return SourceRootProfile.model_validate(
        {"source_index": source_index, "roots": tuple(roots)}
    )


def _profile_from_worker(
    payload: dict[str, Any],
    family: tuple[LabelledRationalPolynomial, ...],
) -> CommonInterlacingProfile:
    raw_source_factors = payload.get("source_factors", [])
    raw_factor_root_counts = payload.get("source_factor_root_counts", [])
    raw_profiles = payload.get("root_profiles")
    if not isinstance(raw_profiles, list):
        raise ValueError("malformed root profiles")
    root_profiles: list[SourceRootProfile] = []
    for value in raw_profiles:
        if not isinstance(value, dict) or type(value.get("source_index")) is not int:
            raise ValueError("malformed root profile source index")
        source_index = value["source_index"]
        if not 0 <= source_index < len(family):
            raise ValueError("worker root profile source index is out of range")
        declared = (
            raw_source_factors[source_index]
            if source_index < len(raw_source_factors)
            else []
        )
        root_counts = (
            raw_factor_root_counts[source_index]
            if source_index < len(raw_factor_root_counts)
            else []
        )
        root_profiles.append(_root_profile_from_worker(value, declared, root_counts))
    root_profiles_tuple = tuple(root_profiles)
    outcome = _OUTCOME.validate_python(payload.get("outcome"))
    return CommonInterlacingProfile.model_validate(
        {
            "family": family,
            "root_profiles": root_profiles_tuple,
            "outcome": outcome,
        }
    )


def _domain_error(payload: dict[str, Any]) -> OperationDomainValidationError:
    errors = payload.get("errors")
    if not isinstance(errors, list) or len(errors) != 1:
        raise RuntimeError("common-interlacing worker returned malformed diagnostics")
    error = errors[0]
    if not isinstance(error, dict):
        raise RuntimeError("common-interlacing worker returned malformed diagnostics")
    location = error.get("loc")
    code = error.get("type")
    message = error.get("msg")
    if (
        not isinstance(location, list)
        or not all(isinstance(item, (str, int)) for item in location)
        or not isinstance(code, str)
        or not isinstance(message, str)
    ):
        raise RuntimeError("common-interlacing worker returned malformed diagnostics")
    return OperationDomainValidationError(
        location=tuple(location),
        code=code,
        message=message,
    )


def run_common_interlacing_profile(
    family: tuple[LabelledRationalPolynomial, ...],
) -> CommonInterlacingProfile:
    """Run factorization, isolation, and comparison in one killable worker."""

    from jacobian.process import (
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
    )

    _preflight_family_size(family)
    try:
        _require_family_shape(family)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("family",),
            code=exc.type,
            message=str(exc),
        ) from exc

    execution = current_request_execution()
    started = execution.started_at if execution is not None else time.monotonic()
    owner_deadline = started + _WALL_SECONDS
    deadline = min(
        owner_deadline,
        execution.deadline
        if execution is not None and execution.deadline is not None
        else owner_deadline,
    )
    bind_request_deadline(deadline)
    if deadline <= time.monotonic():
        raise OperationExecutionTimeoutError(
            "request deadline expired before common-interlacing execution"
        )

    request_bytes = json.dumps(
        [source.model_dump(mode="json") for source in family],
        separators=(",", ":"),
    ).encode("utf-8")
    try:
        with TemporaryDirectory(prefix="jacobian-common-interlacing-") as directory:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OperationExecutionTimeoutError(
                    "request deadline expired before common-interlacing backend launch"
                )
            completed = run_bounded_process(
                [sys.executable, str(_WORKER)],
                input_bytes=request_bytes,
                timeout_seconds=remaining,
                environment={
                    **worker_environment(locale="C.UTF-8"),
                    "PYTHONPATH": str(Path(__file__).resolve().parents[4]),
                },
                stdout_limit=_STDOUT_LIMIT,
                stderr_limit=_STDERR_LIMIT,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(remaining)),
                    address_space_bytes=_ADDRESS_SPACE_BYTES,
                    file_size_bytes=1024 * 1024,
                ),
                cwd=directory,
            )
    except OSError as exc:
        raise RuntimeError(
            "bounded common-interlacing worker could not be started"
        ) from exc

    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "request cancelled during common-interlacing execution"
        )
    if completed.timed_out or time.monotonic() >= deadline:
        raise OperationExecutionTimeoutError(
            "request deadline expired during common-interlacing execution"
        )
    if completed.stdout_exceeded or completed.stderr_exceeded:
        raise RuntimeError(
            "bounded common-interlacing worker exceeded its output limit"
        )
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "bounded common-interlacing worker returned malformed output"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            "bounded common-interlacing worker returned malformed output"
        )
    if (
        completed.returncode == 0
        and payload.get("ok") is False
        and payload.get("kind") == "domain"
    ):
        raise _domain_error(payload)
    if completed.returncode != 0 or payload.get("ok") is not True:
        raise RuntimeError(
            "bounded common-interlacing worker did not establish a profile"
        )
    try:
        result = _profile_from_worker(payload, family)
    except (TypeError, ValueError, ValidationError) as exc:
        raise RuntimeError(
            "bounded common-interlacing worker returned a malformed profile"
        ) from exc
    if time.monotonic() >= deadline:
        raise OperationExecutionTimeoutError(
            "request deadline expired during common-interlacing result construction"
        )
    return result


__all__ = ["run_common_interlacing_profile"]
