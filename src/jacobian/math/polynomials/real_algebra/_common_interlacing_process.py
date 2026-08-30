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

from jacobian._exact import CanonicalInteger
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.algebraic_numbers.real import RealAlgebraicValue
from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy
from jacobian.math.polynomials.real_algebra._common_interlacing_models import (
    MAX_COMMON_INTERLACING_FAMILY_SIZE,
    MAX_COMMON_INTERLACING_INPUT_DIGITS,
    MAX_COMMON_INTERLACING_SOURCE_TERMS,
    CommonInterlacingOutcome,
    CommonInterlacingProfile,
    LabelledRationalPolynomial,
    PolynomialRealRoot,
    SourceRootProfile,
)

_WORKER = Path(__file__).resolve().with_name("_common_interlacing_worker.py")
_WALL_SECONDS = 1_800.0
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
    source: LabelledRationalPolynomial,
) -> SourceRootProfile:
    """Parse bounded worker structure without replaying exact factorization."""

    if not isinstance(value, dict) or not isinstance(value.get("roots"), list):
        raise ValueError("malformed root profile")
    source_poly = rational_polynomial_to_sympy(source.polynomial)
    reconstruction = source_poly.one()
    root_factors: dict[tuple[int, ...], tuple[Any, int]] = {}
    roots: list[PolynomialRealRoot] = []
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
        root_poly = source_poly.from_list(
            [int(coefficient) for coefficient in polynomial],
            gens=source_poly.gens,
            domain="QQ",
        )
        if root_poly.degree() < 1:
            raise ValueError("worker root polynomial is constant")
        coefficients = [int(coefficient) for coefficient in polynomial]
        content = 0
        for coefficient in coefficients:
            content = math.gcd(content, abs(coefficient))
        if coefficients[0] <= 0 or content != 1:
            raise ValueError("worker root polynomial is not primitive canonical form")
        if root_index >= root_poly.degree():
            raise ValueError("worker root index is outside its canonical factor")
        if not source_poly.rem(root_poly).is_zero:
            raise ValueError("worker root polynomial is not a factor of its source")
        multiplicity = raw_root.get("multiplicity")
        if type(multiplicity) is not int or multiplicity < 1:
            raise ValueError("malformed root multiplicity")
        poly_key = tuple(int(c) for c in polynomial)
        if poly_key not in root_factors:
            root_factors[poly_key] = (root_poly, multiplicity)
        algebraic_value = RealAlgebraicValue._from_admitted_polynomial(
            polynomial=polynomial,
            real_root_index=root_index,
        )
        root_payload = dict(raw_root)
        root_payload["value"] = algebraic_value
        roots.append(PolynomialRealRoot.model_validate(root_payload))
    reconstruction = source_poly.one()
    for root_poly, mult in root_factors.values():
        reconstruction *= root_poly ** mult
    if not source_poly.div(reconstruction)[1].is_zero:
        raise ValueError(
            "worker root rows do not divide the source polynomial"
        )
    return SourceRootProfile.model_validate(
        {"source_index": value.get("source_index"), "roots": roots}
    )


def _profile_from_worker(
    payload: dict[str, Any],
    family: tuple[LabelledRationalPolynomial, ...],
) -> CommonInterlacingProfile:
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
        root_profiles.append(_root_profile_from_worker(value, family[source_index]))
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

    _preflight_family_size(family)
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
                environment=worker_environment(locale="C.UTF-8"),
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
