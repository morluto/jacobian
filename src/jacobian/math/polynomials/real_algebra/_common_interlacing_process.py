"""Killable request-scoped process boundary for common interlacing."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from fractions import Fraction
from math import gcd, lcm
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
from jacobian.canonical import (
    CanonicalizationError,
    format_canonical_integer,
    loads_strict_json,
    parse_canonical_integer,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.algebraic_numbers.real import (
    MAX_REAL_ALGEBRAIC_DEGREE,
    RealAlgebraicValue,
)
from jacobian.math.polynomials.real_algebra._common_interlacing import (
    _factor_digit_bound,
    _preflight_common_interlacing_sources,
)
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


def _multiply_integer_polynomials(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[int, ...]:
    """Multiply two dense integer polynomials (leading coefficient first)."""

    if not left or not right:
        return ()
    product = [0] * (len(left) + len(right) - 1)
    for i, lc in enumerate(left):
        for j, rc in enumerate(right):
            product[i + j] += lc * rc
    return tuple(product)


def _source_to_dense_int(
    source: LabelledRationalPolynomial,
) -> tuple[int, ...]:
    """Convert a univariate RationalPolynomial to a dense primitive ZZ[x].

    Returns coefficients in descending degree order with positive leading
    coefficient and content one.
    """

    terms = source.polynomial.polynomial.terms
    degree = terms[0].exponents[0] if terms else 0
    dense_rational: list[Fraction] = [Fraction(0)] * (degree + 1)
    for term in terms:
        dense_rational[degree - term.exponents[0]] = term.coefficient.as_fraction()
    common_denominator = 1
    for coeff in dense_rational:
        common_denominator = lcm(common_denominator, coeff.denominator)
    dense_int: list[int] = [
        int(int(coeff.numerator) * (common_denominator // int(coeff.denominator)))
        for coeff in dense_rational
    ]
    content: int = 0
    for coeff in dense_int:  # type: ignore[assignment]
        content = gcd(content, abs(int(coeff)))
    if content > 1:
        dense_int = [c // content for c in dense_int]
    if dense_int and dense_int[0] < 0:
        dense_int = [-c for c in dense_int]
    return tuple(dense_int)


def _verify_declared_factors(  # noqa: C901
    source: LabelledRationalPolynomial,
    declared_factors: list[Any],
) -> None:
    """Verify the worker's declared factors reconstruct the retained source.

    Each declared factor is a [coefficients, multiplicity] pair.  Multiplies
    each factor raised to its multiplicity and checks the product equals the
    source polynomial.  Uses only integer polynomial arithmetic, not SymPy,
    so no kernel work is replayed in the parent.
    """

    if not declared_factors:
        raise ValueError("worker omitted source factor declarations")
    source_dense = _source_to_dense_int(source)
    source_degree = max(len(source_dense) - 1, 0)
    source_height_digits = max(
        len(format_canonical_integer(coefficient).lstrip("-"))
        for coefficient in source_dense
    )
    factor_digit_bound = _factor_digit_bound(source_degree, source_height_digits)
    if len(declared_factors) > source_degree:
        raise ValueError("worker declared more factors than source degree")
    product: tuple[int, ...] = (1,)
    seen_factor_coefficients: set[tuple[int, ...]] = set()
    aggregate_degree = 0
    for entry in declared_factors:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError("malformed source factor declaration")
        factor_coeffs, multiplicity = entry
        if type(multiplicity) is not int or multiplicity < 1:
            raise ValueError("malformed source factor multiplicity")
        if multiplicity > source_degree:
            raise ValueError("worker factor multiplicity exceeds source degree")
        if not isinstance(factor_coeffs, (list, tuple)) or not (
            2 <= len(factor_coeffs) <= source_degree + 1
        ):
            raise ValueError("worker factor degree exceeds source degree")
        if any(
            not isinstance(coefficient, str)
            or len(coefficient.lstrip("-")) > factor_digit_bound
            for coefficient in factor_coeffs
        ):
            raise ValueError("worker factor coefficient exceeds its height bound")
        factor_dense: tuple[int, ...]
        try:
            factor_dense = tuple(
                parse_canonical_integer(coefficient) for coefficient in factor_coeffs
            )
        except (TypeError, ValueError, CanonicalizationError) as exc:
            raise ValueError("worker factor coefficient is not canonical") from exc
        if any(
            format_canonical_integer(value) != coefficient
            for value, coefficient in zip(factor_dense, factor_coeffs, strict=True)
        ):
            raise ValueError("worker factor coefficient is not canonical")
        aggregate_degree += (len(factor_dense) - 1) * multiplicity
        if aggregate_degree > source_degree:
            raise ValueError("worker factor degrees exceed source degree")
        if factor_dense[0] <= 0:
            raise ValueError("worker factor has non-positive leading coefficient")
        content: int = 0
        for coefficient in factor_dense:
            content = gcd(content, abs(coefficient))
        if content != 1:
            raise ValueError("worker factor is not primitive canonical form")
        if factor_dense in seen_factor_coefficients:
            raise ValueError("worker declared a duplicate source factor")
        seen_factor_coefficients.add(factor_dense)
        for _ in range(multiplicity):
            product = _multiply_integer_polynomials(product, factor_dense)
    if product != source_dense:
        raise ValueError("declared factors do not reconstruct the source polynomial")


def _root_profile_from_worker(  # noqa: C901
    value: object,
    declared_factors: list[Any],
    factor_root_counts: list[Any],
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
    if not isinstance(factor_root_counts, list):
        raise ValueError("worker factor root counts are malformed")
    source_index = value.get("source_index")
    if type(source_index) is not int:
        raise ValueError("malformed root profile source index")
    declared_factor_set = {
        tuple(parse_canonical_integer(c) for c in entry[0])
        for entry in declared_factors
    }
    if not declared_factor_set:
        raise ValueError("worker omitted source factor declarations")

    roots: list[PolynomialRealRoot] = []
    seen_identities: set[tuple[tuple[str, ...], int]] = set()
    factor_multiplicities: dict[tuple[int, ...], int] = {}
    factor_row_counts: dict[tuple[int, ...], int] = {}
    factor_root_indices: dict[tuple[int, ...], set[int]] = {}
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
            raise ValueError(
                "worker root polynomial has non-positive leading coefficient"
            )
        content: int = 0
        for coefficient in coefficients:
            content = math.gcd(content, abs(coefficient))
        if content != 1:
            raise ValueError("worker root polynomial is not primitive canonical form")
        if root_index >= len(coefficients) - 1:
            raise ValueError("worker root index is outside its canonical factor")
        if len(coefficients) - 1 > MAX_REAL_ALGEBRAIC_DEGREE:
            raise ValueError(
                "worker root factor exceeds the algebraic value degree bound"
            )
        poly_key = tuple(int(c) for c in polynomial)
        if poly_key not in declared_factor_set:
            raise ValueError(
                "worker root polynomial is not a declared irreducible factor"
            )
        multiplicity = raw_root.get("multiplicity")
        if type(multiplicity) is not int or multiplicity < 1:
            raise ValueError("malformed root multiplicity")
        declared_multiplicity = next(
            int(entry[1])
            for entry in declared_factors
            if tuple(parse_canonical_integer(c) for c in entry[0]) == poly_key
        )
        if multiplicity != declared_multiplicity:
            raise ValueError("worker root multiplicity disagrees with its factor")
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

        # Count rows per factor for the completeness check.
        factor_row_counts[poly_key] = factor_row_counts.get(poly_key, 0) + 1
        factor_root_indices.setdefault(poly_key, set()).add(root_index)

        algebraic_value = RealAlgebraicValue._from_admitted_polynomial(
            polynomial=polynomial,
            real_root_index=root_index,
        )
        root_payload = dict(raw_root)
        root_payload["value"] = algebraic_value
        root_payload["multiplicity"] = multiplicity
        roots.append(PolynomialRealRoot.model_validate(root_payload))

    # Require factor-root-count projection to be complete and fail closed.
    if not factor_root_counts or len(factor_root_counts) != len(declared_factors):
        raise ValueError("worker omitted or truncated factor root counts")
    for idx, entry in enumerate(declared_factors):
        fk = tuple(int(c) for c in entry[0])
        expected = factor_root_counts[idx]
        if type(expected) is not int or not 0 <= expected <= len(fk) - 1:
            raise ValueError("worker factor root count is malformed")
        actual = factor_row_counts.get(fk, 0)
        if expected != actual:
            raise ValueError("worker omitted real roots of a source factor")
        if factor_root_indices.get(fk, set()) != set(range(expected)):
            raise ValueError("worker root indices do not match projected real roots")

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

    # Verify declared factors reconstruct each retained source.
    if not isinstance(raw_source_factors, list) or len(raw_source_factors) != len(
        family
    ):
        raise ValueError("worker source factor declarations are missing or malformed")
    for source_index, source in enumerate(family):
        declared = raw_source_factors[source_index]
        _verify_declared_factors(source, declared)

    # Require factor-root-count projection aligned one-for-one with family.
    if not isinstance(raw_factor_root_counts, list) or len(
        raw_factor_root_counts
    ) != len(family):
        raise ValueError("worker factor root counts are missing or malformed")
    if len(raw_profiles) != len(family):
        raise ValueError("worker root profiles are missing or malformed")
    _require_family_shape(family)

    root_profiles: list[SourceRootProfile] = []
    for value in raw_profiles:
        if not isinstance(value, dict) or type(value.get("source_index")) is not int:
            raise ValueError("malformed root profile source index")
        source_index = value["source_index"]
        if not 0 <= source_index < len(family):
            raise ValueError("worker root profile source index is out of range")
        declared = raw_source_factors[source_index]
        root_counts = raw_factor_root_counts[source_index]
        root_profiles.append(_root_profile_from_worker(value, declared, root_counts))
    root_profiles_tuple = tuple(root_profiles)
    outcome = _OUTCOME.validate_python(payload.get("outcome"))
    try:
        CommonInterlacingProfile.model_validate(
            {
                "family": family,
                "root_profiles": root_profiles_tuple,
                "outcome": outcome,
            }
        )
    except (PydanticCustomError, ValidationError) as exc:
        raise ValueError("worker returned an inconsistent root profile") from exc
    return CommonInterlacingProfile._from_kernel(
        family=family,
        root_profiles=root_profiles_tuple,
        outcome=outcome,
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
    _preflight_common_interlacing_sources(family)

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
    request_digest = hashlib.sha256(request_bytes).hexdigest()
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
        payload = loads_strict_json(completed.stdout)
    except (CanonicalizationError, ValueError) as exc:
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
    if payload.get("request_digest") != request_digest:
        raise RuntimeError(
            "bounded common-interlacing worker returned a result for another request"
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
