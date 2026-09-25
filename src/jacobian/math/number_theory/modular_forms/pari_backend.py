"""Exact PARI adapter for rational Gamma0 modular-form basis prefixes."""

from __future__ import annotations

import hashlib
import math
import sys
import time
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.canonical import (
    encode_strict_json,
    format_canonical_integer,
    loads_strict_json,
    parse_canonical_integer,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.cyclic_linear._models import RationalCyclotomicField
from jacobian.math.number_theory.characters.operations import (
    require_complete_character_group,
)
from jacobian.math.number_theory.characters.values import DirichletCharacter
from jacobian.math.number_theory.modular_forms.values import ModularFormSpace

PARI_STURM_RREF_BASIS_ID: Literal["gamma0-rational-gamma0-sturm-rref-v1"] = (
    "gamma0-rational-gamma0-sturm-rref-v1"
)
MAX_PARI_BASIS_LEVEL = 10_000
MAX_PARI_BASIS_PRECISION = 128
MAX_PARI_BASIS_WEIGHT = 120
MAX_PARI_BASIS_DIMENSION = 32
MAX_PARI_BASIS_COEFFICIENT_DIGITS = 512
MAX_PARI_BASIS_ALLOCATION_BYTES = 8 * 1024 * 1024
MAX_PARI_BASIS_WORK = 50_000_000
_WORKER = Path(__file__).resolve().with_name("_pari_basis_worker.py")
_ATKIN_WORKER = Path(__file__).resolve().with_name("_pari_atkin_worker.py")
_WORKER_TIMEOUT_SECONDS = 30.0
_WORKER_ADDRESS_SPACE_BYTES = 2 * 1024 * 1024 * 1024
_WORKER_FILE_SIZE_BYTES = 8 * 1024 * 1024
_WORKER_STDERR_BYTES = 64 * 1024


def _rref_prefix(
    vectors: tuple[tuple[Fraction, ...], ...], pivots_through: int
) -> tuple[tuple[Fraction, ...], ...]:
    """Canonicalize a subspace of Q[[q]] by its reduced q-prefix frame."""

    rows = [list(vector) for vector in vectors]
    pivot_row = 0
    for column in range(pivots_through):
        pivot = next(
            (row for row in range(pivot_row, len(rows)) if rows[row][column]), None
        )
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        scale = rows[pivot_row][column]
        rows[pivot_row] = [value / scale for value in rows[pivot_row]]
        for row in range(len(rows)):
            if row == pivot_row or not rows[row][column]:
                continue
            scale = rows[row][column]
            rows[row] = [
                left - scale * right
                for left, right in zip(rows[row], rows[pivot_row], strict=True)
            ]
        pivot_row += 1
        if pivot_row == len(rows):
            break
    if pivot_row != len(rows):
        raise RuntimeError("PARI basis is not independent through the Sturm bound")
    return tuple(tuple(row) for row in rows)


def _rref_coefficient_digit_bound(dimension: int) -> int:
    """Bound reduced minors after clearing all raw rational denominators."""

    if dimension == 0:
        return 1
    determinant_sum_digits = len(str(math.factorial(dimension)))
    return (
        2 * dimension * dimension * MAX_PARI_BASIS_COEFFICIENT_DIGITS
        + determinant_sum_digits
    )


def pari_gamma0_rational_basis(
    space: ModularFormSpace,
    precision: int,
    sturm_precision: int,
    expected_dimension: int,
    *,
    admitted_work: int,
    admitted_rref_digits: int,
) -> tuple[tuple[Fraction, ...], ...]:
    """Return PARI's exact basis normalized to Jacobian's q-Sturm RREF frame.

    All scalar and aggregate bounds are checked by the caller before this
    adapter enters PARI. The backend's basis ordering and scaling are erased by
    exact row reduction through a Sturm determining prefix.
    """

    if admitted_work > MAX_PARI_BASIS_WORK:
        raise OperationResourceAdmissionError(
            location=("space",),
            code="modular_form.pari_basis_work_bound",
            message="PARI modular-form basis work exceeds its admitted envelope",
        )
    if space.character != "TRIVIAL" or space.coefficient_domain != "QQ":
        raise OperationDomainValidationError(
            location=("space",),
            code="modular_form.pari_basis_parent",
            message="the PARI basis adapter supports rational trivial-character spaces",
        )
    response = _run_basis_worker(space, precision, expected_dimension)
    if response.get("kind") != "complete":
        raise RuntimeError("PARI modular-form basis worker did not complete")
    backend_dimension = response.get("backend_dimension")
    if type(backend_dimension) is not int:
        raise RuntimeError(
            "PARI modular-form basis worker returned an invalid dimension"
        )
    if backend_dimension != expected_dimension:
        raise RuntimeError(
            "PARI and the exact Gamma0 dimension formula disagree "
            f"({backend_dimension} != {expected_dimension})"
        )
    if backend_dimension == 0:
        return ()
    vectors = _decode_basis_vectors(
        response.get("vectors"), expected_dimension, precision
    )
    normalized = _rref_prefix(vectors, sturm_precision)
    max_digits = max(
        (
            max(
                len(str(abs(coefficient.numerator))),
                len(str(coefficient.denominator)),
            )
            for vector in normalized
            for coefficient in vector
        ),
        default=1,
    )
    if max_digits > admitted_rref_digits:
        raise OperationResourceAdmissionError(
            location=("space",),
            code="modular_form.pari_basis_coefficient_growth",
            message="PARI basis coefficient exceeds the exact digit envelope",
        )
    # This check is deliberately redundant with admission: it protects the
    # canonical encoder if PARI returns an unexpected rational representation.
    estimated_bytes = expected_dimension * precision * (2 * admitted_rref_digits + 32)
    if estimated_bytes > MAX_PARI_BASIS_ALLOCATION_BYTES:
        raise OperationResourceAdmissionError(
            location=("space",),
            code="modular_form.pari_basis_output_bound",
            message="PARI basis coefficients exceed the bounded result size",
        )
    return normalized


def _pari_character_request(space: ModularFormSpace) -> dict[str, object]:
    """Serialize a character using Jacobian's complete unit-coordinate data.

    The worker maps this character to the generators returned by its own
    ``znstar(N, 1)`` instance. No ordering agreement between the two group
    presentations is assumed.
    """

    character = space.character
    field = space.coefficient_domain
    if not isinstance(character, DirichletCharacter) or not isinstance(
        field, RationalCyclotomicField
    ):
        raise OperationDomainValidationError(
            location=("space",),
            code="modular_form.pari_character_parent",
            message="PARI character serialization requires an explicit character and cyclotomic field",
        )
    if (
        getattr(field, "domain", None) != "QQ_CYCLOTOMIC"
        or type(getattr(field, "order", None)) is not int
        or getattr(field, "generator", None) != "CLASS_OF_X"
    ):
        raise OperationDomainValidationError(
            location=("space", "coefficient_domain"),
            code="modular_form.pari_character_field_invalid",
            message="PARI character serialization requires a canonical cyclotomic field",
        )
    group = require_complete_character_group(character.group)
    if group.modulus != space.level:
        raise OperationDomainValidationError(
            location=("space", "character"),
            code="modular_form.pari_character_level",
            message="PARI character modulus must equal the modular-form level",
        )
    field_order = getattr(field, "order", None)
    group_exponent = group.exponent
    if (
        type(field_order) is not int
        or field_order < 1
        or field_order > 128
        or type(group_exponent) is not int
        or group_exponent < 1
    ):
        raise OperationResourceAdmissionError(
            location=("space", "coefficient_domain"),
            code="modular_form.pari_character_field_bound",
            message="PARI character root order exceeds the bounded bridge envelope",
        )
    coordinates = getattr(character, "coordinates", None)
    if (
        type(coordinates) is not tuple
        or len(coordinates) != len(group.generator_orders)
        or any(
            type(coordinate) is not int or not 0 <= coordinate < order
            for coordinate, order in zip(
                coordinates, group.generator_orders, strict=True
            )
        )
    ):
        raise OperationDomainValidationError(
            location=("space", "character", "coordinates"),
            code="modular_form.pari_character_coordinates",
            message="PARI character coordinates must lie on every complete dual-group axis",
        )
    from math import gcd, lcm

    character_order = 1
    for coordinate, order in zip(
        character.coordinates, group.generator_orders, strict=True
    ):
        character_order = lcm(character_order, order // gcd(coordinate, order))
    if field.order % character_order:
        raise OperationDomainValidationError(
            location=("space", "coefficient_domain"),
            code="modular_form.pari_character_field_mismatch",
            message="explicit coefficient-field root order must contain all character values",
        )
    verification_work = len(group.unit_residues) * max(1, len(group.generator_orders))
    if verification_work > 100_000:
        raise OperationResourceAdmissionError(
            location=("space", "character"),
            code="modular_form.pari_character_check_work_bound",
            message="exhaustive PARI character agreement check exceeds its work envelope",
        )
    return {
        "coefficient_field_order": field.order,
        "character": {
            "modulus": group.modulus,
            "unit_residues": list(group.unit_residues),
            "generator_orders": list(group.generator_orders),
            "unit_coordinates": [list(row) for row in group.unit_coordinates],
            "coordinates": list(character.coordinates),
        },
    }


def pari_gamma0_atkin_matrix(
    space: ModularFormSpace,
    divisor: int,
    precision: int,
    basis_vectors: tuple[tuple[Fraction, ...], ...],
    *,
    admitted_work: int,
    admitted_allocation_bytes: int,
) -> tuple[tuple[Fraction, ...], ...]:
    """Return W_Q's exact matrix in Jacobian's admitted basis."""
    if admitted_work > MAX_PARI_BASIS_WORK:
        raise OperationResourceAdmissionError(
            location=("form",),
            code="modular_form.atkin_lehner_work_bound",
            message="Atkin-Lehner work exceeds its admitted envelope",
        )
    if admitted_allocation_bytes > MAX_PARI_BASIS_ALLOCATION_BYTES:
        raise OperationResourceAdmissionError(
            location=("form",),
            code="modular_form.atkin_lehner_output_bound",
            message="Atkin-Lehner output exceeds its admitted envelope",
        )
    try:
        import cypari  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise OperationResourceAdmissionError(
            location=("form", "space"),
            code="modular_form.pari_backend_unavailable",
            message="Atkin-Lehner transforms require the PARI backend",
        ) from exc
    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return pari_gamma0_atkin_matrix(
                space,
                divisor,
                precision,
                basis_vectors,
                admitted_work=admitted_work,
                admitted_allocation_bytes=admitted_allocation_bytes,
            )
    if execution.deadline is not None:
        deadline = execution.deadline
        worker_wall_seconds = deadline - execution.started_at
    else:
        deadline = execution.started_at + _WORKER_TIMEOUT_SECONDS
        worker_wall_seconds = _WORKER_TIMEOUT_SECONDS
    bind_request_deadline(deadline)
    request_checkpoint("before PARI Atkin-Lehner worker")
    payload: dict[str, object] = {
        "level": space.level,
        "weight": space.weight,
        "kind": space.kind,
        "divisor": divisor,
        "precision": precision,
        "basis_vectors": [
            [
                [
                    format_canonical_integer(value.numerator),
                    format_canonical_integer(value.denominator),
                ]
                for value in vector
            ]
            for vector in basis_vectors
        ],
    }
    input_bytes = encode_strict_json(payload)
    digest = hashlib.sha256(input_bytes).hexdigest()
    dimension = len(basis_vectors)
    stdout_limit = dimension * dimension * (2 * 512 + 64) + 4096
    try:
        with TemporaryDirectory(prefix="jacobian-mfatkin-") as worker_directory:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OperationExecutionTimeoutError(
                    "request deadline expired before Atkin-Lehner worker launch"
                )
            from jacobian.process import (
                ProcessResourceLimits,
                run_checked_worker_process,
                worker_environment,
            )

            response = run_checked_worker_process(
                [sys.executable, str(_ATKIN_WORKER)],
                input_bytes=input_bytes,
                timeout_seconds=remaining,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=stdout_limit,
                stderr_limit=_WORKER_STDERR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(
                        math.ceil(_WORKER_TIMEOUT_SECONDS),
                        math.ceil(worker_wall_seconds),
                    ),
                    address_space_bytes=_WORKER_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_WORKER_FILE_SIZE_BYTES,
                ),
                cwd=worker_directory,
                decode_result=_decode_worker_json,
            )
    except (OperationExecutionCancelledError, OperationExecutionTimeoutError):
        raise
    except OSError as exc:
        request_checkpoint("during PARI Atkin-Lehner worker startup")
        raise RuntimeError("bounded Atkin-Lehner worker could not start") from exc
    request_checkpoint("after PARI Atkin-Lehner worker")
    if not isinstance(response, dict) or response.get("request_digest") != digest:
        raise RuntimeError("PARI Atkin-Lehner worker response digest mismatch")
    raw = response.get("matrix")
    if (
        type(raw) is not list
        or len(raw) != dimension
        or any(type(row) is not list or len(row) != dimension for row in raw)
    ):
        raise RuntimeError("PARI Atkin-Lehner worker returned a malformed matrix")
    matrix = []
    for row in raw:
        values = []
        for pair in row:
            if (
                type(pair) is not list
                or len(pair) != 2
                or any(type(part) is not str for part in pair)
            ):
                raise RuntimeError(
                    "PARI Atkin-Lehner worker returned an invalid rational"
                )
            value = Fraction(
                parse_canonical_integer(pair[0]), parse_canonical_integer(pair[1])
            )
            if max(len(str(abs(value.numerator))), len(str(value.denominator))) > 512:
                raise RuntimeError("PARI Atkin-Lehner matrix exceeds its height cap")
            values.append(value)
        matrix.append(tuple(values))
    return tuple(matrix)


def _run_basis_worker(
    space: ModularFormSpace,
    precision: int,
    expected_dimension: int,
    *,
    character_request: dict[str, object] | None = None,
) -> dict[str, object]:
    try:
        import cypari  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on installed runtime
        raise OperationResourceAdmissionError(
            location=("space",),
            code="modular_form.pari_backend_unavailable",
            message="Gamma0 basis construction requires the PARI backend",
        ) from exc
    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return _run_basis_worker(
                space,
                precision,
                expected_dimension,
                character_request=character_request,
            )
    if execution.deadline is not None:
        deadline = execution.deadline
        worker_wall_seconds = deadline - execution.started_at
    else:
        deadline = execution.started_at + _WORKER_TIMEOUT_SECONDS
        worker_wall_seconds = _WORKER_TIMEOUT_SECONDS
    bind_request_deadline(deadline)
    request_checkpoint("before PARI modular-form basis worker")
    payload: dict[str, object] = {
        "level": space.level,
        "weight": space.weight,
        "kind": space.kind,
        "precision": precision,
        "expected_dimension": expected_dimension,
    }
    if character_request is not None:
        payload.update(character_request)
    input_bytes = encode_strict_json(payload)
    digest = hashlib.sha256(input_bytes).hexdigest()
    stdout_limit = expected_dimension * precision * (2 * 512 + 64) + 4096
    if character_request is not None:
        stdout_limit *= 4
    try:
        with TemporaryDirectory(prefix="jacobian-mfbasis-") as worker_directory:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OperationExecutionTimeoutError(
                    "request deadline expired before modular-form basis worker launch"
                )
            from jacobian.process import (
                ProcessResourceLimits,
                run_checked_worker_process,
                worker_environment,
            )

            response = run_checked_worker_process(
                [sys.executable, str(_WORKER)],
                input_bytes=input_bytes,
                timeout_seconds=remaining,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=stdout_limit,
                stderr_limit=_WORKER_STDERR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(
                        math.ceil(_WORKER_TIMEOUT_SECONDS),
                        math.ceil(worker_wall_seconds),
                    ),
                    address_space_bytes=_WORKER_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_WORKER_FILE_SIZE_BYTES,
                ),
                cwd=worker_directory,
                decode_result=_decode_worker_json,
            )
    except (OperationExecutionCancelledError, OperationExecutionTimeoutError):
        raise
    except OSError as exc:
        request_checkpoint("during PARI modular-form basis worker startup")
        raise RuntimeError("bounded modular-form basis worker could not start") from exc
    request_checkpoint("after PARI modular-form basis worker")
    if not isinstance(response, dict) or response.get("request_digest") != digest:
        raise RuntimeError("PARI modular-form basis worker response digest mismatch")
    return response


def pari_character_basis(
    space: ModularFormSpace,
    precision: int,
    expected_dimension: int,
    *,
    character_request: dict[str, object] | None = None,
) -> tuple[tuple[tuple[Fraction, ...], ...], ...]:
    """Return character-valued q-prefixes in the declared cyclotomic power basis."""
    request = {
        **(
            _pari_character_request(space)
            if character_request is None
            else character_request
        ),
        "bridge_only": False,
    }
    response = _run_basis_worker(
        space,
        precision,
        expected_dimension,
        character_request=request,
    )
    if response.get("kind") != "character_complete":
        raise RuntimeError("PARI character basis worker did not complete")
    if response.get("coefficient_field_order") != space.coefficient_domain.order:  # type: ignore[union-attr]
        raise RuntimeError("PARI character basis worker changed its coefficient field")
    raw = response.get("vectors")
    degree = space.coefficient_domain.degree  # type: ignore[union-attr]
    if (
        type(raw) is not list
        or len(raw) != expected_dimension
        or any(type(vector) is not list or len(vector) != precision for vector in raw)
    ):
        raise RuntimeError("PARI returned a malformed character basis")
    vectors = []
    for vector in raw:
        terms = []
        for term in vector:
            if type(term) is not list or len(term) != degree:
                raise RuntimeError("PARI returned a malformed cyclotomic coefficient")
            coordinates = []
            for pair in term:
                if (
                    type(pair) is not list
                    or len(pair) != 2
                    or any(type(part) is not str for part in pair)
                ):
                    raise RuntimeError("PARI returned an invalid cyclotomic rational")
                coefficient = Fraction(
                    parse_canonical_integer(pair[0]),
                    parse_canonical_integer(pair[1]),
                )
                if (
                    max(
                        len(str(abs(coefficient.numerator))),
                        len(str(coefficient.denominator)),
                    )
                    > 30
                ):
                    raise RuntimeError(
                        "PARI character coefficient exceeds its height cap"
                    )
                coordinates.append(coefficient)
            terms.append(tuple(coordinates))
        vectors.append(tuple(terms))
    return tuple(vectors)


def _decode_basis_vectors(
    raw_vectors: object, dimension: int, precision: int
) -> tuple[tuple[Fraction, ...], ...]:
    if (
        type(raw_vectors) is not list
        or len(raw_vectors) != dimension
        or any(
            type(vector) is not list or len(vector) != precision
            for vector in raw_vectors
        )
    ):
        raise RuntimeError("PARI modular-form basis worker returned malformed vectors")
    vectors = []
    for vector in raw_vectors:
        if any(
            type(coefficient) is not list
            or len(coefficient) != 2
            or any(type(part) is not str for part in coefficient)
            for coefficient in vector
        ):
            raise RuntimeError(
                "PARI modular-form basis worker returned invalid rationals"
            )
        vectors.append(
            tuple(
                Fraction(
                    parse_canonical_integer(coefficient[0]),
                    parse_canonical_integer(coefficient[1]),
                )
                for coefficient in vector
            )
        )
    return tuple(vectors)


def _decode_worker_json(value: object) -> dict[str, object]:
    parsed = loads_strict_json(encode_strict_json(value))
    if not isinstance(parsed, dict):
        raise ValueError("PARI modular-form basis worker result must be an object")
    return parsed


__all__ = [
    "MAX_PARI_BASIS_ALLOCATION_BYTES",
    "MAX_PARI_BASIS_COEFFICIENT_DIGITS",
    "MAX_PARI_BASIS_DIMENSION",
    "MAX_PARI_BASIS_LEVEL",
    "MAX_PARI_BASIS_PRECISION",
    "MAX_PARI_BASIS_WEIGHT",
    "MAX_PARI_BASIS_WORK",
    "PARI_STURM_RREF_BASIS_ID",
    "pari_gamma0_rational_basis",
]
