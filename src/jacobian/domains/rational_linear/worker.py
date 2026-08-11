"""Isolated Python-FLINT worker for rational-linear candidates."""

from __future__ import annotations

import importlib
import sys
from fractions import Fraction
from typing import Any

from pydantic import ValidationError

from jacobian.canonical import (
    CanonicalizationError,
    canonicalize_json,
    format_canonical_integer,
    loads_strict_json,
)
from jacobian.contracts.exact import CanonicalRational
from jacobian.domains.rational_linear.protocol import (
    RationalLinearCertificateProduced,
    RationalLinearInconsistencyWorkerRequest,
    RationalLinearInconsistencyWorkerResponse,
    RationalLinearNoCertificateProduced,
    RationalLinearNoSolutionProduced,
    RationalLinearSolutionProduced,
    RationalLinearSolutionWorkerResponse,
    RationalLinearWorkerFailure,
    RationalLinearWorkerRequest,
    parse_rational_linear_worker_request,
)


def _solve(
    coefficients: list[list[Fraction]], rhs: list[Fraction], flint: Any
) -> list[Any] | None:
    augmented = flint.fmpq_mat(
        [
            [flint.fmpq(value.numerator, value.denominator) for value in row]
            + [flint.fmpq(bound.numerator, bound.denominator)]
            for row, bound in zip(coefficients, rhs, strict=True)
        ]
    )
    reduced, _ = augmented.rref()
    columns = len(coefficients[0])
    values = [flint.fmpq(0) for _ in range(columns)]
    for row in range(reduced.nrows()):
        pivot = next(
            (column for column in range(columns) if reduced[row, column] != 0), None
        )
        if pivot is None:
            if reduced[row, columns] != 0:
                return None
            continue
        values[pivot] = reduced[row, columns]
    return values


def _canonical_rational(value: Any) -> CanonicalRational:
    """Convert a backend-native fmpq through the canonical wire representation."""

    return CanonicalRational(
        num=format_canonical_integer(int(value.numerator)),
        den=format_canonical_integer(int(value.denominator)),
    )


def _run(
    worker_request: RationalLinearWorkerRequest,
) -> RationalLinearSolutionWorkerResponse | RationalLinearInconsistencyWorkerResponse:
    system = worker_request.request.system
    coefficients = [
        [value.as_fraction() for value in row] for row in system.coefficients.entries
    ]
    bounds = [value.as_fraction() for value in system.rhs]
    flint: Any = importlib.import_module("flint")
    if getattr(flint, "__version__", None) != "0.9.0":
        raise ValueError("unsupported Python-FLINT version")
    if isinstance(worker_request, RationalLinearInconsistencyWorkerRequest):
        row_count = len(coefficients)
        column_count = len(coefficients[0])
        dual = [
            [coefficients[row][column] for row in range(row_count)]
            for column in range(column_count)
        ]
        dual.append(bounds)
        values = _solve(dual, [Fraction(0)] * column_count + [Fraction(1)], flint)
        if values is None:
            return RationalLinearNoCertificateProduced(
                protocol="jacobian.rational-linear-inconsistency-worker/v1",
                status="NO_CERTIFICATE_PRODUCED",
            )
        return RationalLinearCertificateProduced(
            protocol="jacobian.rational-linear-inconsistency-worker/v1",
            status="CERTIFICATE_PRODUCED",
            left_witness=tuple(_canonical_rational(value) for value in values),
            rhs_pairing=CanonicalRational(num="1", den="1"),
        )
    values = _solve(coefficients, bounds, flint)
    if values is None:
        return RationalLinearNoSolutionProduced(
            protocol="jacobian.rational-linear-solution-worker/v1",
            status="NO_SOLUTION_PRODUCED",
        )
    return RationalLinearSolutionProduced(
        protocol="jacobian.rational-linear-solution-worker/v1",
        status="SOLUTION_PRODUCED",
        values=tuple(_canonical_rational(value) for value in values),
    )


def main() -> int:
    worker_request: RationalLinearWorkerRequest | None = None
    try:
        worker_request = parse_rational_linear_worker_request(
            loads_strict_json(sys.stdin.buffer.read())
        )
        result = _run(worker_request)
        sys.stdout.buffer.write(
            canonicalize_json(result.model_dump(mode="json")) + b"\n"
        )
        return 0
    except (CanonicalizationError, ValidationError):
        failure = RationalLinearWorkerFailure(
            status="ERROR",
            error_code="INVALID_REQUEST",
        )
        sys.stdout.buffer.write(
            canonicalize_json(failure.model_dump(mode="json")) + b"\n"
        )
        return 2
    except Exception:  # pragma: no cover - process boundary
        failure = RationalLinearWorkerFailure(
            status="ERROR",
            error_code=(
                "INVALID_REQUEST" if worker_request is None else "EXECUTION_FAILED"
            ),
        )
        sys.stdout.buffer.write(
            canonicalize_json(failure.model_dump(mode="json")) + b"\n"
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
