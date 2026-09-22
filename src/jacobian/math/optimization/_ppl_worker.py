"""One-shot exact PPL worker for standard-form rational linear programs."""

from __future__ import annotations

import sys
from fractions import Fraction
from importlib.metadata import version
from typing import Any

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
from jacobian._worker_protocol import encode_worker_result_frame
from jacobian.canonical import (
    CanonicalLimits,
    format_canonical_integer,
    loads_strict_json,
    parse_canonical_integer,
)
from jacobian.math.optimization._models import (
    MAX_LINEAR_PROGRAM_CONSTRAINTS,
    MAX_LINEAR_PROGRAM_VARIABLES,
)
from jacobian.math.optimization._ppl import ExactLinearOutcome, solve_standard_form

_PROTOCOL_VERSION = 1
_SUPPORTED_PPLPY_VERSION = "0.9.0"
# The normalized matrix has at most 64*32 entries. Each rational has two
# canonical components bounded by the admitted result-height envelope.
_MAX_INPUT_BYTES = 192 * 1024 * 1024


def _fraction(value: Any) -> Fraction:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("PPL worker rational has invalid shape")
    numerator, denominator = value
    if not isinstance(numerator, str) or not isinstance(denominator, str):
        raise ValueError("PPL worker rational components must be strings")
    if (
        len(numerator.lstrip("-")) > MAX_CANONICAL_RATIONAL_DIGITS
        or len(denominator) > MAX_CANONICAL_RATIONAL_DIGITS
    ):
        raise ValueError("PPL worker rational exceeds the source digit bound")
    result = Fraction(
        parse_canonical_integer(numerator), parse_canonical_integer(denominator)
    )
    if (
        format_canonical_integer(result.numerator) != numerator
        or format_canonical_integer(result.denominator) != denominator
    ):
        raise ValueError("PPL worker rational is not canonical")
    return result


type _StandardFormData = tuple[
    tuple[Fraction, ...],
    tuple[tuple[Fraction, ...], ...],
    tuple[Fraction, ...],
]


def _decode_program(payload: Any) -> _StandardFormData:
    if not isinstance(payload, dict) or set(payload) != {
        "objective",
        "coefficients",
        "rhs",
    }:
        raise ValueError("PPL worker program has invalid fields")
    objective_value, coefficients_value, rhs_value = (
        payload["objective"],
        payload["coefficients"],
        payload["rhs"],
    )
    if (
        not isinstance(objective_value, list)
        or len(objective_value) > MAX_LINEAR_PROGRAM_VARIABLES
    ):
        raise ValueError("PPL worker objective has invalid dimension")
    if (
        not isinstance(rhs_value, list)
        or len(rhs_value) > MAX_LINEAR_PROGRAM_CONSTRAINTS
    ):
        raise ValueError("PPL worker rhs has invalid dimension")
    if not isinstance(coefficients_value, list) or len(coefficients_value) != len(
        rhs_value
    ):
        raise ValueError("PPL worker matrix has invalid row count")
    if any(
        not isinstance(row, list) or len(row) != len(objective_value)
        for row in coefficients_value
    ):
        raise ValueError("PPL worker matrix has invalid row width")
    return (
        tuple(_fraction(value) for value in objective_value),
        tuple(tuple(_fraction(value) for value in row) for row in coefficients_value),
        tuple(_fraction(value) for value in rhs_value),
    )


def _decode(payload: Any) -> tuple[_StandardFormData, ...]:
    if not isinstance(payload, dict) or set(payload) != {
        "protocol_version",
        "programs",
    }:
        raise ValueError("PPL worker request has invalid fields")
    if payload["protocol_version"] != _PROTOCOL_VERSION:
        raise ValueError("PPL worker protocol version is unsupported")
    programs_value = payload["programs"]
    if (
        not isinstance(programs_value, list)
        or not programs_value
        or len(programs_value) > MAX_LINEAR_PROGRAM_VARIABLES
    ):
        raise ValueError("PPL worker program batch has invalid size")
    programs = tuple(_decode_program(program) for program in programs_value)
    variables = sum(len(objective) for objective, _coefficients, _rhs in programs)
    equations = sum(len(rhs) for _objective, _coefficients, rhs in programs)
    cells = sum(len(objective) * len(rhs) for objective, _coefficients, rhs in programs)
    if (
        variables > MAX_LINEAR_PROGRAM_VARIABLES
        or equations > MAX_LINEAR_PROGRAM_CONSTRAINTS
        or cells > MAX_LINEAR_PROGRAM_VARIABLES * MAX_LINEAR_PROGRAM_CONSTRAINTS
    ):
        raise ValueError("PPL worker aggregate batch dimensions are invalid")
    return programs


def _encode_vector(values: tuple[Fraction, ...]) -> list[list[str]]:
    return [
        [
            format_canonical_integer(value.numerator),
            format_canonical_integer(value.denominator),
        ]
        for value in values
    ]


def _encode(outcome: ExactLinearOutcome) -> dict[str, Any]:
    return {
        "status": outcome.status,
        "point": _encode_vector(outcome.point),
        "dual": _encode_vector(outcome.dual),
        "witness": _encode_vector(outcome.witness),
        "ray": _encode_vector(outcome.ray),
    }


def _solve_program_batch(
    programs: tuple[_StandardFormData, ...],
) -> list[dict[str, Any]]:
    """Return the solved prefix, stopping at a definitive infeasibility witness."""

    outcomes: list[dict[str, Any]] = []
    for objective, coefficients, rhs in programs:
        outcome = solve_standard_form(objective, coefficients, rhs)
        outcomes.append(_encode(outcome))
        if outcome.status == "INFEASIBLE":
            break
    return outcomes


def main() -> int:
    if version("pplpy") != _SUPPORTED_PPLPY_VERSION:
        raise RuntimeError("unsupported pplpy worker version")
    input_bytes = sys.stdin.buffer.read(_MAX_INPUT_BYTES + 1)
    if len(input_bytes) > _MAX_INPUT_BYTES:
        return 2
    programs = _decode(
        loads_strict_json(
            input_bytes, limits=CanonicalLimits(max_input_bytes=_MAX_INPUT_BYTES)
        )
    )
    sys.stdout.buffer.write(
        encode_worker_result_frame(
            {
                "protocol_version": _PROTOCOL_VERSION,
                "outcomes": _solve_program_batch(programs),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
