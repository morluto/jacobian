"""Bounded private Singular adapter for exact ideal operations over ``QQ``."""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.math._singular import (
    SingularProtocolReader,
    UnsupportedSingularVersionError,
    format_singular_version,
    read_singular_version,
    run_bounded_singular,
    singular_version_preamble,
)
from jacobian.math.commutative_algebra_ops._models import IdealComputationBudget
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

_PROTOCOL_HEADER = "JACOBIAN_SINGULAR_IDEAL_V1"
_COEFFICIENT = re.compile(r"^(0|-?[1-9][0-9]*)(?:/([1-9][0-9]*))?$")

SingularOperation = Literal["radical", "quotient", "saturation"]
SingularOutcome = Literal[
    "COMPUTED",
    "UNAVAILABLE",
    "TIMEOUT",
    "CANCELLED",
    "LIMIT_EXCEEDED",
    "ERROR",
]


class _ResultLimitExceededError(ValueError):
    """The backend returned an exact ideal outside the declared result bound."""


@dataclass(frozen=True, slots=True)
class SingularIdealResult:
    outcome: SingularOutcome
    ideal: RationalPolynomialIdeal | None = None
    backend_version: str | None = None
    detail: str | None = None


def _singular_polynomial(polynomial: RationalPolynomial) -> str:
    """Encode canonical terms using only fixed internal Singular identifiers."""

    if not polynomial.polynomial.terms:
        return "0"
    encoded_terms: list[str] = []
    for term in polynomial.polynomial.terms:
        numerator, denominator = term.coefficient.as_integer_ratio()
        coefficient = f"({numerator}/{denominator})"
        monomial = "*".join(
            f"jv{index + 1}^{exponent}"
            for index, exponent in enumerate(term.exponents)
            if exponent
        )
        encoded_terms.append(f"{coefficient}*{monomial}" if monomial else coefficient)
    return "+".join(encoded_terms)


def _singular_ideal(name: str, ideal: RationalPolynomialIdeal) -> str:
    generators = ",".join(
        _singular_polynomial(generator) for generator in ideal.generators
    )
    return f"ideal {name}={generators};"


def _script(
    operation: SingularOperation,
    left: RationalPolynomialIdeal,
    right: RationalPolynomialIdeal | None,
) -> bytes:
    variable_count = len(left.variables)
    variables = ",".join(f"jv{index + 1}" for index in range(variable_count))
    exponent_fields = '+","+'.join(
        f"string(jacobian_exponents[{index + 1}])" for index in range(variable_count)
    )
    declarations = [_singular_ideal("jacobian_left", left)]
    if operation == "radical":
        operation_line = "ideal jacobian_result=radical(jacobian_left);"
        libs = ['LIB "primdec.lib";']
    elif operation == "saturation":
        if right is None:
            raise ValueError("saturation requires a denominator ideal")
        declarations.append(_singular_ideal("jacobian_right", right))
        operation_line = (
            "list jacobian_sat_result=sat(jacobian_left,jacobian_right); "
            "ideal jacobian_result=jacobian_sat_result[1];"
        )
        libs = ['LIB "elim.lib";']
    else:
        if right is None:
            raise ValueError("quotient requires a divisor ideal")
        declarations.append(_singular_ideal("jacobian_right", right))
        operation_line = "ideal jacobian_result=quotient(jacobian_left,jacobian_right);"
        libs = ['LIB "primdec.lib";']
    source = "\n".join(
        [
            *singular_version_preamble(_PROTOCOL_HEADER),
            *libs,
            "option(redSB);",
            f"ring jacobian_ring=0,({variables}),dp;",
            *declarations,
            operation_line,
            "jacobian_result=std(jacobian_result);",
            "print(size(jacobian_result));",
            "int jacobian_i;",
            "poly jacobian_poly;",
            "intvec jacobian_exponents;",
            "for (jacobian_i=1; jacobian_i<=size(jacobian_result); "
            "jacobian_i=jacobian_i+1)",
            "{",
            '  print("GENERATOR");',
            "  jacobian_poly=jacobian_result[jacobian_i];",
            "  while (jacobian_poly != 0)",
            "  {",
            "    jacobian_exponents=leadexp(jacobian_poly);",
            f'    print(string(leadcoef(jacobian_poly))+"|"+{exponent_fields});',
            "    jacobian_poly=jacobian_poly-lead(jacobian_poly);",
            "  }",
            '  print("END_GENERATOR");',
            "}",
            'print("END");',
            "quit;",
            "",
        ]
    )
    return source.encode("ascii")


def _parse_coefficient(text: str) -> CanonicalRational:
    match = _COEFFICIENT.fullmatch(text)
    if match is None:
        raise ValueError("Singular returned a non-rational coefficient")
    numerator_text, denominator_text = match.groups()
    if len(numerator_text.lstrip("-")) > MAX_CANONICAL_RATIONAL_DIGITS or (
        denominator_text is not None
        and len(denominator_text) > MAX_CANONICAL_RATIONAL_DIGITS
    ):
        raise _ResultLimitExceededError(
            "Singular coefficient exceeds the canonical exact-result digit limit"
        )
    return CanonicalRational.from_fraction(
        Fraction(int(numerator_text), int(denominator_text or "1"))
    )


def _parse_term(line: str, variable_count: int) -> RationalPolynomialTerm:
    coefficient_text, separator, exponent_text = line.partition("|")
    if not separator:
        raise ValueError("Singular output contains an invalid term record")
    exponent_parts = exponent_text.split(",")
    if len(exponent_parts) != variable_count:
        raise ValueError("Singular term does not match the declared ring")
    try:
        exponents = tuple(int(part) for part in exponent_parts)
    except ValueError as exc:
        raise ValueError("Singular output contains a non-integer exponent") from exc
    if any(exponent < 0 for exponent in exponents):
        raise ValueError("Singular output contains a negative exponent")
    return RationalPolynomialTerm(
        coefficient=_parse_coefficient(coefficient_text),
        exponents=exponents,
    )


def _parse_generator(
    reader: SingularProtocolReader,
    variables: tuple[str, ...],
) -> tuple[RationalPolynomial, int]:
    reader.expect("GENERATOR")
    terms: list[RationalPolynomialTerm] = []
    while True:
        line = reader.pop()
        if line == "END_GENERATOR":
            break
        terms.append(_parse_term(line, len(variables)))
    return (
        RationalPolynomial(
            variables=variables,
            polynomial=SparseRationalPolynomial(
                terms=tuple(
                    sorted(terms, key=lambda term: term.exponents, reverse=True)
                )
            ),
        ),
        len(terms),
    )


def _parse_output(
    output: bytes,
    *,
    variables: tuple[str, ...],
    budget: IdealComputationBudget,
) -> tuple[RationalPolynomialIdeal, str]:
    try:
        text = output.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("Singular output is not ASCII") from exc
    reader = SingularProtocolReader(text.splitlines())
    version_number = read_singular_version(
        reader,
        protocol_header=_PROTOCOL_HEADER,
    )
    try:
        generator_count = int(reader.pop())
    except ValueError as exc:
        raise ValueError("Singular output has invalid numeric metadata") from exc
    if not 0 <= generator_count <= budget.maximum_output_generators:
        raise _ResultLimitExceededError(
            "Singular generator count exceeds the exact-result limit"
        )

    total_terms = 0
    generators: list[RationalPolynomial] = []
    for _ in range(generator_count):
        generator, term_count = _parse_generator(reader, variables)
        generators.append(generator)
        total_terms += term_count
        if total_terms > budget.maximum_output_terms:
            raise _ResultLimitExceededError(
                "Singular terms exceed the exact-result limit"
            )
    if not generators:
        generators.append(
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(terms=()),
            )
        )
    reader.expect("END")
    if not reader.finished():
        raise ValueError("Singular output has invalid trailing data")
    return (
        RationalPolynomialIdeal(variables=variables, generators=tuple(generators)),
        format_singular_version(version_number),
    )


def run_singular_ideal_operation(
    operation: SingularOperation,
    left: RationalPolynomialIdeal,
    right: RationalPolynomialIdeal | None,
    budget: IdealComputationBudget,
) -> SingularIdealResult:
    """Run one exact ideal operation in a bounded, request-scoped process."""

    completed = run_bounded_singular(
        _script(operation, left, right),
        wall_seconds=budget.wall_seconds,
    )
    if completed is None:
        return SingularIdealResult(
            outcome="UNAVAILABLE",
            detail="The supported Singular 4.4 backend is not installed.",
        )
    if completed.timed_out:
        return SingularIdealResult(
            outcome="TIMEOUT",
            detail="Singular exceeded the declared wall-time limit.",
        )
    if completed.cancelled:
        return SingularIdealResult(
            outcome="CANCELLED",
            detail="Singular execution was cancelled before producing a result.",
        )
    if completed.stdout_exceeded or completed.stderr_exceeded:
        return SingularIdealResult(
            outcome="LIMIT_EXCEEDED" if completed.stdout_exceeded else "ERROR",
            detail=(
                "The exact Singular ideal exceeds the declared result bound."
                if completed.stdout_exceeded
                else "Singular exceeded the diagnostic-output limit."
            ),
        )
    if completed.returncode != 0 or completed.stderr:
        return SingularIdealResult(
            outcome="ERROR",
            detail="Singular failed without producing an exact ideal.",
        )
    try:
        ideal, version = _parse_output(
            completed.stdout,
            variables=left.variables,
            budget=budget,
        )
    except UnsupportedSingularVersionError:
        return SingularIdealResult(
            outcome="UNAVAILABLE",
            detail="The installed Singular release is unsupported.",
        )
    except _ResultLimitExceededError:
        return SingularIdealResult(
            outcome="LIMIT_EXCEEDED",
            detail="The exact Singular ideal exceeds the declared result bound.",
        )
    except ValueError:
        return SingularIdealResult(
            outcome="ERROR",
            detail="Singular returned an invalid or unsupported result encoding.",
        )
    return SingularIdealResult(
        outcome="COMPUTED",
        ideal=ideal,
        backend_version=version,
    )


__all__ = ["SingularIdealResult", "run_singular_ideal_operation"]
