"""Bounded private Singular adapter for exact ideal operations over ``QQ``."""

from __future__ import annotations

import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import CanonicalLimits, parse_canonical_integer
from jacobian.math._singular import (
    SINGULAR_ARGUMENTS,
    SingularProtocolReader,
    UnsupportedSingularVersionError,
    format_singular_version,
    read_singular_version,
    run_bounded_singular,
    singular_version_preamble,
)
from jacobian.math.polynomials.ideals._models import IdealComputationBudget
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MAX_POLYNOMIAL_TERMS,
    RationalPolynomial,
    RationalPolynomialIdeal,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)
from jacobian.process import (
    ProcessPlatformTools,
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_PROTOCOL_HEADER = "JACOBIAN_SINGULAR_IDEAL_V1"
_COEFFICIENT = re.compile(r"^(0|-?[1-9][0-9]*)(?:/([1-9][0-9]*))?$")
_STDOUT_LIMIT = 512 * 1024
_STDERR_LIMIT = 64 * 1024
_CANONICAL_OUTPUT_LIMIT = CanonicalLimits().max_output_bytes

SingularOperation = Literal["radical", "quotient", "saturation"]
SingularOutcome = Literal[
    "COMPUTED",
    "UNAVAILABLE",
    "TIMEOUT",
    "CANCELLED",
    "LIMIT_EXCEEDED",
    "ERROR",
]
SaturationVerificationVerdict = Literal[
    "VERIFIED", "REFUTED", "UNAVAILABLE", "TIMEOUT", "CANCELLED", "ERROR"
]
MinimalPrimesVerificationVerdict = Literal[
    "VERIFIED", "REFUTED", "UNAVAILABLE", "TIMEOUT", "CANCELLED", "ERROR"
]


class _ResultLimitExceededError(ValueError):
    """The backend returned an exact ideal outside the declared result bound."""


@dataclass(frozen=True, slots=True)
class SingularIdealResult:
    outcome: SingularOutcome
    ideal: RationalPolynomialIdeal | None = None
    backend_version: str | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class SingularMinimalPrimesResult:
    """One complete minimal-prime family returned by Singular."""

    outcome: SingularOutcome
    components: tuple[RationalPolynomialIdeal, ...] | None = None
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
        Fraction(
            parse_canonical_integer(numerator_text),
            parse_canonical_integer(denominator_text or "1"),
        )
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
    if any(exponent > MAX_POLYNOMIAL_EXPONENT for exponent in exponents):
        raise _ResultLimitExceededError(
            "Singular exponent exceeds the exact-result representation limit"
        )
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
    if len(terms) > MAX_POLYNOMIAL_TERMS:
        raise _ResultLimitExceededError(
            "Singular generator exceeds the exact-result term representation limit"
        )
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


def _minimal_primes_script(source_ideal: RationalPolynomialIdeal) -> bytes:
    """Encode ``minAssGTZE`` and its complete prime-family result.

    The hermetic version preamble runs before any library load or algebra,
    so an unsupported release quits with the typed unavailability protocol
    instead of failing inside the kernel. ``minAssGTZE`` differs from
    ``minAssGTZ`` only on the unit ideal: it returns the empty family rather
    than an invalid unit ``ideal(1)`` entry. That is precisely the empty
    intersection convention needed for a complete minimal-prime result.
    """

    variable_count = len(source_ideal.variables)
    variables = ",".join(f"jv{index + 1}" for index in range(variable_count))
    exponent_fields = '+","+'.join(
        f"string(jacobian_exponents[{index + 1}])" for index in range(variable_count)
    )
    return "\n".join(
        [
            *singular_version_preamble(_PROTOCOL_HEADER),
            'LIB "primdec.lib";',
            "option(redSB);",
            f"ring jacobian_ring=0,({variables}),dp;",
            _singular_ideal("jacobian_source", source_ideal),
            "list jacobian_primes=minAssGTZE(jacobian_source);",
            "print(size(jacobian_primes));",
            "int jacobian_component_index,jacobian_generator_index;",
            "ideal jacobian_component;",
            "poly jacobian_poly;",
            "intvec jacobian_exponents;",
            "for (jacobian_component_index=1; "
            "jacobian_component_index<=size(jacobian_primes); "
            "jacobian_component_index=jacobian_component_index+1)",
            "{",
            '  print("COMPONENT");',
            "  jacobian_component=std(jacobian_primes[jacobian_component_index]);",
            "  print(size(jacobian_component));",
            "  for (jacobian_generator_index=1; "
            "jacobian_generator_index<=size(jacobian_component); "
            "jacobian_generator_index=jacobian_generator_index+1)",
            "  {",
            '    print("GENERATOR");',
            "    jacobian_poly=jacobian_component[jacobian_generator_index];",
            "    while (jacobian_poly != 0)",
            "    {",
            "      jacobian_exponents=leadexp(jacobian_poly);",
            f'      print(string(leadcoef(jacobian_poly))+"|"+{exponent_fields});',
            "      jacobian_poly=jacobian_poly-lead(jacobian_poly);",
            "    }",
            '    print("END_GENERATOR");',
            "  }",
            '  print("END_COMPONENT");',
            "}",
            'print("END");',
            "quit;",
            "",
        ]
    ).encode("ascii")


def _ideal_key(ideal: RationalPolynomialIdeal) -> str:
    """Return the stable public serialization used to order prime families."""

    return ideal.model_dump_json()


def _minimal_primes_stdout_limit(
    source_ideal: RationalPolynomialIdeal,
    budget: IdealComputationBudget,
) -> int:
    """Size the capture ceiling from the admitted exact-result envelope.

    Every family the decoder can admit fits inside these protocol bytes: at
    most ``budget.maximum_output_terms`` term records, each carrying a
    canonical rational with a signed numerator and denominator of up to
    ``MAX_CANONICAL_RATIONAL_DIGITS`` digits apiece plus one exponent field
    of up to ``len(str(MAX_POLYNOMIAL_EXPONENT))`` digits per ring variable;
    at most ``budget.maximum_output_generators`` generator marker pairs
    across at most that many components with their count lines; and the
    fixed version, top-level count, and end scaffolding. A capture limit
    derived from this admitted envelope lets every result within the
    canonical transport limit reach ``_parse_minimal_primes_output`` intact.
    The transport cap is included here as an early result bound; the operation
    also checks the complete serialized result envelope after decoding because
    protocol bytes do not include the public JSON framing.
    """

    coefficient_width = (
        1 + MAX_CANONICAL_RATIONAL_DIGITS + 1 + MAX_CANONICAL_RATIONAL_DIGITS
    )
    variable_count = len(source_ideal.variables)
    exponent_width = variable_count * len(str(MAX_POLYNOMIAL_EXPONENT)) + max(
        variable_count - 1, 0
    )
    term_record = coefficient_width + 1 + exponent_width + 1
    generator_scaffolding = len("GENERATOR\n") + len("END_GENERATOR\n")
    component_scaffolding = len("COMPONENT\n") + 3 + len("END_COMPONENT\n")
    scaffolding = len(_PROTOCOL_HEADER) + 1 + 8 + 3 + len("END\n")
    return min(
        _CANONICAL_OUTPUT_LIMIT,
        budget.maximum_output_terms * term_record
        + budget.maximum_output_generators
        * (generator_scaffolding + component_scaffolding)
        + scaffolding,
    )


def _format_version(version_number: int) -> str:
    major, remainder = divmod(version_number, 10_000)
    minor, patch_code = divmod(remainder, 1_000)
    patch = patch_code // 100
    return f"{major}.{minor}.{patch}"


def _parse_minimal_primes_output(
    output: bytes,
    *,
    variables: tuple[str, ...],
    budget: IdealComputationBudget,
) -> tuple[tuple[RationalPolynomialIdeal, ...], str]:
    """Decode a bounded canonical family of minimal prime ideals."""

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
        component_count = int(reader.pop())
    except ValueError as exc:
        raise ValueError("Singular output has invalid numeric metadata") from exc
    if not 0 <= component_count <= budget.maximum_output_generators:
        raise _ResultLimitExceededError(
            "Singular component count exceeds the exact-result limit"
        )

    total_generators = 0
    total_terms = 0
    components: list[RationalPolynomialIdeal] = []
    for _ in range(component_count):
        reader.expect("COMPONENT")
        try:
            generator_count = int(reader.pop())
        except ValueError as exc:
            raise ValueError("Singular output has invalid component metadata") from exc
        if not 0 <= generator_count <= budget.maximum_output_generators:
            raise _ResultLimitExceededError(
                "Singular component generator count exceeds the exact-result limit"
            )
        total_generators += generator_count
        if total_generators > budget.maximum_output_generators:
            raise _ResultLimitExceededError(
                "Singular component generators exceed the exact-result limit"
            )
        generators: list[RationalPolynomial] = []
        for _ in range(generator_count):
            generator, term_count = _parse_generator(reader, variables)
            generators.append(generator)
            total_terms += term_count
            if total_terms > budget.maximum_output_terms:
                raise _ResultLimitExceededError(
                    "Singular component terms exceed the exact-result limit"
                )
        reader.expect("END_COMPONENT")
        if not generators:
            total_generators += 1
            if total_generators > budget.maximum_output_generators:
                raise _ResultLimitExceededError(
                    "Singular component generators exceed the exact-result limit"
                )
            generators.append(
                RationalPolynomial(
                    variables=variables,
                    polynomial=SparseRationalPolynomial(terms=()),
                )
            )
        components.append(
            RationalPolynomialIdeal(variables=variables, generators=tuple(generators))
        )
    reader.expect("END")
    if not reader.finished():
        raise ValueError("Singular output has invalid trailing data")
    return tuple(sorted(components, key=_ideal_key)), _format_version(version_number)


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


def run_singular_minimal_primes(
    source_ideal: RationalPolynomialIdeal,
    budget: IdealComputationBudget,
    *,
    wall_seconds: float | None = None,
) -> SingularMinimalPrimesResult:
    """Compute minimal primes over ``QQ`` in one bounded Singular process.

    ``wall_seconds`` charges this call to a caller-owned operation deadline
    by narrowing the enforced allowance below ``budget.wall_seconds``; an
    exhausted or nonpositive allowance returns the typed TIMEOUT outcome
    without launching Singular.
    """

    allowance = (
        float(budget.wall_seconds) if wall_seconds is None else float(wall_seconds)
    )
    if not math.isfinite(allowance) or allowance <= 0:
        return SingularMinimalPrimesResult(
            outcome="TIMEOUT",
            detail="Singular exceeded the declared wall-time limit.",
        )
    resolved = shutil.which("Singular")
    if resolved is None:
        return SingularMinimalPrimesResult(
            outcome="UNAVAILABLE",
            detail="The supported Singular 4.4 backend is not installed.",
        )
    resolved = str(Path(resolved).resolve())
    prlimit = shutil.which("prlimit")
    if prlimit is not None:
        prlimit = str(Path(prlimit).resolve())
    try:
        with tempfile.TemporaryDirectory(prefix="jacobian-singular-") as directory:
            completed = run_bounded_process(
                [resolved, *SINGULAR_ARGUMENTS],
                input_bytes=_minimal_primes_script(source_ideal),
                timeout_seconds=allowance,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_minimal_primes_stdout_limit(source_ideal, budget),
                stderr_limit=_STDERR_LIMIT,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=math.ceil(allowance),
                    address_space_bytes=1024 * 1024 * 1024,
                    file_size_bytes=1024 * 1024,
                ),
                platform_tools=ProcessPlatformTools(prlimit_executable=prlimit),
                cwd=directory,
            )
    except OSError:
        return SingularMinimalPrimesResult(
            outcome="UNAVAILABLE",
            detail="The supported Singular backend could not be started.",
        )
    if completed.timed_out:
        return SingularMinimalPrimesResult(
            outcome="TIMEOUT",
            detail="Singular exceeded the declared wall-time limit.",
        )
    if completed.cancelled:
        return SingularMinimalPrimesResult(
            outcome="CANCELLED",
            detail="Singular execution was cancelled before producing a result.",
        )
    if completed.stdout_exceeded or completed.stderr_exceeded:
        return SingularMinimalPrimesResult(
            outcome="LIMIT_EXCEEDED" if completed.stdout_exceeded else "ERROR",
            detail=(
                "The exact Singular minimal-prime family exceeds the declared "
                "result bound."
                if completed.stdout_exceeded
                else "Singular exceeded the diagnostic-output limit."
            ),
        )
    if completed.returncode != 0 or completed.stderr:
        return SingularMinimalPrimesResult(
            outcome="ERROR",
            detail="Singular failed without producing an exact minimal-prime family.",
        )
    try:
        components, version = _parse_minimal_primes_output(
            completed.stdout,
            variables=source_ideal.variables,
            budget=budget,
        )
    except UnsupportedSingularVersionError:
        return SingularMinimalPrimesResult(
            outcome="UNAVAILABLE",
            detail="The installed Singular release is unsupported.",
        )
    except _ResultLimitExceededError:
        return SingularMinimalPrimesResult(
            outcome="LIMIT_EXCEEDED",
            detail="The exact Singular minimal-prime family exceeds the declared result bound.",
        )
    except ValueError:
        return SingularMinimalPrimesResult(
            outcome="ERROR",
            detail="Singular returned an invalid or unsupported minimal-prime encoding.",
        )
    return SingularMinimalPrimesResult(
        outcome="COMPUTED",
        components=components,
        backend_version=version,
    )


def _minimal_primes_verification_script(
    source: RationalPolynomialIdeal,
    claimed: tuple[RationalPolynomialIdeal, ...],
) -> bytes:
    """Script deciding the defining minimal-prime invariants inside Singular.

    The hermetic version preamble runs before any library load or algebra,
    so an unsupported release quits with the typed unavailability protocol.
    Repetition of one deterministic kernel establishes reproducibility only,
    so this pass decides each defining claim by independent evidence: every
    claimed component is prime and no component contains another (pairwise
    non-containment), the components' intersection equals the source radical
    (mutual Groebner reduction), and the independent characteristic-set
    decomposition ``minAssCharE`` returns the same family. Degenerate
    sources are decided structurally without the heavy kernels.
    """

    variable_count = len(source.variables)
    variables = ",".join(f"jv{index + 1}" for index in range(variable_count))
    if claimed:
        family = ",".join(
            "ideal("
            + ",".join(
                _singular_polynomial(generator) for generator in component.generators
            )
            + ")"
            for component in claimed
        )
        family_declaration = f"list jacobian_family={family};"
    else:
        family_declaration = "list jacobian_family;"
    source_lines = [
        *singular_version_preamble(_PROTOCOL_HEADER),
        'LIB "primdec.lib";',
        "option(redSB);",
        f"ring jacobian_ring=0,({variables}),dp;",
        _singular_ideal("jacobian_source", source),
        family_declaration,
        "int jacobian_ok=1;",
        "int jacobian_i,jacobian_j,jacobian_k;",
        "int jacobian_contained,jacobian_matched,jacobian_same;",
        "list jacobian_std_family;",
        "list jacobian_char;",
        "ideal jacobian_inter;",
        "ideal jacobian_rad;",
        "ideal jacobian_si;",
        "ideal jacobian_std_source=std(jacobian_source);",
        "if (size(jacobian_std_source)==1 && jacobian_std_source[1]==1)",
        "{",
        "  if (size(jacobian_family)!=0) { jacobian_ok=0; }",
        "}",
        "else",
        "{",
        "  if (size(jacobian_std_source)==0)",
        "  {",
        "    if (size(jacobian_family)!=1) { jacobian_ok=0; }",
        "    else",
        "    {",
        "      if (size(std(jacobian_family[1]))!=0) { jacobian_ok=0; }",
        "    }",
        "  }",
        "  else",
        "  {",
        "    for (jacobian_i=1; jacobian_i<=size(jacobian_family); "
        "jacobian_i=jacobian_i+1)",
        "    { jacobian_std_family[jacobian_i]=std(jacobian_family[jacobian_i]); }",
        "    if (size(jacobian_family)==0)",
        "    { jacobian_inter=ideal(1); }",
        "    else",
        "    {",
        "      jacobian_inter=jacobian_std_family[1];",
        "      for (jacobian_i=2; jacobian_i<=size(jacobian_family); "
        "jacobian_i=jacobian_i+1)",
        "      { "
        "jacobian_inter=intersect(jacobian_inter,jacobian_std_family[jacobian_i]); "
        "}",
        "      jacobian_inter=std(jacobian_inter);",
        "    }",
        "    jacobian_rad=std(radical(jacobian_source));",
        "    for (jacobian_i=1; jacobian_i<=size(jacobian_rad); "
        "jacobian_i=jacobian_i+1)",
        "    { if (reduce(jacobian_rad[jacobian_i],jacobian_inter)!=0) "
        "{ jacobian_ok=0; } }",
        "    for (jacobian_i=1; jacobian_i<=size(jacobian_inter); "
        "jacobian_i=jacobian_i+1)",
        "    { if (reduce(jacobian_inter[jacobian_i],jacobian_rad)!=0) "
        "{ jacobian_ok=0; } }",
        "    if (size(jacobian_family)>=2)",
        "    {",
        "      for (jacobian_i=1; jacobian_i<=size(jacobian_family); "
        "jacobian_i=jacobian_i+1)",
        "      {",
        "        for (jacobian_j=1; jacobian_j<=size(jacobian_family); "
        "jacobian_j=jacobian_j+1)",
        "        {",
        "          if (jacobian_i!=jacobian_j)",
        "          {",
        "            jacobian_contained=1;",
        "            for (jacobian_k=1; "
        "jacobian_k<=size(jacobian_std_family[jacobian_i]); "
        "jacobian_k=jacobian_k+1)",
        "            {",
        "              if "
        "(reduce(jacobian_std_family[jacobian_i][jacobian_k],"
        "jacobian_std_family[jacobian_j])!=0) { jacobian_contained=0; }",
        "            }",
        "            if (jacobian_contained==1) { jacobian_ok=0; }",
        "          }",
        "        }",
        "      }",
        "    }",
        "    if (jacobian_ok==1 && size(jacobian_rad)!=0)",
        "    {",
        "      jacobian_char=minAssCharE(jacobian_source);",
        "      if (size(jacobian_char)!=size(jacobian_family)) { jacobian_ok=0; }",
        "      else",
        "      {",
        "        for (jacobian_i=1; jacobian_i<=size(jacobian_char); "
        "jacobian_i=jacobian_i+1)",
        "        {",
        "          jacobian_si=std(jacobian_char[jacobian_i]);",
        "          jacobian_matched=0;",
        "          for (jacobian_j=1; jacobian_j<=size(jacobian_std_family); "
        "jacobian_j=jacobian_j+1)",
        "          {",
        "            jacobian_same=1;",
        "            for (jacobian_k=1; jacobian_k<=size(jacobian_si); "
        "jacobian_k=jacobian_k+1)",
        "            {",
        "              if (reduce(jacobian_si[jacobian_k],"
        "jacobian_std_family[jacobian_j])!=0) { jacobian_same=0; }",
        "            }",
        "            for (jacobian_k=1; "
        "jacobian_k<=size(jacobian_std_family[jacobian_j]); "
        "jacobian_k=jacobian_k+1)",
        "            {",
        "              if (reduce(jacobian_std_family[jacobian_j][jacobian_k],"
        "jacobian_si)!=0) { jacobian_same=0; }",
        "            }",
        "            if (jacobian_same==1) { jacobian_matched=1; }",
        "          }",
        "          if (jacobian_matched==0) { jacobian_ok=0; }",
        "        }",
        "      }",
        "    }",
        "  }",
        "}",
        'print("VERDICT "+string(jacobian_ok));',
        'print("END");',
        "quit;",
        "",
    ]
    return "\n".join(source_lines).encode("ascii")


def _parse_minimal_primes_verdict(output: bytes) -> MinimalPrimesVerificationVerdict:
    """Decode the bounded verification output into a verification verdict."""
    try:
        text = output.decode("ascii")
    except UnicodeDecodeError:
        return "ERROR"
    lines = SingularProtocolReader(text.splitlines())
    try:
        read_singular_version(lines, protocol_header=_PROTOCOL_HEADER)
        verdict_line = lines.pop()
        if not verdict_line.startswith("VERDICT "):
            return "ERROR"
        verdict = int(verdict_line.removeprefix("VERDICT "))
        lines.expect("END")
        if not lines.finished() or verdict not in (0, 1):
            return "ERROR"
    except UnsupportedSingularVersionError:
        return "UNAVAILABLE"
    except ValueError:
        return "ERROR"
    return "VERIFIED" if verdict == 1 else "REFUTED"


def run_singular_minimal_primes_verification(
    source_ideal: RationalPolynomialIdeal,
    claimed_components: tuple[RationalPolynomialIdeal, ...],
    budget: IdealComputationBudget,
    *,
    wall_seconds: float | None = None,
) -> MinimalPrimesVerificationVerdict:
    """Decide the minimal-prime defining invariants in one bounded process.

    Returns ``"VERIFIED"``, ``"REFUTED"``, or an execution outcome
    (``UNAVAILABLE``/``TIMEOUT``/``CANCELLED``/``ERROR``) when the bounded
    backend could not decide. ``wall_seconds`` charges this call to a
    caller-owned operation deadline exactly like the producing pass.
    """

    allowance = (
        float(budget.wall_seconds) if wall_seconds is None else float(wall_seconds)
    )
    if not math.isfinite(allowance) or allowance <= 0:
        return "TIMEOUT"
    resolved = shutil.which("Singular")
    if resolved is None:
        return "UNAVAILABLE"
    resolved = str(Path(resolved).resolve())
    prlimit = shutil.which("prlimit")
    if prlimit is not None:
        prlimit = str(Path(prlimit).resolve())
    try:
        with tempfile.TemporaryDirectory(prefix="jacobian-singular-") as directory:
            completed = run_bounded_process(
                [resolved, *SINGULAR_ARGUMENTS],
                input_bytes=_minimal_primes_verification_script(
                    source_ideal, claimed_components
                ),
                timeout_seconds=allowance,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_STDOUT_LIMIT,
                stderr_limit=_STDERR_LIMIT,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=math.ceil(allowance),
                    address_space_bytes=1024 * 1024 * 1024,
                    file_size_bytes=1024 * 1024,
                ),
                platform_tools=ProcessPlatformTools(prlimit_executable=prlimit),
                cwd=directory,
            )
    except OSError:
        return "UNAVAILABLE"
    if completed.timed_out:
        return "TIMEOUT"
    if completed.cancelled:
        return "CANCELLED"
    if (
        completed.returncode != 0
        or completed.stderr
        or completed.stdout_exceeded
        or completed.stderr_exceeded
    ):
        return "ERROR"
    return _parse_minimal_primes_verdict(completed.stdout)


def _verification_script(
    source: RationalPolynomialIdeal,
    saturator: RationalPolynomialIdeal,
    claimed: RationalPolynomialIdeal,
) -> bytes:
    """Script deciding ``source : saturator^inf == claimed`` inside Singular.

    The hermetic version preamble runs before any library load or algebra,
    so an unsupported release quits with the typed unavailability protocol.
    """
    variable_count = len(source.variables)
    variables = ",".join(f"jv{index + 1}" for index in range(variable_count))
    declarations = [
        _singular_ideal("jacobian_source", source),
        _singular_ideal("jacobian_saturator", saturator),
        _singular_ideal("jacobian_claimed", claimed),
    ]
    source_lines = [
        *singular_version_preamble(_PROTOCOL_HEADER),
        'LIB "primdec.lib";',
        "option(redSB);",
        f"ring jacobian_ring=0,({variables}),dp;",
        *declarations,
        # The defining equality I : d^inf == J, decided by mutual exact
        # reduction of Groebner bases computed inside this bounded process.
        "list jacobian_sat=sat(jacobian_source,jacobian_saturator);",
        "ideal jacobian_true=std(jacobian_sat[1]);",
        "ideal jacobian_std_claimed=std(jacobian_claimed);",
        "int jacobian_equal=1;",
        "int jacobian_i;",
        "for (jacobian_i=1; jacobian_i<=size(jacobian_true); jacobian_i=",
        "jacobian_i+1)",
        "{",
        "  if (reduce(jacobian_true[jacobian_i],jacobian_std_claimed) != 0)",
        "  { jacobian_equal=0; }",
        "}",
        "for (jacobian_i=1; jacobian_i<=size(jacobian_claimed); jacobian_i=",
        "jacobian_i+1)",
        "{",
        "  if (reduce(jacobian_claimed[jacobian_i],jacobian_true) != 0)",
        "  { jacobian_equal=0; }",
        "}",
        'print("VERDICT "+string(jacobian_equal));',
        'print("END");',
        "quit;",
        "",
    ]
    return "\n".join(source_lines).encode("ascii")


def _parse_verification_verdict(output: bytes) -> SaturationVerificationVerdict:
    """Decode the bounded process output into a verification verdict."""
    try:
        text = output.decode("ascii")
    except UnicodeDecodeError:
        return "ERROR"
    lines = SingularProtocolReader(text.splitlines())
    try:
        read_singular_version(lines, protocol_header=_PROTOCOL_HEADER)
        verdict_line = lines.pop()
        if not verdict_line.startswith("VERDICT "):
            return "ERROR"
        verdict = int(verdict_line.removeprefix("VERDICT "))
        lines.expect("END")
        if not lines.finished() or verdict not in (0, 1):
            return "ERROR"
    except UnsupportedSingularVersionError:
        return "UNAVAILABLE"
    except ValueError:
        return "ERROR"
    return "VERIFIED" if verdict == 1 else "REFUTED"


def run_singular_saturation_verification(
    source: RationalPolynomialIdeal,
    saturator: RationalPolynomialIdeal,
    claimed: RationalPolynomialIdeal,
    budget: IdealComputationBudget,
) -> SaturationVerificationVerdict:
    """Decide the saturation's defining equality in a bounded subprocess.

    Returns ``"VERIFIED"``, ``"REFUTED"``, or an execution outcome
    (``UNAVAILABLE``/``TIMEOUT``/``CANCELLED``/``ERROR``) when the bounded
    backend could not decide. Running inside the supervised process keeps
    the replay under the same wall-time and memory limits as the operation
    itself.
    """
    resolved = shutil.which("Singular")
    if resolved is None:
        return "UNAVAILABLE"
    resolved = str(Path(resolved).resolve())
    prlimit = shutil.which("prlimit")
    if prlimit is not None:
        prlimit = str(Path(prlimit).resolve())
    try:
        with tempfile.TemporaryDirectory(prefix="jacobian-singular-") as directory:
            completed = run_bounded_process(
                [resolved, *SINGULAR_ARGUMENTS],
                input_bytes=_verification_script(source, saturator, claimed),
                timeout_seconds=float(budget.wall_seconds),
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_STDOUT_LIMIT,
                stderr_limit=_STDERR_LIMIT,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=budget.wall_seconds,
                    address_space_bytes=1024 * 1024 * 1024,
                    file_size_bytes=1024 * 1024,
                ),
                platform_tools=ProcessPlatformTools(prlimit_executable=prlimit),
                cwd=directory,
            )
    except OSError:
        return "UNAVAILABLE"
    if completed.timed_out:
        return "TIMEOUT"
    if completed.cancelled:
        return "CANCELLED"
    if (
        completed.returncode != 0
        or completed.stderr
        or completed.stdout_exceeded
        or completed.stderr_exceeded
    ):
        return "ERROR"
    return _parse_verification_verdict(completed.stdout)


__all__ = [
    "MinimalPrimesVerificationVerdict",
    "SaturationVerificationVerdict",
    "SingularIdealResult",
    "SingularMinimalPrimesResult",
    "run_singular_ideal_operation",
    "run_singular_minimal_primes",
    "run_singular_minimal_primes_verification",
    "run_singular_saturation_verification",
]


def run_bounded_stdin_python_kernel(
    script: str,
    payload_json: str,
    *,
    wall_seconds: float,
    stdout_limit: int,
    stderr_limit: int,
) -> tuple[bool, str, bool]:
    """Run one bounded Python-kernel worker and return (timed_out, stdout,
    output_limit_exceeded).

    The child process is terminated on wall-budget expiry, so an admitted
    request cannot leave detached computations running inside the server.
    This owner owns every external-executable lookup and the killable
    process launch for the domain's exact kernels.
    """
    import sys
    import tempfile

    from jacobian.process import (
        ProcessPlatformTools,
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
    )

    # Deliberately not resolved: following the interpreter symlink would
    # reparent the worker onto the base prefix without the environment's
    # site-packages.
    resolved = shutil.which(sys.executable) or sys.executable
    prlimit = shutil.which("prlimit")
    if prlimit is not None:
        prlimit = str(Path(prlimit).resolve())
    with tempfile.TemporaryDirectory(prefix="jacobian-sympy-") as directory:
        completed = run_bounded_process(
            [resolved, "-I", "-c", script],
            input_bytes=payload_json.encode("ascii"),
            timeout_seconds=float(wall_seconds),
            environment=worker_environment(locale="C.UTF-8"),
            stdout_limit=stdout_limit,
            stderr_limit=stderr_limit,
            resource_limits=ProcessResourceLimits(
                cpu_seconds=int(wall_seconds),
                address_space_bytes=2 * 1024 * 1024 * 1024,
                file_size_bytes=1024 * 1024,
            ),
            platform_tools=ProcessPlatformTools(prlimit_executable=prlimit),
            cwd=directory,
        )
    if completed.timed_out:
        return True, "", False
    exceeded = completed.stdout_exceeded or completed.stderr_exceeded
    return False, completed.stdout.decode("ascii", errors="replace"), exceeded
