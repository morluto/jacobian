"""Bounded private Singular adapter for polynomial-map generic fibers."""

from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Literal

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials.maps._models import (
    MAX_GENERIC_FIBER_COEFFICIENT_TERMS,
    GenericDegreeComputationBudget,
    GenericFiberCertificate,
    GenericFiberPolynomial,
    GenericFiberTerm,
)
from jacobian.math.polynomials.maps.values import RationalPolynomialMap
from jacobian.math.polynomials.values import (
    RationalFunction,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)
from jacobian.process import (
    ProcessPlatformTools,
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_GENERIC_FIBER_PROTOCOL_HEADER = "JACOBIAN_SINGULAR_GENERIC_FIBER_V1"
_SUPPORTED_VERSION_MIN = 44000
_SUPPORTED_VERSION_MAX = 45000
_PARAMETER_FACTOR = re.compile(r"^jtp([1-9][0-9]*)(?:\^([1-9][0-9]*))?$")
_STDOUT_LIMIT = 512 * 1024
_STDERR_LIMIT = 64 * 1024

SingularOutcome = Literal[
    "COMPUTED", "UNAVAILABLE", "TIMEOUT", "LIMIT_EXCEEDED", "ERROR"
]


class _ResultLimitExceededError(ValueError):
    """The backend returned evidence outside the declared result bound."""


@dataclass(frozen=True, slots=True)
class SingularGenericFiberResult:
    outcome: SingularOutcome
    certificate: GenericFiberCertificate | None = None
    dimension: int | None = None
    vector_dimension: int | None = None
    backend_version: str | None = None
    detail: str | None = None


@dataclass(slots=True)
class _ProtocolReader:
    lines: list[str]
    cursor: int = 0

    def pop(self) -> str:
        if self.cursor >= len(self.lines):
            raise ValueError("Singular output ended unexpectedly")
        line = self.lines[self.cursor]
        self.cursor += 1
        return line

    def expect(self, expected: str) -> None:
        if self.pop() != expected:
            raise ValueError(f"Singular output is missing {expected!r}")

    def finished(self) -> bool:
        return self.cursor == len(self.lines)


def _singular_map_polynomial(
    polynomial: RationalPolynomial,
    *,
    backend_variables: tuple[str, ...],
) -> str:
    """Encode one map component in the fixed internal generic-fiber ring."""

    canonical_positions = {
        variable: index for index, variable in enumerate(polynomial.variables)
    }
    if not polynomial.polynomial.terms:
        return "0"
    encoded_terms: list[str] = []
    for term in polynomial.polynomial.terms:
        numerator, denominator = term.coefficient.as_integer_ratio()
        coefficient = f"({numerator}/{denominator})"
        monomial = "*".join(
            f"jv{backend_index + 1}^{term.exponents[canonical_positions[variable]]}"
            for backend_index, variable in enumerate(backend_variables)
            if term.exponents[canonical_positions[variable]]
        )
        encoded_terms.append(f"{coefficient}*{monomial}" if monomial else coefficient)
    return "+".join(encoded_terms)


def _generic_fiber_script(polynomial_map: RationalPolynomialMap) -> bytes:
    backend_variables = polynomial_map.input_variables
    source_variables = ",".join(
        f"jv{index + 1}" for index in range(len(backend_variables))
    )
    target_parameters = ",".join(
        f"jtp{index + 1}" for index in range(len(polynomial_map.output_polynomials))
    )
    exponent_fields = '+","+'.join(
        f"string(jacobian_exponents[{index + 1}])"
        for index in range(len(backend_variables))
    )
    generators = ",".join(
        f"({_singular_map_polynomial(component, backend_variables=backend_variables)})"
        f"-jtp{index + 1}"
        for index, component in enumerate(polynomial_map.output_polynomials)
    )
    source = "\n".join(
        [
            "option(redSB);",
            f"ring jacobian_ring=(0,{target_parameters}),({source_variables}),lp;",
            f"ideal jacobian_source={generators};",
            "matrix jacobian_transformation;",
            "ideal jacobian_basis=liftstd(jacobian_source,jacobian_transformation);",
            "int jacobian_dimension=dim(jacobian_basis);",
            "int jacobian_vector_dimension=-1;",
            "if (jacobian_dimension==0)",
            "{",
            "  jacobian_vector_dimension=vdim(jacobian_basis);",
            "}",
            "int jacobian_i;",
            "int jacobian_j;",
            "number jacobian_leading_coefficient;",
            "for (jacobian_i=1; jacobian_i<=size(jacobian_basis); "
            "jacobian_i=jacobian_i+1)",
            "{",
            "  jacobian_leading_coefficient=leadcoef(jacobian_basis[jacobian_i]);",
            "  jacobian_basis[jacobian_i]=jacobian_basis[jacobian_i]"
            "/jacobian_leading_coefficient;",
            "  for (jacobian_j=1; jacobian_j<=nrows(jacobian_transformation); "
            "jacobian_j=jacobian_j+1)",
            "  {",
            "    jacobian_transformation[jacobian_j,jacobian_i]="
            "jacobian_transformation[jacobian_j,jacobian_i]"
            "/jacobian_leading_coefficient;",
            "  }",
            "}",
            "proc jacobian_print_polynomial(poly jacobian_poly)",
            "{",
            '  print("POLYNOMIAL");',
            "  intvec jacobian_exponents;",
            "  number jacobian_coefficient;",
            "  while (jacobian_poly != 0)",
            "  {",
            "    jacobian_exponents=leadexp(jacobian_poly);",
            "    jacobian_coefficient=leadcoef(jacobian_poly);",
            f"    print({exponent_fields});",
            "    print(string(numerator(jacobian_coefficient)));",
            "    print(string(denominator(jacobian_coefficient)));",
            "    jacobian_poly=jacobian_poly-lead(jacobian_poly);",
            "  }",
            '  print("END_POLYNOMIAL");',
            "}",
            f'print("{_GENERIC_FIBER_PROTOCOL_HEADER}");',
            'system("version");',
            "print(jacobian_dimension);",
            "print(jacobian_vector_dimension);",
            "print(size(jacobian_basis));",
            "print(nrows(jacobian_transformation));",
            "for (jacobian_i=1; jacobian_i<=size(jacobian_basis); "
            "jacobian_i=jacobian_i+1)",
            "{",
            "  jacobian_print_polynomial(jacobian_basis[jacobian_i]);",
            "}",
            "for (jacobian_i=1; jacobian_i<=nrows(jacobian_transformation); "
            "jacobian_i=jacobian_i+1)",
            "{",
            "  for (jacobian_j=1; jacobian_j<=ncols(jacobian_transformation); "
            "jacobian_j=jacobian_j+1)",
            "  {",
            "    jacobian_print_polynomial("
            "jacobian_transformation[jacobian_i,jacobian_j]);",
            "  }",
            "}",
            'print("END");',
            "quit;",
            "",
        ]
    )
    return source.encode("ascii")


def _parse_parameter_polynomial(
    text: str,
    *,
    parameter_count: int,
) -> dict[tuple[int, ...], int]:
    """Parse Singular's fixed polynomial printer without evaluating its text."""

    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    if (
        not text
        or "(" in text
        or ")" in text
        or any(character.isspace() for character in text)
    ):
        raise ValueError("Singular returned an invalid parameter polynomial")
    if text == "0":
        return {}

    pieces = re.findall(r"[+-]?[^+-]+", text)
    if not pieces or "".join(pieces) != text:
        raise ValueError("Singular returned an invalid parameter polynomial")
    terms: dict[tuple[int, ...], int] = {}
    for index, piece in enumerate(pieces):
        sign = 1
        if piece[0] in "+-":
            if index == 0 and piece[0] == "+":
                raise ValueError("Singular returned an invalid parameter polynomial")
            sign = -1 if piece[0] == "-" else 1
            piece = piece[1:]
        if not piece:
            raise ValueError("Singular returned an invalid parameter polynomial")
        exponent_tuple, coefficient = _parse_parameter_monomial(
            piece,
            parameter_count=parameter_count,
        )
        if exponent_tuple in terms:
            raise ValueError("Singular returned duplicate parameter monomials")
        terms[exponent_tuple] = sign * coefficient
    if len(terms) > 256:
        raise _ResultLimitExceededError(
            "Singular parameter support exceeds the exact-result limit"
        )
    return terms


def _parse_parameter_monomial(
    text: str,
    *,
    parameter_count: int,
) -> tuple[tuple[int, ...], int]:
    coefficient = 1
    coefficient_seen = False
    exponents = [0] * parameter_count
    for factor in text.split("*"):
        if not factor:
            raise ValueError("Singular returned an invalid parameter polynomial")
        if factor.isdigit():
            if len(factor) > 128:
                raise _ResultLimitExceededError(
                    "Singular coefficient exceeds the exact-result limit"
                )
            if coefficient_seen or factor.startswith("0"):
                raise ValueError("Singular returned an invalid parameter polynomial")
            coefficient = int(factor)
            coefficient_seen = True
            continue
        match = _PARAMETER_FACTOR.fullmatch(factor)
        if match is None:
            raise ValueError("Singular returned an invalid parameter polynomial")
        parameter_text, exponent_text = match.groups()
        if len(parameter_text) > 2:
            raise ValueError("Singular returned an invalid parameter polynomial")
        parameter_index = int(parameter_text) - 1
        if not 0 <= parameter_index < parameter_count:
            raise ValueError("Singular parameter does not match the declared field")
        exponent = int(exponent_text or "1")
        if exponent > 64:
            raise _ResultLimitExceededError(
                "Singular parameter exponent exceeds the exact-result limit"
            )
        if exponents[parameter_index]:
            raise ValueError("Singular returned a repeated parameter factor")
        exponents[parameter_index] = exponent
    return tuple(exponents), coefficient


def _sparse_parameter_polynomial(
    terms: dict[tuple[int, ...], int],
    *,
    scale: int,
) -> SparseRationalPolynomial:
    return SparseRationalPolynomial(
        terms=tuple(
            RationalPolynomialTerm(
                coefficient=CanonicalRational.from_fraction(Fraction(value, scale)),
                exponents=exponents,
            )
            for exponents, value in sorted(terms.items(), reverse=True)
        )
    )


def _parse_generic_fiber_coefficient(
    numerator_text: str,
    denominator_text: str,
    *,
    target_parameters: tuple[str, ...],
) -> tuple[RationalFunction, int]:
    numerator = _parse_parameter_polynomial(
        numerator_text,
        parameter_count=len(target_parameters),
    )
    denominator = _parse_parameter_polynomial(
        denominator_text,
        parameter_count=len(target_parameters),
    )
    if not numerator or not denominator:
        raise ValueError("Singular returned a zero generic-fiber coefficient")
    denominator_leading = denominator[max(denominator)]
    value = RationalFunction(
        variables=target_parameters,
        numerator=_sparse_parameter_polynomial(
            numerator,
            scale=denominator_leading,
        ),
        denominator=_sparse_parameter_polynomial(
            denominator,
            scale=denominator_leading,
        ),
    )
    return value, len(numerator) + len(denominator)


def _parse_source_exponents(text: str, source_count: int) -> tuple[int, ...]:
    parts = text.split(",")
    if len(parts) != source_count or any(
        re.fullmatch(r"0|[1-9][0-9]*", part) is None for part in parts
    ):
        raise ValueError("Singular monomial does not match the declared source ring")
    if any(len(part) > 2 for part in parts):
        raise _ResultLimitExceededError(
            "Singular source exponent exceeds the exact-result limit"
        )
    exponents = tuple(int(part) for part in parts)
    if any(exponent > 64 for exponent in exponents):
        raise _ResultLimitExceededError(
            "Singular source exponent exceeds the exact-result limit"
        )
    return exponents


def _parse_generic_fiber_polynomial(
    reader: _ProtocolReader,
    *,
    source_count: int,
    target_parameters: tuple[str, ...],
) -> tuple[GenericFiberPolynomial, int]:
    reader.expect("POLYNOMIAL")
    terms: list[GenericFiberTerm] = []
    coefficient_terms = 0
    while True:
        exponent_text = reader.pop()
        if exponent_text == "END_POLYNOMIAL":
            break
        if len(terms) == 256:
            raise _ResultLimitExceededError(
                "Singular polynomial support exceeds the exact-result limit"
            )
        coefficient, support = _parse_generic_fiber_coefficient(
            reader.pop(),
            reader.pop(),
            target_parameters=target_parameters,
        )
        terms.append(
            GenericFiberTerm(
                coefficient=coefficient,
                source_exponents=_parse_source_exponents(
                    exponent_text,
                    source_count,
                ),
            )
        )
        coefficient_terms += support
    return GenericFiberPolynomial(terms=tuple(terms)), coefficient_terms


def _derive_standard_monomials(
    basis: tuple[GenericFiberPolynomial, ...],
    *,
    source_count: int,
    maximum_standard_monomials: int,
) -> tuple[tuple[int, ...], ...] | None:
    if any(not polynomial.terms for polynomial in basis):
        raise ValueError("Singular returned a zero generic-fiber basis polynomial")
    leading_exponents = tuple(
        polynomial.terms[0].source_exponents for polynomial in basis
    )
    bounds: list[int] = []
    for variable_index in range(source_count):
        powers = [
            exponents[variable_index]
            for exponents in leading_exponents
            if exponents[variable_index]
            and all(
                exponent == 0
                for index, exponent in enumerate(exponents)
                if index != variable_index
            )
        ]
        if not powers:
            return None
        bounds.append(min(powers))
    candidate_count = 1
    for bound in bounds:
        candidate_count *= bound
    if candidate_count > maximum_standard_monomials:
        raise _ResultLimitExceededError(
            "Singular quotient dimension exceeds the exact-result limit"
        )
    return tuple(
        exponents
        for exponents in product(*(range(bound) for bound in bounds))
        if not any(
            all(a <= b for a, b in zip(leading, exponents, strict=True))
            for leading in leading_exponents
        )
    )


def _parse_generic_fiber_metadata(
    reader: _ProtocolReader,
    *,
    polynomial_map: RationalPolynomialMap,
    budget: GenericDegreeComputationBudget,
) -> tuple[int, int, int, int, int]:
    try:
        version_number = int(reader.pop())
        dimension = int(reader.pop())
        vector_dimension = int(reader.pop())
        basis_count = int(reader.pop())
        source_row_count = int(reader.pop())
    except ValueError as exc:
        raise ValueError("Singular output has invalid numeric metadata") from exc
    if not _SUPPORTED_VERSION_MIN <= version_number < _SUPPORTED_VERSION_MAX:
        raise ValueError("Singular backend version is unsupported")
    source_count = len(polynomial_map.input_variables)
    if not -1 <= dimension <= source_count:
        raise ValueError("Singular returned an impossible generic-fiber dimension")
    if not 1 <= basis_count <= budget.maximum_basis_polynomials:
        raise _ResultLimitExceededError("Singular basis exceeds the exact-result limit")
    if source_row_count != len(polynomial_map.output_polynomials):
        raise ValueError("Singular transformation does not match the source ideal")
    return (
        version_number,
        dimension,
        vector_dimension,
        basis_count,
        source_row_count,
    )


def _require_dimension_metadata(
    basis: tuple[GenericFiberPolynomial, ...],
    *,
    dimension: int,
    vector_dimension: int,
    source_count: int,
    maximum_standard_monomials: int,
) -> tuple[tuple[tuple[int, ...], ...], int | None]:
    unit_exponents = (0,) * source_count
    is_unit = (
        len(basis) == 1
        and len(basis[0].terms) == 1
        and basis[0].terms[0].source_exponents == unit_exponents
    )
    standard_monomials = _derive_standard_monomials(
        basis,
        source_count=source_count,
        maximum_standard_monomials=maximum_standard_monomials,
    )
    if dimension == -1:
        if not is_unit or vector_dimension != -1:
            raise ValueError("Singular returned inconsistent empty-fiber metadata")
        return (), None
    if dimension == 0:
        if standard_monomials is None or not standard_monomials:
            raise ValueError("Singular returned an inconsistent finite generic fiber")
        if vector_dimension != len(standard_monomials):
            raise ValueError("Singular quotient dimension disagrees with its basis")
        return standard_monomials, vector_dimension
    if is_unit or standard_monomials is not None or vector_dimension != -1:
        raise ValueError(
            "Singular returned inconsistent positive-dimensional fiber metadata"
        )
    return (), None


def _order_basis_columns(
    basis: list[GenericFiberPolynomial],
    transformation: list[tuple[GenericFiberPolynomial, ...]],
) -> tuple[
    tuple[GenericFiberPolynomial, ...],
    tuple[tuple[GenericFiberPolynomial, ...], ...],
]:
    if any(not polynomial.terms for polynomial in basis):
        raise ValueError("Singular returned a zero generic-fiber basis polynomial")
    order = tuple(
        sorted(
            range(len(basis)),
            key=lambda index: basis[index].terms[0].source_exponents,
        )
    )
    return (
        tuple(basis[index] for index in order),
        tuple(tuple(row[index] for index in order) for row in transformation),
    )


def _parse_generic_fiber_output(
    output: bytes,
    *,
    polynomial_map: RationalPolynomialMap,
    budget: GenericDegreeComputationBudget,
) -> tuple[GenericFiberCertificate, int, int | None, str]:
    try:
        text = output.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("Singular output is not ASCII") from exc
    reader = _ProtocolReader(text.splitlines())
    reader.expect(_GENERIC_FIBER_PROTOCOL_HEADER)
    (
        version_number,
        dimension,
        vector_dimension_record,
        basis_count,
        source_row_count,
    ) = _parse_generic_fiber_metadata(
        reader,
        polynomial_map=polynomial_map,
        budget=budget,
    )
    source_count = len(polynomial_map.input_variables)

    target_parameters = tuple(
        f"t{index + 1}" for index in range(len(polynomial_map.output_polynomials))
    )
    certificate_terms = 0
    coefficient_terms = 0
    basis: list[GenericFiberPolynomial] = []
    for _ in range(basis_count):
        polynomial, support = _parse_generic_fiber_polynomial(
            reader,
            source_count=source_count,
            target_parameters=target_parameters,
        )
        basis.append(polynomial)
        certificate_terms += len(polynomial.terms)
        coefficient_terms += support
    transformation: list[tuple[GenericFiberPolynomial, ...]] = []
    for _ in range(source_row_count):
        row: list[GenericFiberPolynomial] = []
        for _ in range(basis_count):
            polynomial, support = _parse_generic_fiber_polynomial(
                reader,
                source_count=source_count,
                target_parameters=target_parameters,
            )
            row.append(polynomial)
            certificate_terms += len(polynomial.terms)
            coefficient_terms += support
        transformation.append(tuple(row))
    if certificate_terms > budget.maximum_certificate_terms:
        raise _ResultLimitExceededError(
            "Singular certificate support exceeds the exact-result limit"
        )
    if coefficient_terms > MAX_GENERIC_FIBER_COEFFICIENT_TERMS:
        raise _ResultLimitExceededError(
            "Singular coefficient support exceeds the exact-result limit"
        )
    reader.expect("END")
    if not reader.finished():
        raise ValueError("Singular output has invalid trailing data")

    basis_tuple, transformation_tuple = _order_basis_columns(basis, transformation)
    standard_monomials, vector_dimension = _require_dimension_metadata(
        basis_tuple,
        dimension=dimension,
        vector_dimension=vector_dimension_record,
        source_count=source_count,
        maximum_standard_monomials=budget.maximum_standard_monomials,
    )

    certificate = GenericFiberCertificate(
        target_parameters=target_parameters,
        source_variable_order=polynomial_map.input_variables,
        basis=basis_tuple,
        basis_from_source=transformation_tuple,
        standard_monomials=standard_monomials,
    )
    return certificate, dimension, vector_dimension, _format_version(version_number)


def _format_version(version_number: int) -> str:
    major, remainder = divmod(version_number, 10_000)
    minor, patch_code = divmod(remainder, 1_000)
    patch = patch_code // 100
    return f"{major}.{minor}.{patch}"


def run_singular_generic_fiber(
    polynomial_map: RationalPolynomialMap,
    budget: GenericDegreeComputationBudget,
) -> SingularGenericFiberResult:
    """Compute one exact generic-fiber certificate in a bounded process."""

    resolved = shutil.which("Singular")
    if resolved is None:
        return SingularGenericFiberResult(
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
                [resolved, "-q"],
                input_bytes=_generic_fiber_script(polynomial_map),
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
        return SingularGenericFiberResult(
            outcome="UNAVAILABLE",
            detail="The supported Singular backend could not be started.",
        )
    if completed.timed_out:
        return SingularGenericFiberResult(
            outcome="TIMEOUT",
            detail="Singular exceeded the declared wall-time limit.",
        )
    if completed.cancelled:
        return SingularGenericFiberResult(
            outcome="ERROR",
            detail="Singular execution was cancelled before producing a result.",
        )
    if completed.stdout_exceeded or completed.stderr_exceeded:
        return SingularGenericFiberResult(
            outcome="LIMIT_EXCEEDED" if completed.stdout_exceeded else "ERROR",
            detail=(
                "The exact Singular generic-fiber certificate exceeds the declared "
                "result bound."
                if completed.stdout_exceeded
                else "Singular exceeded the diagnostic-output limit."
            ),
        )
    if completed.returncode != 0 or completed.stderr:
        return SingularGenericFiberResult(
            outcome="ERROR",
            detail=(
                "Singular failed without producing an exact generic-fiber certificate."
            ),
        )
    try:
        certificate, dimension, vector_dimension, version = _parse_generic_fiber_output(
            completed.stdout,
            polynomial_map=polynomial_map,
            budget=budget,
        )
    except _ResultLimitExceededError:
        return SingularGenericFiberResult(
            outcome="LIMIT_EXCEEDED",
            detail=(
                "The exact Singular generic-fiber certificate exceeds the declared "
                "result bound."
            ),
        )
    except ValueError:
        return SingularGenericFiberResult(
            outcome="ERROR",
            detail=(
                "Singular returned an invalid or unsupported generic-fiber certificate."
            ),
        )
    return SingularGenericFiberResult(
        outcome="COMPUTED",
        certificate=certificate,
        dimension=dimension,
        vector_dimension=vector_dimension,
        backend_version=version,
    )


__all__ = ["SingularGenericFiberResult", "run_singular_generic_fiber"]
