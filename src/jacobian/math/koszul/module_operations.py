"""Exact finite-module Koszul construction and homology."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from fractions import Fraction
from itertools import combinations
from math import comb
from typing import Any

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    canonical_rational_component_digits,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.koszul.module_models import (
    BasedFiniteModule,
    FiniteCommutativeAlgebra,
    ModuleChainMapMatrix,
    ModuleDifferential,
    ModuleKoszulChainMap,
    ModuleKoszulComplex,
    ModuleKoszulDifferentialRequest,
    ModuleKoszulDifferentialValue,
    ModuleKoszulDirectSumRequest,
    ModuleKoszulDirectSumValue,
    ModuleKoszulExactnessProfile,
    ModuleKoszulHomology,
    ModuleKoszulHomologyDegree,
    ModuleKoszulHomologyRequest,
    ModuleKoszulMapRequest,
    ModuleKoszulRequest,
    ModuleKoszulSequencePermutation,
    ModuleKoszulSequencePermutationRequest,
    ModuleKoszulTopHomology,
    ModuleKoszulTopHomologyRequest,
    ModuleKoszulUnitContraction,
    ModuleKoszulUnitContractionRequest,
    ModuleKoszulZeroExtension,
    ModuleKoszulZeroExtensionRequest,
    ModuleQuotientValue,
)

MAX_KOSZUL_HOMOLOGY_COEFFICIENT_DIGITS = 128
MAX_KOSZUL_HOMOLOGY_WORK = 1 << 40
MAX_KOSZUL_HOMOLOGY_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_KOSZUL_SEQUENCE_TRANSFORM_WORK = 2_000_000
MAX_KOSZUL_SEQUENCE_TRANSFORM_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_KOSZUL_UNIT_CONTRACTION_WORK = 2_000_000
MAX_KOSZUL_UNIT_CONTRACTION_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_KOSZUL_ZERO_EXTENSION_WORK = 2_000_000
MAX_KOSZUL_ZERO_EXTENSION_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_KOSZUL_MODULE_MAP_WORK = 2_000_000
MAX_KOSZUL_MODULE_MAP_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_KOSZUL_MODULE_MAP_CELLS = 4_096
MAX_KOSZUL_TOP_HOMOLOGY_WORK = 1 << 40
MAX_KOSZUL_TOP_HOMOLOGY_OUTPUT_BYTES = 8 * 1024 * 1024
_MAX_KOSZUL_HOMOLOGY_COEFFICIENT = 10**MAX_KOSZUL_HOMOLOGY_COEFFICIENT_DIGITS


def _f(value: CanonicalRational) -> Fraction:
    return Fraction(value.num, value.den)


def _matrix_rank(matrix: list[list[Fraction]]) -> int:
    if not matrix or not matrix[0]:
        return 0
    reduced = [row[:] for row in matrix]
    rows, columns = len(reduced), len(reduced[0])
    rank = 0
    for column in range(columns):
        pivot = next((row for row in range(rank, rows) if reduced[row][column]), None)
        if pivot is None:
            continue
        reduced[rank], reduced[pivot] = reduced[pivot], reduced[rank]
        scale = reduced[rank][column]
        reduced[rank] = [value / scale for value in reduced[rank]]
        for row in range(rows):
            if row != rank and reduced[row][column]:
                factor = reduced[row][column]
                reduced[row] = [
                    left - factor * right
                    for left, right in zip(reduced[row], reduced[rank], strict=True)
                ]
        rank += 1
        if rank == rows:
            break
    return rank


def _rref_with_pivots(
    matrix: list[list[Fraction]], width: int
) -> tuple[list[list[Fraction]], list[int]]:
    """Return reduced rows and pivot columns for a matrix with known width."""
    if not matrix:
        return [], []
    reduced = [row[:] for row in matrix]
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(width):
        pivot = next(
            (row for row in range(pivot_row, len(reduced)) if reduced[row][column]),
            None,
        )
        if pivot is None:
            continue
        reduced[pivot_row], reduced[pivot] = reduced[pivot], reduced[pivot_row]
        scale = reduced[pivot_row][column]
        reduced[pivot_row] = [entry / scale for entry in reduced[pivot_row]]
        for row in range(len(reduced)):
            if row != pivot_row and reduced[row][column]:
                factor = reduced[row][column]
                reduced[row] = [
                    left - factor * right
                    for left, right in zip(
                        reduced[row], reduced[pivot_row], strict=True
                    )
                ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(reduced):
            break
    return reduced[:pivot_row], pivot_columns


def _nullspace(matrix: list[list[Fraction]], width: int) -> list[list[Fraction]]:
    """Return a deterministic basis for the kernel in source coordinates."""
    reduced, pivots = _rref_with_pivots(matrix, width)
    pivot_rows = dict(zip(pivots, reduced, strict=True))
    basis: list[list[Fraction]] = []
    for free_column in (column for column in range(width) if column not in pivot_rows):
        vector = [Fraction(0) for _ in range(width)]
        vector[free_column] = Fraction(1)
        for pivot_column, row in pivot_rows.items():
            vector[pivot_column] = -row[free_column]
        basis.append(vector)
    return basis


def _column_space(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
    """Return a deterministic basis of column vectors, including zero-row maps."""
    if not matrix:
        return []
    transposed = [list(column) for column in zip(*matrix, strict=True)]
    return _row_space_basis(transposed)


def _extend_basis(
    initial: list[list[Fraction]], candidates: list[list[Fraction]], width: int
) -> list[list[Fraction]]:
    """Append independent candidates using incremental exact elimination."""
    echelon: dict[int, list[Fraction]] = {}
    for source in (*initial,):
        vector = source[:]
        for pivot in sorted(echelon):
            if vector[pivot]:
                factor = vector[pivot]
                vector = [
                    left - factor * right
                    for left, right in zip(vector, echelon[pivot], strict=True)
                ]
        pivot = next((index for index, value in enumerate(vector) if value), None)
        if pivot is not None:
            scale = vector[pivot]
            echelon[pivot] = [value / scale for value in vector]
    appended: list[list[Fraction]] = []
    for source in candidates:
        vector = source[:]
        for pivot in sorted(echelon):
            if vector[pivot]:
                factor = vector[pivot]
                vector = [
                    left - factor * right
                    for left, right in zip(vector, echelon[pivot], strict=True)
                ]
        pivot = next((index for index, value in enumerate(vector) if value), None)
        if pivot is not None:
            scale = vector[pivot]
            echelon[pivot] = [value / scale for value in vector]
            appended.append(source)
        if len(echelon) == width:
            break
    return appended


def _structure(algebra: FiniteCommutativeAlgebra) -> list[list[list[Fraction]]]:
    return [
        [[_f(value) for value in cell] for cell in row]
        for row in algebra.multiplication
    ]


def _basis_element(index: int, dimension: int) -> tuple[CanonicalRational, ...]:
    return tuple(
        CanonicalRational.from_fraction(Fraction(1 if coordinate == index else 0))
        for coordinate in range(dimension)
    )


def _action_matrix(
    module: BasedFiniteModule, element: tuple[CanonicalRational, ...]
) -> list[list[Fraction]]:
    dimension = len(module.basis)
    result = [[Fraction(0) for _ in range(dimension)] for _ in range(dimension)]
    for coefficient, matrix in zip(element, module.action, strict=True):
        scale = _f(coefficient)
        for row in range(dimension):
            for column in range(dimension):
                result[row][column] += scale * _f(matrix[row][column])
    return result


def _build_module_koszul_differential(
    value: ModuleKoszulRequest,
    degree: int,
    actions: tuple[list[list[Fraction]], ...] | None = None,
) -> ModuleDifferential:
    """Build one canonical degree differential from cached element actions."""
    length = len(value.sequence)
    module_dimension = len(value.module.basis)
    source_wedges = tuple(combinations(range(length), degree))
    target_wedges = tuple(combinations(range(length), degree - 1))
    target_index = {wedge: index for index, wedge in enumerate(target_wedges)}
    if actions is None:
        actions = tuple(
            _action_matrix(value.module, element) for element in value.sequence
        )
    entries: list[tuple[int, int, CanonicalRational]] = []
    for wedge_column, wedge in enumerate(source_wedges):
        for position, sequence_index in enumerate(wedge):
            sign = -1 if position % 2 else 1
            target = wedge[:position] + wedge[position + 1 :]
            for row in range(module_dimension):
                for column in range(module_dimension):
                    coefficient = sign * actions[sequence_index][row][column]
                    if coefficient:
                        entries.append(
                            (
                                target_index[target] * module_dimension + row,
                                wedge_column * module_dimension + column,
                                CanonicalRational.from_fraction(coefficient),
                            )
                        )
    return ModuleDifferential(
        row_count=len(target_wedges) * module_dimension,
        column_count=len(source_wedges) * module_dimension,
        entries=tuple(sorted(entries, key=lambda entry: (entry[0], entry[1]))),
    )


def _admit(
    module: BasedFiniteModule, sequence: tuple[tuple[CanonicalRational, ...], ...]
) -> None:
    # This is mathematical admission for the Koszul postcondition, not merely
    # a shape check.  It is intentionally rerun by consumers of authored
    # complexes: serialized/model_construct values carry no trusted provenance.
    if len(sequence) > 6 or len(module.basis) * (2 ** len(sequence)) > 256:
        raise OperationResourceAdmissionError(
            location=("sequence",),
            code="koszul.module.budget",
            message="finite-module Koszul complex exceeds its admitted envelope",
        )
    algebra = _structure(module.algebra)
    algebra_dimension = len(algebra)
    module_dimension = len(module.basis)
    for first in range(algebra_dimension):
        for second in range(algebra_dimension):
            if any(
                algebra[first][second][target] != algebra[second][first][target]
                for target in range(algebra_dimension)
            ):
                raise OperationDomainValidationError(
                    location=("algebra",),
                    code="koszul.module.noncommutative",
                    message="the finite algebra must be commutative",
                )
            for third in range(algebra_dimension):
                for target in range(algebra_dimension):
                    left = sum(
                        algebra[first][second][middle] * algebra[middle][third][target]
                        for middle in range(algebra_dimension)
                    )
                    right = sum(
                        algebra[second][third][middle] * algebra[first][middle][target]
                        for middle in range(algebra_dimension)
                    )
                    if left != right:
                        raise OperationDomainValidationError(
                            location=("algebra",),
                            code="koszul.module.nonassociative",
                            message="the finite algebra must be associative",
                        )
    for first in range(algebra_dimension):
        for second in range(algebra_dimension):
            product = tuple(
                CanonicalRational.from_fraction(algebra[first][second][target])
                for target in range(algebra_dimension)
            )
            left_action = _action_matrix(
                module, _basis_element(first, algebra_dimension)
            )
            right_action = _action_matrix(
                module, _basis_element(second, algebra_dimension)
            )
            composed = [
                [
                    sum(
                        left_action[row][middle] * right_action[middle][column]
                        for middle in range(module_dimension)
                    )
                    for column in range(module_dimension)
                ]
                for row in range(module_dimension)
            ]
            if composed != _action_matrix(module, product):
                raise OperationDomainValidationError(
                    location=("module",),
                    code="koszul.module.action",
                    message="module action does not respect algebra multiplication",
                )


def _dense(differential: ModuleDifferential) -> list[list[Fraction]]:
    matrix = [
        [Fraction(0) for _ in range(differential.column_count)]
        for _ in range(differential.row_count)
    ]
    for row, column, value in differential.entries:
        matrix[row][column] = _f(value)
    return matrix


def _as_request(
    request: ModuleKoszulRequest | Mapping[str, Any],
) -> ModuleKoszulRequest:
    try:
        payload = (
            request.model_dump()
            if isinstance(request, ModuleKoszulRequest)
            else request
        )
        # Reparse typed values as well: model_construct can forge nested parent
        # and matrix shapes that a trusted instance would otherwise bypass.
        return ModuleKoszulRequest.model_validate(payload)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="koszul.module.request_shape",
            message="the module Koszul request is not canonical",
        ) from exc


def module_koszul_complex(
    request: ModuleKoszulRequest | Mapping[str, Any],
) -> ModuleKoszulComplex:
    value = _as_request(request)
    _admit(value.module, value.sequence)
    return _build_module_koszul_complex(value)


def _build_module_koszul_complex(
    value: ModuleKoszulRequest,
    *,
    check_square: bool = True,
) -> ModuleKoszulComplex:
    """Build from an already parsed and admitted request."""
    sequence_length = len(value.sequence)
    module_dimension = len(value.module.basis)
    bases = tuple(
        tuple(combinations(range(sequence_length), degree))
        for degree in range(sequence_length + 1)
    )
    basis_sizes = tuple(module_dimension * len(basis) for basis in bases)
    actions = tuple(_action_matrix(value.module, element) for element in value.sequence)
    differentials = [
        _build_module_koszul_differential(value, degree, actions)
        for degree in range(1, sequence_length + 1)
    ]
    if check_square:
        for index in range(1, len(differentials)):
            outer = _dense(differentials[index - 1])
            inner = _dense(differentials[index])
            for row in range(len(outer)):
                for column in range(len(inner[0]) if inner else 0):
                    if sum(
                        outer[row][middle] * inner[middle][column]
                        for middle in range(len(inner))
                    ):
                        raise OperationDomainValidationError(
                            location=("sequence",),
                            code="koszul.module.differential_square",
                            message="module Koszul differential does not square to zero",
                        )
    return ModuleKoszulComplex.model_construct(
        algebra=value.algebra,
        module=value.module,
        sequence=value.sequence,
        basis_sizes=basis_sizes,
        differentials=tuple(differentials),
        square_zero=True,
    )


def _admit_complex(
    value: ModuleKoszulComplex, *, admit_homology: bool = True
) -> ModuleKoszulComplex:
    """Revalidate authored complexes before any dense matrix allocation.

    ``model_construct`` is intentionally available to internal result builders,
    so homology is also an admission boundary for typed values supplied by a
    caller.  Revalidating the serialized shape catches forged differential
    coordinates, duplicate/unsorted sparse keys, and parent/axis mismatches.
    """
    try:
        candidate = ModuleKoszulComplex.model_validate(value.model_dump())
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("complex",),
            code="koszul.module.complex_shape",
            message="the supplied module Koszul complex is not canonical",
        ) from exc
    sequence_length = len(candidate.sequence)
    module_dimension = len(candidate.module.basis)
    if sequence_length > 6 or (
        admit_homology and module_dimension * (2**sequence_length) > 256
    ):
        raise OperationResourceAdmissionError(
            location=("complex", "sequence"),
            code="koszul.module.budget",
            message="finite-module Koszul complex exceeds its admitted envelope",
        )
    expected_sizes = tuple(
        module_dimension * comb(sequence_length, degree)
        for degree in range(sequence_length + 1)
    )
    if candidate.basis_sizes != expected_sizes or any(
        not isinstance(size, int) or isinstance(size, bool) or size < 0
        for size in candidate.basis_sizes
    ):
        raise OperationDomainValidationError(
            location=("complex", "basis_sizes"),
            code="koszul.module.result_shape",
            message="complex basis sizes must be the canonical Koszul dimensions",
        )
    if any(
        len(element) != len(candidate.algebra.basis) for element in candidate.sequence
    ):
        raise OperationDomainValidationError(
            location=("complex", "sequence"),
            code="koszul.module.sequence_shape",
            message="sequence coordinates must use the algebra basis",
        )
    if admit_homology:
        _admit_homology_output(candidate)
        # Validate coefficient/work estimates before semantic source replay can
        # multiply caller-supplied rationals, and well before square checks/RREF.
        _admit_homology_rank_work(candidate)
    # A homology consumer relies on this being a Koszul complex over the
    # retained module parent, not merely an arbitrary square-zero matrix.
    _admit(candidate.module, candidate.sequence)
    return candidate


def _require_square_zero(value: ModuleKoszulComplex) -> None:
    for index in range(1, len(value.differentials)):
        outer = _dense(value.differentials[index - 1])
        inner = _dense(value.differentials[index])
        for row in range(len(outer)):
            for column in range(len(inner[0]) if inner else 0):
                if sum(
                    outer[row][middle] * inner[middle][column]
                    for middle in range(len(inner))
                ):
                    raise OperationDomainValidationError(
                        location=("complex", "differentials"),
                        code="koszul.module.differential_square",
                        message="the supplied complex does not satisfy d^2=0",
                    )


def _algebra_rationals(
    algebra: FiniteCommutativeAlgebra,
) -> Iterator[CanonicalRational]:
    yield from (
        coefficient
        for row in algebra.multiplication
        for cell in row
        for coefficient in cell
    )
    if algebra.unit is not None:
        yield from algebra.unit


def _homology_rationals(value: ModuleKoszulComplex) -> Iterator[CanonicalRational]:
    yield from _algebra_rationals(value.algebra)
    # The result echoes both source fields, including the module's nested
    # algebra value, so count it twice for its transport bound.
    yield from _algebra_rationals(value.module.algebra)
    yield from (
        coefficient
        for action in value.module.action
        for row in action
        for coefficient in row
    )
    yield from (coefficient for element in value.sequence for coefficient in element)
    yield from (
        coefficient
        for differential in value.differentials
        for _, _, coefficient in differential.entries
    )


def _decimal_digits_upper_bound(value: int) -> int:
    bits = abs(value).bit_length()
    # log10(2) < 30103/100000; this integer-only estimate safely bounds the
    # decimal digits serialized for any numerator or denominator.
    return (bits * 30_103) // 100_000 + 1


def _admit_homology_output(value: ModuleKoszulComplex) -> None:
    """Bound echoed exact data and representative bases before semantic replay."""

    digit_count = 0
    rational_count = 0
    for coefficient in _homology_rationals(value):
        numerator, denominator = abs(coefficient.num), coefficient.den
        if (
            numerator >= _MAX_KOSZUL_HOMOLOGY_COEFFICIENT
            or denominator >= _MAX_KOSZUL_HOMOLOGY_COEFFICIENT
        ):
            raise OperationResourceAdmissionError(
                location=("complex",),
                code="koszul.module.homology_coefficient_budget",
                message=(
                    "homology coefficients exceed the admitted exact-rank "
                    "component digit bound"
                ),
            )
        digit_count += _decimal_digits_upper_bound(numerator)
        digit_count += _decimal_digits_upper_bound(denominator)
        rational_count += 1

    entry_count = sum(len(matrix.entries) for matrix in value.differentials)
    # Basis elimination uses only differential entries; echoed algebra/action
    # coefficients do not participate in RREF and must not inflate this bound.
    max_input_digits = max(
        (
            canonical_rational_component_digits(coefficient)
            for differential in value.differentials
            for _, _, coefficient in differential.entries
        ),
        default=1,
    )
    max_dimension = max(value.basis_sizes, default=0)
    # RREF coordinates are ratios of minors. This conservative decimal bound
    # covers the cycle, image, and quotient bases returned in every degree.
    basis_coefficient_digits = 4 * max_dimension * (max_input_digits + 4) + 32
    representative_scalars = 3 * sum(value.basis_sizes)
    labels = (
        *value.algebra.basis,
        *value.module.algebra.basis,
        *value.module.basis,
    )
    escaped_label_bytes = sum(12 * len(label) for label in labels)
    # Each rational needs its JSON keys/quotes, each sparse cell needs row,
    # column and list syntax, and the remaining fixed object/list structure is
    # covered by 4096 bytes. String labels are conservatively JSON-escaped.
    output_bytes = (
        digit_count
        + 32 * rational_count
        + 24 * entry_count
        + representative_scalars * (2 * basis_coefficient_digits + 48)
        + escaped_label_bytes
        + 16_384
    )
    if output_bytes > MAX_KOSZUL_HOMOLOGY_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="koszul.module.homology_output_budget",
            message="serialized finite-module Koszul homology exceeds its output bound",
        )


def _admit_homology_rank_work(value: ModuleKoszulComplex) -> None:
    """Bound exact matrix arithmetic before square checks and rational RREF.

    For each sparse matrix, clear denominators row by row using their product
    as a conservative common-denominator bound. Every cleared integer entry
    then has fewer than ``entry_bits`` bits. Hadamard's inequality bounds each
    pivot minor by ``k * (entry_bits + ceil(log2(k)))`` bits. Gauss-Jordan
    entries are Schur complements, hence ratios of adjacent-size minors; the
    extra row denominator is included in ``minor_bits`` below. Products and
    subtractions of those reduced fractions need at most ``4 * minor_bits + 2``
    transient component bits. Finally, the operation count dominates the
    dense scan, pivot normalization, and two rational operations per update.

    The squared bit width is a conservative schoolbook bound for multiplication
    and Euclidean normalization. This is an admission estimate, not a claim
    about wall-clock completion.
    """

    work = 0
    max_transient_bits = 1
    for differential in value.differentials:
        rows, columns = differential.row_count, differential.column_count
        rank_bound = min(rows, columns)
        if rank_bound == 0:
            continue

        row_denominator_bits = [0] * rows
        row_numerator_bits = [0] * rows
        for row, _, coefficient in differential.entries:
            numerator, denominator = abs(coefficient.num), coefficient.den
            if (
                numerator >= _MAX_KOSZUL_HOMOLOGY_COEFFICIENT
                or denominator >= _MAX_KOSZUL_HOMOLOGY_COEFFICIENT
            ):
                raise OperationResourceAdmissionError(
                    location=("complex", "differentials"),
                    code="koszul.module.homology_coefficient_budget",
                    message=(
                        "homology coefficients exceed the admitted exact-rank "
                        "component digit bound"
                    ),
                )
            row_numerator_bits[row] = max(
                row_numerator_bits[row], numerator.bit_length()
            )
            # log2(d) is less than bit_length(d). The denominator one needs
            # no row scaling, which keeps integer matrices tightly estimated.
            if denominator != 1:
                row_denominator_bits[row] += denominator.bit_length()

        row_scale_bits = max(row_denominator_bits, default=0)
        entry_bits = max(
            (
                numerator_bits + denominator_bits
                for numerator_bits, denominator_bits in zip(
                    row_numerator_bits, row_denominator_bits, strict=True
                )
            ),
            default=1,
        )
        augmented_size = rank_bound + 1
        minor_bits = (
            augmented_size * (entry_bits + augmented_size.bit_length()) + row_scale_bits
        )
        transient_bits = 4 * minor_bits + 2
        max_transient_bits = max(max_transient_bits, transient_bits)
        rational_operations = (
            rows * columns + rank_bound * columns + 2 * rank_bound * rows * columns
        )
        work += rational_operations * transient_bits * transient_bits * 16

    # The exact d^2 replay also runs before rank. A product entry has at most
    # `middle` products of two reduced fractions; summing them with Fraction
    # arithmetic is bounded by multiplying the component-bit bound by that
    # number of terms, plus a carry allowance.
    square_operations = 0
    square_peak_bits = max_transient_bits
    for outer, inner in zip(value.differentials, value.differentials[1:], strict=False):
        middle = inner.row_count
        square_operations += outer.row_count * inner.column_count * 2 * middle
        if middle:
            square_peak_bits = max(
                square_peak_bits,
                2 * middle * max_transient_bits + middle * middle.bit_length() + 2,
            )
    work += square_operations * square_peak_bits * square_peak_bits * 16

    # Homology representatives require one kernel RREF, one image basis
    # elimination, and incremental extension of image bases by cycles.
    # Four times the rank estimate bounds these additional exact operations.
    work *= 4

    if work > MAX_KOSZUL_HOMOLOGY_WORK:
        raise OperationResourceAdmissionError(
            location=("complex", "differentials"),
            code="koszul.module.homology_work_budget",
            message=(
                "exact Koszul homology matrix work exceeds the admitted "
                "coefficient-aware bound"
            ),
        )


def module_koszul_homology(
    complex_value: ModuleKoszulComplex | Mapping[str, Any],
) -> ModuleKoszulHomology:
    try:
        value = (
            complex_value
            if isinstance(complex_value, ModuleKoszulComplex)
            else ModuleKoszulComplex.model_validate(complex_value)
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("complex",),
            code="koszul.module.complex_shape",
            message="the supplied module Koszul complex is not canonical",
        ) from exc
    value = _admit_complex(value)
    _require_square_zero(value)
    cycles: list[int] = []
    boundaries: list[int] = []
    dimensions: list[int] = []
    degrees: list[ModuleKoszulHomologyDegree] = []
    for degree, size in enumerate(value.basis_sizes):
        outgoing = _dense(value.differentials[degree - 1]) if degree > 0 else []
        cycle_basis = _nullspace(outgoing, size)
        boundary_basis = (
            _column_space(_dense(value.differentials[degree]))
            if degree < len(value.differentials)
            else []
        )
        homology_basis = _extend_basis(boundary_basis, cycle_basis, size)
        cycle_dimension = len(cycle_basis)
        boundary_dimension = len(boundary_basis)
        cycles.append(cycle_dimension)
        boundaries.append(boundary_dimension)
        dimensions.append(len(homology_basis))

        def to_exact(
            basis: list[list[Fraction]],
        ) -> tuple[tuple[CanonicalRational, ...], ...]:
            return tuple(
                tuple(CanonicalRational.from_fraction(value) for value in vector)
                for vector in basis
            )

        degrees.append(
            ModuleKoszulHomologyDegree.model_construct(
                degree=degree,
                cycle_basis=to_exact(cycle_basis),
                boundary_basis=to_exact(boundary_basis),
                homology_basis=to_exact(homology_basis),
            )
        )
    return ModuleKoszulHomology.model_construct(
        complex=value,
        dimensions=tuple(dimensions),
        cycle_dimensions=tuple(cycles),
        boundary_dimensions=tuple(boundaries),
        degrees=tuple(degrees),
    )


def _admit_top_homology(value: ModuleKoszulComplex) -> None:
    """Bound retained source context and one top-kernel elimination before work."""
    retained_coefficients = (
        *_algebra_rationals(value.algebra),
        *_algebra_rationals(value.module.algebra),
        *(
            coefficient
            for action in value.module.action
            for row in action
            for coefficient in row
        ),
        *(coefficient for element in value.sequence for coefficient in element),
        *(
            coefficient
            for differential in value.differentials[-1:]
            for _, _, coefficient in differential.entries
        ),
    )
    for coefficient in retained_coefficients:
        if (
            abs(coefficient.num) >= _MAX_KOSZUL_HOMOLOGY_COEFFICIENT
            or coefficient.den >= _MAX_KOSZUL_HOMOLOGY_COEFFICIENT
        ):
            raise OperationResourceAdmissionError(
                location=("complex",),
                code="koszul.module.homology_coefficient_budget",
                message="top Koszul homology coefficients exceed the exact digit bound",
            )

    echoed_bytes = (
        len(value.algebra.model_dump_json().encode("utf-8"))
        + len(value.module.model_dump_json().encode("utf-8"))
        + (
            sum(
                2 * canonical_rational_component_digits(item) + 24
                for element in value.sequence
                for item in element
            )
        )
        + (
            len(value.differentials[-1].model_dump_json().encode("utf-8"))
            if value.differentials
            else 0
        )
        + 256
    )
    module_dimension = len(value.module.basis)
    differential = value.differentials[-1] if value.differentials else None
    basis_digits = 1
    work = 0
    if differential is not None:
        rows, columns = differential.row_count, differential.column_count
        rank_bound = min(rows, columns)
        row_denominator_bits = [0] * rows
        row_numerator_bits = [0] * rows
        max_input_digits = 1
        for row, _, coefficient in differential.entries:
            numerator, denominator = abs(coefficient.num), coefficient.den
            row_numerator_bits[row] = max(
                row_numerator_bits[row], numerator.bit_length()
            )
            if denominator != 1:
                row_denominator_bits[row] += denominator.bit_length()
            max_input_digits = max(
                max_input_digits, canonical_rational_component_digits(coefficient)
            )
        row_scale_bits = max(row_denominator_bits, default=0)
        entry_bits = max(
            (
                numerator_bits + denominator_bits
                for numerator_bits, denominator_bits in zip(
                    row_numerator_bits, row_denominator_bits, strict=True
                )
            ),
            default=1,
        )
        minor_bits = (rank_bound + 1) * (
            entry_bits + (rank_bound + 1).bit_length()
        ) + row_scale_bits
        transient_bits = 4 * minor_bits + 2
        operations = (
            rows * columns + rank_bound * columns + 2 * rank_bound * rows * columns
        )
        work = operations * transient_bits * transient_bits * 16
        basis_digits = 4 * max(rows, columns) * (max_input_digits + 4) + 32
    if work > MAX_KOSZUL_TOP_HOMOLOGY_WORK:
        raise OperationResourceAdmissionError(
            location=("complex", "differentials"),
            code="koszul.module.top_homology_work_budget",
            message="top Koszul homology kernel exceeds its exact work bound",
        )

    # At most two module-dimension-square bases are returned. The coefficient
    # bound is the same minor bound used for deterministic rational RREF.
    output_bytes = (
        echoed_bytes + 2 * module_dimension**2 * (2 * basis_digits + 48) + 4096
    )
    if output_bytes > MAX_KOSZUL_TOP_HOMOLOGY_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="koszul.module.top_homology_output_budget",
            message="top Koszul homology value exceeds its serialized output bound",
        )


def module_koszul_top_homology(
    request: ModuleKoszulTopHomologyRequest | Mapping[str, Any],
) -> ModuleKoszulTopHomology:
    """Identify top Koszul homology with the common annihilator of the sequence.

    In top degree there is a single exterior basis wedge, and its differential
    has the signed action matrices of all sequence entries as its row blocks.
    Thus its kernel is precisely ``{m : f_i m = 0 for every i}``. The source
    differential is reconstructed from the retained module action before this
    identity is used.
    """
    try:
        payload = (
            request.model_dump()
            if isinstance(request, ModuleKoszulTopHomologyRequest)
            else request
        )
        parsed_request = ModuleKoszulTopHomologyRequest.model_validate(payload)
        value = ModuleKoszulComplex.model_validate(parsed_request.complex.model_dump())
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="koszul.module.top_homology_request_shape",
            message="the top-homology request is not canonical",
        ) from exc

    # The top-kernel contract needs no elimination in lower degrees. Admit the
    # retained input and this one matrix before rebuilding or densifying it.
    _admit_top_homology(value)
    value = _admit_complex(value, admit_homology=False)
    source_request = ModuleKoszulRequest(
        algebra=value.algebra, module=value.module, sequence=value.sequence
    )
    dimension = len(value.module.basis)
    top_differential = None
    if value.sequence:
        actions = tuple(
            _action_matrix(value.module, element) for element in value.sequence
        )
        top_differential = _build_module_koszul_differential(
            source_request, len(value.sequence), actions
        )
        if top_differential != value.differentials[-1]:
            raise OperationDomainValidationError(
                location=("complex", "differentials"),
                code="koszul.module.source_complex_mismatch",
                message="top homology requires the top differential induced by the retained sequence",
            )
    kernel = _nullspace(
        _dense(top_differential) if top_differential is not None else [], dimension
    )
    exact_basis = tuple(
        tuple(CanonicalRational.from_fraction(coefficient) for coefficient in vector)
        for vector in kernel
    )
    return ModuleKoszulTopHomology.model_construct(
        algebra=value.algebra,
        module=value.module,
        sequence=value.sequence,
        top_differential=top_differential,
        annihilator_basis=exact_basis,
        top_homology_basis=exact_basis,
    )


def module_koszul_exactness_profile(
    request: ModuleKoszulHomologyRequest | Mapping[str, Any],
) -> ModuleKoszulExactnessProfile:
    """Return positive-degree acyclicity and its first exact class witness.

    The profile describes this complete finite Koszul complex only. Acyclicity
    above degree zero is not promoted to a theorem about regular sequences.
    The homology operation's coefficient, work, and output admission also
    bounds this smaller projection before any exact elimination is performed.
    """
    try:
        value = (
            request
            if isinstance(request, ModuleKoszulHomologyRequest)
            else ModuleKoszulHomologyRequest.model_validate(request)
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="koszul.module.homology_request_shape",
            message="the Koszul exactness request is not canonical",
        ) from exc
    homology = module_koszul_homology(value.complex)
    first_nonzero = next(
        (
            degree
            for degree, dimension in enumerate(homology.dimensions[1:], start=1)
            if dimension
        ),
        None,
    )
    representative = (
        homology.degrees[first_nonzero].homology_basis[0]
        if first_nonzero is not None
        else None
    )
    return ModuleKoszulExactnessProfile.model_construct(
        complex=homology.complex,
        homology_dimensions=homology.dimensions,
        acyclic_above_zero=first_nonzero is None,
        first_nonzero_degree=first_nonzero,
        first_nonzero_class=representative,
    )


def _permutation_map(
    basis_sizes: tuple[int, ...], module_dimension: int, old_to_new: tuple[int, ...]
) -> tuple[ModuleDifferential, ...]:
    """Build exterior permutation matrices in wedge-major basis order."""
    length = len(old_to_new)
    degree_wedges = tuple(
        tuple(combinations(range(length), degree)) for degree in range(length + 1)
    )
    target_positions = tuple(
        {wedge: index for index, wedge in enumerate(wedges)} for wedges in degree_wedges
    )
    maps: list[ModuleDifferential] = []
    for degree, wedges in enumerate(degree_wedges):
        entries: list[tuple[int, int, CanonicalRational]] = []
        for source_wedge_index, source_wedge in enumerate(wedges):
            permuted = tuple(old_to_new[index] for index in source_wedge)
            target_wedge = tuple(sorted(permuted))
            inversions = sum(
                left > right
                for position, left in enumerate(permuted)
                for right in permuted[position + 1 :]
            )
            sign = -1 if inversions % 2 else 1
            target_wedge_index = target_positions[degree][target_wedge]
            for module_index in range(module_dimension):
                entries.append(
                    (
                        target_wedge_index * module_dimension + module_index,
                        source_wedge_index * module_dimension + module_index,
                        CanonicalRational.from_fraction(Fraction(sign)),
                    )
                )
        maps.append(
            ModuleDifferential(
                row_count=basis_sizes[degree],
                column_count=basis_sizes[degree],
                entries=tuple(sorted(entries, key=lambda entry: (entry[0], entry[1]))),
            )
        )
    return tuple(maps)


def _sparse_columns(
    matrix: ModuleDifferential,
) -> tuple[dict[int, Fraction], ...]:
    columns: list[dict[int, Fraction]] = [{} for _ in range(matrix.column_count)]
    for row, column, coefficient in matrix.entries:
        columns[column][row] = _f(coefficient)
    return tuple(columns)


def _apply_sparse_columns(
    columns: tuple[dict[int, Fraction], ...], vector: Mapping[int, Fraction]
) -> dict[int, Fraction]:
    result: dict[int, Fraction] = {}
    for column, source in vector.items():
        if source:
            for row, coefficient in columns[column].items():
                result[row] = result.get(row, Fraction(0)) + coefficient * source
    return {index: coefficient for index, coefficient in result.items() if coefficient}


def _compose_sparse_maps(
    outer: tuple[dict[int, Fraction], ...],
    inner: tuple[dict[int, Fraction], ...],
) -> tuple[tuple[int, int, Fraction], ...]:
    """Return the sparse matrix for ``outer * inner`` in canonical order."""
    entries: list[tuple[int, int, Fraction]] = []
    for column, middle_values in enumerate(inner):
        result: dict[int, Fraction] = {}
        for middle, inner_coefficient in middle_values.items():
            for row, outer_coefficient in outer[middle].items():
                result[row] = result.get(row, Fraction(0)) + (
                    outer_coefficient * inner_coefficient
                )
        entries.extend(
            (row, column, coefficient)
            for row, coefficient in result.items()
            if coefficient
        )
    return tuple(sorted(entries, key=lambda entry: (entry[0], entry[1])))


def _verify_permutation_chain_map(
    source: ModuleKoszulComplex,
    target: ModuleKoszulComplex,
    forward: tuple[ModuleDifferential, ...],
    backward: tuple[ModuleDifferential, ...],
) -> None:
    for degree, (source_differential, target_differential) in enumerate(
        zip(source.differentials, target.differentials, strict=True), start=1
    ):
        source_columns = _sparse_columns(source_differential)
        target_columns = _sparse_columns(target_differential)
        forward_lower_columns = _sparse_columns(forward[degree - 1])
        forward_upper_columns = _sparse_columns(forward[degree])
        for column in range(source_differential.column_count):
            basis_vector = {column: Fraction(1)}
            left = _apply_sparse_columns(
                forward_lower_columns,
                _apply_sparse_columns(source_columns, basis_vector),
            )
            right = _apply_sparse_columns(
                target_columns,
                _apply_sparse_columns(forward_upper_columns, basis_vector),
            )
            if left != right:
                raise OperationDomainValidationError(
                    location=("permutation", "chain_map"),
                    code="koszul.module.sequence_permutation_chain_map",
                    message="the exterior permutation map does not commute with d",
                )
    for forward_map, backward_map in zip(forward, backward, strict=True):
        backward_columns: dict[int, tuple[int, Fraction]] = {}
        for row, column, coefficient in backward_map.entries:
            backward_columns[column] = (row, _f(coefficient))
        for row, column, coefficient in forward_map.entries:
            inverse_row, inverse_coefficient = backward_columns[row]
            if inverse_row != column or inverse_coefficient * _f(coefficient) != 1:
                raise OperationDomainValidationError(
                    location=("permutation", "inverse"),
                    code="koszul.module.sequence_permutation_inverse",
                    message="the returned degreewise maps are not mutual inverses",
                )


def _admit_sequence_permutation(
    value: ModuleKoszulSequencePermutationRequest,
) -> ModuleKoszulComplex:
    # The request model has already canonicalized the typed source payload.
    # Bound this operation's doubled complexes and maps before semantic replay.
    complex_value = value.complex
    algebra_dimension = len(complex_value.algebra.basis)
    module_dimension = len(complex_value.module.basis)
    sequence_length = len(complex_value.sequence)
    differential_entry_bound = (
        sequence_length * module_dimension**2 * (1 << (sequence_length - 1))
        if sequence_length
        else 0
    )
    source_entry_count = sum(
        len(differential.entries) for differential in complex_value.differentials
    )
    if source_entry_count > differential_entry_bound:
        raise OperationDomainValidationError(
            location=("complex", "differentials"),
            code="koszul.module.source_complex_mismatch",
            message="source differential sparsity cannot arise from its retained sequence",
        )
    algebra_digit_bound = max(
        (
            canonical_rational_component_digits(coefficient)
            for coefficient in _algebra_rationals(complex_value.algebra)
        ),
        default=1,
    )
    action_digit_bound = max(
        (
            canonical_rational_component_digits(coefficient)
            for action in complex_value.module.action
            for row in action
            for coefficient in row
        ),
        default=1,
    )
    sequence_digit_bound = max(
        (
            canonical_rational_component_digits(coefficient)
            for element in complex_value.sequence
            for coefficient in element
        ),
        default=1,
    )
    source_differential_digit_bound = max(
        (
            canonical_rational_component_digits(coefficient)
            for differential in complex_value.differentials
            for _, _, coefficient in differential.entries
        ),
        default=1,
    )
    differential_digit_bound = (
        algebra_dimension * (action_digit_bound + sequence_digit_bound)
        + len(str(algebra_dimension))
        + 4
        if sequence_length
        else 1
    )
    if differential_digit_bound > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationResourceAdmissionError(
            location=("complex", "sequence"),
            code="koszul.module.sequence_permutation_coefficient_budget",
            message="permuted Koszul differential coefficients exceed the exact digit limit",
        )

    basis_total = sum(complex_value.basis_sizes)
    differential_products = sum(
        complex_value.basis_sizes[degree - 2]
        * complex_value.basis_sizes[degree - 1]
        * complex_value.basis_sizes[degree]
        for degree in range(2, sequence_length + 1)
    )
    differential_build_work = (
        sequence_length
        * (1 << max(sequence_length - 1, 0))
        * algebra_dimension
        * module_dimension**2
        + differential_entry_bound
        + 2 * differential_products
    )
    semantic_operation_count = (
        3 * algebra_dimension**5
        + algebra_dimension**4
        + algebra_dimension**3
        + algebra_dimension**2 * module_dimension**3
        + algebra_dimension**3 * module_dimension**2
    )
    semantic_validation_work = (
        semantic_operation_count
        * max(algebra_digit_bound, action_digit_bound, sequence_digit_bound) ** 2
    )
    work_bound = (
        (2 * differential_build_work + 4 * basis_total + 4 * differential_entry_bound)
        * max(1, differential_digit_bound) ** 2
        + semantic_validation_work
        + differential_entry_bound * source_differential_digit_bound
    )

    complex_rational_items = (
        2 * (algebra_dimension**3 + algebra_dimension)
        + algebra_dimension * module_dimension**2
        + sequence_length * algebra_dimension
        + differential_entry_bound
    )
    map_entry_bound = 2 * basis_total
    rational_items = 2 * complex_rational_items + map_entry_bound
    maximum_digits = max(
        algebra_digit_bound,
        action_digit_bound,
        sequence_digit_bound,
        source_differential_digit_bound,
        differential_digit_bound,
        1,
    )
    labels = (
        *complex_value.algebra.basis,
        *complex_value.module.algebra.basis,
        *complex_value.module.basis,
    )
    label_bytes = sum(12 * len(label) for label in labels) * 2
    output_bytes = (
        rational_items * (2 * maximum_digits + 48)
        + 2 * differential_entry_bound * 48
        + map_entry_bound * 48
        + label_bytes
        + 8_192
    )
    if (
        work_bound > MAX_KOSZUL_SEQUENCE_TRANSFORM_WORK
        or output_bytes > MAX_KOSZUL_SEQUENCE_TRANSFORM_OUTPUT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="koszul.module.sequence_permutation_budget",
            message=(
                "sequence permutation maps, rebuilt complexes, or exact work "
                "exceed the admitted transform envelope"
            ),
        )
    return _admit_complex(complex_value, admit_homology=False)


def module_koszul_sequence_permute(
    request: ModuleKoszulSequencePermutationRequest | Mapping[str, Any],
) -> ModuleKoszulSequencePermutation:
    """Return the induced exterior-power chain isomorphism for a permutation.

    ``new_to_old[j]`` names the source sequence entry used in target position
    ``j``. The basis maps use the corresponding signed permutation on each
    increasing exterior wedge and the identity on the module coordinate.
    """
    try:
        payload = (
            request.model_dump()
            if isinstance(request, ModuleKoszulSequencePermutationRequest)
            else request
        )
        value = ModuleKoszulSequencePermutationRequest.model_validate(payload)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="koszul.module.sequence_permutation_request_shape",
            message="the Koszul sequence permutation request is not canonical",
        ) from exc
    source = _admit_sequence_permutation(value)
    canonical_source = _build_module_koszul_complex(
        ModuleKoszulRequest(
            algebra=source.algebra,
            module=source.module,
            sequence=source.sequence,
        )
    )
    if canonical_source.differentials != source.differentials:
        raise OperationDomainValidationError(
            location=("complex", "differentials"),
            code="koszul.module.source_complex_mismatch",
            message="source differentials must be induced by the retained sequence and action",
        )
    target_sequence = tuple(source.sequence[index] for index in value.new_to_old)
    target = _build_module_koszul_complex(
        ModuleKoszulRequest(
            algebra=source.algebra,
            module=source.module,
            sequence=target_sequence,
        )
    )
    inverse_permutation = tuple(
        value.new_to_old.index(index) for index in range(len(value.new_to_old))
    )
    forward = _permutation_map(
        source.basis_sizes, len(source.module.basis), inverse_permutation
    )
    backward = _permutation_map(
        source.basis_sizes, len(source.module.basis), value.new_to_old
    )
    _verify_permutation_chain_map(source, target, forward, backward)
    return ModuleKoszulSequencePermutation.model_construct(
        source_complex=source,
        target_complex=target,
        new_to_old=value.new_to_old,
        source_to_target=forward,
        target_to_source=backward,
    )


def _admit_unit_contraction(
    value: ModuleKoszulUnitContractionRequest,
) -> ModuleKoszulComplex:
    complex_value = value.complex
    algebra_dimension = len(complex_value.algebra.basis)
    module_dimension = len(complex_value.module.basis)
    sequence_length = len(complex_value.sequence)
    if value.unit_index >= sequence_length:
        raise OperationDomainValidationError(
            location=("unit_index",),
            code="koszul.module.unit_contraction_index",
            message="unit_index must select an entry of the retained sequence",
        )
    source_entries = sum(len(d.entries) for d in complex_value.differentials)
    digit_bound = max(
        (
            canonical_rational_component_digits(item)
            for item in _homology_rationals(complex_value)
        ),
        default=1,
    )
    # Inversion by exact elimination uses a small (at most 6 dimensional)
    # regular representation. Hadamard-style digit allowance bounds its
    # intermediate numerators before the system is expanded.
    inverse_digit_bound = 4 * algebra_dimension**2 * max(1, digit_bound)
    reconstructed_differential_digit_bound = (
        2 * algebra_dimension * max(1, digit_bound) + len(str(algebra_dimension)) + 4
    )
    homotopy_digit_bound = (
        inverse_digit_bound + digit_bound + len(str(algebra_dimension)) + 4
    )
    expanded_digit_bound = max(
        inverse_digit_bound,
        reconstructed_differential_digit_bound,
        homotopy_digit_bound,
    )
    if expanded_digit_bound > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="koszul.module.unit_contraction_coefficient_budget",
            message="inverse or homotopy coefficients exceed the exact digit limit",
        )
    inverse_work = algebra_dimension**4 * max(1, inverse_digit_bound**2)
    homotopy_entries = sum(
        (module_dimension**2) * comb(sequence_length - 1, degree)
        for degree in range(sequence_length)
    )
    verification_work = sum(
        complex_value.basis_sizes[degree]
        * (complex_value.basis_sizes[degree - 1] if degree > 0 else 0)
        * module_dimension
        + complex_value.basis_sizes[degree]
        * (complex_value.basis_sizes[degree + 1] if degree < sequence_length else 0)
        * module_dimension
        for degree in range(sequence_length + 1)
    )
    differential_entry_bound = (
        sequence_length * module_dimension**2 * (1 << max(sequence_length - 1, 0))
    )
    semantic_work = (
        3 * algebra_dimension**5
        + algebra_dimension**4
        + algebra_dimension**3
        + algebra_dimension**2 * module_dimension**3
        + algebra_dimension**3 * module_dimension**2
    ) * max(1, expanded_digit_bound**2)
    reconstruction_work = (
        sequence_length
        * (1 << max(sequence_length - 1, 0))
        * algebra_dimension
        * module_dimension**2
        + differential_entry_bound
        + 2
        * sum(
            complex_value.basis_sizes[degree - 2]
            * complex_value.basis_sizes[degree - 1]
            * complex_value.basis_sizes[degree]
            for degree in range(2, sequence_length + 1)
        )
    ) * max(1, expanded_digit_bound**2)
    work = (
        inverse_work
        + 8 * homotopy_entries * max(1, inverse_digit_bound**2)
        + verification_work
        + semantic_work
        + reconstruction_work
    )
    complex_rational_items = (
        2 * (algebra_dimension**3 + algebra_dimension)
        + algebra_dimension * module_dimension**2
        + sequence_length * algebra_dimension
        + source_entries
    )
    returned_rational_items = (
        complex_rational_items + algebra_dimension + homotopy_entries
    )
    labels = (
        *complex_value.algebra.basis,
        *complex_value.module.algebra.basis,
        *complex_value.module.basis,
    )
    label_bytes = 2 * sum(6 * len(label) + 4 for label in labels)
    output_bytes = (
        returned_rational_items * (2 * expanded_digit_bound + 48)
        + 48 * (source_entries + homotopy_entries)
        + label_bytes
        + 8_192
    )
    if (
        work > MAX_KOSZUL_UNIT_CONTRACTION_WORK
        or output_bytes > MAX_KOSZUL_UNIT_CONTRACTION_OUTPUT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="koszul.module.unit_contraction_budget",
            message="unit contraction work or exact output exceeds the admitted envelope",
        )
    if source_entries > sequence_length * module_dimension**2 * (
        1 << max(sequence_length - 1, 0)
    ):
        raise OperationDomainValidationError(
            location=("complex", "differentials"),
            code="koszul.module.source_complex_mismatch",
            message="source differential sparsity cannot arise from its retained sequence",
        )
    return _admit_complex(complex_value, admit_homology=False)


def _algebra_inverse(
    algebra: FiniteCommutativeAlgebra,
    element: tuple[CanonicalRational, ...],
) -> list[Fraction] | None:
    table = _structure(algebra)
    dimension = len(table)
    # Replace the multiplication-by-basis columns with multiplication by the
    # requested element: L_element * inverse = 1.
    matrix = [
        [
            sum(
                _f(element[basis]) * table[basis][column][row]
                for basis in range(dimension)
            )
            for column in range(dimension)
        ]
        + [_f(algebra.unit[row])]
        for row in range(dimension)
    ]
    for pivot_row, column in enumerate(range(dimension)):
        pivot = next(
            (row for row in range(pivot_row, dimension) if matrix[row][column]), None
        )
        if pivot is None:
            return None
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        scale = matrix[pivot_row][column]
        matrix[pivot_row] = [entry / scale for entry in matrix[pivot_row]]
        for row in range(dimension):
            if row != pivot_row and matrix[row][column]:
                scale = matrix[row][column]
                matrix[row] = [
                    a - scale * b
                    for a, b in zip(matrix[row], matrix[pivot_row], strict=True)
                ]
    inverse = [matrix[row][-1] for row in range(dimension)]
    product = [
        sum(
            _f(element[left]) * inverse[right] * table[left][right][target]
            for left in range(dimension)
            for right in range(dimension)
        )
        for target in range(dimension)
    ]
    return inverse if product == [_f(item) for item in algebra.unit] else None


def module_koszul_unit_contract(
    request: ModuleKoszulUnitContractionRequest | Mapping[str, Any],
) -> ModuleKoszulUnitContraction:
    """Return the signed exterior contraction induced by an invertible entry."""
    try:
        payload = (
            request.model_dump()
            if isinstance(request, ModuleKoszulUnitContractionRequest)
            else request
        )
        value = ModuleKoszulUnitContractionRequest.model_validate(payload)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="koszul.module.unit_contraction_request_shape",
            message="the unit contraction request is not canonical",
        ) from exc
    source = _admit_unit_contraction(value)
    canonical = _build_module_koszul_complex(
        ModuleKoszulRequest(
            algebra=source.algebra, module=source.module, sequence=source.sequence
        )
    )
    if not source.square_zero or canonical.differentials != source.differentials:
        raise OperationDomainValidationError(
            location=("complex", "differentials"),
            code="koszul.module.source_complex_mismatch",
            message="source differentials must be induced by the retained sequence and action",
        )
    algebra = source.algebra
    if algebra.unit is None:
        raise OperationDomainValidationError(
            location=("algebra", "unit"),
            code="koszul.module.algebra_not_unital",
            message="unit contraction requires a retained algebra unit",
        )
    element = source.sequence[value.unit_index]
    inverse = _algebra_inverse(algebra, element)
    if inverse is None:
        raise OperationDomainValidationError(
            location=("sequence", value.unit_index),
            code="koszul.module.sequence_entry_not_unit",
            message="the selected sequence entry is not a unit of the algebra",
        )
    action = _action_matrix(
        source.module, tuple(CanonicalRational.from_fraction(v) for v in inverse)
    )
    length = len(source.sequence)
    wedge_bases = tuple(
        tuple(combinations(range(length), degree)) for degree in range(length + 1)
    )
    homotopy: list[ModuleDifferential] = []
    module_dimension = len(source.module.basis)
    for degree, wedges in enumerate(wedge_bases):
        entries: list[tuple[int, int, CanonicalRational]] = []
        if degree < length:
            next_wedges = wedge_bases[degree + 1]
            for wedge_index, wedge in enumerate(wedges):
                if value.unit_index in wedge:
                    continue
                insertion = sum(index < value.unit_index for index in wedge)
                target_wedge = tuple(sorted((*wedge, value.unit_index)))
                target_wedge_index = next_wedges.index(target_wedge)
                sign = -1 if insertion % 2 else 1
                for target in range(module_dimension):
                    for origin in range(module_dimension):
                        coefficient = sign * action[target][origin]
                        if coefficient:
                            entries.append(
                                (
                                    target_wedge_index * module_dimension + target,
                                    wedge_index * module_dimension + origin,
                                    CanonicalRational.from_fraction(coefficient),
                                )
                            )
        homotopy.append(
            ModuleDifferential(
                row_count=source.basis_sizes[degree + 1] if degree < length else 0,
                column_count=source.basis_sizes[degree],
                entries=tuple(sorted(entries, key=lambda entry: (entry[0], entry[1]))),
            )
        )
    maps = tuple(homotopy)
    for degree, basis_size in enumerate(source.basis_sizes):
        out_d = _sparse_columns(source.differentials[degree]) if degree < length else ()
        h_in = _sparse_columns(maps[degree - 1]) if degree > 0 else ()
        h_out = _sparse_columns(maps[degree])
        in_d = _sparse_columns(source.differentials[degree - 1]) if degree > 0 else ()
        for column in range(basis_size):
            basis = {column: Fraction(1)}
            dh = (
                _apply_sparse_columns(out_d, _apply_sparse_columns(h_out, basis))
                if degree < length
                else {}
            )
            hd = (
                _apply_sparse_columns(h_in, _apply_sparse_columns(in_d, basis))
                if degree > 0
                else {}
            )
            if {
                i: a + hd.get(i, Fraction(0))
                for i, a in dh.items()
                if a + hd.get(i, Fraction(0))
            } | {i: b for i, b in hd.items() if i not in dh and b} != basis:
                raise OperationDomainValidationError(
                    location=("homotopy", degree),
                    code="koszul.module.unit_contraction_identity",
                    message="the returned maps do not satisfy dH + Hd = identity",
                )
    return ModuleKoszulUnitContraction.model_construct(
        complex=source,
        unit_index=value.unit_index,
        inverse=tuple(CanonicalRational.from_fraction(v) for v in inverse),
        homotopy=maps,
    )


def module_koszul_append_zero(  # noqa: C901
    request: ModuleKoszulZeroExtensionRequest | Mapping[str, Any],
) -> ModuleKoszulZeroExtension:
    """Split K(f_1,...,f_r,0) as K(f) plus its degree shift."""
    try:
        payload = (
            request.model_dump()
            if isinstance(request, ModuleKoszulZeroExtensionRequest)
            else request
        )
        value = ModuleKoszulZeroExtensionRequest.model_validate(payload)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="koszul.module.zero_extension_request_shape",
            message="the zero-extension request is not canonical",
        ) from exc

    source = value.complex
    algebra_dimension = len(source.algebra.basis)
    module_dimension = len(source.module.basis)
    source_length = len(source.sequence)
    if source_length >= 6:
        raise OperationResourceAdmissionError(
            location=("complex", "sequence"),
            code="koszul.module.zero_extension_budget",
            message="zero extension is admitted only when the extended sequence has length at most 6",
        )
    source_entries = sum(len(d.entries) for d in source.differentials)
    source_total = sum(source.basis_sizes)
    target_total = module_dimension * (1 << (source_length + 1))
    target_entry_bound = (
        (source_length + 1) * module_dimension**2 * (1 << source_length)
    )
    map_entry_bound = 4 * source_total
    digit_bound = max(
        (
            canonical_rational_component_digits(item)
            for item in _homology_rationals(source)
        ),
        default=1,
    )
    differential_digit_bound = (
        2 * algebra_dimension * max(1, digit_bound) + len(str(algebra_dimension)) + 4
    )
    if differential_digit_bound > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="koszul.module.zero_extension_coefficient_budget",
            message="rebuilt differential coefficients exceed the exact digit limit",
        )
    semantic_work = (
        3 * algebra_dimension**5
        + algebra_dimension**4
        + algebra_dimension**3
        + algebra_dimension**2 * module_dimension**3
        + algebra_dimension**3 * module_dimension**2
    ) * max(1, differential_digit_bound**2)
    build_work = (
        (source_length + 1)
        * (1 << source_length)
        * algebra_dimension
        * module_dimension**2
        + target_entry_bound
        + 2
        * sum(
            module_dimension**3
            * comb(source_length + 1, degree)
            * comb(source_length + 1, degree - 1)
            for degree in range(2, source_length + 2)
        )
    ) * max(1, differential_digit_bound**2)
    source_rational_items = (
        2 * (algebra_dimension**3 + algebra_dimension)
        + algebra_dimension * module_dimension**2
        + source_length * algebra_dimension
        + source_entries
    )
    target_rational_items = (
        2 * (algebra_dimension**3 + algebra_dimension)
        + algebra_dimension * module_dimension**2
        + (source_length + 1) * algebra_dimension
        + target_entry_bound
    )
    labels = (
        *source.algebra.basis,
        *source.module.algebra.basis,
        *source.module.basis,
    )
    label_bytes = 2 * sum(6 * len(label) + 4 for label in labels)
    output_bound = (
        (source_rational_items + target_rational_items + map_entry_bound)
        * (2 * differential_digit_bound + 48)
        + 48 * (source_entries + target_entry_bound + map_entry_bound)
        + label_bytes
        + 8_192
    )
    work_bound = semantic_work + build_work + 8 * map_entry_bound + 4 * target_total
    if (
        work_bound > MAX_KOSZUL_ZERO_EXTENSION_WORK
        or output_bound > MAX_KOSZUL_ZERO_EXTENSION_OUTPUT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="koszul.module.zero_extension_budget",
            message="zero extension complex, splitting maps, or exact work exceed the admitted envelope",
        )

    admitted = _admit_complex(source, admit_homology=False)
    canonical_source = _build_module_koszul_complex(
        ModuleKoszulRequest(
            algebra=admitted.algebra,
            module=admitted.module,
            sequence=admitted.sequence,
        )
    )
    if (
        not admitted.square_zero
        or canonical_source.differentials != admitted.differentials
    ):
        raise OperationDomainValidationError(
            location=("complex", "differentials"),
            code="koszul.module.source_complex_mismatch",
            message="source differentials must be induced by the retained sequence and action",
        )
    zero = tuple(
        CanonicalRational.from_fraction(Fraction(0)) for _ in range(algebra_dimension)
    )
    target_sequence = (*admitted.sequence, zero)
    target = _build_module_koszul_complex(
        ModuleKoszulRequest(
            algebra=admitted.algebra,
            module=admitted.module,
            sequence=target_sequence,
        )
    )
    source_wedges = tuple(
        tuple(combinations(range(source_length), degree))
        for degree in range(source_length + 1)
    )
    target_wedges = tuple(
        tuple(combinations(range(source_length + 1), degree))
        for degree in range(source_length + 2)
    )
    unshifted_inclusions: list[ModuleDifferential] = []
    unshifted_projections: list[ModuleDifferential] = []
    shifted_inclusions: list[ModuleDifferential] = []
    for degree in range(source_length + 1):
        unshifted_in: list[tuple[int, int, CanonicalRational]] = []
        unshifted_out: list[tuple[int, int, CanonicalRational]] = []
        shifted: list[tuple[int, int, CanonicalRational]] = []
        for wedge_index, wedge in enumerate(source_wedges[degree]):
            target_wedge_index = target_wedges[degree].index(wedge)
            shifted_wedge_index = target_wedges[degree + 1].index(
                (*wedge, source_length)
            )
            for module_index in range(module_dimension):
                source_index = wedge_index * module_dimension + module_index
                target_index = target_wedge_index * module_dimension + module_index
                shifted_target_index = (
                    shifted_wedge_index * module_dimension + module_index
                )
                unshifted_in.append(
                    (
                        target_index,
                        source_index,
                        CanonicalRational.from_fraction(Fraction(1)),
                    )
                )
                unshifted_out.append(
                    (
                        source_index,
                        target_index,
                        CanonicalRational.from_fraction(Fraction(1)),
                    )
                )
                shifted_sign = -1 if degree % 2 else 1
                shifted.append(
                    (
                        shifted_target_index,
                        source_index,
                        CanonicalRational.from_fraction(Fraction(shifted_sign)),
                    )
                )
        unshifted_inclusions.append(
            ModuleDifferential(
                row_count=target.basis_sizes[degree],
                column_count=admitted.basis_sizes[degree],
                entries=tuple(unshifted_in),
            )
        )
        unshifted_projections.append(
            ModuleDifferential(
                row_count=admitted.basis_sizes[degree],
                column_count=target.basis_sizes[degree],
                entries=tuple(unshifted_out),
            )
        )
        shifted_inclusions.append(
            ModuleDifferential(
                row_count=target.basis_sizes[degree + 1],
                column_count=admitted.basis_sizes[degree],
                entries=tuple(shifted),
            )
        )
    for degree in range(source_length + 1, source_length + 2):
        unshifted_projections.append(
            ModuleDifferential(
                row_count=0,
                column_count=target.basis_sizes[degree],
                entries=(),
            )
        )
    shifted_projections: list[ModuleDifferential] = []
    for target_degree in range(source_length + 2):
        entries = []
        if target_degree:
            source_degree = target_degree - 1
            shifted_sign = -1 if source_degree % 2 else 1
            for wedge_index, wedge in enumerate(source_wedges[source_degree]):
                target_wedge_index = target_wedges[target_degree].index(
                    (*wedge, source_length)
                )
                for module_index in range(module_dimension):
                    entries.append(
                        (
                            wedge_index * module_dimension + module_index,
                            target_wedge_index * module_dimension + module_index,
                            CanonicalRational.from_fraction(Fraction(shifted_sign)),
                        )
                    )
        shifted_projections.append(
            ModuleDifferential(
                row_count=admitted.basis_sizes[target_degree - 1]
                if target_degree
                else 0,
                column_count=target.basis_sizes[target_degree],
                entries=tuple(entries),
            )
        )

    def require_equal_maps(
        left: ModuleDifferential, right: ModuleDifferential, location: tuple[str, ...]
    ) -> None:
        left_columns = _sparse_columns(left)
        right_columns = _sparse_columns(right)
        if len(left_columns) != len(right_columns):
            raise OperationDomainValidationError(
                location=location,
                code="koszul.module.zero_extension_identity",
                message="zero-extension maps do not have matching source axes",
            )
        for column in range(len(left_columns)):
            if left_columns[column] != right_columns[column]:
                raise OperationDomainValidationError(
                    location=location,
                    code="koszul.module.zero_extension_identity",
                    message="zero-extension maps failed an exact chain or splitting identity",
                )

    # Verify both summand chain maps, inverse projections, and the degreewise
    # direct-sum identity. The shifted summand carries differential -d.
    for degree in range(1, source_length + 2):
        target_d = target.differentials[degree - 1]
        if degree <= source_length:
            source_d = admitted.differentials[degree - 1]
            require_equal_maps(
                ModuleDifferential(
                    row_count=target_d.row_count,
                    column_count=source_d.column_count,
                    entries=tuple(
                        (row, column, CanonicalRational.from_fraction(value))
                        for row, column, value in _compose_sparse_maps(
                            _sparse_columns(target_d),
                            _sparse_columns(unshifted_inclusions[degree]),
                        )
                    ),
                ),
                ModuleDifferential(
                    row_count=target_d.row_count,
                    column_count=source_d.column_count,
                    entries=tuple(
                        (row, column, CanonicalRational.from_fraction(value))
                        for row, column, value in _compose_sparse_maps(
                            _sparse_columns(unshifted_inclusions[degree - 1]),
                            _sparse_columns(source_d),
                        )
                    ),
                ),
                ("unshifted_inclusions", str(degree)),
            )
        source_row_count = admitted.basis_sizes[degree - 1]
        projection_rhs = (
            _compose_sparse_maps(
                _sparse_columns(admitted.differentials[degree - 1]),
                _sparse_columns(unshifted_projections[degree]),
            )
            if degree <= source_length
            else ()
        )
        require_equal_maps(
            ModuleDifferential(
                row_count=source_row_count,
                column_count=target_d.column_count,
                entries=tuple(
                    (row, column, CanonicalRational.from_fraction(value))
                    for row, column, value in _compose_sparse_maps(
                        _sparse_columns(unshifted_projections[degree - 1]),
                        _sparse_columns(target_d),
                    )
                ),
            ),
            ModuleDifferential(
                row_count=source_row_count,
                column_count=target_d.column_count,
                entries=(
                    tuple(
                        (row, column, CanonicalRational.from_fraction(value))
                        for row, column, value in projection_rhs
                    )
                    if degree <= source_length
                    else ()
                ),
            ),
            ("unshifted_projections", str(degree)),
        )

    for degree in range(1, source_length + 1):
        source_d = _sparse_columns(admitted.differentials[degree - 1])
        target_d = _sparse_columns(target.differentials[degree])
        shifted_upper = _sparse_columns(shifted_inclusions[degree])
        shifted_lower = _sparse_columns(shifted_inclusions[degree - 1])
        for column in range(admitted.basis_sizes[degree]):
            basis = {column: Fraction(1)}
            lhs = _apply_sparse_columns(
                target_d, _apply_sparse_columns(shifted_upper, basis)
            )
            rhs = {
                index: -coefficient
                for index, coefficient in _apply_sparse_columns(
                    shifted_lower, _apply_sparse_columns(source_d, basis)
                ).items()
            }
            if lhs != rhs:
                raise OperationDomainValidationError(
                    location=("shifted_inclusions", str(degree)),
                    code="koszul.module.zero_extension_chain_map",
                    message="shifted inclusion does not commute with the shifted differential",
                )

    for target_degree in range(1, source_length + 2):
        target_d = _sparse_columns(target.differentials[target_degree - 1])
        lhs = _compose_sparse_maps(
            _sparse_columns(shifted_projections[target_degree - 1]), target_d
        )
        if target_degree > 1:
            source_d = _sparse_columns(admitted.differentials[target_degree - 2])
            rhs = tuple(
                (row, column, -coefficient)
                for row, column, coefficient in _compose_sparse_maps(
                    source_d, _sparse_columns(shifted_projections[target_degree])
                )
            )
        else:
            rhs = ()
        if lhs != rhs:
            raise OperationDomainValidationError(
                location=("shifted_projections", str(target_degree)),
                code="koszul.module.zero_extension_chain_map",
                message="shifted projection does not commute with the shifted differential",
            )

    for target_degree, target_size in enumerate(target.basis_sizes):
        for column in range(target_size):
            basis = {column: Fraction(1)}
            image: dict[int, Fraction] = {}
            if target_degree <= source_length:
                image.update(
                    _apply_sparse_columns(
                        _sparse_columns(unshifted_inclusions[target_degree]),
                        _apply_sparse_columns(
                            _sparse_columns(unshifted_projections[target_degree]), basis
                        ),
                    )
                )
            if target_degree > 0:
                shifted = _apply_sparse_columns(
                    _sparse_columns(shifted_inclusions[target_degree - 1]),
                    _apply_sparse_columns(
                        _sparse_columns(shifted_projections[target_degree]), basis
                    ),
                )
                for row, coefficient in shifted.items():
                    image[row] = image.get(row, Fraction(0)) + coefficient
            image = {
                row: coefficient for row, coefficient in image.items() if coefficient
            }
            if image != basis:
                raise OperationDomainValidationError(
                    location=("splitting", str(target_degree)),
                    code="koszul.module.zero_extension_splitting",
                    message="the two summands do not split the target basis exactly",
                )
    return ModuleKoszulZeroExtension.model_construct(
        source_complex=admitted,
        target_complex=target,
        unshifted_inclusions=tuple(unshifted_inclusions),
        unshifted_projections=tuple(unshifted_projections),
        shifted_inclusions=tuple(shifted_inclusions),
        shifted_projections=tuple(shifted_projections),
    )


def _row_space_basis(rows: list[list[Fraction]]) -> list[list[Fraction]]:
    """Return the unique nonzero RREF rows spanning ``rows``."""
    if not rows:
        return []
    reduced = [row[:] for row in rows]
    width = len(reduced[0])
    pivot_row = 0
    for column in range(width):
        pivot = next(
            (row for row in range(pivot_row, len(reduced)) if reduced[row][column]),
            None,
        )
        if pivot is None:
            continue
        reduced[pivot_row], reduced[pivot] = reduced[pivot], reduced[pivot_row]
        scale = reduced[pivot_row][column]
        reduced[pivot_row] = [value / scale for value in reduced[pivot_row]]
        for row in range(len(reduced)):
            if row != pivot_row and reduced[row][column]:
                factor = reduced[row][column]
                reduced[row] = [
                    left - factor * right
                    for left, right in zip(
                        reduced[row], reduced[pivot_row], strict=True
                    )
                ]
        pivot_row += 1
        if pivot_row == len(reduced):
            break
    return [row for row in reduced if any(row)]


def _inverse(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
    """Invert a nonsingular square matrix by exact Gauss-Jordan elimination."""
    size = len(matrix)
    augmented = [
        row[:] + [Fraction(int(row_index == column)) for column in range(size)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if augmented[row][column]), None
        )
        if pivot is None:
            raise OperationDomainValidationError(
                location=("sequence",),
                code="koszul.module.quotient_basis",
                message="quotient complement failed to form a basis",
            )
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row != column and augmented[row][column]:
                factor = augmented[row][column]
                augmented[row] = [
                    left - factor * right
                    for left, right in zip(
                        augmented[row], augmented[column], strict=True
                    )
                ]
    return [row[size:] for row in augmented]


def module_koszul_quotient(
    request: ModuleKoszulRequest | Mapping[str, Any],
) -> ModuleQuotientValue:
    """Construct ``M/(f_1,...,f_r)M`` as a based algebra module.

    The quotient basis is chosen deterministically by extending the canonical
    RREF basis of the relation submodule with standard source basis vectors.
    The returned projection and representatives make the quotient map explicit.
    """
    value = _as_request(request)
    source_dimension = len(value.module.basis)
    algebra_dimension = len(value.algebra.basis)
    # Bound exact arithmetic before checking algebra/module identities or
    # expanding the relation space. A sequence-action entry is a sum of at
    # most n products, hence has at most 2*n*d+O(log n) component digits.
    # Clearing row denominators and bounding determinant minors for an m by m
    # rational basis change gives the subsequent conservative quadratic m
    # factor. Algebra/module identity checks use at most n^2 triple products.
    inputs = (
        [
            coefficient
            for first in value.algebra.multiplication
            for row in first
            for coefficient in row
        ]
        + [
            coefficient
            for matrix in value.module.action
            for row in matrix
            for coefficient in row
        ]
        + [coefficient for element in value.sequence for coefficient in element]
    )
    max_digits = max(
        (canonical_rational_component_digits(coefficient) for coefficient in inputs),
        default=1,
    )
    relation_entry_digits = 2 * algebra_dimension * max_digits + 8
    elimination_bound = (
        4 * source_dimension * source_dimension * relation_entry_digits
        + 2 * max_digits
        + 32
    )
    validation_bound = 6 * algebra_dimension * algebra_dimension * max_digits + 32
    growth_bound = max(elimination_bound, validation_bound)
    if growth_bound > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationResourceAdmissionError(
            location=("module", "action"),
            code="koszul.module.quotient_coefficient_budget",
            message="exact quotient elimination exceeds the admitted coefficient growth",
        )
    _admit(value.module, value.sequence)

    relation_generators: list[list[Fraction]] = []
    for element in value.sequence:
        action = _action_matrix(value.module, element)
        relation_generators.extend(
            [action[row][column] for row in range(source_dimension)]
            for column in range(source_dimension)
        )
    relation_basis = _row_space_basis(relation_generators)

    # A stable quotient basis is obtained by adding standard vectors whenever
    # they increase the span; coordinates are then recovered from one square
    # basis matrix. This choice is independent of generator order.
    basis_columns = [row[:] for row in relation_basis]
    quotient_representatives: list[list[Fraction]] = []
    for coordinate in range(source_dimension):
        standard = [Fraction(int(i == coordinate)) for i in range(source_dimension)]
        if _matrix_rank([*basis_columns, standard]) > len(basis_columns):
            basis_columns.append(standard)
            quotient_representatives.append(standard)
    full_basis = [
        [basis_columns[column][row] for column in range(source_dimension)]
        for row in range(source_dimension)
    ]
    inverse_basis = _inverse(full_basis)
    quotient_dimension = len(quotient_representatives)

    def quotient_coordinates(vector: list[Fraction]) -> list[Fraction]:
        coordinates = [
            sum(
                inverse_basis[row][column] * vector[column]
                for column in range(source_dimension)
            )
            for row in range(source_dimension)
        ]
        return coordinates[len(relation_basis) :]

    projection = [
        quotient_coordinates(
            [Fraction(int(row == column)) for row in range(source_dimension)]
        )
        for column in range(source_dimension)
    ]
    # projection above is source-major; the public matrix is target by source.
    projection = [
        [projection[source][target] for source in range(source_dimension)]
        for target in range(quotient_dimension)
    ]
    quotient_actions: list[tuple[tuple[CanonicalRational, ...], ...]] = []
    for action in value.module.action:
        source_action = [[_f(entry) for entry in row] for row in action]
        induced = [
            [Fraction(0) for _ in range(quotient_dimension)]
            for _ in range(quotient_dimension)
        ]
        for source_coordinate, representative in enumerate(quotient_representatives):
            image = [
                sum(
                    source_action[row][column] * representative[column]
                    for column in range(source_dimension)
                )
                for row in range(source_dimension)
            ]
            quotient_image = quotient_coordinates(image)
            for target_coordinate, coefficient in enumerate(quotient_image):
                induced[target_coordinate][source_coordinate] = coefficient
        quotient_actions.append(
            tuple(
                tuple(CanonicalRational.from_fraction(entry) for entry in row)
                for row in induced
            )
        )

    quotient = BasedFiniteModule(
        algebra=value.algebra,
        basis=tuple(f"q{index}" for index in range(quotient_dimension)),
        action=tuple(quotient_actions),
    )
    return ModuleQuotientValue(
        algebra=value.algebra,
        source_module=value.module,
        sequence=value.sequence,
        relation_basis=tuple(
            tuple(CanonicalRational.from_fraction(entry) for entry in row)
            for row in relation_basis
        ),
        quotient_module=quotient,
        quotient_basis_representatives=tuple(
            tuple(CanonicalRational.from_fraction(entry) for entry in vector)
            for vector in quotient_representatives
        ),
        projection=tuple(
            tuple(CanonicalRational.from_fraction(entry) for entry in row)
            for row in projection
        ),
    )


def module_koszul_direct_sum(
    request: ModuleKoszulDirectSumRequest | Mapping[str, Any],
) -> ModuleKoszulDirectSumValue:
    """Return the direct-sum Koszul complex with its canonical inclusions.

    The aggregate wedge work for both summands and their sum is bounded before
    constructing any complex. The degreewise inclusions are checked as chain
    maps; their images partition each target basis, so together they give the
    canonical chain isomorphism from the summand direct sum.
    """
    try:
        payload = (
            request.model_dump()
            if isinstance(request, ModuleKoszulDirectSumRequest)
            else request
        )
        value = ModuleKoszulDirectSumRequest.model_validate(payload)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="koszul.module.direct_sum_request",
            message="the module direct-sum request is not canonical",
        ) from exc

    left_dimension = len(value.left.basis)
    right_dimension = len(value.right.basis)
    total_dimension = left_dimension + right_dimension
    length = len(value.sequence)
    # Three complexes are returned. Bound their aggregate basis coordinates
    # and differential contributions before any wedge or matrix expansion.
    aggregate_basis = (left_dimension + right_dimension + total_dimension) * (
        1 << length
    )
    aggregate_entries = (
        length
        * (1 << max(0, length - 1))
        * (
            left_dimension * left_dimension
            + right_dimension * right_dimension
            + total_dimension * total_dimension
        )
    )
    if aggregate_basis > 768 or aggregate_entries > 100_000:
        raise OperationResourceAdmissionError(
            location=("sequence",),
            code="koszul.module.direct_sum_budget",
            message="the aggregate direct-sum complex and chain maps exceed the admitted envelope",
        )

    _admit(value.left, value.sequence)
    _admit(value.right, value.sequence)
    direct_basis = tuple(f"L:{label}" for label in value.left.basis) + tuple(
        f"R:{label}" for label in value.right.basis
    )
    left_actions = [
        [[_f(item) for item in row] for row in matrix] for matrix in value.left.action
    ]
    right_actions = [
        [[_f(item) for item in row] for row in matrix] for matrix in value.right.action
    ]
    actions = []
    for left, right in zip(left_actions, right_actions, strict=True):
        matrix = [
            [Fraction(0) for _ in range(total_dimension)]
            for _ in range(total_dimension)
        ]
        for row in range(left_dimension):
            matrix[row][:left_dimension] = left[row]
        for row in range(right_dimension):
            matrix[left_dimension + row][left_dimension:] = right[row]
        actions.append(
            tuple(
                tuple(CanonicalRational.from_fraction(item) for item in row)
                for row in matrix
            )
        )
    direct_module = BasedFiniteModule(
        algebra=value.algebra, basis=direct_basis, action=tuple(actions)
    )
    left_request = ModuleKoszulRequest(
        algebra=value.algebra, module=value.left, sequence=value.sequence
    )
    right_request = ModuleKoszulRequest(
        algebra=value.algebra, module=value.right, sequence=value.sequence
    )
    sum_request = ModuleKoszulRequest(
        algebra=value.algebra, module=direct_module, sequence=value.sequence
    )
    left_complex = module_koszul_complex(left_request)
    right_complex = module_koszul_complex(right_request)
    sum_complex = module_koszul_complex(sum_request)

    length = len(value.sequence)
    left_maps = []
    right_maps = []
    for degree in range(length + 1):
        wedges = tuple(combinations(range(length), degree))
        wedge_count = len(wedges)
        target_size = total_dimension * wedge_count
        left_map = [
            [Fraction(0) for _ in range(left_dimension * wedge_count)]
            for _ in range(target_size)
        ]
        right_map = [
            [Fraction(0) for _ in range(right_dimension * wedge_count)]
            for _ in range(target_size)
        ]
        for wedge_index in range(wedge_count):
            for basis_index in range(left_dimension):
                left_map[wedge_index * total_dimension + basis_index][
                    wedge_index * left_dimension + basis_index
                ] = Fraction(1)
            for basis_index in range(right_dimension):
                right_map[wedge_index * total_dimension + left_dimension + basis_index][
                    wedge_index * right_dimension + basis_index
                ] = Fraction(1)
        left_maps.append(
            tuple(
                tuple(CanonicalRational.from_fraction(item) for item in row)
                for row in left_map
            )
        )
        right_maps.append(
            tuple(
                tuple(CanonicalRational.from_fraction(item) for item in row)
                for row in right_map
            )
        )

    # Verify the two inclusions commute with each adjacent differential.
    for degree in range(1, length + 1):
        target_d = _dense(sum_complex.differentials[degree - 1])
        for source_complex, inclusions in (
            (left_complex, left_maps),
            (right_complex, right_maps),
        ):
            source_d = _dense(source_complex.differentials[degree - 1])
            higher = [[_f(item) for item in row] for row in inclusions[degree]]
            lower = [[_f(item) for item in row] for row in inclusions[degree - 1]]
            for row in range(len(target_d)):
                for column in range(len(higher[0]) if higher else 0):
                    lhs = sum(
                        target_d[row][middle] * higher[middle][column]
                        for middle in range(len(higher))
                    )
                    rhs = sum(
                        lower[row][middle] * source_d[middle][column]
                        for middle in range(len(source_d))
                    )
                    if lhs != rhs:
                        raise OperationDomainValidationError(
                            location=("sequence",),
                            code="koszul.module.direct_sum_chain_map",
                            message="canonical summand inclusion failed the chain-map identity",
                        )

    return ModuleKoszulDirectSumValue(
        algebra=value.algebra,
        left=value.left,
        right=value.right,
        sequence=value.sequence,
        direct_sum_module=direct_module,
        left_complex=left_complex,
        right_complex=right_complex,
        direct_sum_complex=sum_complex,
        left_inclusions=tuple(left_maps),
        right_inclusions=tuple(right_maps),
    )


def module_koszul_differential(
    request: ModuleKoszulDifferentialRequest | Mapping[str, Any],
) -> ModuleKoszulDifferentialValue:
    """Compute one exact degree of a finite-module Koszul differential.

    Only the requested differential and its predecessor are materialized. The
    latter is used to replay the adjacent square-zero identity.
    """
    try:
        payload = (
            request.model_dump()
            if isinstance(request, ModuleKoszulDifferentialRequest)
            else request
        )
        value = ModuleKoszulDifferentialRequest.model_validate(payload)
        source = _as_request(value.request)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="koszul.module.differential_request",
            message="the degree-specific module Koszul request is not canonical",
        ) from exc
    _admit(source.module, source.sequence)
    degree = value.degree
    length = len(source.sequence)
    module_dimension = len(source.module.basis)

    def differential_at(
        k: int,
    ) -> tuple[
        tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], ModuleDifferential
    ]:
        source_wedges = tuple(combinations(range(length), k))
        target_wedges = tuple(combinations(range(length), k - 1))
        target_index = {wedge: index for index, wedge in enumerate(target_wedges)}
        entries: list[tuple[int, int, CanonicalRational]] = []
        actions = {
            index: _action_matrix(source.module, element)
            for index, element in enumerate(source.sequence)
        }
        for wedge_column, wedge in enumerate(source_wedges):
            for position, sequence_index in enumerate(wedge):
                sign = -1 if position % 2 else 1
                target = wedge[:position] + wedge[position + 1 :]
                for row in range(module_dimension):
                    for column in range(module_dimension):
                        coefficient = sign * actions[sequence_index][row][column]
                        if coefficient:
                            entries.append(
                                (
                                    target_index[target] * module_dimension + row,
                                    wedge_column * module_dimension + column,
                                    CanonicalRational.from_fraction(coefficient),
                                )
                            )
        return (
            source_wedges,
            target_wedges,
            ModuleDifferential(
                row_count=len(target_wedges) * module_dimension,
                column_count=len(source_wedges) * module_dimension,
                entries=tuple(sorted(entries, key=lambda entry: (entry[0], entry[1]))),
            ),
        )

    source_wedges, target_wedges, differential = differential_at(degree)
    if degree > 1:
        _, _, predecessor = differential_at(degree - 1)
        outer, inner = _dense(predecessor), _dense(differential)
        for row in range(len(outer)):
            for column in range(len(inner[0]) if inner else 0):
                if sum(
                    outer[row][middle] * inner[middle][column]
                    for middle in range(len(inner))
                ):
                    raise OperationDomainValidationError(
                        location=("request", "degree"),
                        code="koszul.module.differential_square",
                        message="the selected differential fails the exact adjacent d^2=0 check",
                    )
    return ModuleKoszulDifferentialValue(
        algebra=source.algebra,
        module=source.module,
        sequence=source.sequence,
        degree=degree,
        source_wedges=source_wedges,
        target_wedges=target_wedges,
        differential=differential,
    )


def module_koszul_map(
    request: ModuleKoszulMapRequest | Mapping[str, Any],
) -> ModuleKoszulChainMap:
    """Induce a degreewise chain map from one exact module homomorphism.

    Both module actions and every chain-map square are checked in the
    operation. Returned values retain the source map and both complexes.
    """
    try:
        payload = (
            request.model_dump()
            if isinstance(request, ModuleKoszulMapRequest)
            else request
        )
        value = ModuleKoszulMapRequest.model_validate(payload)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="koszul.module.map_request",
            message="the module map request is not canonical",
        ) from exc

    source_dimension = len(value.source.basis)
    target_dimension = len(value.target.basis)
    algebra_dimension = len(value.algebra.basis)
    length = len(value.sequence)
    wedge_total = 1 << length
    map_cells = source_dimension * target_dimension * wedge_total
    differential_terms = (
        length * (1 << max(0, length - 1)) * (source_dimension**2 + target_dimension**2)
    )
    action_check_work = (
        algebra_dimension
        * source_dimension
        * target_dimension
        * (source_dimension + target_dimension)
    )
    source_validation_work = (
        2 * algebra_dimension**4
        + algebra_dimension**2 * (source_dimension**3 + target_dimension**3)
        + 3 * algebra_dimension**3 * (source_dimension**2 + target_dimension**2)
    )
    estimated_work = (
        map_cells
        + differential_terms * (algebra_dimension + source_dimension + target_dimension)
        + action_check_work
        + source_validation_work
    )
    input_bytes = len(value.model_dump_json().encode("utf-8"))
    input_coefficients = (
        tuple(_algebra_rationals(value.algebra))
        + tuple(
            item
            for row in value.source.action
            for matrix_row in row
            for item in matrix_row
        )
        + tuple(
            item
            for row in value.target.action
            for matrix_row in row
            for item in matrix_row
        )
        + tuple(item for element in value.sequence for item in element)
        + tuple(item for row in value.map_matrix for item in row)
    )
    max_input_digits = max(
        (canonical_rational_component_digits(item) for item in input_coefficients),
        default=1,
    )
    action_digits = (
        2 * algebra_dimension * max_input_digits + algebra_dimension.bit_length() + 2
    )
    map_digits = max(
        (
            canonical_rational_component_digits(item)
            for row in value.map_matrix
            for item in row
        ),
        default=1,
    )
    linearity_intermediate_digits = (
        2 * max(source_dimension, target_dimension) * (map_digits + max_input_digits)
        + max(source_dimension, target_dimension).bit_length()
        + 2
    )
    chain_intermediate_digits = (
        2 * max(source_dimension, target_dimension) * (map_digits + action_digits)
        + max(source_dimension, target_dimension).bit_length()
        + 2
    )
    algebra_intermediate_digits = (
        2 * algebra_dimension * max_input_digits + algebra_dimension.bit_length() + 2
    )
    module_action_intermediate_digits = (
        2
        * algebra_dimension
        * (2 * algebra_dimension * max_input_digits + max_input_digits)
        + 2 * algebra_dimension * algebra_dimension.bit_length()
        + algebra_dimension.bit_length()
        + 2
    )
    estimated_output_bytes = (
        5 * input_bytes
        + map_cells * (2 * map_digits + 64)
        + differential_terms * (2 * action_digits + 64)
        + 4096
    )
    if (
        map_cells > MAX_KOSZUL_MODULE_MAP_CELLS
        or estimated_work > MAX_KOSZUL_MODULE_MAP_WORK
        or estimated_output_bytes > MAX_KOSZUL_MODULE_MAP_OUTPUT_BYTES
        or linearity_intermediate_digits > MAX_CANONICAL_RATIONAL_DIGITS
        or chain_intermediate_digits > MAX_CANONICAL_RATIONAL_DIGITS
        or algebra_intermediate_digits > MAX_CANONICAL_RATIONAL_DIGITS
        or module_action_intermediate_digits > MAX_CANONICAL_RATIONAL_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=("sequence",),
            code="koszul.module.map_budget",
            message=(
                "the module-induced chain map exceeds its admitted work, "
                "intermediate-growth, or output bound"
            ),
        )

    _admit(value.source, value.sequence)
    _admit(value.target, value.sequence)
    phi = [[_f(item) for item in row] for row in value.map_matrix]
    # A module homomorphism must intertwine the action of every algebra basis
    # element. Checking basis actions suffices by linearity.
    for algebra_index in range(algebra_dimension):
        source_action = [
            [_f(item) for item in row] for row in value.source.action[algebra_index]
        ]
        target_action = [
            [_f(item) for item in row] for row in value.target.action[algebra_index]
        ]
        left = [
            [
                sum(
                    phi[row][middle] * source_action[middle][column]
                    for middle in range(source_dimension)
                )
                for column in range(source_dimension)
            ]
            for row in range(target_dimension)
        ]
        right = [
            [
                sum(
                    target_action[row][middle] * phi[middle][column]
                    for middle in range(target_dimension)
                )
                for column in range(source_dimension)
            ]
            for row in range(target_dimension)
        ]
        if left != right:
            raise OperationDomainValidationError(
                location=("map_matrix",),
                code="koszul.module.map_not_linear",
                message="the supplied linear map does not commute with the algebra action",
            )

    source_request = ModuleKoszulRequest(
        algebra=value.algebra, module=value.source, sequence=value.sequence
    )
    target_request = ModuleKoszulRequest(
        algebra=value.algebra, module=value.target, sequence=value.sequence
    )
    # `_admit` proves the algebra is commutative and each action respects its
    # multiplication table. These laws imply d^2=0, so skip the redundant
    # dense square replay for these freshly constructed complexes.
    source_request = _as_request(source_request)
    target_request = _as_request(target_request)
    source_complex = _build_module_koszul_complex(source_request, check_square=False)
    target_complex = _build_module_koszul_complex(target_request, check_square=False)
    degree_maps: list[ModuleChainMapMatrix] = []
    for degree in range(length + 1):
        wedge_count = comb(length, degree)
        rows = target_dimension * wedge_count
        columns = source_dimension * wedge_count
        entries = tuple(
            (
                wedge * target_dimension + target_index,
                wedge * source_dimension + source_index,
                CanonicalRational.from_fraction(phi[target_index][source_index]),
            )
            for wedge in range(wedge_count)
            for target_index in range(target_dimension)
            for source_index in range(source_dimension)
            if phi[target_index][source_index]
        )
        degree_maps.append(
            ModuleChainMapMatrix(row_count=rows, column_count=columns, entries=entries)
        )

    def sparse(
        matrix: ModuleDifferential | ModuleChainMapMatrix,
    ) -> dict[tuple[int, int], Fraction]:
        return {
            (row, column): _f(coefficient)
            for row, column, coefficient in matrix.entries
        }

    def compose(
        left: ModuleDifferential | ModuleChainMapMatrix,
        right: ModuleDifferential | ModuleChainMapMatrix,
    ) -> dict[tuple[int, int], Fraction]:
        right_by_row: dict[int, list[tuple[int, Fraction]]] = {}
        for (row, column), coefficient in sparse(right).items():
            right_by_row.setdefault(row, []).append((column, coefficient))
        result: dict[tuple[int, int], Fraction] = {}
        for (row, middle), coefficient in sparse(left).items():
            for column, right_coefficient in right_by_row.get(middle, ()):
                key = (row, column)
                result[key] = (
                    result.get(key, Fraction(0)) + coefficient * right_coefficient
                )
        return {key: coefficient for key, coefficient in result.items() if coefficient}

    for degree in range(1, length + 1):
        target_then_map = compose(
            target_complex.differentials[degree - 1], degree_maps[degree]
        )
        map_then_source = compose(
            degree_maps[degree - 1], source_complex.differentials[degree - 1]
        )
        if target_then_map != map_then_source:
            raise OperationDomainValidationError(
                location=("map_matrix",),
                code="koszul.module.map_chain_relation",
                message="the induced degree maps do not commute with the Koszul differentials",
            )

    return ModuleKoszulChainMap(
        algebra=value.algebra,
        source=value.source,
        target=value.target,
        sequence=value.sequence,
        module_map=value.map_matrix,
        source_complex=source_complex,
        target_complex=target_complex,
        degree_maps=tuple(degree_maps),
    )
