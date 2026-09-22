"""Exact finite-module Koszul construction and homology."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from itertools import combinations
from math import comb
from typing import Any

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.koszul.module_models import (
    BasedFiniteModule,
    FiniteCommutativeAlgebra,
    ModuleDifferential,
    ModuleKoszulComplex,
    ModuleKoszulHomology,
    ModuleKoszulRequest,
)


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
    sequence_length = len(value.sequence)
    module_dimension = len(value.module.basis)
    bases = tuple(
        tuple(combinations(range(sequence_length), degree))
        for degree in range(sequence_length + 1)
    )
    basis_sizes = tuple(module_dimension * len(basis) for basis in bases)
    differentials: list[ModuleDifferential] = []
    for degree in range(1, sequence_length + 1):
        entries: list[tuple[int, int, CanonicalRational]] = []
        for wedge_column, indices in enumerate(bases[degree]):
            for position, sequence_index in enumerate(indices):
                action = _action_matrix(value.module, value.sequence[sequence_index])
                sign = -1 if position % 2 else 1
                target_wedge = indices[:position] + indices[position + 1 :]
                wedge_row = bases[degree - 1].index(target_wedge)
                for target in range(module_dimension):
                    for source in range(module_dimension):
                        coefficient = sign * action[target][source]
                        if coefficient:
                            entries.append(
                                (
                                    wedge_row * module_dimension + target,
                                    wedge_column * module_dimension + source,
                                    CanonicalRational.from_fraction(coefficient),
                                )
                            )
        differentials.append(
            ModuleDifferential(
                row_count=basis_sizes[degree - 1],
                column_count=basis_sizes[degree],
                entries=tuple(sorted(entries, key=lambda entry: (entry[0], entry[1]))),
            )
        )
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


def _admit_complex(value: ModuleKoszulComplex) -> ModuleKoszulComplex:
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
    if sequence_length > 6 or module_dimension * (2**sequence_length) > 256:
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
    incoming = [None, *value.differentials]
    outgoing = [*value.differentials, None]
    cycles: list[int] = []
    boundaries: list[int] = []
    dimensions: list[int] = []
    for degree, size in enumerate(value.basis_sizes):
        outgoing_differential = outgoing[degree]
        incoming_differential = incoming[degree]
        outgoing_rank = (
            0
            if outgoing_differential is None
            else _matrix_rank(_dense(outgoing_differential))
        )
        incoming_rank = (
            0
            if incoming_differential is None
            else _matrix_rank(_dense(incoming_differential))
        )
        cycle_dimension = size - outgoing_rank
        cycles.append(cycle_dimension)
        boundaries.append(incoming_rank)
        dimensions.append(cycle_dimension - incoming_rank)
    return ModuleKoszulHomology.model_construct(
        complex=value,
        dimensions=tuple(dimensions),
        cycle_dimensions=tuple(cycles),
        boundary_dimensions=tuple(boundaries),
    )
