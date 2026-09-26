"""Bounded composition for admitted filtered chain-map values."""

from __future__ import annotations

from fractions import Fraction

from jacobian._models import StrictModel
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.topology.chain_complexes._filtered_models import (
    MAX_FILTER_AMBIENT_DIMENSION,
)
from jacobian.math.topology.chain_complexes._filtered_operations import (
    _admit_filtered_semantics,
    _fail,
    _parse_entry,
    _serialize_scalar,
)
from jacobian.math.topology.chain_complexes.filtered_extensions import (
    FilteredChainMapRequest,
    FilteredChainMapResult,
    _check_filtered_chain_map_axes,
    _filtered_map_admitted,
    _mul,
)
from jacobian.math.topology.chain_complexes.values import (
    MAX_CHAIN_COMPLEX_COEFFICIENT_DIGITS,
    MAX_CHAIN_MAP_CELLS,
    MAX_CHAIN_MAP_ENTRY_CHARS,
)

MAX_COMPOSITION_WORK = MAX_CHAIN_MAP_CELLS * MAX_FILTER_AMBIENT_DIMENSION


class FilteredChainMapCompositionRequest(StrictModel):
    """Two composable filtered chain maps, in application order."""

    first: FilteredChainMapResult
    second: FilteredChainMapResult


def _digits(value: int | Fraction) -> tuple[int, int]:
    value = value if isinstance(value, Fraction) else Fraction(value)
    numerator, denominator = abs(value.numerator), value.denominator
    if numerator.bit_length() > 13_608 or denominator.bit_length() > 13_608:
        raise OperationResourceAdmissionError(
            location=("maps",),
            code="filtered_chain_map.coefficient_exceeded",
            message="an input coefficient exceeds the exact chain-map digit limit",
        )
    return len(str(numerator)), len(str(denominator))


def _parse_map(request: FilteredChainMapRequest, label: str, prime: int | None):
    cells = chars = 0
    parsed = []
    for degree, matrix in enumerate(request.maps):
        rows, columns = (
            request.target.basis_sizes[degree],
            request.source.basis_sizes[degree],
        )
        if len(matrix) != rows or any(len(row) != columns for row in matrix):
            raise _fail(
                (label, "maps", degree),
                "filtered_chain_map.shape_invalid",
                "map matrix axes are invalid",
            )
        cells += rows * columns
        output_matrix = []
        for row in matrix:
            output_row = []
            for value in row:
                if prime is not None and (
                    type(value) is not int or not 0 <= value < prime
                ):
                    raise _fail(
                        (label, "maps", degree),
                        "filtered_chain_map.entry_invalid",
                        "finite-field map entries must be canonical residues",
                    )
                numerator_digits, denominator_digits = _digits(value)
                if (
                    max(numerator_digits, denominator_digits)
                    > MAX_CHAIN_COMPLEX_COEFFICIENT_DIGITS
                ):
                    raise OperationResourceAdmissionError(
                        location=(label, "maps", degree),
                        code="filtered_chain_map.coefficient_exceeded",
                        message="an input map coefficient exceeds the exact chain-map digit limit",
                    )
                chars += numerator_digits + denominator_digits + 1
                output_row.append(_parse_entry(value, prime))
            output_matrix.append(output_row)
        parsed.append(output_matrix)
    if cells > MAX_CHAIN_MAP_CELLS or chars > MAX_CHAIN_MAP_ENTRY_CHARS:
        raise OperationResourceAdmissionError(
            location=(label, "maps"),
            code="filtered_chain_map.input_envelope_exceeded",
            message="the degreewise map exceeds its admitted cell or coefficient-character envelope",
        )
    return parsed


def _sum_digits(terms: list[tuple[int | Fraction, int | Fraction]]) -> int:
    sizes = []
    for left, right in terms:
        ln, ld = _digits(left)
        rn, rd = _digits(right)
        lu, ru = (
            Fraction(left).numerator in (1, -1) and ld == 1,
            Fraction(right).numerator in (1, -1) and rd == 1,
        )
        sizes.append(
            (
                rn if lu else ln if ru else ln + rn,
                1 if ld == rd == 1 else ld + rd,
                ld == rd == 1,
            )
        )
    denominator = (
        1 if all(item[2] for item in sizes) else sum(item[1] for item in sizes)
    )
    numerator = max(n + denominator - td for n, td, _ in sizes) + (
        len(str(len(terms))) if len(terms) > 1 else 0
    )
    if max(numerator, denominator) > MAX_CHAIN_COMPLEX_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("maps",),
            code="filtered_chain_map.composition_coefficient_exceeded",
            message="a composed coefficient may exceed the exact chain-map digit limit",
        )
    return numerator + denominator + 1


def filtered_chain_map_compose(
    request: FilteredChainMapCompositionRequest,
) -> FilteredChainMapResult:
    """Recheck and compose two exact filtration-preserving chain maps."""
    first_value = FilteredChainMapResult.model_validate(request.first.model_dump())
    second_value = FilteredChainMapResult.model_validate(request.second.model_dump())
    first = FilteredChainMapRequest(
        source=first_value.source,
        source_filtration=first_value.source_filtration,
        target=first_value.target,
        target_filtration=first_value.target_filtration,
        maps=first_value.maps,
    )
    second = FilteredChainMapRequest(
        source=second_value.source,
        source_filtration=second_value.source_filtration,
        target=second_value.target,
        target_filtration=second_value.target_filtration,
        maps=second_value.maps,
    )
    if first.target != second.source:
        raise _fail(
            ("second",),
            "filtered_chain_map.composition_middle_mismatch",
            "the target of the first map must equal the source of the second",
        )
    if (
        first.source.coefficient_ring != second.target.coefficient_ring
        or first.source.prime != second.target.prime
    ):
        raise _fail(
            ("second",),
            "filtered_chain_map.composition_coefficient_mismatch",
            "the composite must retain one coefficient field",
        )
    prime = first.source.prime
    _check_filtered_chain_map_axes(first)
    _check_filtered_chain_map_axes(second)
    for label, request_value in (("first", first), ("second", second)):
        input_cells = sum(
            rows * columns
            for rows, columns in zip(
                request_value.target.basis_sizes,
                request_value.source.basis_sizes,
                strict=True,
            )
        )
        if input_cells > MAX_CHAIN_MAP_CELLS:
            raise OperationResourceAdmissionError(
                location=(label, "maps"),
                code="filtered_chain_map.input_envelope_exceeded",
                message="the input map exceeds its admitted cell envelope",
            )
    left, right = _parse_map(first, "first", prime), _parse_map(second, "second", prime)
    cells = sum(
        a * b
        for a, b in zip(
            second.target.basis_sizes, first.source.basis_sizes, strict=True
        )
    )
    work = sum(
        a * b * c
        for a, b, c in zip(
            second.target.basis_sizes,
            first.source.basis_sizes,
            first.target.basis_sizes,
            strict=True,
        )
    )
    if cells > MAX_CHAIN_MAP_CELLS or work > MAX_COMPOSITION_WORK:
        raise OperationResourceAdmissionError(
            location=("maps",),
            code="filtered_chain_map.composition_work_exceeded",
            message="the composed map exceeds its admitted cell or multiplication-work envelope",
        )
    output_chars = 0
    for lmat, rmat in zip(right, left, strict=True):
        for row in lmat:
            for column in zip(*rmat, strict=False):
                terms = [(a, b) for a, b in zip(row, column, strict=True) if a and b]
                output_chars += (
                    (len(str(prime - 1)) if prime is not None else _sum_digits(terms))
                    if terms
                    else 1
                )
    if output_chars > MAX_CHAIN_MAP_ENTRY_CHARS:
        raise OperationResourceAdmissionError(
            location=("maps",),
            code="filtered_chain_map.composition_output_exceeded",
            message="the composed map exceeds its admitted output coefficient-character limit",
        )
    source_admission = _admit_filtered_semantics(first.source, first.source_filtration)
    middle_first = _admit_filtered_semantics(first.target, first.target_filtration)
    middle_second = (
        middle_first
        if first.target_filtration == second.source_filtration
        else _admit_filtered_semantics(second.source, second.source_filtration)
    )
    if middle_first.bases != middle_second.bases:
        raise _fail(
            ("second", "source_filtration"),
            "filtered_chain_map.composition_middle_filtration_mismatch",
            "the middle filtrations must define the same subspaces",
        )
    target_admission = _admit_filtered_semantics(
        second.target, second.target_filtration
    )
    for source_request, source_space, target_space in (
        (first, source_admission, middle_first),
        (second, middle_second, target_admission),
    ):
        result = _filtered_map_admitted(source_request, source_space, target_space)
        if not result.chain_map or not result.filtration_preserving:
            raise _fail(
                ("maps",),
                "filtered_chain_map.composition_input_invalid",
                "both input maps must be chain maps that preserve their filtrations",
            )
    maps = tuple(
        tuple(
            tuple(_serialize_scalar(value, prime) for value in row)
            for row in _mul(
                right[degree],
                left[degree],
                prime,
                output_width=first.source.basis_sizes[degree],
            )
        )
        for degree in range(len(left))
    )
    return FilteredChainMapResult(
        source=first.source,
        target=second.target,
        source_filtration=first.source_filtration,
        target_filtration=second.target_filtration,
        maps=maps,
        filtration_preserving=True,
        chain_map=True,
    )


__all__ = ["FilteredChainMapCompositionRequest", "filtered_chain_map_compose"]
