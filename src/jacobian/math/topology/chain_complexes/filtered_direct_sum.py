"""Direct sums of finite filtered chain complexes."""

from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import model_validator

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.chain_complexes._filtered_models import (
    MAX_FILTER_AMBIENT_DIMENSION,
    MAX_FILTER_LEVELS,
    MAX_FILTER_VECTORS_PER_GROUP,
    FilteredChainComplexRequest,
    FilteredSubspace,
    FiltrationLevel,
)
from jacobian.math.topology.chain_complexes._filtered_operations import (
    _admit_filtered_semantics,
)
from jacobian.math.topology.chain_complexes.values import (
    MAX_MATRIX_ENTRY_CHARS,
    ChainCoefficient,
    ChainComplexValue,
    CoefficientRing,
)

MAX_FILTERED_DIRECT_SUM_MATRIX_CELLS = 4096
MAX_FILTERED_DIRECT_SUM_OUTPUT_CELLS = 250_000
MAX_FILTERED_DIRECT_SUM_OUTPUT_CHARS = 20_000_000
MAX_FILTERED_DIRECT_SUM_WORK = 50_000_000


class FilteredDirectSumRequest(StrictModel):
    left: FilteredChainComplexRequest
    right: FilteredChainComplexRequest


def _require_canonical_inclusions(
    left_maps: tuple[tuple[tuple[ChainCoefficient, ...], ...], ...],
    right_maps: tuple[tuple[tuple[ChainCoefficient, ...], ...], ...],
    left_sizes: tuple[int, ...],
    right_sizes: tuple[int, ...],
    output_sizes: tuple[int, ...],
) -> None:
    for degree, dimension in enumerate(output_sizes):
        left = left_maps[degree]
        right = right_maps[degree]
        left_size = left_sizes[degree]
        right_size = right_sizes[degree]
        if (
            len(left) != dimension
            or len(right) != dimension
            or any(len(row) != left_size for row in left)
            or any(len(row) != right_size for row in right)
            or left_size + right_size != dimension
        ):
            raise ValueError("direct-sum inclusion axes are inconsistent")
        for row_index, (left_row, right_row) in enumerate(
            zip(left, right, strict=True)
        ):
            if any(
                value != (1 if row_index == column else 0)
                for column, value in enumerate(left_row)
            ) or any(
                value != (1 if row_index - left_size == column else 0)
                for column, value in enumerate(right_row)
            ):
                raise ValueError(
                    "direct-sum inclusions must be canonical summand embeddings"
                )


def _require_block_sum_differentials(
    left: ChainComplexValue,
    right: ChainComplexValue,
    output: ChainComplexValue,
) -> None:
    for index, (matrix_left, matrix_right) in enumerate(
        zip(left.differential_matrices, right.differential_matrices, strict=True)
    ):
        output_matrix = output.differential_matrices[index]
        left_rows = len(matrix_left)
        left_columns = left.basis_sizes[index + 1]
        for row_index, row in enumerate(output_matrix):
            if row_index < left_rows:
                source_row = matrix_left[row_index]
                if any(
                    row[column] != source_row[column] for column in range(left_columns)
                ) or any(row[column] for column in range(left_columns, len(row))):
                    raise ValueError(
                        "direct-sum output differentials must be block diagonal"
                    )
            else:
                source_row = matrix_right[row_index - left_rows]
                if any(row[column] for column in range(left_columns)) or any(
                    row[left_columns + column] != source_row[column]
                    for column in range(len(source_row))
                ):
                    raise ValueError(
                        "direct-sum output differentials must be block diagonal"
                    )
        if len(output_matrix) != len(matrix_left) + len(matrix_right):
            raise ValueError("direct-sum output differentials must be block diagonal")


def _require_block_sum_filtration(
    left: FilteredChainComplexRequest,
    right: FilteredChainComplexRequest,
    output: FilteredChainComplexRequest,
) -> None:
    for level_index, (level_left, level_right) in enumerate(
        zip(left.filtration, right.filtration, strict=True)
    ):
        output_level = output.filtration[level_index]
        for degree, (space_left, space_right) in enumerate(
            zip(level_left.subspaces, level_right.subspaces, strict=True)
        ):
            left_size = left.complex.basis_sizes[degree]
            output_vectors = output_level.subspaces[degree].vectors
            if len(output_vectors) != len(space_left.vectors) + len(
                space_right.vectors
            ):
                raise ValueError("direct-sum output filtration must be the block sum")
            for index, source_vector in enumerate(space_left.vectors):
                vector = output_vectors[index]
                if any(
                    vector[column] != source_vector[column]
                    for column in range(left_size)
                ) or any(vector[column] for column in range(left_size, len(vector))):
                    raise ValueError(
                        "direct-sum output filtration must be the block sum"
                    )
            right_offset = len(space_left.vectors)
            for index, source_vector in enumerate(space_right.vectors):
                vector = output_vectors[right_offset + index]
                if any(vector[column] for column in range(left_size)) or any(
                    vector[left_size + column] != source_vector[column]
                    for column in range(len(source_vector))
                ):
                    raise ValueError(
                        "direct-sum output filtration must be the block sum"
                    )


class FilteredDirectSumResult(StrictModel):
    """A filtered direct sum and its canonical summand inclusions."""

    left: FilteredChainComplexRequest
    right: FilteredChainComplexRequest
    filtered_complex: FilteredChainComplexRequest
    left_inclusions: tuple[tuple[tuple[ChainCoefficient, ...], ...], ...]
    right_inclusions: tuple[tuple[tuple[ChainCoefficient, ...], ...], ...]

    @model_validator(mode="after")
    def require_inclusion_axes(self) -> Self:
        combined = self.filtered_complex.complex
        left_complex = self.left.complex
        right_complex = self.right.complex
        degrees = len(combined.basis_sizes)
        if (
            len(left_complex.basis_sizes) != degrees
            or len(right_complex.basis_sizes) != degrees
            or len(self.left_inclusions) != degrees
            or len(self.right_inclusions) != degrees
            or len(self.left.filtration) != len(self.filtered_complex.filtration)
            or len(self.right.filtration) != len(self.filtered_complex.filtration)
            or combined.coefficient_ring != left_complex.coefficient_ring
            or combined.coefficient_ring != right_complex.coefficient_ring
            or combined.prime != left_complex.prime
            or combined.prime != right_complex.prime
            or combined.degree_min != left_complex.degree_min
            or combined.degree_min != right_complex.degree_min
            or combined.degree_max != left_complex.degree_max
            or combined.degree_max != right_complex.degree_max
        ):
            raise ValueError("direct-sum source and output axes are inconsistent")
        _require_canonical_inclusions(
            self.left_inclusions,
            self.right_inclusions,
            left_complex.basis_sizes,
            right_complex.basis_sizes,
            combined.basis_sizes,
        )
        for index, (left_size, right_size) in enumerate(
            zip(left_complex.basis_sizes, right_complex.basis_sizes, strict=True)
        ):
            if combined.basis_sizes[index] != left_size + right_size:
                raise ValueError(
                    "direct-sum output basis sizes must be the summand block sum"
                )

        _require_block_sum_differentials(left_complex, right_complex, combined)
        _require_block_sum_filtration(self.left, self.right, self.filtered_complex)
        return self


def _fail(
    location: tuple[str | int, ...], code: str, message: str
) -> OperationDomainValidationError:
    return OperationDomainValidationError(location=location, code=code, message=message)


def _revalidate(
    value: FilteredChainComplexRequest, label: str
) -> FilteredChainComplexRequest:
    if not isinstance(value, FilteredChainComplexRequest):
        raise _fail(
            (label,),
            "filtered_chain_complex.type_invalid",
            "expected a filtered chain complex",
        )
    try:
        # Re-admit native values too: model_construct and mutated nested objects
        # must not bypass the structural wire contract.
        return FilteredChainComplexRequest.model_validate(
            value.model_dump(mode="python")
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise _fail(
            (label,), "filtered_chain_complex.structure_invalid", str(exc)
        ) from exc


def _work_bound(
    complex_value: ChainComplexValue, filtration: tuple[FiltrationLevel, ...]
) -> int:
    sizes = complex_value.basis_sizes
    square = sum(
        sizes[index - 1] * sizes[index] * sizes[index + 1]
        for index in range(1, len(sizes) - 1)
    )
    level_work = 0
    for level_index, level in enumerate(filtration):
        for degree, subspace in enumerate(level.subspaces):
            dimension = sizes[degree]
            vectors = len(subspace.vectors)
            # Covers row reduction, containment, and boundary-preservation
            # elimination with a conservative cubic-in-dimension bound.
            level_work += (vectors + 1) * max(1, dimension) ** 2
            if level_index:
                level_work += (
                    len(filtration[level_index - 1].subspaces[degree].vectors)
                    * max(1, dimension) ** 3
                )
            if degree:
                source_dimension = sizes[degree - 1]
                level_work += vectors * max(1, dimension, source_dimension) ** 3
    return square + level_work


def _integer_digits_upper(value: int) -> int:
    """Bound decimal digits from bit length without converting huge integers."""
    magnitude = abs(value)
    if magnitude == 0:
        return 1
    return (magnitude.bit_length() * 30_103 + 99_999) // 100_000


def _coefficient_chars_upper(value: object) -> int:
    if type(value) is int:
        return _integer_digits_upper(value) + int(value < 0)
    if isinstance(value, Fraction):
        numerator = _integer_digits_upper(value.numerator) + int(value.numerator < 0)
        if value.denominator == 1:
            return numerator
        return numerator + 1 + _integer_digits_upper(value.denominator)
    # The structural revalidation rejects this before resource estimation.
    raise TypeError("filtered coefficients must be exact integers or Fractions")


def _require_shared_context(
    left: FilteredChainComplexRequest, right: FilteredChainComplexRequest
) -> tuple[int, ...]:
    a = left.complex
    b = right.complex
    if (
        a.coefficient_ring is CoefficientRing.INTEGER
        or b.coefficient_ring is CoefficientRing.INTEGER
    ):
        raise _fail(
            ("left", "complex"),
            "filtered_chain_complex.integer_coefficients_unsupported",
            "filtered complexes support QQ and GF(p)",
        )
    if a.coefficient_ring != b.coefficient_ring or a.prime != b.prime:
        raise _fail(
            ("right", "complex"),
            "filtered_direct_sum.coefficient_context_mismatch",
            "summands must use the same coefficient field",
        )
    if (a.degree_min, a.degree_max) != (b.degree_min, b.degree_max):
        raise _fail(
            ("right", "complex"),
            "filtered_direct_sum.degree_axis_mismatch",
            "summands must use the same degree interval",
        )
    if len(left.filtration) != len(right.filtration):
        raise _fail(
            ("right", "filtration"),
            "filtered_direct_sum.filtration_axis_mismatch",
            "summands must use the same filtration levels",
        )

    output_sizes = tuple(
        x + y for x, y in zip(a.basis_sizes, b.basis_sizes, strict=True)
    )
    if any(size > MAX_FILTER_AMBIENT_DIMENSION for size in output_sizes):
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="filtered_direct_sum.ambient_dimension_exceeded",
            message=f"direct-sum chain groups must have dimension at most {MAX_FILTER_AMBIENT_DIMENSION}",
        )
    return output_sizes


def _preflight_filtration(
    left: FilteredChainComplexRequest,
    right: FilteredChainComplexRequest,
    output_sizes: tuple[int, ...],
) -> int:
    a = left.complex
    b = right.complex
    if len(left.filtration) > MAX_FILTER_LEVELS:
        raise OperationResourceAdmissionError(
            location=("filtration",),
            code="filtered_direct_sum.level_budget_exceeded",
            message="direct-sum filtration exceeds the admitted level count",
        )

    output_filtration_cells = 0
    input_filtration_cells = 0
    input_filtration_chars = 0
    for level_a, level_b in zip(left.filtration, right.filtration, strict=True):
        for degree, (space_a, space_b) in enumerate(
            zip(level_a.subspaces, level_b.subspaces, strict=True)
        ):
            count = len(space_a.vectors) + len(space_b.vectors)
            if count > MAX_FILTER_VECTORS_PER_GROUP:
                raise OperationResourceAdmissionError(
                    location=("filtration", degree),
                    code="filtered_direct_sum.vector_count_exceeded",
                    message=f"a direct-sum filtration subspace has {count} spanning vectors; limit is {MAX_FILTER_VECTORS_PER_GROUP}",
                )
            output_filtration_cells += count * output_sizes[degree]
            input_filtration_cells += (
                len(space_a.vectors) * a.basis_sizes[degree]
                + len(space_b.vectors) * b.basis_sizes[degree]
            )
            input_filtration_chars += sum(
                _coefficient_chars_upper(value)
                for vector in (*space_a.vectors, *space_b.vectors)
                for value in vector
            )
    result_filtration_cells = output_filtration_cells + input_filtration_cells
    if result_filtration_cells > MAX_FILTERED_DIRECT_SUM_OUTPUT_CELLS:
        raise OperationResourceAdmissionError(
            location=("filtration",),
            code="filtered_direct_sum.output_budget_exceeded",
            message=f"direct-sum result filtration requires {result_filtration_cells} scalar cells; limit is {MAX_FILTERED_DIRECT_SUM_OUTPUT_CELLS}",
        )
    output_chars = input_filtration_chars + (
        output_filtration_cells - input_filtration_cells
    )
    result_chars = output_chars + input_filtration_chars
    if result_chars > MAX_FILTERED_DIRECT_SUM_OUTPUT_CHARS:
        raise OperationResourceAdmissionError(
            location=("filtration",),
            code="filtered_direct_sum.output_character_budget_exceeded",
            message=f"direct-sum result filtrations may require {result_chars} characters; limit is {MAX_FILTERED_DIRECT_SUM_OUTPUT_CHARS}",
        )
    return result_chars


def _preflight_matrices(
    left: FilteredChainComplexRequest,
    right: FilteredChainComplexRequest,
    output_sizes: tuple[int, ...],
) -> tuple[int, int]:
    a = left.complex
    b = right.complex
    matrix_cells = sum(
        output_sizes[index] * output_sizes[index + 1]
        for index in range(len(output_sizes) - 1)
    )
    if matrix_cells > MAX_FILTERED_DIRECT_SUM_MATRIX_CELLS:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="filtered_direct_sum.matrix_budget_exceeded",
            message=f"direct-sum differential has {matrix_cells} cells; limit is {MAX_FILTERED_DIRECT_SUM_MATRIX_CELLS}",
        )
    inclusion_cells = sum(size * size for size in output_sizes)
    input_matrix_chars = sum(
        _coefficient_chars_upper(value)
        for complex_value in (a, b)
        for matrix in complex_value.differential_matrices
        for row in matrix
        for value in row
    )
    input_matrix_cells = sum(
        size[index] * size[index + 1]
        for size in (a.basis_sizes, b.basis_sizes)
        for index in range(len(size) - 1)
    )
    output_matrix_chars = input_matrix_chars + matrix_cells - input_matrix_cells
    result_matrix_chars = input_matrix_chars + output_matrix_chars
    if output_matrix_chars > MAX_MATRIX_ENTRY_CHARS:
        raise OperationResourceAdmissionError(
            location=("complex",),
            code="filtered_direct_sum.matrix_character_budget_exceeded",
            message=f"direct-sum differential may require {output_matrix_chars} characters; limit is {MAX_MATRIX_ENTRY_CHARS}",
        )
    result_chars = result_matrix_chars + inclusion_cells
    if result_chars > MAX_FILTERED_DIRECT_SUM_OUTPUT_CHARS:
        raise OperationResourceAdmissionError(
            location=("result",),
            code="filtered_direct_sum.result_character_budget_exceeded",
            message=f"direct-sum result may require {result_chars} coefficient characters; limit is {MAX_FILTERED_DIRECT_SUM_OUTPUT_CHARS}",
        )
    return matrix_cells, result_chars


def _preflight(
    left: FilteredChainComplexRequest, right: FilteredChainComplexRequest
) -> None:
    a = left.complex
    b = right.complex
    output_sizes = _require_shared_context(left, right)
    filtration_chars = _preflight_filtration(left, right, output_sizes)
    _, matrix_and_map_chars = _preflight_matrices(left, right, output_sizes)
    result_chars = filtration_chars + matrix_and_map_chars
    if result_chars > MAX_FILTERED_DIRECT_SUM_OUTPUT_CHARS:
        raise OperationResourceAdmissionError(
            location=("result",),
            code="filtered_direct_sum.result_character_budget_exceeded",
            message=f"direct-sum result may require {result_chars} coefficient characters; limit is {MAX_FILTERED_DIRECT_SUM_OUTPUT_CHARS}",
        )
    estimated_work = (
        _work_bound(a, left.filtration)
        + _work_bound(b, right.filtration)
        + filtration_chars
        + matrix_and_map_chars
    )
    if estimated_work > MAX_FILTERED_DIRECT_SUM_WORK:
        raise OperationResourceAdmissionError(
            location=("left", "filtration"),
            code="filtered_direct_sum.work_budget_exceeded",
            message=f"filtered direct-sum admission is estimated at {estimated_work} units; limit is {MAX_FILTERED_DIRECT_SUM_WORK}",
        )


def filtered_direct_sum(
    left_value: FilteredChainComplexRequest,
    right_value: FilteredChainComplexRequest,
) -> FilteredDirectSumResult:
    if (
        type(left_value) is not FilteredChainComplexRequest
        or type(right_value) is not FilteredChainComplexRequest
    ):
        raise _fail(("request",), "filtered_chain_complex.type_invalid", "expected two filtered chain complexes")
    try:
        # Price the combined payload before dumping/rebuilding either nested
        # filtration. A rejection can then happen without duplicating its scalars.
        _preflight(left_value, right_value)
    except OperationDomainValidationError:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise _fail(
            ("request",), "filtered_chain_complex.structure_invalid", str(exc)
        ) from exc
    left = _revalidate(left_value, "left")
    right = _revalidate(right_value, "right")

    _admit_filtered_semantics(left.complex, left.filtration)
    _admit_filtered_semantics(right.complex, right.filtration)
    a = left.complex
    b = right.complex
    sizes = tuple(x + y for x, y in zip(a.basis_sizes, b.basis_sizes, strict=True))
    prime = a.prime
    zero = 0

    differentials = []
    for index, (matrix_a, matrix_b) in enumerate(
        zip(a.differential_matrices, b.differential_matrices, strict=True)
    ):
        cols_a = a.basis_sizes[index + 1]
        cols_b = b.basis_sizes[index + 1]
        rows = [tuple(row) + (zero,) * cols_b for row in matrix_a]
        rows.extend((zero,) * cols_a + tuple(row) for row in matrix_b)
        differentials.append(tuple(rows))

    complex_value = ChainComplexValue(
        coefficient_ring=a.coefficient_ring,
        prime=prime,
        degree_min=a.degree_min,
        degree_max=a.degree_max,
        basis_sizes=sizes,
        differential_matrices=tuple(differentials),
    )
    filtration = []
    for level_a, level_b in zip(left.filtration, right.filtration, strict=True):
        subspaces = []
        for degree, (space_a, space_b) in enumerate(
            zip(level_a.subspaces, level_b.subspaces, strict=True)
        ):
            dimension_a = a.basis_sizes[degree]
            dimension_b = b.basis_sizes[degree]
            vectors = tuple(
                tuple(vector) + (zero,) * dimension_b for vector in space_a.vectors
            ) + tuple(
                (zero,) * dimension_a + tuple(vector) for vector in space_b.vectors
            )
            subspaces.append(FilteredSubspace(vectors=vectors))
        filtration.append(FiltrationLevel(subspaces=tuple(subspaces)))
    output = FilteredChainComplexRequest(
        complex=complex_value, filtration=tuple(filtration)
    )

    left_inclusions = tuple(
        tuple(
            tuple(1 if row == column else 0 for column in range(size_a))
            for row in range(size_a + size_b)
        )
        for size_a, size_b in zip(a.basis_sizes, b.basis_sizes, strict=True)
    )
    right_inclusions = tuple(
        tuple(
            tuple(1 if row - size_a == column else 0 for column in range(size_b))
            for row in range(size_a + size_b)
        )
        for size_a, size_b in zip(a.basis_sizes, b.basis_sizes, strict=True)
    )
    # The block-diagonal boundary and componentwise subspaces prove the output
    # chain and filtration laws from the already admitted summands.
    return FilteredDirectSumResult.model_construct(
        left=left,
        right=right,
        filtered_complex=output,
        left_inclusions=left_inclusions,
        right_inclusions=right_inclusions,
    )


__all__ = ["FilteredDirectSumResult", "filtered_direct_sum"]
