"""Typed wire contracts for chain complex operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.chain_complexes.values import (
    MAX_TENSOR_GROUP_DIMENSION,
    MAX_TENSOR_TOTAL_CELLS,
    ChainComplexValue,
    CoefficientField,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"chain_complex.{reason}", message)


class ChainComplexAdmissionError(ValueError):
    """Native admission failure for chain-complex tensor operations."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class ConstructChainComplexRequest(StrictModel):
    """Construct a chain complex from differential matrices.

    A valid request contains exactly one fewer differential matrix than
    basis sizes; matrix ``i`` has shape ``basis_sizes[i] x
    basis_sizes[i+1]``, and adjacent matrices must compose to zero so the
    constructed value satisfies d^2 = 0.
    """

    coefficient_field: CoefficientField = CoefficientField.RATIONAL
    prime: int | None = Field(default=None, ge=2)
    basis_sizes: tuple[int, ...] = Field(
        min_length=1,
        description=(
            "One dimension per chain group ordered by increasing degree; "
            "there must be exactly one more basis size than differential "
            "matrices."
        ),
    )
    differential_matrices: tuple[tuple[tuple[str, ...], ...], ...] = Field(
        description=(
            "Exactly one fewer dense row-major differential matrix than "
            "basis sizes. Matrix i maps chain group i+1 into chain group i "
            "and must have shape basis_sizes[i] x basis_sizes[i+1]; "
            "adjacent matrices must compose to zero (d^2 = 0). Each entry "
            "is one canonical coefficient string: an integer with no "
            "leading zeros and no negative zero ('0', '5', '-3'), or for "
            "QQ a fully reduced fraction with denominator >= 2 ('-1/2'); "
            "for GF(p) only integer residues in [0, p) are accepted. "
            "Parsing is plain integer/fraction string parsing and never "
            "evaluates input."
        )
    )

    @model_validator(mode="after")
    def require_consistent_dimensions(self) -> Self:
        if len(self.basis_sizes) != len(self.differential_matrices) + 1:
            raise _validation_error(
                "basis_differential_count_mismatch",
                "need one more basis size than differential matrices",
            )
        # Full canonical admission: the constructed value must satisfy the
        # chain-complex value contract here rather than failing inside
        # execution.
        from jacobian.math.chain_complexes.values import ChainComplexValue

        ChainComplexValue(
            coefficient_field=self.coefficient_field,
            prime=self.prime,
            degree_min=0,
            degree_max=len(self.basis_sizes) - 1,
            basis_sizes=self.basis_sizes,
            differential_matrices=self.differential_matrices,
        )
        # A chain complex must satisfy d^2 = 0; unchecked candidate data
        # would let the public operation label arbitrary matrices as an
        # exact chain complex.
        from jacobian.math.chain_complexes.operations import (
            _matrix_multiply,
            _matrix_to_fractions,
        )

        prime = self.prime
        differentials = [
            _matrix_to_fractions(
                matrix,
                self.basis_sizes[index],
                self.basis_sizes[index + 1],
                prime=prime,
            )
            for index, matrix in enumerate(self.differential_matrices)
        ]
        for index in range(len(differentials) - 1):
            composite = _matrix_multiply(
                differentials[index],
                differentials[index + 1],
                prime=prime,
                # Declared group widths keep zero-row/zero-width composites
                # shape-faithful: an empty 0 x n map must not be re-inferred
                # as zero-width during the construct-time replay.
                left_declared_columns=self.basis_sizes[index + 1],
                result_columns=self.basis_sizes[index + 2],
            )
            if any(any(entry != 0 for entry in row) for row in composite):
                raise _validation_error(
                    "differential_not_square_zero",
                    "differential matrices must satisfy d^2 = 0: the "
                    f"composite of degrees {index + 1} and {index} is "
                    "nonzero",
                )
        return self


class VerifyDifferentialRequest(StrictModel):
    """Verify that d^2 = 0 for a chain complex."""

    complex: ChainComplexValue


def _require_component_entry_grammar(
    coefficient_field: CoefficientField,
    matrix: tuple[tuple[str, ...], ...],
    *,
    prime: int | None = None,
) -> tuple[int, int]:
    """Validate one component's entries; return its (cells, characters)."""
    from jacobian.math.chain_complexes.values import (
        _require_rational_entry_grammar,
    )

    for row in matrix:
        for entry in row:
            # Shape alone does not make an entry parseable: the exact
            # kernels parse entries with Fraction/int and would turn an
            # accepted request into a host exception.
            _require_rational_entry_grammar(coefficient_field, entry, prime=prime)
    return (
        sum(len(row) for row in matrix),
        sum(len(entry) for row in matrix for entry in row),
    )


def _require_chain_map_components(
    source: ChainComplexValue,
    target: ChainComplexValue,
    map_matrices: tuple[tuple[tuple[str, ...], ...], ...],
    *,
    label: str,
) -> None:
    """Admit only complete, correctly shaped degree-aligned chain maps.

    One component per source degree is required; component ``i`` must have
    exactly ``target.basis_sizes[i]`` rows and ``source.basis_sizes[i]``
    columns. Degree intervals must coincide so tuple indices are actual
    chain degrees.
    """
    if source.coefficient_field != target.coefficient_field:
        raise _validation_error(
            "chain_map_field_mismatch",
            f"{label} requires equal coefficient fields "
            f"({source.coefficient_field} vs {target.coefficient_field})",
        )
    if source.prime != target.prime:
        raise _validation_error(
            "chain_map_prime_mismatch",
            f"{label} requires equal prime moduli ({source.prime} vs {target.prime})",
        )
    if (source.degree_min, source.degree_max) != (
        target.degree_min,
        target.degree_max,
    ):
        raise _validation_error(
            "chain_map_degree_interval_mismatch",
            f"{label} requires source and target complexes concentrated on "
            "the same degree interval "
            f"({source.degree_min}..{source.degree_max} vs "
            f"{target.degree_min}..{target.degree_max})",
        )
    expected_count = len(source.basis_sizes)
    if len(map_matrices) != expected_count:
        raise _validation_error(
            "chain_map_component_count_mismatch",
            f"{label} requires one map component per chain degree "
            f"({expected_count}), got {len(map_matrices)}",
        )
    from jacobian.math.chain_complexes.values import (
        MAX_CHAIN_MAP_CELLS,
        MAX_CHAIN_MAP_ENTRY_CHARS,
    )

    total_map_cells = 0
    total_entry_chars = 0
    for index, matrix in enumerate(map_matrices):
        rows = target.basis_sizes[index]
        cols = source.basis_sizes[index]
        if len(matrix) != rows or any(len(row) != cols for row in matrix):
            raise _validation_error(
                "chain_map_component_shape_mismatch",
                f"{label} map component {index} must have shape "
                f"{rows}x{cols} (target rows x source columns)",
            )
        cells, chars = _require_component_entry_grammar(
            source.coefficient_field, matrix, prime=source.prime
        )
        total_map_cells += cells
        total_entry_chars += chars
    if total_map_cells > MAX_CHAIN_MAP_CELLS:
        raise _validation_error(
            "chain_map_cell_budget_exceeded",
            f"{label} map components total {total_map_cells} cells, "
            f"exceeding the {MAX_CHAIN_MAP_CELLS}-cell aggregate budget",
        )
    if total_entry_chars > MAX_CHAIN_MAP_ENTRY_CHARS:
        raise _validation_error(
            "chain_map_entry_budget_exceeded",
            f"{label} map components total {total_entry_chars} entry "
            f"characters, exceeding the {MAX_CHAIN_MAP_ENTRY_CHARS}-character aggregate budget",
        )


class VerifyChainMapRequest(StrictModel):
    """Verify that a chain map commutes with differentials."""

    source: ChainComplexValue
    target: ChainComplexValue
    map_matrices: tuple[tuple[tuple[str, ...], ...], ...] = Field(
        description=(
            "One dense component per chain degree, each shaped "
            "(target basis size) x (source basis size). Entries follow the "
            "same canonical coefficient grammar as differential matrices: "
            "integers without leading zeros, reduced QQ fractions, and "
            "GF(p) residues in [0, p); strings are parsed, never evaluated."
        )
    )

    @model_validator(mode="after")
    def require_admissible_map_components(self) -> Self:
        _require_chain_map_components(
            self.source,
            self.target,
            self.map_matrices,
            label="chain-map verification",
        )
        return self


def _require_square_zero_at_admission(
    complex_value: ChainComplexValue, *, label: str
) -> None:
    """Homology-type outputs are defined only for genuine complexes."""
    from jacobian.math.chain_complexes.operations import (
        _parsed_differentials,
        _require_square_zero,
    )

    _require_square_zero(
        _parsed_differentials(complex_value),
        complex_value.prime,
        label=label,
        group_columns=list(complex_value.basis_sizes),
        degree_min=complex_value.degree_min,
    )


class ComputeHomologyRequest(StrictModel):
    """Compute homology of a chain complex."""

    complex: ChainComplexValue

    @model_validator(mode="after")
    def require_genuine_chain_complex(self) -> Self:
        # Homology is defined only when d^2 = 0; checking here keeps an
        # unbounded execution failure out of math.run's typed contract.
        _require_square_zero_at_admission(self.complex, label="homology")
        return self


def _entry_character_count(complex_value: ChainComplexValue) -> int:
    """Total printed characters across one complex's differential cells."""
    return sum(
        len(entry)
        for matrix in complex_value.differential_matrices
        for row in matrix
        for entry in row
    )


def _require_admissible_cone_value(
    source: ChainComplexValue,
    target: ChainComplexValue,
    map_matrices: tuple[tuple[tuple[str, ...], ...], ...],
) -> None:
    """Bound the derived mapping-cone work before any allocation.

    The cone becomes a first-class ``ChainComplexValue``, so its degree
    interval, group dimensions, dense cell budget, and serialization
    envelope must be established at admission rather than discovered
    during execution.
    """
    from jacobian.math.chain_complexes.operations import _cone_group_sizes
    from jacobian.math.chain_complexes.values import (
        MAX_MATRIX_ENTRY_CHARS,
        ChainComplexValue,
    )

    cone_basis_sizes = _cone_group_sizes(source, target)
    degree_min = source.degree_min
    placeholder_diffs = tuple(
        tuple(("0",) * cone_basis_sizes[deg + 1] for _ in range(cone_basis_sizes[deg]))
        for deg in range(max(0, len(cone_basis_sizes) - 1))
    )
    ChainComplexValue(
        coefficient_field=source.coefficient_field,
        prime=source.prime,
        degree_min=degree_min,
        degree_max=degree_min + len(cone_basis_sizes) - 1,
        basis_sizes=cone_basis_sizes,
        differential_matrices=placeholder_diffs,
    )
    # Every populated cone cell copies one admitted coefficient string
    # (possibly gaining a leading '-') and every remaining cell prints
    # "0", so this bounds the derived serialization from above.
    cone_cells = sum(
        cone_basis_sizes[i] * cone_basis_sizes[i + 1]
        for i in range(len(cone_basis_sizes) - 1)
    )
    worst_case_chars = (
        _entry_character_count(source)
        + _entry_character_count(target)
        + sum(len(entry) for matrix in map_matrices for row in matrix for entry in row)
        + cone_cells
    )
    if worst_case_chars > MAX_MATRIX_ENTRY_CHARS:
        raise _validation_error(
            "mapping_cone_output_budget_exceeded",
            "mapping cone serialization exceeds the canonical output "
            f"ceiling ({worst_case_chars} characters against "
            f"{MAX_MATRIX_ENTRY_CHARS}); supply smaller coefficients",
        )


class MappingConeRequest(StrictModel):
    """Compute the mapping cone of a chain map."""

    source: ChainComplexValue
    target: ChainComplexValue
    map_matrices: tuple[tuple[tuple[str, ...], ...], ...] = Field(
        description=(
            "One dense component per chain degree, each shaped "
            "(target basis size) x (source basis size). Entries follow the "
            "same canonical coefficient grammar as differential matrices: "
            "integers without leading zeros, reduced QQ fractions, and "
            "GF(p) residues in [0, p); strings are parsed, never evaluated."
        )
    )

    @model_validator(mode="after")
    def require_admissible_map_components(self) -> Self:
        _require_chain_map_components(
            self.source,
            self.target,
            self.map_matrices,
            label="mapping cone",
        )
        _require_square_zero_at_admission(self.source, label="mapping cone source")
        _require_square_zero_at_admission(self.target, label="mapping cone target")
        # The mapping cone is defined only for a genuine chain map, so the
        # commutation relation d_target * f_{i+1} == f_i * d_source is part
        # of the accepted request domain: checking it here keeps an accepted
        # math.run request from dying in execution.
        from jacobian.math.chain_complexes.operations import (
            _matrix_to_fractions,
            _require_chain_map_relation,
        )

        prime = self.source.prime
        source_diffs = [
            _matrix_to_fractions(
                m,
                self.source.basis_sizes[i],
                self.source.basis_sizes[i + 1],
                prime,
            )
            for i, m in enumerate(self.source.differential_matrices)
        ]
        target_diffs = [
            _matrix_to_fractions(
                m,
                self.target.basis_sizes[i],
                self.target.basis_sizes[i + 1],
                prime,
            )
            for i, m in enumerate(self.target.differential_matrices)
        ]
        map_mats = [
            _matrix_to_fractions(
                m,
                self.target.basis_sizes[i],
                self.source.basis_sizes[i],
                prime,
            )
            for i, m in enumerate(self.map_matrices)
        ]
        _require_chain_map_relation(
            source_diffs,
            target_diffs,
            map_mats,
            prime,
            list(self.source.basis_sizes),
            list(self.target.basis_sizes),
        )
        # The returned cone is exposed as a first-class canonical value, so
        # its derived bounds are part of the accepted request domain.
        _require_admissible_cone_value(self.source, self.target, self.map_matrices)
        return self


def _require_serializable_entries(*complex_values: ChainComplexValue) -> None:
    """Tensor inputs stay within the serialization envelope: printed
    entries are products/sums of two coefficients, so each component is
    capped at 512 digits."""
    for complex_value in complex_values:
        for matrix in complex_value.differential_matrices:
            for row in matrix:
                for entry in row:
                    numerator, _, denominator = entry.partition("/")
                    if (
                        len(numerator.lstrip("-")) > 512
                        or len(denominator.lstrip("-")) > 512
                    ):
                        raise ChainComplexAdmissionError(
                            "tensor_coefficient_digit_budget_exceeded",
                            "tensor product inputs are limited to "
                            "512-digit coefficients",
                        )


def _require_admissible_tensor_work(
    left: ChainComplexValue, right: ChainComplexValue
) -> None:
    """Bound the derived tensor work before any allocation.

    Shared by the request model and the result validator's construction
    replay: each tensor-product group dimension, the total group cells,
    and the dense differential cells actually allocated between
    consecutive groups stay within conservative budgets derived from the
    input bounds.
    """
    if left.coefficient_field != right.coefficient_field or left.prime != right.prime:
        raise ChainComplexAdmissionError(
            "tensor_context_mismatch",
            "tensor product requires same coefficient field and prime",
        )
    group_count = len(left.basis_sizes) + len(right.basis_sizes) - 1
    group_sizes: list[int] = []
    for degree in range(group_count):
        size = 0
        for i in range(min(degree + 1, len(left.basis_sizes))):
            j = degree - i
            if j < len(right.basis_sizes):
                size += left.basis_sizes[i] * right.basis_sizes[j]
        if size > MAX_TENSOR_GROUP_DIMENSION:
            raise ChainComplexAdmissionError(
                "tensor_group_dimension_budget_exceeded",
                f"tensor product group dimension {size} exceeds the "
                f"{MAX_TENSOR_GROUP_DIMENSION}-dimension work bound",
            )
        group_sizes.append(size)
    total = sum(group_sizes)
    allocated_cells = sum(
        group_sizes[degree - 1] * group_sizes[degree]
        for degree in range(1, group_count)
    )
    if total > MAX_TENSOR_TOTAL_CELLS or allocated_cells > MAX_TENSOR_TOTAL_CELLS:
        raise ChainComplexAdmissionError(
            "tensor_cell_budget_exceeded",
            f"tensor product allocates {max(total, allocated_cells)} "
            f"cells, exceeding the {MAX_TENSOR_TOTAL_CELLS}-cell work bound",
        )
    _require_serializable_entries(left, right)
    _require_square_zero_at_admission(left, label="tensor product left")
    _require_square_zero_at_admission(right, label="tensor product right")
    # Admission guarantees the derived complex value is canonical: the
    # degree interval must fit the shared chain-degree bounds, so
    # constructing it here fails at the boundary rather than inside
    # execution when the result is exposed as a ChainComplexValue.
    from jacobian.math.chain_complexes.values import ChainComplexValue

    tensor_degree_min = left.degree_min + right.degree_min
    # Shape-correct zero placeholders: differential deg has
    # group_sizes[deg] rows and group_sizes[deg+1] columns.
    placeholder_diffs = tuple(
        tuple(("0",) * group_sizes[deg + 1] for _ in range(group_sizes[deg]))
        for deg in range(max(0, group_count - 1))
    )
    ChainComplexValue(
        coefficient_field=left.coefficient_field,
        prime=left.prime,
        degree_min=tensor_degree_min,
        degree_max=tensor_degree_min + group_count - 1,
        basis_sizes=tuple(group_sizes),
        differential_matrices=placeholder_diffs,
    )
    # Every populated tensor cell copies one admitted coefficient string
    # (a Koszul negation may add a leading '-') and every remaining cell
    # prints "0", so the expanded differentials print at most this many
    # characters. The derived value enforces MAX_MATRIX_ENTRY_CHARS
    # against its real coefficients, so admission must couple the same
    # budget to the expansion instead of to shape alone.
    from jacobian.math.chain_complexes.values import MAX_MATRIX_ENTRY_CHARS

    def _max_entry_length(complex_value: ChainComplexValue) -> int:
        return max(
            (
                len(entry)
                for matrix in complex_value.differential_matrices
                for row in matrix
                for entry in row
            ),
            default=1,
        )

    worst_entry_chars = max(_max_entry_length(left), _max_entry_length(right)) + 1
    expanded_entry_chars = allocated_cells * worst_entry_chars
    if expanded_entry_chars > MAX_MATRIX_ENTRY_CHARS:
        raise ChainComplexAdmissionError(
            "tensor_output_budget_exceeded",
            "tensor product serialization exceeds the canonical "
            f"{MAX_MATRIX_ENTRY_CHARS}-character budget: {allocated_cells} "
            f"expanded cells x ~{worst_entry_chars} characters per copied "
            "coefficient; supply smaller coefficients",
        )


class TensorProductRequest(StrictModel):
    """Compute the tensor product of two chain complexes."""

    left: ChainComplexValue
    right: ChainComplexValue

    @model_validator(mode="after")
    def require_admissible_tensor_work(self) -> Self:
        try:
            _require_admissible_tensor_work(self.left, self.right)
        except ChainComplexAdmissionError as exc:
            raise _validation_error(exc.reason, str(exc)) from None
        return self


__all__ = [
    "ComputeHomologyRequest",
    "ConstructChainComplexRequest",
    "MappingConeRequest",
    "TensorProductRequest",
    "VerifyChainMapRequest",
    "VerifyDifferentialRequest",
]
