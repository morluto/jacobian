"""Typed wire contracts for chain complex operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.topology.chain_complexes.values import (
    MAX_MATRIX_CELLS,
    MAX_OPERATION_MATRIX_CELLS,
    ChainComplexValue,
    CoefficientField,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"chain_complex.{reason}", message)


def _matrix_cells(complex_value: ChainComplexValue) -> int:
    return sum(
        complex_value.basis_sizes[index] * complex_value.basis_sizes[index + 1]
        for index in range(len(complex_value.basis_sizes) - 1)
    )


def _require_complex_cell_budget(
    complex_value: ChainComplexValue, *, maximum: int, label: str
) -> None:
    cells = _matrix_cells(complex_value)
    if cells > maximum:
        raise _validation_error(
            "operation_matrix_cell_budget_exceeded",
            f"{label} has {cells} differential cells, exceeding the "
            f"{maximum}-cell operation budget",
        )


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


class VerifyDifferentialRequest(StrictModel):
    """Verify that d^2 = 0 for a chain complex."""

    complex: ChainComplexValue

    @model_validator(mode="after")
    def require_operation_budget(self) -> Self:
        _require_complex_cell_budget(
            self.complex,
            maximum=MAX_OPERATION_MATRIX_CELLS,
            label="differential verification input",
        )
        return self


def _require_component_entry_grammar(
    coefficient_field: CoefficientField,
    matrix: tuple[tuple[str, ...], ...],
    *,
    prime: int | None = None,
) -> tuple[int, int]:
    """Validate one component's entries; return its (cells, characters)."""
    from jacobian.math.topology.chain_complexes.values import (
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
    from jacobian.math.topology.chain_complexes.values import (
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
        for label, complex_value in (("source", self.source), ("target", self.target)):
            _require_complex_cell_budget(
                complex_value,
                maximum=MAX_OPERATION_MATRIX_CELLS,
                label=f"chain-map {label}",
            )
        _require_chain_map_components(
            self.source,
            self.target,
            self.map_matrices,
            label="chain-map verification",
        )
        return self


class ComputeHomologyRequest(StrictModel):
    """Compute homology of a chain complex."""

    complex: ChainComplexValue

    @model_validator(mode="after")
    def require_homology_budget(self) -> Self:
        _require_complex_cell_budget(
            self.complex,
            maximum=MAX_MATRIX_CELLS,
            label="homology input",
        )
        return self


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
    def require_input_budgets(self) -> Self:
        for label, complex_value in (("source", self.source), ("target", self.target)):
            _require_complex_cell_budget(
                complex_value,
                maximum=MAX_OPERATION_MATRIX_CELLS,
                label=f"mapping-cone {label}",
            )
        return self


class TensorProductRequest(StrictModel):
    """Compute the tensor product of two chain complexes."""

    left: ChainComplexValue
    right: ChainComplexValue

    @model_validator(mode="after")
    def require_input_budgets(self) -> Self:
        for label, complex_value in (("left", self.left), ("right", self.right)):
            _require_complex_cell_budget(
                complex_value,
                maximum=MAX_OPERATION_MATRIX_CELLS,
                label=f"tensor-product {label}",
            )
        return self


__all__ = [
    "ComputeHomologyRequest",
    "ConstructChainComplexRequest",
    "MappingConeRequest",
    "TensorProductRequest",
    "VerifyChainMapRequest",
    "VerifyDifferentialRequest",
]
