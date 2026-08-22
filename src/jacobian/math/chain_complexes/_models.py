"""Typed wire contracts for chain complex operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.chain_complexes.values import (
    MAX_TENSOR_GROUP_DIMENSION,
    MAX_TENSOR_SERIALIZED_CHARS,
    MAX_TENSOR_TOTAL_CELLS,
    ChainComplexValue,
    CoefficientField,
)


class ConstructChainComplexRequest(StrictModel):
    """Construct a chain complex from differential matrices."""

    coefficient_field: CoefficientField = CoefficientField.RATIONAL
    prime: int | None = Field(default=None, ge=2)
    basis_sizes: tuple[int, ...] = Field(min_length=1)
    differential_matrices: tuple[tuple[tuple[str, ...], ...], ...]

    @model_validator(mode="after")
    def require_consistent_dimensions(self) -> Self:
        if len(self.basis_sizes) != len(self.differential_matrices) + 1:
            raise ValueError("need one more basis size than differential matrices")
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
        return self


class VerifyDifferentialRequest(StrictModel):
    """Verify that d^2 = 0 for a chain complex."""

    complex: ChainComplexValue


def _require_component_entry_grammar(coefficient_field, matrix):
    """Validate one component's entries; return its (cells, characters)."""
    from jacobian.math.chain_complexes.values import (
        _require_rational_entry_grammar,
    )

    for row in matrix:
        for entry in row:
            # Shape alone does not make an entry parseable: the exact
            # kernels parse entries with Fraction/int and would turn an
            # accepted request into a host exception.
            _require_rational_entry_grammar(coefficient_field, entry)
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
        raise ValueError(
            f"{label} requires equal coefficient fields "
            f"({source.coefficient_field} vs {target.coefficient_field})"
        )
    if source.prime != target.prime:
        raise ValueError(
            f"{label} requires equal prime moduli ({source.prime} vs {target.prime})"
        )
    if (source.degree_min, source.degree_max) != (
        target.degree_min,
        target.degree_max,
    ):
        raise ValueError(
            f"{label} requires source and target complexes concentrated on "
            "the same degree interval "
            f"({source.degree_min}..{source.degree_max} vs "
            f"{target.degree_min}..{target.degree_max})"
        )
    expected_count = len(source.basis_sizes)
    if len(map_matrices) != expected_count:
        raise ValueError(
            f"{label} requires one map component per chain degree "
            f"({expected_count}), got {len(map_matrices)}"
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
            raise ValueError(
                f"{label} map component {index} must have shape "
                f"{rows}x{cols} (target rows x source columns)"
            )
        cells, chars = _require_component_entry_grammar(
            source.coefficient_field, matrix
        )
        total_map_cells += cells
        total_entry_chars += chars
    if total_map_cells > MAX_CHAIN_MAP_CELLS:
        raise ValueError(
            f"{label} map components total {total_map_cells} cells, "
            f"exceeding the {MAX_CHAIN_MAP_CELLS}-cell aggregate budget"
        )
    if total_entry_chars > MAX_CHAIN_MAP_ENTRY_CHARS:
        raise ValueError(
            f"{label} map components total {total_entry_chars} entry "
            f"characters, exceeding the {MAX_CHAIN_MAP_ENTRY_CHARS}-character aggregate budget"
        )


class VerifyChainMapRequest(StrictModel):
    """Verify that a chain map commutes with differentials."""

    source: ChainComplexValue
    target: ChainComplexValue
    map_matrices: tuple[tuple[tuple[str, ...], ...], ...]

    @model_validator(mode="after")
    def require_admissible_map_components(self) -> Self:
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


class MappingConeRequest(StrictModel):
    """Compute the mapping cone of a chain map."""

    source: ChainComplexValue
    target: ChainComplexValue
    map_matrices: tuple[tuple[tuple[str, ...], ...], ...]

    @model_validator(mode="after")
    def require_admissible_map_components(self) -> Self:
        _require_chain_map_components(
            self.source,
            self.target,
            self.map_matrices,
            label="mapping cone",
        )
        return self


def _require_serializable_entries(*complex_values) -> None:
    """Tensor inputs stay within the serialization envelope: printed
    entries are products/sums of two coefficients, so each component is
    capped at 512 digits."""
    for complex_value in complex_values:
        for matrix in complex_value.differential_matrices:
            for row in matrix:
                for entry in row:
                    numerator, slash, denominator = entry.partition("/")
                    if (
                        len(numerator.lstrip("-")) > 512
                        or len(denominator.lstrip("-")) > 512
                    ):
                        raise ValueError(
                            "tensor product inputs are limited to "
                            "512-digit coefficients"
                        )


class TensorProductRequest(StrictModel):
    """Compute the tensor product of two chain complexes."""

    left: ChainComplexValue
    right: ChainComplexValue

    @model_validator(mode="after")
    def require_admissible_tensor_work(self) -> Self:
        if (
            self.left.coefficient_field != self.right.coefficient_field
            or self.left.prime != self.right.prime
        ):
            raise ValueError("tensor product requires same coefficient field and prime")
        # Bound the derived tensor work before any allocation: each
        # tensor-product group dimension, the total group cells, and the
        # dense differential cells actually allocated between consecutive
        # groups all stay within conservative budgets derived from the
        # input bounds.
        group_count = len(self.left.basis_sizes) + len(self.right.basis_sizes) - 1
        group_sizes: list[int] = []
        for degree in range(group_count):
            size = 0
            for i in range(min(degree + 1, len(self.left.basis_sizes))):
                j = degree - i
                if j < len(self.right.basis_sizes):
                    size += self.left.basis_sizes[i] * self.right.basis_sizes[j]
            if size > MAX_TENSOR_GROUP_DIMENSION:
                raise ValueError(
                    f"tensor product group dimension {size} exceeds the "
                    f"{MAX_TENSOR_GROUP_DIMENSION}-dimension work bound"
                )
            group_sizes.append(size)
        total = sum(group_sizes)
        allocated_cells = sum(
            group_sizes[degree - 1] * group_sizes[degree]
            for degree in range(1, group_count)
        )
        if total > MAX_TENSOR_TOTAL_CELLS or allocated_cells > MAX_TENSOR_TOTAL_CELLS:
            raise ValueError(
                f"tensor product allocates {max(total, allocated_cells)} "
                f"cells, exceeding the {MAX_TENSOR_TOTAL_CELLS}-cell work bound"
            )
        _require_serializable_entries(self.left, self.right)
        # Admission guarantees the derived complex value is canonical: the
        # degree interval must fit the shared chain-degree bounds, so
        # constructing it here fails at the boundary rather than inside
        # execution when the result is exposed as a ChainComplexValue.
        from jacobian.math.chain_complexes.values import ChainComplexValue

        tensor_degree_min = self.left.degree_min + self.right.degree_min
        # Shape-correct zero placeholders: differential deg has
        # group_sizes[deg] rows and group_sizes[deg+1] columns.
        placeholder_diffs = tuple(
            tuple(("0",) * group_sizes[deg + 1] for _ in range(group_sizes[deg]))
            for deg in range(max(0, group_count - 1))
        )
        ChainComplexValue(
            coefficient_field=self.left.coefficient_field,
            prime=self.left.prime,
            degree_min=tensor_degree_min,
            degree_max=tensor_degree_min + group_count - 1,
            basis_sizes=group_sizes,
            differential_matrices=placeholder_diffs,
        )
        max_entry_chars = 2 * 512 + 8
        if max(allocated_cells, 1) * max_entry_chars > MAX_TENSOR_SERIALIZED_CHARS:
            raise ValueError(
                "tensor product serialization exceeds the canonical output "
                f"ceiling ({allocated_cells} cells x ~{max_entry_chars} "
                "characters); supply smaller coefficients"
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
