"""Typed wire contracts for chain complex operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.chain_complexes.values import (
    MAX_TENSOR_GROUP_DIMENSION,
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
    for index, matrix in enumerate(map_matrices):
        rows = target.basis_sizes[index]
        cols = source.basis_sizes[index]
        if len(matrix) != rows or any(len(row) != cols for row in matrix):
            raise ValueError(
                f"{label} map component {index} must have shape "
                f"{rows}x{cols} (target rows x source columns)"
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
        # Bound the derived tensor dimensions before any allocation: each
        # tensor-product group and the total cell count stay within a
        # conservative budget derived from the input bounds.
        group_count = len(self.left.basis_sizes) + len(self.right.basis_sizes) - 1
        total = 0
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
            total += size
        if total > MAX_TENSOR_TOTAL_CELLS:
            raise ValueError(
                f"tensor product totals {total} cells, exceeding the "
                f"{MAX_TENSOR_TOTAL_CELLS}-cell work bound"
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
