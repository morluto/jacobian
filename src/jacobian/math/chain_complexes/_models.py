"""Typed wire contracts for based finite chain complex operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel


class MatrixEntry(StrictModel):
    """One entry of a sparse differential matrix."""

    row: int = Field(ge=0)
    col: int = Field(ge=0)
    value: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def require_canonical_integer(self) -> Self:
        import re

        if not re.match(r"^-?(0|[1-9][0-9]*)$", self.value):
            raise ValueError("matrix value must be a canonical integer string")
        # Reject non-canonical "-0"
        if self.value == "-0":
            raise ValueError("matrix value must be a canonical integer string")
        return self


class ChainComplex(StrictModel):
    """A bounded homological chain complex over a prime field GF(p).

    The complex has groups C_{n_min}, ..., C_{n_max} with differentials
    d_n : C_n -> C_{n-1}. Each group has a finite basis (dimension).
    """

    prime: int = Field(ge=2, le=10_000)
    min_degree: int = Field(ge=-10, le=10)
    max_degree: int = Field(ge=-10, le=10)
    dimensions: tuple[int, ...] = Field(min_length=1, max_length=21)
    differentials: tuple[tuple[MatrixEntry, ...], ...] = Field(
        min_length=0, max_length=21
    )

    @model_validator(mode="after")
    def require_valid_complex(self) -> Self:
        import sympy

        if not sympy.isprime(self.prime):
            raise ValueError("prime must be prime")
        if self.max_degree < self.min_degree:
            raise ValueError("max_degree must be >= min_degree")
        expected_length = self.max_degree - self.min_degree + 1
        if len(self.dimensions) != expected_length:
            raise ValueError("dimensions must cover the degree range")
        if any(d < 0 for d in self.dimensions):
            raise ValueError("dimensions must be non-negative")
        if self.differentials:
            if len(self.differentials) != expected_length - 1:
                raise ValueError("differentials must cover the degree gaps")
            for i, diff in enumerate(self.differentials):
                # Homological convention d_{min+i+1}: C_{min+i+1} -> C_{min+i}
                # So differential i has source = dimensions[i+1], target = dimensions[i]
                source_dim = self.dimensions[i + 1]
                target_dim = self.dimensions[i]
                for entry in diff:
                    if entry.row >= target_dim:
                        raise ValueError("differential entry row exceeds target dimension")
                    if entry.col >= source_dim:
                        raise ValueError("differential entry col exceeds source dimension")
            # Enforce d_{n-1} * d_n = 0 for all n (d^2 = 0)
            # Build dense matrices and check composition.
            from fractions import Fraction

            def _build_dense(entries, rows, cols):
                mat = [[0] * cols for _ in range(rows)]
                for e in entries:
                    mat[e.row][e.col] = int(e.value) % self.prime
                return mat

            def _mat_mul(a, b):
                if not a or not b or not a[0] or not b[0]:
                    return [[0] * len(b[0]) if b and b[0] else 0 for _ in range(len(a))] if a else []
                n_rows = len(a)
                n_cols = len(b[0])
                n_inner = len(b)
                result = [[0] * n_cols for _ in range(n_rows)]
                for r in range(n_rows):
                    for k in range(n_inner):
                        if a[r][k] == 0:
                            continue
                        for c in range(n_cols):
                            result[r][c] = (result[r][c] + a[r][k] * b[k][c]) % self.prime
                return result

            for i in range(len(self.differentials) - 1):
                # d_{i+1}: C_{i+1} -> C_i, dims[i] x dims[i+1]
                # d_{i+2}: C_{i+2} -> C_{i+1}, dims[i+1] x dims[i+2]
                # Composition d_{i+1} * d_{i+2}: dims[i] x dims[i+2] should be zero
                a = _build_dense(self.differentials[i], self.dimensions[i], self.dimensions[i + 1])
                b = _build_dense(self.differentials[i + 1], self.dimensions[i + 1], self.dimensions[i + 2])
                prod = _mat_mul(a, b)
                if any(any(v % self.prime != 0 for v in row) for row in prod):
                    raise ValueError("differentials must satisfy d^2 = 0")
        return self


class HomologyRequest(StrictModel):
    """Compute the homology of a chain complex."""

    complex: ChainComplex


class HomologyGroup(StrictModel):
    """One homology group Betti number."""

    degree: int
    betti: int = Field(ge=0)
    dimension: int = Field(ge=0)
    boundary_rank: int = Field(ge=0)
    cycle_rank: int = Field(ge=0)


class HomologyResult(StrictModel):
    """Homology groups of a chain complex."""

    groups: tuple[HomologyGroup, ...] = Field(min_length=1)
    prime: int = Field(ge=2, le=10_000)
    min_degree: int = Field(ge=-10, le=10)
    max_degree: int = Field(ge=-10, le=10)

    @model_validator(mode="after")
    def require_consistent_groups(self) -> Self:
        if len(self.groups) != self.max_degree - self.min_degree + 1:
            raise ValueError("homology groups must cover the degree range")
        return self


class MappingConeRequest(StrictModel):
    """Compute the mapping cone of a chain map f: C -> D."""

    source: ChainComplex
    target: ChainComplex
    chain_map: tuple[tuple[MatrixEntry, ...], ...] = Field(min_length=0, max_length=21)


class MappingConeResult(StrictModel):
    """The mapping cone complex of a chain map."""

    cone: ChainComplex


__all__ = [
    "ChainComplex",
    "HomologyGroup",
    "HomologyRequest",
    "HomologyResult",
    "MappingConeRequest",
    "MappingConeResult",
    "MatrixEntry",
]
