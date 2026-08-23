"""Typed wire contracts for based finite chain complex operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel


MAX_CHAIN_GROUP_DIMENSION = 512
"""Conservative per-group dimension bound derived from the dense work budget.

The exact kernels build dense GF(p) matrices over the declared group
dimensions and run Gaussian elimination whose work is O(d^3) in the largest
group dimension; d = 512 keeps any accepted request's elimination inside a
bounded elementary-operation envelope instead of allowing an admitted
request to declare billion-wide groups.
"""


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
    min_degree: int = Field(ge=-10, le=11)
    max_degree: int = Field(ge=-10, le=11)
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
        if any(d > MAX_CHAIN_GROUP_DIMENSION for d in self.dimensions):
            raise ValueError(
                "group dimensions must not exceed the dense-work bound "
                f"{MAX_CHAIN_GROUP_DIMENSION}"
            )
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
    min_degree: int = Field(ge=-10, le=11)
    max_degree: int = Field(ge=-10, le=11)

    @model_validator(mode="after")
    def require_consistent_groups(self) -> Self:
        if len(self.groups) != self.max_degree - self.min_degree + 1:
            raise ValueError("homology groups must cover the degree range")
        return self


def _chain_map_target_dim(target: ChainComplex, degree: int) -> int:
    """Dimension of the target group at ``degree``; zero outside its range."""
    if target.min_degree <= degree <= target.max_degree:
        return target.dimensions[degree - target.min_degree]
    return 0


class MappingConeRequest(StrictModel):
    """Compute the mapping cone of a chain map f: C -> D."""

    source: ChainComplex
    target: ChainComplex
    chain_map: tuple[tuple[MatrixEntry, ...], ...] = Field(min_length=0, max_length=21)

    @model_validator(mode="after")
    def require_valid_chain_map(self) -> Self:
        if self.source.prime != self.target.prime:
            raise ValueError("source and target must have same prime")
        if len(self.chain_map) != len(self.source.dimensions):
            raise ValueError("chain_map must have one entry per source degree")
        # Each chain map entry f_n: C_n -> D_n must respect dimensions.
        # Target groups are indexed by mathematical degree, not tuple
        # position: a source group at degree n maps into the target group at
        # the same degree, and zero when that degree is outside the target.
        for i, entries in enumerate(self.chain_map):
            degree = self.source.min_degree + i
            s_dim = self.source.dimensions[i]
            t_dim = _chain_map_target_dim(self.target, degree)
            for e in entries:
                if e.row >= t_dim:
                    raise ValueError("chain_map entry row exceeds target dimension")
                if e.col >= s_dim:
                    raise ValueError("chain_map entry col exceeds source dimension")
        # Verify chain-map commutes with differentials: d^D_{n} * f_n = f_{n-1} * d^C_n
        prime = self.source.prime

        def _build(entries, rows, cols):
            mat = [[0] * cols for _ in range(rows)]
            for ent in entries:
                mat[ent.row][ent.col] = int(ent.value) % prime
            return mat

        def _mul(a, b):
            if not a or not b or not a[0] or not b[0]:
                return [[0] * (len(b[0]) if b and b[0] else 0) for _ in range(len(a))] if a else []
            n_rows, n_cols, n_inner = len(a), len(b[0]), len(b)
            res = [[0] * n_cols for _ in range(n_rows)]
            for r in range(n_rows):
                for k in range(n_inner):
                    if a[r][k] == 0:
                        continue
                    for c in range(n_cols):
                        res[r][c] = (res[r][c] + a[r][k] * b[k][c]) % prime
            return res

        for n in range(self.source.min_degree + 1, self.source.max_degree + 1):
            i = n - self.source.min_degree
            # need to handle different degree ranges for source/target; assume same min
            if i <= 0 or i >= len(self.source.dimensions):
                continue
            # d^C_n: C_n -> C_{n-1}, matrix s_dims[i-1] x s_dims[i]
            # f_n: C_n -> D_n
            # f_{n-1}: C_{n-1} -> D_{n-1}
            # d^D_n: D_n -> D_{n-1}
            # Check if n is within target range
            if n < self.target.min_degree or n > self.target.max_degree:
                continue
            # Build matrices
            # d^C_n
            s_idx = i  # because differentials[i-1] is d_n
            t_idx = n - self.target.min_degree
            # need to ensure indices are valid
            if s_idx - 1 < 0 or s_idx - 1 >= len(self.source.differentials):
                dC = []
            else:
                dC = _build(self.source.differentials[s_idx - 1], self.source.dimensions[i - 1], self.source.dimensions[i])
            if t_idx - 1 < 0 or t_idx - 1 >= len(self.target.differentials):
                dD = []
            else:
                # t_idx is index for D_n, so d^D_n is at t_idx-1
                dD = _build(self.target.differentials[t_idx - 1], self.target.dimensions[t_idx - 1] if t_idx - 1 >= 0 else 0, self.target.dimensions[t_idx])
            f_n = _build(self.chain_map[i], self.target.dimensions[t_idx] if t_idx < len(self.target.dimensions) else 0, self.source.dimensions[i])
            f_n_minus_1 = _build(self.chain_map[i - 1], self.target.dimensions[t_idx - 1] if t_idx - 1 >= 0 and t_idx - 1 < len(self.target.dimensions) else 0, self.source.dimensions[i - 1] if i - 1 >= 0 else 0)
            left = _mul(dD, f_n)
            right = _mul(f_n_minus_1, dC)
            # Compare left and right; they should be equal matrices
            # Empty matrices compare equal elementwise, so only a
            # non-empty disagreement breaks commutativity.
            if left != right and (left or right):
                raise ValueError(f"chain map does not commute at degree {n}")
        return self


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
