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


def _require_admissible_dimensions(dimensions: tuple[int, ...]) -> None:
    """Bound every chain group to the dense-work elimination envelope."""
    if any(d < 0 for d in dimensions):
        raise ValueError("dimensions must be non-negative")
    if any(d > MAX_CHAIN_GROUP_DIMENSION for d in dimensions):
        raise ValueError(
            "group dimensions must not exceed the dense-work bound "
            f"{MAX_CHAIN_GROUP_DIMENSION}"
        )


def _require_differential_bounds(
    differentials: tuple[tuple[MatrixEntry, ...], ...],
    dimensions: tuple[int, ...],
) -> None:
    """Keep every differential entry inside its source and target group."""
    for i, diff in enumerate(differentials):
        # Homological convention d_{min+i+1}: C_{min+i+1} -> C_{min+i}
        # So differential i has source = dimensions[i+1], target = dimensions[i]
        source_dim = dimensions[i + 1]
        target_dim = dimensions[i]
        for entry in diff:
            if entry.row >= target_dim:
                raise ValueError("differential entry row exceeds target dimension")
            if entry.col >= source_dim:
                raise ValueError("differential entry col exceeds source dimension")


def _dense_gfprime(entries: tuple[MatrixEntry, ...], rows: int, cols: int, prime: int):
    mat = [[0] * cols for _ in range(rows)]
    for e in entries:
        mat[e.row][e.col] = int(e.value) % prime
    return mat


def _gfprime_mat_mul(a, b, prime: int):
    if not a or not b or not a[0] or not b[0]:
        return (
            [[0] * len(b[0]) if b and b[0] else 0 for _ in range(len(a))] if a else []
        )
    n_rows = len(a)
    n_cols = len(b[0])
    n_inner = len(b)
    result = [[0] * n_cols for _ in range(n_rows)]
    for r in range(n_rows):
        for k in range(n_inner):
            if a[r][k] == 0:
                continue
            for c in range(n_cols):
                result[r][c] = (result[r][c] + a[r][k] * b[k][c]) % prime
    return result


def _require_differential_squared_zero(
    differentials: tuple[tuple[MatrixEntry, ...], ...],
    dimensions: tuple[int, ...],
    prime: int,
) -> None:
    """Verify d^2 = 0 by dense GF(prime) composition of adjacent differentials."""
    for i in range(len(differentials) - 1):
        # d_{i+1}: C_{i+1} -> C_i, dims[i] x dims[i+1]
        # d_{i+2}: C_{i+2} -> C_{i+1}, dims[i+1] x dims[i+2]
        # Composition d_{i+1} * d_{i+2}: dims[i] x dims[i+2] should be zero
        a = _dense_gfprime(differentials[i], dimensions[i], dimensions[i + 1], prime)
        b = _dense_gfprime(
            differentials[i + 1],
            dimensions[i + 1],
            dimensions[i + 2],
            prime,
        )
        prod = _gfprime_mat_mul(a, b, prime)
        if any(any(v % prime != 0 for v in row) for row in prod):
            raise ValueError("differentials must satisfy d^2 = 0")


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
        _require_admissible_dimensions(self.dimensions)
        if self.differentials:
            if len(self.differentials) != expected_length - 1:
                raise ValueError("differentials must cover the degree gaps")
            _require_differential_bounds(self.differentials, self.dimensions)
            # Enforce d_{n-1} * d_n = 0 for all n (d^2 = 0) over GF(prime).
            _require_differential_squared_zero(
                self.differentials, self.dimensions, self.prime
            )
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
    """Homology groups of a chain complex.

    Retains its source complex so every group's degree, dimension, ranks, and
    Betti number replay against the exact kernel instead of trusting an
    independently authored table.
    """

    complex: ChainComplex
    groups: tuple[HomologyGroup, ...] = Field(min_length=1)
    prime: int = Field(ge=2, le=10_000)
    min_degree: int = Field(ge=-10, le=11)
    max_degree: int = Field(ge=-10, le=11)

    @model_validator(mode="after")
    def require_consistent_groups(self) -> Self:
        if len(self.groups) != self.max_degree - self.min_degree + 1:
            raise ValueError("homology groups must cover the degree range")
        from jacobian.math.chain_complexes._operations import _homology_groups

        if (
            self.prime != self.complex.prime
            or self.min_degree != self.complex.min_degree
            or self.max_degree != self.complex.max_degree
        ):
            raise ValueError("homology result fields must match the retained complex")
        expected = _homology_groups(self.complex)
        if self.groups != expected:
            raise ValueError(
                "homology groups must be the exact homology of the retained complex"
            )
        return self


def _chain_map_target_dim(target: ChainComplex, degree: int) -> int:
    """Dimension of the target group at ``degree``; zero outside its range."""
    if target.min_degree <= degree <= target.max_degree:
        return target.dimensions[degree - target.min_degree]
    return 0


def _chain_map_degree(complex_: ChainComplex, deg: int) -> int:
    if complex_.min_degree <= deg <= complex_.max_degree:
        return complex_.dimensions[deg - complex_.min_degree]
    return 0


def _dense_matrix(
    entries: tuple[MatrixEntry, ...], rows: int, cols: int, prime: int
) -> list[list[int]]:
    """A dense rows x cols matrix; zero-filled when no entries exist."""
    mat = [[0] * cols for _ in range(rows)]
    for ent in entries:
        mat[ent.row][ent.col] = int(ent.value) % prime
    return mat


def _dense_product(
    a: list[list[int]],
    b: list[list[int]],
    inner: int,
    cols: int,
    prime: int,
) -> list[list[int]]:
    """Dense product of an len(a) x inner by inner x cols matrix."""
    res = [[0] * cols for _ in range(len(a))]
    for r in range(len(a)):
        for k in range(inner):
            if a[r][k] == 0:
                continue
            for c in range(cols):
                res[r][c] = (res[r][c] + a[r][k] * b[k][c]) % prime
    return res


def _require_chain_map_commutativity(
    source: ChainComplex,
    target: ChainComplex,
    chain_map: tuple[tuple[MatrixEntry, ...], ...],
) -> None:
    """Verify d^D_n * f_n = f_{n-1} * d^C_n at every source degree.

    Groups outside a complex's degree range are zero-dimensional, so the
    equation is still checked where one side's differential or map is the
    shaped zero matrix - in particular at the target's boundary degrees.
    """
    prime = source.prime
    s_min = source.min_degree

    def dim(complex_: ChainComplex, deg: int) -> int:
        return _chain_map_degree(complex_, deg)

    def diff(entries, rows, cols):
        return _dense_matrix(entries, rows, cols, prime)

    # At the source's bottom degree the source differential leaves the
    # retained range, so commutativity requires d^D_{s_min} * f_{s_min} = 0
    # exactly; skipping it would accept maps that already fail at s_min.
    bottom = diff(
        (
            target.differentials[s_min - target.min_degree - 1]
            if target.min_degree < s_min <= target.max_degree
            and s_min - target.min_degree - 1 < len(target.differentials)
            else ()
        ),
        dim(target, s_min - 1),
        dim(target, s_min),
    )
    f_bottom = diff(chain_map[0], dim(target, s_min), dim(source, s_min))
    left = _dense_product(
        bottom, f_bottom, dim(target, s_min), dim(source, s_min), prime
    )
    if any(any(v % prime != 0 for v in row) for row in left):
        raise ValueError(f"chain map does not commute at degree {s_min}")

    for n in range(s_min + 1, source.max_degree + 1):
        i = n - s_min
        # d^C_n: C_n -> C_{n-1}, always inside the source range.
        d_c = diff(
            (source.differentials[i - 1] if i - 1 < len(source.differentials) else ()),
            dim(source, n - 1),
            dim(source, n),
        )
        # d^D_n: D_n -> D_{n-1}; zero-dimensional when n leaves the target.
        d_d = diff(
            (
                target.differentials[n - target.min_degree - 1]
                if target.min_degree < n <= target.max_degree
                and n - target.min_degree - 1 < len(target.differentials)
                else ()
            ),
            dim(target, n - 1),
            dim(target, n),
        )
        f_n = diff(chain_map[i], dim(target, n), dim(source, n))
        f_prev = diff(chain_map[i - 1], dim(target, n - 1), dim(source, n - 1))
        left = _dense_product(d_d, f_n, dim(target, n), dim(source, n), prime)
        right = _dense_product(f_prev, d_c, dim(source, n - 1), dim(source, n), prime)
        if left != right:
            raise ValueError(f"chain map does not commute at degree {n}")


def _require_cone_within_domain(source: ChainComplex, target: ChainComplex) -> None:
    """The derived cone must stay inside ChainComplex's representable domain."""
    cone_min = min(source.min_degree + 1, target.min_degree)
    cone_max = max(source.max_degree + 1, target.max_degree)

    def group_dim(complex_: ChainComplex, deg: int) -> int:
        return _chain_map_degree(complex_, deg)

    if cone_min < -10 or cone_max > 11:
        raise ValueError(
            "mapping cone degrees must stay within the supported "
            f"[-10, 11] range; this request derives [{cone_min}, "
            f"{cone_max}] because the shifted source extends past it"
        )
    for deg in range(cone_min, cone_max + 1):
        cone_group = group_dim(source, deg - 1) + group_dim(target, deg)
        if cone_group > MAX_CHAIN_GROUP_DIMENSION:
            raise ValueError(
                "mapping cone group dimension exceeds the dense-work "
                f"bound {MAX_CHAIN_GROUP_DIMENSION} at degree {deg}; "
                "overlapping source and target groups sum beyond it"
            )


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
        _require_chain_map_commutativity(self.source, self.target, self.chain_map)
        # The cone complex Cone(f)_n = C_{n-1} + D_n must itself stay inside
        # the representable degree range and per-group dimension budget, so
        # admission rejects here instead of the construction raising after
        # validation on an accepted request.
        _require_cone_within_domain(self.source, self.target)
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
