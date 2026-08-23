"""Typed wire contracts for based finite chain complex operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix, multiply

MAX_CHAIN_GROUP_DIMENSION = 256
"""Per-group dimension bound inherited from the shared prime-field kernel.

Differentials are canonical ``PrimeFieldMatrix`` values, so every matrix a
request or derived cone carries is bounded by the kernel's own 256-dimension
cap; admitting larger groups would create a second, wider encoding of the
same public value. Rank and composition work stay inside one bounded
envelope instead of allowing an admitted request to declare billion-wide
groups.
"""


MAX_CHAIN_DEGREES = 21
"""Schema-visible cap on the consecutive groups one complex may carry.

Degrees are admitted in the window [-10, 11] and a complex holds at most one
group per degree, so 21 consecutive groups is the representable maximum.
Derived complexes must satisfy the same cap: the mapping cone of a source
concentrated at degree 11 would otherwise derive degree 12, and disjoint
endpoints can derive a 22-group span that no ChainComplex can represent.
"""

MIN_DEGREE = -10
MAX_DEGREE = 11

# Aggregate boundedness proof (AGENTS.md "Mathematical boundedness is a proof
# obligation").
#
# MAX_AGGREGATE_CELLS bounds one request's total parsed matrix cells: every
# differential of both complexes, every chain-map matrix, and (for mapping-
# cone requests) the predicted cone differentials this operation will emit.
# Parsing plus canonical-residue validation are linear in cells, so admission
# stays linear before any backend expansion.
#
# The work constants bound the field operations behind those cells. For
# matrices with row, inner, column counts <= MAX_CHAIN_GROUP_DIMENSION:
#   composition work  rows * inner * cols
#                     <= MAX_CHAIN_GROUP_DIMENSION * (rows * inner
#                      + inner * cols) / 2 <= 128 * adjacent cells,
#   elimination work  rows * cols * min(rows, cols) <= 256 * cells.
# Summed over a request (adjacent-cell pairs share each cell at most twice),
# both are provably <= MAX_CHAIN_GROUP_DIMENSION * MAX_AGGREGATE_CELLS; the
# constants below adopt half that maximum as conservative named envelopes.
# Every product and rank runs on the maintained SymPy/FLINT DomainMatrix
# kernel, so at the worst admitted envelope the backend performs at most
# ~3.4e7 field operations. Wall time remains only an execution safety net;
# these named quantities are the mathematical work bound. Adversarial
# envelopes that previously passed per-matrix checks (20 full all-ones
# differentials composing through ~2.55e9 Python inner-loop updates) now fail
# the aggregate cell budget before any product runs.

MAX_AGGREGATE_CELLS = 262_144
MAX_AGGREGATE_MULTIPLICATION_WORK = (
    MAX_AGGREGATE_CELLS * MAX_CHAIN_GROUP_DIMENSION // 2
)
MAX_AGGREGATE_ELIMINATION_WORK = MAX_AGGREGATE_CELLS * MAX_CHAIN_GROUP_DIMENSION // 2


def _matrix_cells(matrix: PrimeFieldMatrix) -> int:
    return len(matrix.entries) * matrix.columns


def _composition_work(left: PrimeFieldMatrix, right: PrimeFieldMatrix) -> int:
    """Predicted field multiplications for ``multiply(left, right)``."""

    return len(left.entries) * left.columns * right.columns


def _elimination_work(matrix: PrimeFieldMatrix) -> int:
    """Predicted field work for ``rank(matrix)`` elimination."""

    rows = len(matrix.entries)
    return rows * matrix.columns * min(rows, matrix.columns)


def _is_zero(matrix: PrimeFieldMatrix) -> bool:
    return all(value == 0 for row in matrix.entries for value in row)


def _zero_block(rows: int, columns: int, prime: int) -> PrimeFieldMatrix:
    return PrimeFieldMatrix(
        prime=prime,
        entries=tuple((0,) * columns for _ in range(rows)),
        columns=columns,
    )


def _require_aggregate_budget(
    matrices: tuple[PrimeFieldMatrix, ...],
    predicted_cells: int,
    predicted_work: int,
) -> None:
    """Gate aggregate parse cells and predicted multiplication work."""

    total_cells = sum(_matrix_cells(matrix) for matrix in matrices) + predicted_cells
    if total_cells > MAX_AGGREGATE_CELLS:
        raise ValueError(
            "request carries "
            f"{total_cells} matrix cells across all differentials and chain "
            f"maps; admission bounds one request at {MAX_AGGREGATE_CELLS} so "
            "validation work stays bounded before any backend expansion"
        )
    if predicted_work > MAX_AGGREGATE_MULTIPLICATION_WORK:
        raise ValueError(
            "request predicts "
            f"{predicted_work} field multiplications for differential "
            "composition and chain-map equations; admission bounds one "
            f"request at {MAX_AGGREGATE_MULTIPLICATION_WORK}"
        )


def _d_squared_work(differentials: tuple[PrimeFieldMatrix, ...]) -> int:
    """Predicted composition work for the d^2 = 0 check."""

    return sum(
        _composition_work(differentials[i], differentials[i + 1])
        for i in range(len(differentials) - 1)
    )


def _check_d_squared(differentials: tuple[PrimeFieldMatrix, ...]) -> None:
    """Verify d_i . d_{i+1} == 0 within the aggregate work budget.

    Each differential is stored as its (target x source) matrix, so the
    composition acting on C_{i+1} is the product M_i . M_{i+1}.
    """

    if not differentials:
        return
    prime = differentials[0].prime
    for i in range(len(differentials) - 1):
        square = multiply(differentials[i], differentials[i + 1])
        if not _is_zero(square):
            raise ValueError(
                "differentials must satisfy d^2 = 0 "
                f"(gap {i} failed over GF({prime}))"
            )


def _require_admissible_dimensions(dimensions: tuple[int, ...]) -> None:
    """Bound every chain group to the dense-work elimination envelope."""
    if any(d < 0 for d in dimensions):
        raise ValueError("dimensions must be non-negative")
    if any(d > MAX_CHAIN_GROUP_DIMENSION for d in dimensions):
        raise ValueError(
            "group dimensions must not exceed the dense-work bound "
            f"{MAX_CHAIN_GROUP_DIMENSION}"
        )


class ChainComplex(StrictModel):
    """A bounded homological chain complex over a prime field GF(p).

    The complex has groups C_{n_min}, ..., C_{n_max} with differentials
    d_n : C_n -> C_{n-1}. Each group has a finite basis (dimension).
    ``differentials[i]`` is the boundary map d_{min_degree + i + 1}, stored
    as its canonical (target x source) matrix with ``dimensions[i]`` rows and
    ``dimensions[i + 1]`` columns.
    """

    prime: int = Field(ge=2, le=10_000)
    min_degree: int = Field(ge=-10, le=MAX_DEGREE)
    max_degree: int = Field(ge=MIN_DEGREE, le=MAX_DEGREE)
    dimensions: tuple[int, ...] = Field(min_length=1, max_length=MAX_CHAIN_DEGREES)
    differentials: tuple[PrimeFieldMatrix, ...]

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
            for i, diff in enumerate(self.differentials):
                if diff.prime != self.prime:
                    raise ValueError(
                        "differential prime must match the complex prime"
                    )
                if len(diff.entries) != self.dimensions[i]:
                    raise ValueError(
                        f"differential {i} must have {self.dimensions[i]} rows"
                    )
                if diff.columns != self.dimensions[i + 1]:
                    raise ValueError(
                        f"differential {i} must have "
                        f"{self.dimensions[i + 1]} columns"
                    )
            # Gate the aggregate budget before any composition expands, then
            # enforce d^2 = 0 through the shared kernel.
            _require_aggregate_budget(
                self.differentials, 0, _d_squared_work(self.differentials)
            )
            _check_d_squared(self.differentials)
        return self


class HomologyRequest(StrictModel):
    """Compute the homology of a chain complex."""

    complex: ChainComplex

    @model_validator(mode="after")
    def require_bounded_elimination(self) -> Self:
        work = sum(_elimination_work(diff) for diff in self.complex.differentials)
        if work > MAX_AGGREGATE_ELIMINATION_WORK:
            raise ValueError(
                "request predicts "
                f"{work} elimination operations for the rank profile; "
                "admission bounds one request at "
                f"{MAX_AGGREGATE_ELIMINATION_WORK}"
            )
        return self


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
    min_degree: int = Field(ge=MIN_DEGREE, le=MAX_DEGREE)
    max_degree: int = Field(ge=MIN_DEGREE, le=MAX_DEGREE)

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


def _boundary_matrix(complex_: ChainComplex, deg: int) -> PrimeFieldMatrix:
    """d_deg : C_deg -> C_{deg-1}; the shaped zero map when undefined.

    A boundary leaving a degree above the complex has an empty domain
    (columns), one arriving below the complex has an empty codomain (rows);
    inside the range a declared-but-missing differential is the all-zero map.
    """

    prime = complex_.prime
    if deg <= complex_.min_degree:
        return _zero_block(0, _chain_map_degree(complex_, deg), prime)
    if deg > complex_.max_degree:
        return _zero_block(_chain_map_degree(complex_, deg - 1), 0, prime)
    idx = deg - complex_.min_degree - 1
    if idx >= len(complex_.differentials):
        return _zero_block(
            _chain_map_degree(complex_, deg - 1),
            _chain_map_degree(complex_, deg),
            prime,
        )
    return complex_.differentials[idx]


def _chain_map_matrix(
    chain_map: tuple[PrimeFieldMatrix, ...],
    source: ChainComplex,
    target: ChainComplex,
    degree: int,
) -> PrimeFieldMatrix:
    """f_degree : C_degree -> D_degree; the shaped zero map when undefined."""

    if degree < source.min_degree or degree > source.max_degree:
        # The source group is absent, so the map lands in an empty codomain.
        return _zero_block(_chain_map_target_dim(target, degree), 0, source.prime)
    idx = degree - source.min_degree
    return chain_map[idx]


def _validate_chain_map_shapes(
    source: ChainComplex,
    target: ChainComplex,
    chain_map: tuple[PrimeFieldMatrix, ...],
) -> None:
    """Each f_n must carry exactly the (target x source) shape at degree n."""

    if len(chain_map) != len(source.dimensions):
        raise ValueError("chain_map must have one entry per source degree")
    for i, map_matrix in enumerate(chain_map):
        degree = source.min_degree + i
        s_dim = source.dimensions[i]
        t_dim = _chain_map_target_dim(target, degree)
        if map_matrix.prime != source.prime:
            raise ValueError("chain-map prime must match the complex prime")
        if len(map_matrix.entries) != t_dim:
            raise ValueError(
                f"chain_map[{i}] must have {t_dim} rows at degree {degree}"
            )
        if map_matrix.columns != s_dim:
            raise ValueError(
                f"chain_map[{i}] must have {s_dim} columns at degree {degree}"
            )


def _chain_map_commutativity_work(
    source: ChainComplex,
    target: ChainComplex,
    chain_map: tuple[PrimeFieldMatrix, ...],
) -> int:
    """Predicted multiplication work of the commutativity equations."""

    work = 0
    bottom = source.min_degree
    work += _composition_work(_boundary_matrix(target, bottom), chain_map[0])
    for n in range(bottom + 1, source.max_degree + 1):
        i = n - bottom
        d_c = _boundary_matrix(source, n)
        d_d = _boundary_matrix(target, n)
        work += _composition_work(d_d, chain_map[i])
        work += _composition_work(chain_map[i - 1], d_c)
    return work


def _require_chain_map_commutativity(
    source: ChainComplex,
    target: ChainComplex,
    chain_map: tuple[PrimeFieldMatrix, ...],
) -> None:
    """Verify d^D_n . f_n = f_{n-1} . d^C_n at every source degree.

    Groups outside a complex's degree range are zero-dimensional, so the
    equation is still checked where one side's differential or map is the
    shaped zero matrix - in particular at the target's boundary degrees.
    """
    s_min = source.min_degree

    # At the source's bottom degree the source differential leaves the
    # retained range, so commutativity requires d^D_{s_min} . f_{s_min} = 0
    # exactly; skipping it would accept maps that already fail at s_min.
    bottom_product = multiply(_boundary_matrix(target, s_min), chain_map[0])
    if not _is_zero(bottom_product):
        raise ValueError(f"chain map does not commute at degree {s_min}")

    for n in range(s_min + 1, source.max_degree + 1):
        i = n - s_min
        left = multiply(
            _boundary_matrix(target, n),
            chain_map[i],
        )
        right = multiply(
            chain_map[i - 1],
            _boundary_matrix(source, n),
        )
        if left != right:
            raise ValueError(f"chain map does not commute at degree {n}")


def _cone_span(source: ChainComplex, target: ChainComplex) -> tuple[int, int]:
    cone_min = min(source.min_degree + 1, target.min_degree)
    cone_max = max(source.max_degree + 1, target.max_degree)
    return cone_min, cone_max


def _cone_dimension_at(
    source: ChainComplex, target: ChainComplex, deg: int
) -> int:
    """Dimension of Cone(f)_deg = C_{deg-1} (+) D_deg."""

    return _chain_map_degree(source, deg - 1) + _chain_map_degree(target, deg)


def _require_cone_within_domain(source: ChainComplex, target: ChainComplex) -> None:
    """The derived cone must stay inside ChainComplex's representable domain."""
    cone_min, cone_max = _cone_span(source, target)

    if cone_min < MIN_DEGREE or cone_max > MAX_DEGREE:
        raise ValueError(
            "mapping cone degrees must stay within the supported "
            f"[{MIN_DEGREE}, {MAX_DEGREE}] range; this request derives "
            f"[{cone_min}, {cone_max}] because the shifted source extends "
            "past it"
        )
    span = cone_max - cone_min + 1
    if span > MAX_CHAIN_DEGREES:
        raise ValueError(
            "mapping cone spans "
            f"{span} consecutive degrees [{cone_min}, {cone_max}]; a chain "
            f"complex represents at most {MAX_CHAIN_DEGREES} consecutive "
            "groups, so admission rejects the derived span here instead of "
            "failing construction on an accepted request"
        )
    for deg in range(cone_min, cone_max + 1):
        cone_group = _cone_dimension_at(source, target, deg)
        if cone_group > MAX_CHAIN_GROUP_DIMENSION:
            raise ValueError(
                "mapping cone group dimension exceeds the dense-work "
                f"bound {MAX_CHAIN_GROUP_DIMENSION} at degree {deg}; "
                "overlapping source and target groups sum beyond it"
            )


def _cone_work_predictions(
    source: ChainComplex, target: ChainComplex
) -> tuple[int, int]:
    """Predicted aggregate cells and composition work of the cone differentials."""

    cone_min, cone_max = _cone_span(source, target)
    dims = [
        _cone_dimension_at(source, target, deg) for deg in range(cone_min, cone_max + 1)
    ]
    predicted_cells = sum(dims[g] * dims[g + 1] for g in range(len(dims) - 1))
    predicted_work = sum(
        dims[g] * dims[g + 1] * dims[g + 2] for g in range(len(dims) - 2)
    )
    return predicted_cells, predicted_work


class MappingConeRequest(StrictModel):
    """Compute the mapping cone of a chain map f: C -> D."""

    source: ChainComplex
    target: ChainComplex
    chain_map: tuple[PrimeFieldMatrix, ...]

    @model_validator(mode="after")
    def require_valid_chain_map(self) -> Self:
        if self.source.prime != self.target.prime:
            raise ValueError("source and target must have same prime")
        _validate_chain_map_shapes(self.source, self.target, self.chain_map)
        predicted_cone_cells, predicted_cone_work = _cone_work_predictions(
            self.source, self.target
        )
        law_work = _chain_map_commutativity_work(
            self.source, self.target, self.chain_map
        )
        # Gate everything (both complexes, the chain map, and the predicted
        # cone differentials execution will emit) before any product runs.
        _require_aggregate_budget(
            (*self.source.differentials, *self.target.differentials, *self.chain_map),
            predicted_cone_cells,
            _d_squared_work(self.source.differentials)
            + _d_squared_work(self.target.differentials)
            + law_work
            + predicted_cone_work,
        )
        _require_chain_map_commutativity(self.source, self.target, self.chain_map)
        # The cone complex Cone(f)_n = C_{n-1} + D_n must itself stay inside
        # the representable degree range and per-group dimension budget, so
        # admission rejects here instead of the construction raising after
        # validation on an accepted request.
        _require_cone_within_domain(self.source, self.target)
        return self


class MappingConeResult(StrictModel):
    """The mapping cone complex of a chain map.

    Retains the full request (source and target complexes plus the chain map)
    so an authoritative cone replays the exact construction at validation; a
    relayed payload carrying any other complex cannot revalidate.
    """

    request: MappingConeRequest
    cone: ChainComplex

    @model_validator(mode="after")
    def require_source_bound_cone(self) -> Self:
        from jacobian.math.chain_complexes._operations import _build_cone_complex

        expected = _build_cone_complex(self.request)
        if self.cone != expected:
            raise ValueError(
                "cone must be the exact mapping cone of the retained "
                "source complexes and chain map"
            )
        return self


__all__ = [
    "MAX_AGGREGATE_CELLS",
    "MAX_AGGREGATE_ELIMINATION_WORK",
    "MAX_AGGREGATE_MULTIPLICATION_WORK",
    "MAX_CHAIN_DEGREES",
    "MAX_CHAIN_GROUP_DIMENSION",
    "ChainComplex",
    "HomologyGroup",
    "HomologyRequest",
    "HomologyResult",
    "MappingConeRequest",
    "MappingConeResult",
]
