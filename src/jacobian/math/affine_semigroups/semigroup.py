"""Exact bounded positive affine-semigroup carriers and kernels."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import gcd
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, ExactInteger
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)

MAX_AFFINE_ROWS = 8
MAX_AFFINE_GENERATORS = 10
MAX_AFFINE_DIGITS = 8
MAX_AFFINE_FIBER = 50_000
MAX_AFFINE_FIBER_WORK = MAX_AFFINE_FIBER
MAX_AFFINE_GRADING_ROWS = 100_000
MAX_AFFINE_GRAPH_MOVES = 16
MAX_AFFINE_GRAPH_EDGE_CHECKS = 2_000_000
MAX_AFFINE_GRAPH_EDGES = 100_000
MAX_HILBERT_BASIS_DETERMINANT = 1_000
MAX_HILBERT_BASIS_WORK = 1_010_000
MAX_AFFINE_NORMALIZATION_OUTPUT_DIGITS = 64
MAX_AFFINE_NORMALITY_CANDIDATES = MAX_HILBERT_BASIS_DETERMINANT + 1
MAX_AFFINE_NORMALITY_WORK = 2_000_000


def _err(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"affine_semigroup.{reason}", message)


class AffineConfiguration(StrictModel):
    """A labelled integer matrix; columns are the generator axis."""

    row_labels: tuple[str, ...]
    generator_labels: tuple[str, ...]
    entries: tuple[tuple[ExactInteger, ...], ...]

    @model_validator(mode="after")
    def _shape(self) -> Self:
        if not self.row_labels or len(self.row_labels) > MAX_AFFINE_ROWS:
            raise _err("rows", f"row axis must have 1..{MAX_AFFINE_ROWS} labels")
        if (
            not self.generator_labels
            or len(self.generator_labels) > MAX_AFFINE_GENERATORS
        ):
            raise _err(
                "generators",
                f"generator axis must have 1..{MAX_AFFINE_GENERATORS} labels",
            )
        if len(set(self.row_labels)) != len(self.row_labels) or len(
            set(self.generator_labels)
        ) != len(self.generator_labels):
            raise _err("axis_unique", "row and generator labels must be unique")
        if len(self.entries) != len(self.row_labels) or any(
            len(row) != len(self.generator_labels) for row in self.entries
        ):
            raise _err(
                "shape", "entries must cover the labelled row and generator axes"
            )
        if any(
            abs(value) >= 10**MAX_AFFINE_DIGITS for row in self.entries for value in row
        ):
            raise _err(
                "digits",
                f"configuration entries are limited to {MAX_AFFINE_DIGITS} digits",
            )
        return self

    @property
    def rows(self) -> int:
        return len(self.row_labels)

    @property
    def columns(self) -> int:
        return len(self.generator_labels)

    @property
    def columns_vectors(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(self.entries[row][col] for row in range(self.rows))
            for col in range(self.columns)
        )


class AffineHilbertBasis(StrictModel):
    """Hilbert basis of the configuration cone in its ambient integer lattice."""

    configuration: AffineConfiguration
    basis: tuple[tuple[ExactInteger, ExactInteger], ...] = Field(
        max_length=MAX_HILBERT_BASIS_DETERMINANT + 1
    )

    @model_validator(mode="after")
    def _basis_shape(self) -> Self:
        if self.configuration.rows != 2:
            raise _err("hilbert_basis_shape", "Hilbert basis values require two rows")
        if self.basis != tuple(sorted(set(self.basis))):
            raise _err("hilbert_basis_order", "basis vectors must be sorted and unique")
        return self


class PositiveAffineSemigroup(StrictModel):
    configuration: AffineConfiguration
    grading: tuple[CanonicalRational, ...]

    @model_validator(mode="after")
    def _positive(self) -> Self:
        if len(self.grading) != self.configuration.rows:
            raise _err("grading_shape", "grading must have one coordinate per row")
        for column in self.configuration.columns_vectors:
            if (
                sum(
                    (
                        self.grading[i].as_fraction() * column[i]
                        for i in range(len(column))
                    ),
                    Fraction(0),
                )
                <= 0
            ):
                raise _err(
                    "grading_not_positive",
                    "every generator must have strictly positive grading",
                )
        return self


class AffineSemigroupNormalization(StrictModel):
    """The normalization generators, in the source semigroup's ambient axes."""

    semigroup: PositiveAffineSemigroup
    generators: tuple[tuple[ExactInteger, ExactInteger], ...] = Field(
        max_length=MAX_HILBERT_BASIS_DETERMINANT + 1
    )

    @model_validator(mode="after")
    def _normalization_shape(self) -> Self:
        if self.semigroup.configuration.rows != 2:
            raise _err("normalization_shape", "normalization values require two rows")
        if self.generators != tuple(sorted(set(self.generators))):
            raise _err(
                "normalization_order",
                "normalization generators must be sorted and unique",
            )
        return self


class AffineSemigroupNormality(StrictModel):
    """Normality decision bound to a positive affine semigroup."""

    semigroup: PositiveAffineSemigroup
    normal: bool = Field(
        description=(
            "Whether the semigroup equals its cone intersected with its "
            "generated group lattice."
        )
    )
    hole: tuple[ExactInteger, ExactInteger] | None = Field(
        default=None,
        description=(
            "One element of cone(S) intersect gp(S) that is absent from S "
            "when normal is false; uses the retained ambient row axes."
        ),
    )

    @model_validator(mode="after")
    def _normality_shape(self) -> Self:
        if self.semigroup.configuration.rows != 2:
            raise _err("normality_shape", "normality values require two rows")
        if self.normal == (self.hole is not None):
            raise _err(
                "normality_witness",
                "normality is true exactly when no normalization hole is returned",
            )
        return self


class AffineFactorization(StrictModel):
    coordinates: tuple[ExactInteger, ...]
    target: tuple[ExactInteger, ...]


def _factorization_matches(
    semigroup: PositiveAffineSemigroup,
    target: tuple[int, ...],
    factorization: tuple[int, ...],
) -> bool:
    if (
        len(target) != semigroup.configuration.rows
        or len(factorization) != semigroup.configuration.columns
    ):
        return False
    return all(
        sum(
            factorization[column] * semigroup.configuration.entries[row][column]
            for column in range(semigroup.configuration.columns)
        )
        == target[row]
        for row in range(semigroup.configuration.rows)
    )


class AffineFiber(StrictModel):
    semigroup: PositiveAffineSemigroup
    target: tuple[ExactInteger, ...]
    factorizations: tuple[tuple[ExactInteger, ...], ...]
    complete: Literal[True] = True

    @model_validator(mode="after")
    def _canonical(self) -> Self:
        if len(self.target) != self.semigroup.configuration.rows:
            raise _err("target_shape", "target must use the ambient row axis")
        n = self.semigroup.configuration.columns
        if any(len(row) != n or any(x < 0 for x in row) for row in self.factorizations):
            raise _err(
                "fiber_coordinates",
                "fiber coordinates must be nonnegative on the generator axis",
            )
        if any(
            not _factorization_matches(self.semigroup, self.target, row)
            for row in self.factorizations
        ):
            raise _err(
                "fiber_relation",
                "every factorization must evaluate to the retained target",
            )
        if self.factorizations != tuple(sorted(set(self.factorizations))):
            raise _err(
                "fiber_order", "fiber coordinates must be sorted and duplicate-free"
            )
        return self


class AffineFiberGraph(StrictModel):
    """The graph induced on one complete fiber by an explicit relation set.

    Vertices are factorization vectors. Edges are pairs of indices into the
    sorted vertex tuple, so the result retains the labeled generator axis once.
    This value describes this fiber only; it does not assert that the move set
    connects every fiber of the semigroup.
    """

    semigroup: PositiveAffineSemigroup
    target: tuple[ExactInteger, ...]
    moves: tuple[tuple[ExactInteger, ...], ...]
    vertices: tuple[tuple[ExactInteger, ...], ...]
    edges: tuple[tuple[int, int], ...]
    components: tuple[tuple[int, ...], ...]

    @model_validator(mode="after")
    def _graph_shape(self) -> Self:
        n_vertices = len(self.vertices)
        configuration = self.semigroup.configuration
        if self.moves != tuple(sorted(set(self.moves))):
            raise _err("graph_moves", "fiber graph moves must be sorted and unique")
        for move in self.moves:
            if (
                len(move) != configuration.columns
                or not any(move)
                or next(value for value in move if value) < 0
                or any(
                    sum(
                        configuration.entries[row][column] * move[column]
                        for column in range(configuration.columns)
                    )
                    for row in range(configuration.rows)
                )
            ):
                raise _err(
                    "graph_moves", "moves must be normalized nonzero kernel vectors"
                )
        if len(self.moves) > MAX_AFFINE_GRAPH_MOVES:
            raise _err("graph_move_bound", "fiber graph exceeds its move envelope")
        if self.vertices != tuple(sorted(set(self.vertices))):
            raise _err(
                "graph_vertices", "fiber graph vertices must be sorted and unique"
            )
        if any(
            len(vertex) != self.semigroup.configuration.columns
            for vertex in self.vertices
        ):
            raise _err(
                "graph_vertex_axis", "fiber graph vertices must use the generator axis"
            )
        if self.edges != tuple(sorted(set(self.edges))):
            raise _err("graph_edges", "fiber graph edges must be sorted and unique")
        if any(not (0 <= left < right < n_vertices) for left, right in self.edges):
            raise _err(
                "graph_edge_indices", "fiber graph edges must index distinct vertices"
            )
        if self.components != tuple(
            sorted(self.components, key=lambda part: part[0] if part else -1)
        ):
            raise _err(
                "graph_components", "fiber graph components must be canonically ordered"
            )
        flattened = tuple(index for component in self.components for index in component)
        if any(
            not component or component != tuple(sorted(set(component)))
            for component in self.components
        ) or tuple(sorted(flattened)) != tuple(range(n_vertices)):
            raise _err(
                "graph_components", "components must partition the vertex indices"
            )
        if len(self.edges) > MAX_AFFINE_GRAPH_EDGES:
            raise _err("graph_edge_bound", "fiber graph exceeds its edge envelope")
        return self


class AffineMembershipResult(StrictModel):
    semigroup: PositiveAffineSemigroup
    target: tuple[ExactInteger, ...]
    cone_member: bool
    lattice_member: bool
    semigroup_member: bool
    factorization: tuple[ExactInteger, ...] | None = None

    @model_validator(mode="after")
    def _membership_binding(self) -> Self:
        if len(self.target) != self.semigroup.configuration.rows:
            raise _err("target_shape", "target must use the ambient row axis")
        if self.factorization is not None and (
            any(value < 0 for value in self.factorization)
            or not _factorization_matches(
                self.semigroup, self.target, self.factorization
            )
        ):
            raise _err(
                "membership_factorization",
                "membership factorization must evaluate to the retained target",
            )
        if self.semigroup_member != (self.factorization is not None):
            raise _err(
                "membership_binding",
                "semigroup membership must agree with the retained factorization",
            )
        return self


class PositiveGradingResult(StrictModel):
    configuration: AffineConfiguration
    grading: tuple[CanonicalRational, ...]
    positive: bool


def _rank_solve(
    matrix: list[list[Fraction]], rhs: list[Fraction]
) -> tuple[Fraction, ...] | None:
    m, n = len(matrix), (len(matrix[0]) if matrix else 0)
    a = [[*list(row), rhs[i]] for i, row in enumerate(matrix)]
    pivot = 0
    pivots: list[int] = []
    for col in range(n):
        found = next((r for r in range(pivot, m) if a[r][col]), None)
        if found is None:
            continue
        a[pivot], a[found] = a[found], a[pivot]
        q = a[pivot][col]
        a[pivot] = [x / q for x in a[pivot]]
        for r in range(m):
            if r != pivot and a[r][col]:
                q = a[r][col]
                a[r] = [x - q * y for x, y in zip(a[r], a[pivot], strict=True)]
        pivots.append(col)
        pivot += 1
    if any(all(a[r][c] == 0 for c in range(n)) and a[r][-1] != 0 for r in range(m)):
        return None
    if len(pivots) != n:
        return None
    return tuple(a[i][-1] for i in range(n))


def _cone_member(
    config: AffineConfiguration, target: tuple[int, ...]
) -> tuple[Fraction, ...] | None:
    d, n = config.rows, config.columns
    vectors = config.columns_vectors
    for size in range(1, min(d, n) + 1):
        for indices in combinations(range(n), size):
            # solve the selected columns against every row subset
            rowsets = combinations(range(d), size) if size < d else (tuple(range(d)),)
            for rows in rowsets:
                coeff = _rank_solve(
                    [[Fraction(vectors[j][r]) for j in indices] for r in rows],
                    [Fraction(target[r]) for r in rows],
                )
                if coeff is None or any(x < 0 for x in coeff):
                    continue
                if all(
                    sum(coeff[k] * vectors[indices[k]][r] for k in range(size))
                    == target[r]
                    for r in range(d)
                ):
                    full = [Fraction(0)] * n
                    for k, j in enumerate(indices):
                        full[j] = coeff[k]
                    return tuple(full)
    return None


def _lattice_member(config: AffineConfiguration, target: tuple[int, ...]) -> bool:
    from sympy import Matrix
    from sympy.matrices.normalforms import hermite_normal_form

    matrix = Matrix([[int(x) for x in row] for row in config.entries])
    hnf = hermite_normal_form(matrix)
    if hnf.rows == 0 or hnf.cols == 0:
        return all(x == 0 for x in target)
    coeff = _rank_solve(
        [[Fraction(hnf[r, c]) for c in range(hnf.cols)] for r in range(hnf.rows)],
        [Fraction(x) for x in target],
    )
    return coeff is not None and all(x.denominator == 1 for x in coeff)


class _AffineGradingWorkExceededError(Exception):
    pass


def _fourier_motzkin_witness(
    rows: list[tuple[tuple[Fraction, ...], Fraction]], dimension: int
) -> tuple[Fraction, ...] | None:
    """Decide exact rational inequalities and recover one witness."""
    if dimension == 0:
        return () if all(rhs <= 0 for _, rhs in rows) else None
    positive: list[tuple[tuple[Fraction, ...], Fraction]] = []
    negative: list[tuple[tuple[Fraction, ...], Fraction]] = []
    zero: list[tuple[tuple[Fraction, ...], Fraction]] = []
    for coeff, rhs in rows:
        if coeff[0] > 0:
            positive.append((coeff, rhs))
        elif coeff[0] < 0:
            negative.append((coeff, rhs))
        else:
            zero.append((coeff[1:], rhs))
    next_rows = list(zero)
    for lower, lower_rhs in positive:
        p = lower[0]
        for upper, upper_rhs in negative:
            n = upper[0]
            next_rows.append(
                (
                    tuple(
                        -n * a + p * b
                        for a, b in zip(lower[1:], upper[1:], strict=True)
                    ),
                    -n * lower_rhs + p * upper_rhs,
                )
            )
            if len(next_rows) > MAX_AFFINE_GRADING_ROWS:
                raise _AffineGradingWorkExceededError
    tail = _fourier_motzkin_witness(next_rows, dimension - 1)
    if tail is None:
        return None
    lowers = [
        (rhs - sum(a * x for a, x in zip(coeff[1:], tail, strict=True))) / coeff[0]
        for coeff, rhs in positive
    ]
    uppers = [
        (rhs - sum(a * x for a, x in zip(coeff[1:], tail, strict=True))) / coeff[0]
        for coeff, rhs in negative
    ]
    lo = max(lowers) if lowers else None
    hi = min(uppers) if uppers else None
    if lo is not None and hi is not None and lo > hi:
        return None
    chosen = lo if hi is None else hi if lo is None else lo
    return (chosen if chosen is not None else Fraction(0), *tail)


def _find_grading(config: AffineConfiguration) -> tuple[Fraction, ...] | None:
    # Strict positivity is equivalent after rescaling to A^T h >= 1.
    rows = [
        (tuple(Fraction(config.entries[r][c]) for r in range(config.rows)), Fraction(1))
        for c in range(config.columns)
    ]
    try:
        witness = _fourier_motzkin_witness(rows, config.rows)
    except _AffineGradingWorkExceededError as exc:
        raise OperationResourceAdmissionError(
            location=("configuration",),
            code="affine_semigroup.grading_work",
            message="positive-grading feasibility exceeds its work bound",
        ) from exc
    if witness is None:
        return None
    common = 1
    for value in witness:
        common = gcd(common, value.denominator)
    integral = tuple(value * common for value in witness)
    g = 0
    for value in integral:
        g = gcd(g, abs(value.numerator))
    return tuple(Fraction(value / g) for value in integral) if g else tuple(integral)


def _validate_target(
    semigroup: PositiveAffineSemigroup, target: tuple[int, ...]
) -> None:
    if type(target) is not tuple or any(type(value) is not int for value in target):
        raise OperationDomainValidationError(
            location=("target",),
            code="affine_semigroup.target_type",
            message="target must be a tuple of exact integers",
        )
    if len(target) != semigroup.configuration.rows:
        raise OperationDomainValidationError(
            location=("target",),
            code="affine_semigroup.target_shape",
            message="target must match the row axis",
        )


def _admit_fiber(
    semigroup: PositiveAffineSemigroup, target: tuple[int, ...]
) -> tuple[tuple[Fraction, ...], Fraction, tuple[int, ...]]:
    _validate_target(semigroup, target)
    config = semigroup.configuration
    h = tuple(value.as_fraction() for value in semigroup.grading)
    grades = tuple(
        sum((h[r] * config.entries[r][c] for r in range(config.rows)), Fraction(0))
        for c in range(config.columns)
    )
    if any(grade <= 0 for grade in grades):
        raise OperationDomainValidationError(
            location=("semigroup",),
            code="affine_semigroup.grading_not_positive",
            message="every generator must have strictly positive grading",
        )
    target_grade = sum((h[r] * target[r] for r in range(config.rows)), Fraction(0))
    if target_grade < 0:
        return grades, target_grade, tuple(0 for _ in grades)
    maxima = tuple(int(target_grade // grade) for grade in grades)
    work = 1
    for maximum in maxima:
        factor = maximum + 1
        if factor > MAX_AFFINE_FIBER_WORK // work:
            raise OperationResourceAdmissionError(
                location=("target",),
                code="affine_semigroup.fiber_work",
                message=f"target-derived affine fiber work exceeds the {MAX_AFFINE_FIBER_WORK}-state envelope",
            )
        work *= factor
    return grades, target_grade, maxima


def _fiber(
    semigroup: PositiveAffineSemigroup, target: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    grades, target_grade, maxima = _admit_fiber(semigroup, target)
    return _enumerate_fiber(semigroup, target, grades, target_grade, maxima)


def _enumerate_fiber(
    semigroup: PositiveAffineSemigroup,
    target: tuple[int, ...],
    grades: tuple[Fraction, ...],
    target_grade: Fraction,
    maxima: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    if target_grade < 0:
        return ()
    config = semigroup.configuration
    out: list[tuple[int, ...]] = []
    current = [0] * config.columns

    def visit(index: int, remaining: Fraction) -> None:
        if index == config.columns:
            if all(
                sum(current[c] * config.entries[r][c] for c in range(config.columns))
                == target[r]
                for r in range(config.rows)
            ):
                out.append(tuple(current))
            return
        max_value = min(maxima[index], int(remaining // grades[index]))
        for value in range(max_value + 1):
            current[index] = value
            visit(index + 1, remaining - value * grades[index])
        current[index] = 0

    visit(0, target_grade)
    return tuple(sorted(set(out)))


def _admit_configuration(value: object) -> AffineConfiguration:
    if type(value) is not AffineConfiguration:
        raise OperationDomainValidationError(
            location=("configuration",),
            code="affine_semigroup.configuration",
            message="configuration must be a canonical affine configuration",
        )
    try:
        return AffineConfiguration.model_validate(value.model_dump(mode="python"))
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("configuration",),
            code="affine_semigroup.configuration",
            message="configuration is malformed",
        ) from exc


def _admit_semigroup(value: object) -> PositiveAffineSemigroup:
    if type(value) is not PositiveAffineSemigroup:
        raise OperationDomainValidationError(
            location=("semigroup",),
            code="affine_semigroup.semigroup",
            message="semigroup must be a canonical positive affine semigroup",
        )
    try:
        return PositiveAffineSemigroup.model_validate(value.model_dump(mode="python"))
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("semigroup",),
            code="affine_semigroup.semigroup",
            message="semigroup is malformed",
        ) from exc


def positive_grading(config: AffineConfiguration) -> PositiveGradingResult:
    config = _admit_configuration(config)
    grading = _find_grading(config)
    if grading is None:
        return PositiveGradingResult(configuration=config, grading=(), positive=False)
    return PositiveGradingResult(
        configuration=config,
        grading=tuple(CanonicalRational.from_fraction(x) for x in grading),
        positive=True,
    )


def construct(
    config: AffineConfiguration, grading: tuple[CanonicalRational, ...]
) -> PositiveAffineSemigroup:
    config = _admit_configuration(config)
    if type(grading) is not tuple or any(
        type(x) is not CanonicalRational for x in grading
    ):
        raise OperationDomainValidationError(
            location=("grading",),
            code="affine_semigroup.grading",
            message="grading must be a tuple of canonical rationals",
        )
    try:
        return PositiveAffineSemigroup(configuration=config, grading=grading)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("grading",),
            code="affine_semigroup.grading",
            message="grading is malformed or not positive on every generator",
        ) from exc


def _hilbert_rays(
    configuration: AffineConfiguration,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return primitive, positively oriented extreme rays of a 2D cone."""
    if (
        type(configuration.rows) is not int
        or configuration.rows != 2
        or not (1 <= len(configuration.generator_labels) <= MAX_AFFINE_GENERATORS)
        or len(configuration.entries) != 2
        or any(
            len(row) != len(configuration.generator_labels)
            for row in configuration.entries
        )
    ):
        raise ValueError("Hilbert bases require a canonical two-row configuration")
    if any(
        type(value) is not int or abs(value) >= 10**MAX_AFFINE_DIGITS
        for row in configuration.entries
        for value in row
    ):
        raise ValueError("configuration entries must be bounded exact integers")

    vectors = tuple(
        (configuration.entries[0][i], configuration.entries[1][i])
        for i in range(len(configuration.generator_labels))
    )
    primitive: set[tuple[int, int]] = set()
    for x, y in vectors:
        divisor = gcd(abs(x), abs(y))
        if divisor == 0:
            raise ValueError("zero generators do not define a pointed cone")
        primitive.add((x // divisor, y // divisor))

    rays = tuple(sorted(primitive))
    pairs = []
    for u in rays:
        for v in rays:
            if u[0] * v[1] - u[1] * v[0] <= 0:
                continue
            if all(
                u[0] * y - u[1] * x >= 0 and x * v[1] - y * v[0] >= 0
                for x, y in vectors
            ):
                pairs.append((u, v))
    if len(pairs) != 1:
        raise ValueError("generators must span a full-dimensional pointed cone in Z^2")
    return pairs[0]


def _hilbert_basis_admitted(
    configuration: AffineConfiguration,
    rays: tuple[tuple[int, int], tuple[int, int]],
) -> AffineHilbertBasis:
    """Compute a Hilbert basis from configuration and rays already admitted."""
    """Compute the complete Hilbert basis of a bounded pointed 2D cone.

    Every indecomposable lattice point other than a primitive boundary ray lies
    in the semi-open fundamental parallelogram of the two extreme rays. Its
    index is the ray determinant, so this operation examines a finite admitted
    set and tests decomposability in increasing positive cone grading.
    """
    u, v = rays
    determinant = u[0] * v[1] - u[1] * v[0]
    if determinant > MAX_HILBERT_BASIS_DETERMINANT:
        raise OperationResourceAdmissionError(
            location=("configuration",),
            code="affine_semigroup.hilbert_basis_determinant",
            message=(
                f"ray determinant {determinant} exceeds the Hilbert-basis limit "
                f"{MAX_HILBERT_BASIS_DETERMINANT}"
            ),
        )
    work = determinant + (determinant + 1) ** 2
    if work > MAX_HILBERT_BASIS_WORK:
        raise OperationResourceAdmissionError(
            location=("configuration",),
            code="affine_semigroup.hilbert_basis_work",
            message=f"Hilbert-basis work estimate {work} exceeds {MAX_HILBERT_BASIS_WORK}",
        )

    # The lattice generated by u and v has lower-triangular column-Hermite
    # basis ((a,b),(0,c)). Its D coset representatives are a complete set.
    a = gcd(abs(u[0]), abs(v[0]))
    if a == 0:
        raise ValueError("the two extreme rays cannot both be vertical")
    c = determinant // a
    candidates: dict[tuple[int, int], int] = {}
    for x in range(a):
        for y in range(c):
            alpha_num = v[1] * x - v[0] * y
            beta_num = -u[1] * x + u[0] * y
            alpha_floor = alpha_num // determinant
            beta_floor = beta_num // determinant
            point = (
                x - alpha_floor * u[0] - beta_floor * v[0],
                y - alpha_floor * u[1] - beta_floor * v[1],
            )
            if point != (0, 0):
                candidates[point] = alpha_num % determinant + beta_num % determinant
    candidates[u] = determinant
    candidates[v] = determinant

    accepted: list[tuple[int, int]] = []
    for point, _weight in sorted(
        candidates.items(), key=lambda item: (item[1], item[0])
    ):
        decomposable = False
        for summand in accepted:
            remainder = (point[0] - summand[0], point[1] - summand[1])
            if remainder == (0, 0):
                continue
            if (
                remainder[0] * v[1] - remainder[1] * v[0] >= 0
                and u[0] * remainder[1] - u[1] * remainder[0] >= 0
            ):
                decomposable = True
                break
        if not decomposable:
            accepted.append(point)
    return AffineHilbertBasis(
        configuration=configuration, basis=tuple(sorted(accepted))
    )


def hilbert_basis(configuration: AffineConfiguration) -> AffineHilbertBasis:
    """Compute the complete Hilbert basis of a bounded pointed 2D cone."""
    configuration = _admit_configuration(configuration)
    if configuration.rows != 2:
        raise ValueError("Hilbert bases require a two-row configuration")
    return _hilbert_basis_admitted(configuration, _hilbert_rays(configuration))


def normalization(semigroup: PositiveAffineSemigroup) -> AffineSemigroupNormalization:
    """Compute ``cone(S) intersect gp(S)`` for a full-rank 2D semigroup.

    The generated group is first put in a canonical integer basis. In those
    coordinates the normalization is the ordinary two-dimensional cone
    Hilbert basis; its generators are then transported back to the retained
    ambient row axis.
    """
    semigroup = _admit_semigroup(semigroup)
    configuration = semigroup.configuration
    if configuration.rows != 2 or configuration.columns < 2:
        raise ValueError("normalization currently requires a two-row configuration")

    maximum = max(abs(value) for row in configuration.entries for value in row)
    minor_bound = max(
        1, 2 * (configuration.columns * (configuration.columns - 1) // 2) * maximum**2
    )
    if len(str(minor_bound)) > MAX_AFFINE_NORMALIZATION_OUTPUT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("semigroup", "configuration"),
            code="affine_semigroup.normalization_intermediate_digits",
            message="column-lattice intermediate bound exceeds the normalization digit envelope",
        )

    from sympy import Matrix
    from sympy.matrices.normalforms import hermite_normal_form

    matrix = Matrix([[int(value) for value in row] for row in configuration.entries])
    hnf = hermite_normal_form(matrix)
    if hnf.shape != (2, 2):
        raise ValueError("normalization requires a full-rank generated lattice in Z^2")
    a, b = int(hnf[0, 0]), int(hnf[0, 1])
    c, d = int(hnf[1, 0]), int(hnf[1, 1])
    determinant = a * d - b * c
    if determinant == 0:
        raise ValueError("normalization requires a full-rank generated lattice in Z^2")

    coordinate_columns: list[tuple[int, int]] = []
    for x, y in configuration.columns_vectors:
        numerators = (d * x - b * y, -c * x + a * y)
        if any(value % determinant for value in numerators):
            raise ArithmeticError("column-lattice HNF failed to contain a generator")
        coordinate_columns.append(
            (
                numerators[0] // determinant,
                numerators[1] // determinant,
            )
        )

    coordinate_maximum = max(
        abs(value) for vector in coordinate_columns for value in vector
    )
    if coordinate_maximum >= 10**MAX_AFFINE_DIGITS:
        raise OperationResourceAdmissionError(
            location=("semigroup", "configuration"),
            code="affine_semigroup.normalization_coordinate_digits",
            message="generated-lattice coordinates exceed the 8-digit cone envelope",
        )
    coordinate_configuration = AffineConfiguration(
        row_labels=configuration.row_labels,
        generator_labels=configuration.generator_labels,
        entries=tuple(
            tuple(
                coordinate_columns[column][row]
                for column in range(configuration.columns)
            )
            for row in range(2)
        ),
    )
    rays = _hilbert_rays(coordinate_configuration)
    ray_determinant = rays[0][0] * rays[1][1] - rays[0][1] * rays[1][0]
    if ray_determinant > MAX_HILBERT_BASIS_DETERMINANT:
        raise OperationResourceAdmissionError(
            location=("semigroup", "configuration"),
            code="affine_semigroup.normalization_hilbert_determinant",
            message=(
                f"normalization Hilbert determinant {ray_determinant} exceeds "
                f"the limit {MAX_HILBERT_BASIS_DETERMINANT}"
            ),
        )
    # Hilbert points lie in the cone spanned by primitive input rays. Bound
    # their ambient coordinates before running the finite parallelogram search.
    coordinate_bound = max(abs(value) for ray in rays for value in ray)
    output_bound = 4 * minor_bound * max(1, coordinate_bound)
    if len(str(output_bound)) > MAX_AFFINE_NORMALIZATION_OUTPUT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("semigroup", "configuration"),
            code="affine_semigroup.normalization_output_digits",
            message="normalization output bound exceeds the 64-digit result envelope",
        )

    basis = _hilbert_basis_admitted(coordinate_configuration, rays).basis
    transported = tuple(sorted({(a * x + b * y, c * x + d * y) for x, y in basis}))
    if any(
        len(str(abs(value))) > MAX_AFFINE_NORMALIZATION_OUTPUT_DIGITS
        for v in transported
        for value in v
    ):
        raise ArithmeticError("normalization output exceeded its admitted digit bound")
    return AffineSemigroupNormalization(semigroup=semigroup, generators=transported)


def _fiber_has_member(
    semigroup: PositiveAffineSemigroup,
    target: tuple[int, ...],
    grades: tuple[int, ...],
    target_grade: int,
    maxima: tuple[int, ...],
) -> bool:
    """Search one already-admitted fiber and stop at its first factorization."""
    if target_grade < 0:
        return False
    configuration = semigroup.configuration
    coordinates = [0] * configuration.columns

    def visit(index: int, remaining: int) -> bool:
        if index == configuration.columns:
            return all(
                sum(
                    coordinates[column] * configuration.entries[row][column]
                    for column in range(configuration.columns)
                )
                == target[row]
                for row in range(configuration.rows)
            )
        maximum = min(maxima[index], int(remaining // grades[index]))
        for value in range(maximum + 1):
            coordinates[index] = value
            if visit(index + 1, remaining - value * grades[index]):
                return True
        coordinates[index] = 0
        return False

    return visit(0, target_grade)


def normality(semigroup: PositiveAffineSemigroup) -> AffineSemigroupNormality:
    """Decide normality for a full-rank, positive two-dimensional semigroup.

    The complete Hilbert basis of ``cone(S) intersect gp(S)`` generates the
    normalization. Testing its elements in ``S`` proves equality in both
    directions: ``S`` is contained in its normalization, and membership of
    every normalization generator gives the reverse inclusion. A missing
    generator is returned as an exact hole.
    """
    normalized = normalization(semigroup)
    source = normalized.semigroup
    candidates = normalized.generators
    if len(candidates) > MAX_AFFINE_NORMALITY_CANDIDATES:
        raise OperationResourceAdmissionError(
            location=("semigroup",),
            code="affine_semigroup.normality_candidate_bound",
            message="normalization Hilbert basis exceeds the normality candidate bound",
        )

    # An integral covector formed from the two extreme rays is strictly
    # positive on every generator. It bounds coefficients without using the
    # caller's potentially much larger rational grading.
    lower, upper = _hilbert_rays(source.configuration)
    grading = (upper[1] - lower[1], lower[0] - upper[0])
    grades = tuple(
        sum(
            grading[row] * source.configuration.entries[row][column]
            for row in range(source.configuration.rows)
        )
        for column in range(source.configuration.columns)
    )
    if any(grade <= 0 for grade in grades):
        raise ArithmeticError("extreme-ray covector is not positive on the semigroup")

    # Admit all candidate coefficient boxes and the worst-case exact
    # cone/lattice witness checks before starting any coefficient traversal.
    plans: list[tuple[int, tuple[int, ...], int]] = []
    total_work = len(candidates) * (
        source.configuration.columns**2
        + source.configuration.rows * source.configuration.columns
        + 2
    )
    for target in candidates:
        target_grade = sum(
            grading[row] * target[row] for row in range(source.configuration.rows)
        )
        if target_grade < 0:
            raise ArithmeticError("normalization generator lies outside the cone")
        maxima = tuple(target_grade // grade for grade in grades)
        candidate_work = 1
        for maximum in maxima:
            candidate_work *= maximum + 1
            if candidate_work > MAX_AFFINE_FIBER_WORK:
                raise OperationResourceAdmissionError(
                    location=("semigroup",),
                    code="affine_semigroup.normality_candidate_fiber_bound",
                    message=(
                        "a normalization generator coefficient box exceeds "
                        f"the {MAX_AFFINE_FIBER_WORK}-tuple candidate bound"
                    ),
                )
        candidate_work *= (
            source.configuration.columns
            + 1
            + source.configuration.rows * source.configuration.columns
        )
        total_work += candidate_work
        if total_work > MAX_AFFINE_NORMALITY_WORK:
            raise OperationResourceAdmissionError(
                location=("semigroup",),
                code="affine_semigroup.normality_work_bound",
                message=(
                    "normality membership and witness checks exceed the "
                    f"{MAX_AFFINE_NORMALITY_WORK}-unit work envelope"
                ),
            )
        plans.append((target_grade, maxima, candidate_work))

    for target, (target_grade, maxima, _candidate_work) in zip(
        candidates, plans, strict=True
    ):
        if _fiber_has_member(source, target, grades, target_grade, maxima):
            continue
        # The normalization kernel should produce only elements in both the
        # generated cone and lattice. Replay those defining relations for the
        # returned nonnormality witness.
        if _cone_member(source.configuration, target) is None or not _lattice_member(
            source.configuration, target
        ):
            raise ArithmeticError(
                "normalization output failed its cone or generated-lattice invariant"
            )
        return AffineSemigroupNormality(semigroup=source, normal=False, hole=target)
    return AffineSemigroupNormality(semigroup=source, normal=True, hole=None)


def fiber(semigroup: PositiveAffineSemigroup, target: tuple[int, ...]) -> AffineFiber:
    semigroup = _admit_semigroup(semigroup)
    rows = _fiber(semigroup, target)
    try:
        return AffineFiber(semigroup=semigroup, target=target, factorizations=rows)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("target",),
            code="affine_semigroup.fiber",
            message="fiber result is malformed",
        ) from exc


def _canonical_graph_moves(
    semigroup: PositiveAffineSemigroup, moves: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int, ...], ...]:
    config = semigroup.configuration
    if type(moves) is not tuple:
        raise OperationDomainValidationError(
            location=("moves",),
            code="affine_semigroup.graph_move_shape",
            message="moves must be a tuple of exact integer vectors",
        )
    if len(moves) > MAX_AFFINE_GRAPH_MOVES:
        raise OperationResourceAdmissionError(
            location=("moves",),
            code="affine_semigroup.graph_move_count",
            message=f"at most {MAX_AFFINE_GRAPH_MOVES} moves are admitted",
        )
    canonical_moves: set[tuple[int, ...]] = set()
    for move in moves:
        if (
            type(move) is not tuple
            or len(move) != config.columns
            or any(type(value) is not int for value in move)
        ):
            raise OperationDomainValidationError(
                location=("moves",),
                code="affine_semigroup.graph_move_shape",
                message="each move must be an exact integer vector on the generator axis",
            )
        if not any(move):
            raise OperationDomainValidationError(
                location=("moves",),
                code="affine_semigroup.graph_zero_move",
                message="zero is not a fiber-graph move",
            )
        if any(
            sum(
                config.entries[row][column] * move[column]
                for column in range(config.columns)
            )
            for row in range(config.rows)
        ):
            raise OperationDomainValidationError(
                location=("moves",),
                code="affine_semigroup.graph_move_not_relation",
                message="each move must be an exact integer relation of the configuration",
            )
        sign = next(value for value in move if value)
        canonical_moves.add(move if sign > 0 else tuple(-value for value in move))
    return tuple(sorted(canonical_moves))


def _graph_components(
    vertex_count: int, edges: set[tuple[int, int]]
) -> tuple[tuple[int, ...], ...]:
    adjacency: list[set[int]] = [set() for _ in range(vertex_count)]
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen: set[int] = set()
    components: list[tuple[int, ...]] = []
    for start in range(vertex_count):
        if start in seen:
            continue
        seen.add(start)
        stack = [start]
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in adjacency[current] - seen:
                seen.add(neighbour)
                stack.append(neighbour)
        components.append(tuple(sorted(component)))
    return tuple(components)


def fiber_graph(
    semigroup: PositiveAffineSemigroup,
    target: tuple[int, ...],
    moves: tuple[tuple[int, ...], ...],
) -> AffineFiberGraph:
    """Return the graph induced on one complete fiber by supplied kernel moves.

    The operation validates every move against the retained configuration,
    admits the fiber and maximum edge work before enumeration, then returns the
    undirected graph and its connected components. A disconnected result is
    only about this target fiber and this move set.
    """
    semigroup = _admit_semigroup(semigroup)
    ordered_moves = _canonical_graph_moves(semigroup, moves)
    grades, target_grade, maxima = _admit_fiber(semigroup, target)
    candidate_count = 1
    for maximum in maxima:
        candidate_count *= maximum + 1
    edge_checks = candidate_count * len(ordered_moves)
    if edge_checks > MAX_AFFINE_GRAPH_EDGE_CHECKS:
        raise OperationResourceAdmissionError(
            location=("target",),
            code="affine_semigroup.graph_work",
            message=(
                "fiber graph edge work exceeds the "
                f"{MAX_AFFINE_GRAPH_EDGE_CHECKS}-check envelope"
            ),
        )
    if candidate_count * len(ordered_moves) > MAX_AFFINE_GRAPH_EDGES:
        raise OperationResourceAdmissionError(
            location=("target",),
            code="affine_semigroup.graph_output",
            message=(
                "worst-case fiber graph exceeds the "
                f"{MAX_AFFINE_GRAPH_EDGES}-edge output envelope"
            ),
        )
    vertices = _enumerate_fiber(semigroup, target, grades, target_grade, maxima)
    positions = {vertex: index for index, vertex in enumerate(vertices)}
    edges: set[tuple[int, int]] = set()
    for source_index, vertex in enumerate(vertices):
        for move in ordered_moves:
            neighbour = tuple(a + b for a, b in zip(vertex, move, strict=True))
            neighbour_index = positions.get(neighbour)
            if neighbour_index is not None and source_index < neighbour_index:
                edges.add((source_index, neighbour_index))
    return AffineFiberGraph(
        semigroup=semigroup,
        target=target,
        moves=ordered_moves,
        vertices=vertices,
        edges=tuple(sorted(edges)),
        components=_graph_components(len(vertices), edges),
    )


def membership(
    semigroup: PositiveAffineSemigroup, target: tuple[int, ...]
) -> AffineMembershipResult:
    semigroup = _admit_semigroup(semigroup)
    _validate_target(semigroup, target)
    cone = _cone_member(semigroup.configuration, target) is not None
    lattice = _lattice_member(semigroup.configuration, target)
    rows = _fiber(semigroup, target) if cone and lattice else ()
    return AffineMembershipResult(
        semigroup=semigroup,
        target=target,
        cone_member=cone,
        lattice_member=lattice,
        semigroup_member=bool(rows),
        factorization=rows[0] if rows else None,
    )


__all__ = [
    "AffineConfiguration",
    "AffineFactorization",
    "AffineFiber",
    "AffineFiberGraph",
    "AffineMembershipResult",
    "AffineSemigroupNormality",
    "AffineSemigroupNormalization",
    "PositiveAffineSemigroup",
    "PositiveGradingResult",
    "construct",
    "fiber",
    "fiber_graph",
    "membership",
    "normality",
    "normalization",
    "positive_grading",
]
