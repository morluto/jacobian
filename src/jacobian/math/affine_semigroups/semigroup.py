"""Exact bounded positive affine-semigroup carriers and kernels."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import Annotated, Literal, Self

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
MAX_GRADING_INEQUALITIES = 200_000


def _err(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"affine_semigroup.{reason}", message)


AffineCell = Annotated[
    ExactInteger,
    Field(
        json_schema_extra={
            "maxLength": MAX_AFFINE_DIGITS + 1,
            "pattern": rf"^(?:0|-?[1-9][0-9]{{0,{MAX_AFFINE_DIGITS - 1}}})(?![\s\S])",
        }
    ),
]
AffineRow = Annotated[
    tuple[AffineCell, ...],
    Field(min_length=1, max_length=MAX_AFFINE_GENERATORS),
]


class AffineConfiguration(StrictModel):
    """A labelled integer matrix; columns are the generator axis."""

    row_labels: tuple[str, ...] = Field(min_length=1, max_length=MAX_AFFINE_ROWS)
    generator_labels: tuple[str, ...] = Field(
        min_length=1, max_length=MAX_AFFINE_GENERATORS
    )
    entries: tuple[AffineRow, ...] = Field(min_length=1, max_length=MAX_AFFINE_ROWS)

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


class AffineFactorization(StrictModel):
    coordinates: tuple[ExactInteger, ...]
    target: tuple[ExactInteger, ...]


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
        if self.factorizations != tuple(sorted(set(self.factorizations))):
            raise _err(
                "fiber_order", "fiber coordinates must be sorted and duplicate-free"
            )
        return self


class AffineMembershipResult(StrictModel):
    semigroup: PositiveAffineSemigroup
    target: tuple[ExactInteger, ...]
    cone_member: bool
    lattice_member: bool
    semigroup_member: bool
    factorization: tuple[ExactInteger, ...] | None = None


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


def _project_inequalities(
    inequalities: list[tuple[tuple[Fraction, ...], Fraction]],
) -> tuple[list[tuple[tuple[Fraction, ...], Fraction]], bool]:
    """Eliminate the first variable from ``c*x >= b`` inequalities.

    Fourier--Motzkin elimination is complete over the rationals.  Keeping the
    inequalities exact is important here: a failed small witness search is not
    evidence that a positive grading does not exist.
    """
    if not inequalities:
        return [], True
    dimension = len(inequalities[0][0])
    lower = []
    upper = []
    zero = []
    for coefficients, bound in inequalities:
        coefficient = coefficients[0]
        rest = coefficients[1:]
        if coefficient > 0:
            lower.append((tuple(-x / coefficient for x in rest), bound / coefficient))
        elif coefficient < 0:
            # x <= (bound - rest*y)/coefficient, represented as a lower
            # inequality for the projected slack (upper - lower >= 0).
            upper.append((tuple(-x / coefficient for x in rest), bound / coefficient))
        else:
            zero.append((rest, bound))
    projected_count = len(zero) + len(lower) * len(upper)
    if projected_count > MAX_GRADING_INEQUALITIES:
        raise OperationResourceAdmissionError(
            location=("configuration",),
            code="affine_semigroup.grading_work_bound",
            message=(
                "exact positive-grading elimination exceeds the "
                f"{MAX_GRADING_INEQUALITIES} inequality bound"
            ),
        )
    projected = list(zero)
    for lower_coefficients, lower_bound in lower:
        for upper_coefficients, upper_bound in upper:
            projected.append(
                (
                    tuple(
                        upper_coefficients[i] - lower_coefficients[i]
                        for i in range(dimension - 1)
                    ),
                    lower_bound - upper_bound,
                )
            )
    for coefficients, bound in projected:
        if not any(coefficients) and bound > 0:
            return projected, False
    return projected, True


def _solve_positive_grading(
    config: AffineConfiguration,
) -> tuple[Fraction, ...] | None:
    """Find a rational ``h`` with ``h*a_i >= 1`` or prove infeasibility."""
    inequalities = [
        (
            tuple(Fraction(config.entries[row][column]) for row in range(config.rows)),
            Fraction(1),
        )
        for column in range(config.columns)
    ]
    layers: list[list[tuple[tuple[Fraction, ...], Fraction]]] = [inequalities]
    current = inequalities
    for _ in range(config.rows):
        current, feasible = _project_inequalities(current)
        if not feasible:
            return None
        layers.append(current)
    if any(not coefficients and bound > 0 for coefficients, bound in current):
        return None

    # Reconstruct one exact point while reversing the eliminations.  At each
    # layer all lower bounds are below all upper bounds precisely when the
    # projected inequalities hold.
    suffix: tuple[Fraction, ...] = ()
    for layer_index in range(config.rows - 1, -1, -1):
        layer = layers[layer_index]
        lower_values: list[Fraction] = []
        upper_values: list[Fraction] = []
        for coefficients, bound in layer:
            coefficient = coefficients[0]
            rest = coefficients[1:]
            residual = sum(
                (c * x for c, x in zip(rest, suffix, strict=True)), Fraction(0)
            )
            if coefficient > 0:
                lower_values.append((bound - residual) / coefficient)
            elif coefficient < 0:
                upper_values.append((bound - residual) / coefficient)
            elif residual < bound:
                return None
        # A variable may be unrestricted in one direction.  Choosing zero
        # when only an upper bound exists incorrectly rejects valid systems
        # whose witness has a negative coordinate.
        if lower_values:
            lower = max(lower_values)
        elif upper_values:
            lower = min(upper_values)
        else:
            lower = Fraction(0)
        upper = min(upper_values, default=lower)
        if lower > upper:
            return None
        suffix = (lower, *suffix)
    return suffix


def _find_grading(config: AffineConfiguration) -> tuple[Fraction, ...] | None:
    # Strict positivity can be scaled to h*a_i >= 1.  The exact
    # Fourier--Motzkin solver is complete for this finite rational system.
    return _solve_positive_grading(config)


def _admit_semigroup(semigroup: PositiveAffineSemigroup) -> PositiveAffineSemigroup:
    """Re-establish the positive-semigroup invariant for native carriers."""
    try:
        # model_validate(instance) trusts an existing model.  Native callers
        # may supply model_construct() values, so reparse before touching the
        # grading or starting any target-derived work.
        return PositiveAffineSemigroup.model_validate(semigroup.model_dump())
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("semigroup",),
            code="affine_semigroup.invalid",
            message=str(exc),
        ) from exc


def _admit_fiber(
    semigroup: PositiveAffineSemigroup, target: tuple[int, ...]
) -> tuple[PositiveAffineSemigroup, tuple[int, ...]]:
    semigroup = _admit_semigroup(semigroup)
    if type(target) is not tuple or len(target) != semigroup.configuration.rows:
        raise OperationDomainValidationError(
            location=("target",),
            code="affine_semigroup.target_shape",
            message="target must be an exact tuple on the ambient row axis",
        )
    if any(type(value) is not int for value in target):
        raise OperationDomainValidationError(
            location=("target",),
            code="affine_semigroup.target_type",
            message="target coordinates must be exact integers",
        )
    h = tuple(value.as_fraction() for value in semigroup.grading)
    grades = tuple(
        sum(
            (
                h[r] * semigroup.configuration.entries[r][c]
                for r in range(semigroup.configuration.rows)
            ),
            Fraction(0),
        )
        for c in range(semigroup.configuration.columns)
    )
    target_grade = sum(
        (h[r] * target[r] for r in range(semigroup.configuration.rows)), Fraction(0)
    )
    if target_grade < 0:
        return semigroup, ()
    maxima = tuple(int(target_grade // grade) for grade in grades)
    candidate_count = 1
    for maximum in maxima:
        candidate_count *= maximum + 1
        if candidate_count > MAX_AFFINE_FIBER:
            raise OperationResourceAdmissionError(
                location=("target",),
                code="affine_semigroup.fiber_work_bound",
                message=(
                    "target-derived affine fiber enumeration exceeds the "
                    f"{MAX_AFFINE_FIBER} candidate bound"
                ),
            )
    return semigroup, maxima


def _fiber(
    semigroup: PositiveAffineSemigroup, target: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    config = semigroup.configuration
    h = tuple(value.as_fraction() for value in semigroup.grading)
    grades = [
        sum(h[r] * config.entries[r][c] for r in range(config.rows))
        for c in range(config.columns)
    ]
    target_grade = sum((h[r] * target[r] for r in range(config.rows)), Fraction(0))
    if target_grade < 0:
        return ()
    out: list[tuple[int, ...]] = []
    current = [0] * config.columns

    def visit(index: int, remaining: Fraction) -> None:
        if len(out) > MAX_AFFINE_FIBER:
            raise OverflowError("affine fiber exceeds exact materialization bound")
        if index == config.columns:
            if all(
                sum(current[c] * config.entries[r][c] for c in range(config.columns))
                == target[r]
                for r in range(config.rows)
            ):
                out.append(tuple(current))
            return
        max_value = int(remaining // grades[index]) if grades[index] > 0 else 0
        for value in range(max_value + 1):
            current[index] = value
            visit(index + 1, remaining - value * grades[index])
        current[index] = 0

    visit(0, target_grade)
    return tuple(sorted(set(out)))


def positive_grading(config: AffineConfiguration) -> PositiveGradingResult:
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
    return PositiveAffineSemigroup(configuration=config, grading=grading)


def fiber(semigroup: PositiveAffineSemigroup, target: tuple[int, ...]) -> AffineFiber:
    semigroup, _ = _admit_fiber(semigroup, target)
    rows = _fiber(semigroup, target)
    return AffineFiber(semigroup=semigroup, target=target, factorizations=rows)


def membership(
    semigroup: PositiveAffineSemigroup, target: tuple[int, ...]
) -> AffineMembershipResult:
    semigroup, _ = _admit_fiber(semigroup, target)
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
    "AffineMembershipResult",
    "PositiveAffineSemigroup",
    "PositiveGradingResult",
    "construct",
    "fiber",
    "membership",
    "positive_grading",
]
