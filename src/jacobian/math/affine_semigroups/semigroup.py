"""Exact bounded positive affine-semigroup carriers and kernels."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import gcd
from typing import Literal, Self

from pydantic import model_validator
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
    "AffineMembershipResult",
    "PositiveAffineSemigroup",
    "PositiveGradingResult",
    "construct",
    "fiber",
    "membership",
    "positive_grading",
]
