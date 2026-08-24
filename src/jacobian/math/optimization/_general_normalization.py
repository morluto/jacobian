"""Private exact general-form to standard-form LP normalization."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.optimization._general_models import (
    GeneralFormRationalLinearProgram,
)
from jacobian.math.optimization._models import (
    MAX_LINEAR_PROGRAM_RESULT_BYTES,
    StandardFormRationalLinearProgram,
    _result_digit_bound,
)


@dataclass(frozen=True)
class GeneralLinearNormalization:
    """Request-scoped exact map between source and standard coordinates."""

    standard_program: StandardFormRationalLinearProgram
    offsets: tuple[Fraction, ...]
    columns: tuple[tuple[tuple[int, Fraction], ...], ...]
    source_rows: tuple[tuple[int, Fraction], ...]
    upper_rows: tuple[int | None, ...]


def _rational(value: Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(value)


def _source_coordinate_map(
    program: GeneralFormRationalLinearProgram,
) -> tuple[
    list[Fraction],
    list[tuple[tuple[int, Fraction], ...]],
    list[str],
]:
    """Represent every original coordinate by nonnegative private columns."""

    offsets: list[Fraction] = []
    columns: list[tuple[tuple[int, Fraction], ...]] = []
    standard_names: list[str] = []
    for index, variable in enumerate(program.variables):
        lower = variable.lower_bound
        upper = variable.upper_bound
        if lower is not None:
            offsets.append(lower.as_fraction())
            column = len(standard_names)
            standard_names.append(f"_g{index}_lower")
            columns.append(((column, Fraction(1)),))
        elif upper is not None:
            offsets.append(upper.as_fraction())
            column = len(standard_names)
            standard_names.append(f"_g{index}_upper")
            columns.append(((column, Fraction(-1)),))
        else:
            offsets.append(Fraction())
            positive = len(standard_names)
            standard_names.append(f"_g{index}_positive")
            negative = len(standard_names)
            standard_names.append(f"_g{index}_negative")
            columns.append(((positive, Fraction(1)), (negative, Fraction(-1))))
    return offsets, columns, standard_names


def normalize_general_program(
    program: GeneralFormRationalLinearProgram,
) -> GeneralLinearNormalization:
    """Construct the bounded private standard-form source and exact ledger."""

    source_objective = tuple(
        value.as_fraction() for value in program.objective.coefficients
    )
    offsets, columns, standard_names = _source_coordinate_map(program)

    rows: list[list[Fraction]] = []
    rhs: list[Fraction] = []
    source_rows: list[tuple[int, Fraction]] = []
    for row_index, source_row in enumerate(program.constraints):
        coefficients = tuple(value.as_fraction() for value in source_row.coefficients)
        sign = Fraction(-1 if source_row.relation == "GE" else 1)
        normalized_row = [Fraction()] * len(standard_names)
        for source_index, coefficient in enumerate(coefficients):
            for column, multiplier in columns[source_index]:
                normalized_row[column] += sign * coefficient * multiplier
        if source_row.relation != "EQ":
            # Both normalized inequality directions use ``a*z+s=b`` with a
            # nonnegative slack.  Its reduced cost and Farkas coordinate give
            # the source row multiplier its required one-sided sign.
            for existing in rows:
                existing.append(Fraction())
            normalized_row.append(Fraction(1))
            standard_names.append(f"_row{row_index}_slack")
        rows.append(normalized_row)
        rhs.append(
            sign
            * (
                source_row.rhs.as_fraction()
                - sum(
                    (
                        coefficient * offset
                        for coefficient, offset in zip(
                            coefficients, offsets, strict=True
                        )
                    ),
                    Fraction(),
                )
            )
        )
        source_rows.append((row_index, sign))

    upper_rows: list[int | None] = []
    for source_index, variable in enumerate(program.variables):
        lower_bound = variable.lower_bound
        upper_bound = variable.upper_bound
        if lower_bound is None or upper_bound is None:
            upper_rows.append(None)
            continue
        row = [Fraction()] * len(standard_names)
        # A finite lower uses x=l+z.  The extra equality z+s=u-l keeps the
        # upper endpoint exact and gives its multiplier a source interpretation.
        for column, multiplier in columns[source_index]:
            row[column] = multiplier
        row.append(Fraction(1))
        for existing in rows:
            existing.append(Fraction())
        standard_names.append(f"_g{source_index}_upper_slack")
        rows.append(row)
        rhs.append(upper_bound.as_fraction() - lower_bound.as_fraction())
        upper_rows.append(len(rows) - 1)

    sense_sign = Fraction(1 if program.objective.sense == "MINIMIZE" else -1)
    objective = [Fraction()] * len(standard_names)
    for source_index, coefficient in enumerate(source_objective):
        for column, multiplier in columns[source_index]:
            objective[column] += sense_sign * coefficient * multiplier
    if len(standard_names) > 32:
        raise ValueError(
            "general linear-program normalized variables exceed the 32-entry bound"
        )
    if len(rows) > 64:
        raise ValueError(
            "general linear-program normalized rows exceed the 64-entry bound"
        )
    standard = StandardFormRationalLinearProgram(
        variables=tuple(standard_names),
        objective=tuple(_rational(value) for value in objective),
        coefficients=tuple(tuple(_rational(value) for value in row) for row in rows),
        rhs=tuple(_rational(value) for value in rhs),
    )
    return GeneralLinearNormalization(
        standard_program=standard,
        offsets=tuple(offsets),
        columns=tuple(columns),
        source_rows=tuple(source_rows),
        upper_rows=tuple(upper_rows),
    )


def normalized_result_digit_bound(program: GeneralFormRationalLinearProgram) -> int:
    return _result_digit_bound(normalize_general_program(program).standard_program)


def require_admitted_general_normalization(
    program: GeneralFormRationalLinearProgram,
) -> None:
    """Preflight the whole standard expansion and the mapped public result."""

    normalization = normalize_general_program(program)
    digits = _result_digit_bound(normalization.standard_program)
    # An optimal general result has five source-coordinate vectors over variables
    # and two over rows plus two scalar objectives.  This is conservative because
    # only one status-specific certificate family is present in any actual result.
    values = 6 * len(program.variables) + 3 * len(program.constraints) + 2
    if 4096 + values * (2 * digits + 32) > MAX_LINEAR_PROGRAM_RESULT_BYTES:
        raise ValueError(
            "general linear-program mapped result can exceed the "
            f"{MAX_LINEAR_PROGRAM_RESULT_BYTES}-byte result bound"
        )


__all__ = [
    "GeneralLinearNormalization",
    "normalize_general_program",
    "normalized_result_digit_bound",
    "require_admitted_general_normalization",
]
