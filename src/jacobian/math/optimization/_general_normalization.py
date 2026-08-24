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
    MAX_RATIONAL_DIGITS,
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
    standard = StandardFormRationalLinearProgram.admit_derived_intermediate(
        {
            "variables": tuple(standard_names),
            "objective": tuple(_rational(value) for value in objective),
            "coefficients": tuple(
                tuple(_rational(value) for value in row) for row in rows
            ),
            "rhs": tuple(_rational(value) for value in rhs),
        },
        maximum_digits=_standard_intermediate_digit_bound(program),
    )
    return GeneralLinearNormalization(
        standard_program=standard,
        offsets=tuple(offsets),
        columns=tuple(columns),
        source_rows=tuple(source_rows),
        upper_rows=tuple(upper_rows),
    )


def _standard_intermediate_digit_bound(
    program: GeneralFormRationalLinearProgram,
) -> int:
    """Bound every scalar the private standard expansion can produce.

    Non-RHS entries are one signed source scalar.  Each source-row RHS
    subtracts the sum of up to ``n`` coefficient-offset products from the
    source RHS, so its reduced denominator divides the product of the summed
    term denominators while its numerator carries the matching magnitude;
    upper-row RHS values subtract two source bounds.
    """

    product_digits = 2 * MAX_RATIONAL_DIGITS
    terms = 1 + len(program.variables)
    return (terms + 1) * product_digits + len(str(terms))


_MAPPED_RESULT_HEIGHT_SLACK = 16


def _mapped_point_digit_bound(normalization: GeneralLinearNormalization) -> int:
    """Bound mapped coordinates, bound slacks, and mapped recession rays.

    A source coordinate adds its offset to at most two standard columns, and
    each bound slack subtracts one further source bound, so such a value
    stacks at most two input-height products onto one standard coordinate
    before one chained difference.
    """

    return (
        2 * MAX_RATIONAL_DIGITS
        + 2 * _result_digit_bound(normalization.standard_program)
        + _MAPPED_RESULT_HEIGHT_SLACK
    )


def _mapped_residual_digit_bound(normalization: GeneralLinearNormalization) -> int:
    """Bound mapped objectives, residuals, and constraint slacks.

    These sums stack ``n`` coefficient-offset products and ``n``
    coefficient-standard products onto the source RHS.  The contributing
    denominators share no factor, so the reduced denominator grows like the
    product of all contributing source denominators rather than like one
    stacked pair; this branch reserves that common-denominator growth across
    every one of the ``2n + 1`` summed terms.
    """

    variables = len(normalization.offsets)
    summed_terms = 2 * variables + 1
    return (
        variables
        * (
            3 * MAX_RATIONAL_DIGITS
            + _result_digit_bound(normalization.standard_program)
        )
        + MAX_RATIONAL_DIGITS
        + len(str(summed_terms))
        + 1
        + _MAPPED_RESULT_HEIGHT_SLACK
    )


def _mapped_certificate_digit_bound(normalization: GeneralLinearNormalization) -> int:
    """Bound mapped dual, stationarity, and Farkas certificate values.

    Standard multipliers stay within ``_result_digit_bound`` of the private
    program.  Mapping stacks at most two input-height products per value --
    objective and gradient terms, then bound multipliers against bounds -- and
    the dual or Farkas pairing sums up to ``2n`` such products with source
    bounds whose denominators share no factor, so the bound additionally
    reserves that common-denominator growth; the slack covers summation
    carries and chained differences.
    """

    variables = len(normalization.offsets)
    return (
        _result_digit_bound(normalization.standard_program)
        + (2 + 4 * variables) * MAX_RATIONAL_DIGITS
        + _MAPPED_RESULT_HEIGHT_SLACK
    )


def normalized_point_digit_bound(program: GeneralFormRationalLinearProgram) -> int:
    return _mapped_point_digit_bound(normalize_general_program(program))


def normalized_residual_digit_bound(program: GeneralFormRationalLinearProgram) -> int:
    return _mapped_residual_digit_bound(normalize_general_program(program))


def normalized_certificate_digit_bound(
    program: GeneralFormRationalLinearProgram,
) -> int:
    return _mapped_certificate_digit_bound(normalize_general_program(program))


def estimated_mapped_result_bytes(program: GeneralFormRationalLinearProgram) -> int:
    """Upper-bound the wired bytes of any outcome this program can return."""

    normalization = normalize_general_program(program)
    point_unit = 2 * _mapped_point_digit_bound(normalization) + 32
    residual_unit = 2 * _mapped_residual_digit_bound(normalization) + 32
    certificate_unit = 2 * _mapped_certificate_digit_bound(normalization) + 32
    variables = len(normalization.offsets)
    rows = len(program.constraints)
    # Each status carries its full replayable block: optimal adds the primal
    # residual sums plus one certificate family, unbounded swaps duals for the
    # recession ray, infeasible carries only Farkas coordinates, and unknown
    # carries no values at all.  Primal evidence always wires within its own
    # bound, so it anchors the guaranteed typed outcome.
    optimal_bytes = (
        (2 * rows + 1) * residual_unit
        + 3 * variables * point_unit
        + (3 * variables + rows + 1) * certificate_unit
    )
    unbounded_bytes = (2 * rows + 1) * residual_unit + 4 * variables * point_unit
    infeasible_bytes = (2 * variables + rows) * certificate_unit
    return 4096 + max(optimal_bytes, unbounded_bytes, infeasible_bytes)


def require_admitted_general_normalization(
    program: GeneralFormRationalLinearProgram,
) -> None:
    """Preflight the whole standard expansion and the mapped public result."""

    if estimated_mapped_result_bytes(program) > MAX_LINEAR_PROGRAM_RESULT_BYTES:
        raise ValueError(
            "general linear-program mapped result can exceed the "
            f"{MAX_LINEAR_PROGRAM_RESULT_BYTES}-byte result bound"
        )


__all__ = [
    "GeneralLinearNormalization",
    "estimated_mapped_result_bytes",
    "normalize_general_program",
    "normalized_certificate_digit_bound",
    "normalized_point_digit_bound",
    "normalized_residual_digit_bound",
    "require_admitted_general_normalization",
]
