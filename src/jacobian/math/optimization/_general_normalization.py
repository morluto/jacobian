"""Private exact general-form to standard-form LP normalization."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.optimization._general_models import (
    MAX_GENERAL_LINEAR_PROGRAM_CONSTRAINTS,
    MAX_GENERAL_LINEAR_PROGRAM_VARIABLES,
    GeneralFormRationalLinearProgram,
)
from jacobian.math.optimization._models import (
    MAX_RATIONAL_DIGITS,
    StandardFormRationalLinearProgram,
)


@dataclass(frozen=True)
class GeneralLinearNormalization:
    """Request-scoped exact map between source and standard coordinates."""

    standard_program: StandardFormRationalLinearProgram
    offsets: tuple[Fraction, ...]
    columns: tuple[tuple[tuple[int, Fraction], ...], ...]
    source_rows: tuple[tuple[int | None, Fraction], ...]
    upper_rows: tuple[int | None, ...]
    chain: tuple[int, ...] = ()
    chain_rows: tuple[tuple[int, Fraction], ...] = ()


def _ordered_chain(
    program: GeneralFormRationalLinearProgram,
) -> tuple[tuple[int, ...], tuple[tuple[int, Fraction], ...]]:
    """Recognize a full zero-gap chain above one common finite lower bound."""
    size = len(program.variables)
    lower = program.variables[0].lower_bound
    if (
        size < 2
        or lower is None
        or any(
            variable.lower_bound != lower or variable.upper_bound is not None
            for variable in program.variables
        )
    ):
        return (), ()
    outgoing: dict[int, tuple[int, int, Fraction]] = {}
    incoming: set[int] = set()
    for row_index, row in enumerate(program.constraints):
        if row.relation == "EQ" or row.rhs.num != 0:
            continue
        sign = Fraction(-1 if row.relation == "GE" else 1)
        nonzero = [
            (i, sign * value.as_fraction())
            for i, value in enumerate(row.coefficients)
            if value.num
        ]
        if len(nonzero) != 2 or {value for _, value in nonzero} != {-1, 1}:
            continue
        first = next(i for i, value in nonzero if value == 1)
        second = next(i for i, value in nonzero if value == -1)
        if first in outgoing or second in incoming:
            return (), ()
        outgoing[first] = second, row_index, sign
        incoming.add(second)
    starts = set(range(size)) - incoming
    if len(outgoing) != size - 1 or len(starts) != 1:
        return (), ()
    chain = [starts.pop()]
    rows = []
    while chain[-1] in outgoing:
        following, row_index, sign = outgoing[chain[-1]]
        if following in chain:
            return (), ()
        chain.append(following)
        rows.append((row_index, sign))
    return (tuple(chain), tuple(rows)) if len(chain) == size else ((), ())


def _rational(value: Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(value)


def _source_coordinate_map(
    program: GeneralFormRationalLinearProgram,
    chain: tuple[int, ...] = (),
) -> tuple[
    list[Fraction],
    list[tuple[tuple[int, Fraction], ...]],
    list[str],
]:
    """Represent every original coordinate by nonnegative private columns."""

    if chain:
        lower = program.variables[chain[0]].lower_bound
        assert lower is not None
        positions = {source: position for position, source in enumerate(chain)}
        return (
            [lower.as_fraction()] * len(chain),
            [
                tuple((i, Fraction(1)) for i in range(positions[source] + 1))
                for source in range(len(chain))
            ],
            [f"_chain{i}" for i in range(len(chain))],
        )

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
    chain, chain_rows = _ordered_chain(program)
    removed_rows = {index for index, _ in chain_rows}
    normalized_columns = (
        sum(
            2 if v.lower_bound is None and v.upper_bound is None else 1
            for v in program.variables
        )
        + sum(row.relation != "EQ" for row in program.constraints)
        - len(removed_rows)
        + sum(
            v.lower_bound is not None and v.upper_bound is not None
            for v in program.variables
        )
    )
    normalized_rows = (
        len(program.constraints)
        - len(removed_rows)
        + sum(
            v.lower_bound is not None and v.upper_bound is not None
            for v in program.variables
        )
    )
    for reason, count, limit in (
        (
            "normalized_columns",
            normalized_columns,
            MAX_GENERAL_LINEAR_PROGRAM_VARIABLES,
        ),
        ("normalized_rows", normalized_rows, MAX_GENERAL_LINEAR_PROGRAM_CONSTRAINTS),
    ):
        if count > limit:
            raise OperationResourceAdmissionError(
                location=("program",),
                code=f"optimization.linear.{reason}",
                message=(
                    f"Exact LP {reason} exceeded: normalized_columns={normalized_columns}, "
                    f"column_limit={MAX_GENERAL_LINEAR_PROGRAM_VARIABLES}, "
                    f"normalized_rows={normalized_rows}, row_limit={MAX_GENERAL_LINEAR_PROGRAM_CONSTRAINTS}."
                ),
            )

    source_objective = tuple(
        value.as_fraction() for value in program.objective.coefficients
    )
    offsets, columns, standard_names = _source_coordinate_map(program, chain)

    rows: list[list[Fraction]] = []
    rhs: list[Fraction] = []
    source_rows: list[tuple[int | None, Fraction]] = []
    for row_index, source_row in enumerate(program.constraints):
        if row_index in removed_rows:
            source_rows.append((None, Fraction(1)))
            continue
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
        source_rows.append((len(rows) - 1, sign))

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
        chain=chain,
        chain_rows=chain_rows,
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


def _mapped_point_digit_bound(standard_digits: int, terms: int = 2) -> int:
    """Bound mapped coordinates, bound slacks, and mapped recession rays.

    A source coordinate adds its offset to at most two standard columns, and
    each bound slack subtracts one further source bound, so such a value
    stacks at most two input-height products onto one standard coordinate
    before one chained difference.
    """

    return (
        2 * MAX_RATIONAL_DIGITS + terms * standard_digits + _MAPPED_RESULT_HEIGHT_SLACK
    )


def _mapped_residual_digit_bound(
    normalization: GeneralLinearNormalization, standard_digits: int
) -> int:
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
    mapped_terms = max(len(mapping) for mapping in normalization.columns)
    return (
        variables * (3 * MAX_RATIONAL_DIGITS + mapped_terms * standard_digits)
        + MAX_RATIONAL_DIGITS
        + len(str(summed_terms))
        + 1
        + _MAPPED_RESULT_HEIGHT_SLACK
    )


def _mapped_certificate_digit_bound(
    normalization: GeneralLinearNormalization, standard_digits: int
) -> int:
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
        standard_digits
        + (2 + 4 * variables) * MAX_RATIONAL_DIGITS
        + _MAPPED_RESULT_HEIGHT_SLACK
    )


def chain_mapped_digit_bounds(
    program: GeneralFormRationalLinearProgram,
    normalization: GeneralLinearNormalization,
    standard_digits: int,
) -> tuple[int, int, int]:
    """Bound source-coordinate outputs before solving the reduced chain LP.

    All source denominators divide their product D (count each distinct
    denominator once). Standard coordinate denominators divide a product of
    at most n denominators; standard dual denominators a product of at most
    r denominators. A source point therefore has denominator dividing D*P,
    and its objective/residuals D**2*P. A mapped chain multiplier is a sum
    of source coefficients times standard duals, with denominator D*Q;
    stationarity and source-bound pairings divide D**2*Q. The magnitude
    allowance covers two source factors, a standard factor and all sums.
    """
    scalars = [*program.objective.coefficients]
    for row in program.constraints:
        scalars.extend(row.coefficients)
        scalars.append(row.rhs)
    scalars.extend(
        variable.lower_bound
        for variable in program.variables
        if variable.lower_bound is not None
    )
    denominator_digits = sum(len(str(d)) for d in {v.den for v in scalars} if d != 1)
    source_digits = max(len(str(abs(v.num))) for v in scalars)
    n = len(program.variables)
    r = len(normalization.standard_program.coefficients)
    magnitude = 2 * source_digits + standard_digits + 3 * len(str(n + r + 1)) + 4
    point = denominator_digits + n * standard_digits + magnitude
    residual = 2 * denominator_digits + n * standard_digits + magnitude
    certificate = 2 * denominator_digits + r * standard_digits + magnitude
    return point, residual, certificate


def admit_general_normalization(
    program: GeneralFormRationalLinearProgram,
) -> GeneralLinearNormalization:
    """Preflight the whole standard expansion and the mapped public result."""

    return normalize_general_program(program)


__all__ = [
    "GeneralLinearNormalization",
    "admit_general_normalization",
    "normalize_general_program",
]
