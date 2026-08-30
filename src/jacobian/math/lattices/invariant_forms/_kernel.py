"""Exact kernel for integral forms fixed by rational congruence actions."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import ceil, gcd, lcm, log10

from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
)
from jacobian.math.lattices._lattice_ops import hermite_basis, saturated_lattice_basis
from jacobian.math.lattices.invariant_forms._models import (
    MAX_CONSTRAINT_CELLS,
    MAX_CONSTRAINT_DIGIT_WORK,
    MAX_INTEGER_KERNEL_DIGIT_WORK,
    FormKind,
    IntegralBilinearForm,
    InvariantBilinearFormLattice,
    RationalMatrixAction,
    _validation_error,
    constraint_coefficient_count,
)
from jacobian.math.matrices.values import MAX_MATRIX_SCALAR_DIGITS

MAX_CONSTRAINT_COMPONENT_DIGITS = 65_536
MAX_STORED_CONSTRAINT_DIGITS = 2_000_000
MAX_INVARIANT_FORM_RESULT_BYTES = CanonicalLimits().max_output_bytes

type CoefficientPosition = tuple[int, int]
type IntegerConstraint = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _ConstraintPlan:
    """One admitted exact constraint matrix and its coefficient coordinates."""

    positions: tuple[CoefficientPosition, ...]
    constraints: tuple[IntegerConstraint, ...]


def _coefficient_positions(
    dimension: int, kind: FormKind
) -> tuple[CoefficientPosition, ...]:
    if kind == "BILINEAR":
        return tuple(
            (row, column) for row in range(dimension) for column in range(dimension)
        )
    if kind == "SYMMETRIC":
        return tuple(
            (row, column)
            for row in range(dimension)
            for column in range(row, dimension)
        )
    return tuple(
        (row, column)
        for row in range(dimension)
        for column in range(row + 1, dimension)
    )


def _constraint_coefficient(
    matrix: tuple[tuple[Fraction, ...], ...],
    *,
    equation_row: int,
    equation_column: int,
    position: CoefficientPosition,
    kind: FormKind,
) -> Fraction:
    """Return one coefficient of ``A^T Q A - Q`` exactly."""

    row, column = position
    if kind == "BILINEAR":
        transformed = matrix[row][equation_row] * matrix[column][equation_column]
        source = int(row == equation_row and column == equation_column)
        return transformed - source
    if kind == "SYMMETRIC":
        if row == column:
            transformed = matrix[row][equation_row] * matrix[row][equation_column]
            source = int(row == equation_row and row == equation_column)
            return transformed - source
        transformed = (
            matrix[row][equation_row] * matrix[column][equation_column]
            + matrix[column][equation_row] * matrix[row][equation_column]
        )
        source = int(row == equation_row and column == equation_column) + int(
            column == equation_row and row == equation_column
        )
        return transformed - source
    transformed = (
        matrix[row][equation_row] * matrix[column][equation_column]
        - matrix[column][equation_row] * matrix[row][equation_column]
    )
    source = int(row == equation_row and column == equation_column) - int(
        column == equation_row and row == equation_column
    )
    return transformed - source


def _integer_digit_count(value: int) -> int:
    return len(format_canonical_integer(value).lstrip("-"))


def _normalize_constraint(
    coefficients: tuple[Fraction, ...],
) -> tuple[IntegerConstraint | None, int]:
    """Clear denominators and normalize one rational equation primitively."""

    if not any(coefficients):
        return None, 0
    denominator = lcm(*(coefficient.denominator for coefficient in coefficients))
    integers = [
        coefficient.numerator * (denominator // coefficient.denominator)
        for coefficient in coefficients
    ]
    content = gcd(*(abs(value) for value in integers))
    primitive = [value // content for value in integers]
    first_nonzero = next(value for value in primitive if value)
    if first_nonzero < 0:
        primitive = [-value for value in primitive]
    maximum_digits = max(_integer_digit_count(value) for value in primitive)
    if maximum_digits > MAX_CONSTRAINT_COMPONENT_DIGITS:
        raise _validation_error(
            "budget_exceeded",
            "a normalized congruence constraint exceeds the exact intermediate "
            f"bound of {MAX_CONSTRAINT_COMPONENT_DIGITS} decimal digits",
        )
    return tuple(primitive), sum(_integer_digit_count(value) for value in primitive)


def _require_constraint_expansion_envelope(
    action: RationalMatrixAction,
    *,
    kind: FormKind,
    coefficient_count: int,
    coefficient_cells: int,
) -> None:
    """Bound Fraction construction, products, and row normalization by height."""

    if not action.generators or coefficient_count == 0:
        return
    values = tuple(
        value
        for generator in action.generators
        for row in generator.matrix.entries
        for value in row
    )
    numerator_digits = max(len(value.num.lstrip("-")) for value in values)
    all_denominators_are_one = all(value.den == "1" for value in values)
    denominator_growth_digits = (
        0 if all_denominators_are_one else max(len(value.den) for value in values)
    )
    source_component_digits = max(
        numerator_digits,
        max(len(value.den) for value in values),
    )
    transformed_terms = 1 if kind == "BILINEAR" else 2
    # One product has two source numerators and denominators. Combining two
    # products can multiply their denominators; subtracting the source basis
    # coefficient adds at most one further decimal digit to the numerator.
    coefficient_denominator_digits = 2 * transformed_terms * denominator_growth_digits
    coefficient_numerator_digits = (
        max(
            2 * numerator_digits
            + 2 * (transformed_terms - 1) * denominator_growth_digits
            + int(transformed_terms == 2),
            coefficient_denominator_digits,
        )
        + 1
    )
    cleared_component_digits = (
        coefficient_numerator_digits
        + max(coefficient_count - 1, 0) * coefficient_denominator_digits
        + 1
    )
    if cleared_component_digits > MAX_CONSTRAINT_COMPONENT_DIGITS:
        raise _validation_error(
            "budget_exceeded",
            "the source rational heights can expand one denominator-cleared "
            "constraint beyond the exact intermediate bound of "
            f"{MAX_CONSTRAINT_COMPONENT_DIGITS} decimal digits",
        )
    matrix_cells = len(action.generators) * len(action.coordinate_axis) ** 2
    digit_work = matrix_cells * source_component_digits + coefficient_cells * (
        coefficient_numerator_digits
        + coefficient_denominator_digits
        + 2 * cleared_component_digits
    )
    if digit_work > MAX_CONSTRAINT_DIGIT_WORK:
        raise _validation_error(
            "budget_exceeded",
            "congruence expansion and primitive row normalization exceed the "
            f"{MAX_CONSTRAINT_DIGIT_WORK:,}-unit digit-work bound",
        )


def _retained_action_bytes(action: RationalMatrixAction) -> int:
    """Measure the exact retained source before expanding any constraints."""

    try:
        source_bytes = len(encode_strict_json(action.model_dump(mode="json")))
    except CanonicalizationError:
        raise _validation_error(
            "budget_exceeded",
            "the retained rational matrix action exceeds the canonical output limit",
        ) from None
    if source_bytes + 4_096 > MAX_INVARIANT_FORM_RESULT_BYTES:
        raise _validation_error(
            "budget_exceeded",
            "the retained rational matrix action leaves no room for the canonical result",
        )
    return source_bytes


def _kernel_entry_digit_bound(
    *, coefficient_count: int, constraint_count: int, constraint_digits: int
) -> int:
    """Bound canonical integer-kernel HNF coefficients before backend work.

    For constraint rank ``r`` in ``ZZ^m``, Cramer's rule gives an integer
    rational-kernel basis whose entries are rank minors. Hadamard bounds each
    such minor by ``(sqrt(m) H)^r``. The saturated row-HNF entries are bounded
    by the maximal minors of that kernel basis, hence by another Hadamard
    factor in kernel dimension ``m-r``. Admission maximizes this deliberately
    loose bound over every possible non-full constraint rank; full rank has no
    basis entries to serialize.
    """

    if coefficient_count <= 1 or constraint_count == 0:
        return 1
    rank_ceiling = min(constraint_count, coefficient_count - 1)
    logarithmic_bound = max(
        (coefficient_count - rank)
        * (
            rank * (constraint_digits + 0.5 * log10(coefficient_count))
            + 0.5 * log10(coefficient_count - rank)
        )
        + 2
        for rank in range(1, rank_ceiling + 1)
    )
    return max(1, ceil(logarithmic_bound))


def _require_result_envelope(
    action: RationalMatrixAction,
    *,
    coefficient_count: int,
    constraints: tuple[IntegerConstraint, ...],
    source_bytes: int,
) -> None:
    """Prove a conservative exact-output envelope before nullspace work."""

    if constraints:
        constraint_digits = max(
            _integer_digit_count(value)
            for constraint in constraints
            for value in constraint
        )
        maximum_basis_count = max(coefficient_count - 1, 0)
    else:
        constraint_digits = 1
        maximum_basis_count = coefficient_count
    entry_digits = _kernel_entry_digit_bound(
        coefficient_count=coefficient_count,
        constraint_count=len(constraints),
        constraint_digits=constraint_digits,
    )
    if maximum_basis_count and entry_digits > MAX_MATRIX_SCALAR_DIGITS:
        raise _validation_error(
            "budget_exceeded",
            "the exact invariant-form basis can exceed the canonical integer "
            f"component bound of {MAX_MATRIX_SCALAR_DIGITS} decimal digits",
        )
    axis_bytes = len(encode_strict_json(list(action.coordinate_axis)))
    dimension = len(action.coordinate_axis)
    predicted_result_bytes = (
        source_bytes
        + 4_096
        + maximum_basis_count
        * (axis_bytes + 1_024 + dimension * dimension * (entry_digits + 5))
    )
    if predicted_result_bytes > MAX_INVARIANT_FORM_RESULT_BYTES:
        raise _validation_error(
            "budget_exceeded",
            "the conservative exact invariant-form basis bound exceeds the "
            f"{MAX_INVARIANT_FORM_RESULT_BYTES}-byte canonical output limit",
        )


def _require_nullspace_work_envelope(plan: _ConstraintPlan) -> None:
    """Admit FLINT's dense integer nullspace from the realized system."""

    if not plan.constraints or not plan.positions:
        return
    rows = len(plan.constraints)
    columns = len(plan.positions)
    rank_bound = min(rows, columns)
    scalar_digits = max(
        _integer_digit_count(value)
        for constraint in plan.constraints
        for value in constraint
    )
    digit_work = rows * columns * rank_bound * scalar_digits
    if digit_work > MAX_INTEGER_KERNEL_DIGIT_WORK:
        raise _validation_error(
            "budget_exceeded",
            "the realized congruence matrix exceeds the exact nullspace "
            f"digit-work bound of {MAX_INTEGER_KERNEL_DIGIT_WORK:,} units",
        )


def _require_normal_form_work_envelope(
    plan: _ConstraintPlan, raw_basis: list[list[int]]
) -> None:
    """Admit saturation and both Hermite normalizations before they run."""

    if not raw_basis:
        return
    coefficient_count = len(plan.positions)
    constraint_digits = max(
        _integer_digit_count(value)
        for constraint in plan.constraints
        for value in constraint
    )
    predicted_digits = _kernel_entry_digit_bound(
        coefficient_count=coefficient_count,
        constraint_count=len(plan.constraints),
        constraint_digits=constraint_digits,
    )
    raw_digits = max(_integer_digit_count(value) for row in raw_basis for value in row)
    scalar_digits = max(predicted_digits, raw_digits)
    # Saturation performs rank/SNF work on a basis_rank x coefficient_count
    # matrix, inverts the ambient Smith transform, then applies SymPy column
    # HNF and FLINT row HNF. Four dense cubic passes conservatively cover those
    # mandatory phases under the repository's scalar-digit work convention.
    digit_work = 4 * coefficient_count**3 * scalar_digits
    if digit_work > MAX_INTEGER_KERNEL_DIGIT_WORK:
        raise _validation_error(
            "budget_exceeded",
            "integer-kernel saturation and Hermite normalization exceed the "
            f"{MAX_INTEGER_KERNEL_DIGIT_WORK:,}-unit digit-work bound",
        )


def _build_constraint_plan(
    action: RationalMatrixAction, kind: FormKind
) -> _ConstraintPlan:
    dimension = len(action.coordinate_axis)
    positions = _coefficient_positions(dimension, kind)
    source_bytes = _retained_action_bytes(action)
    if not positions:
        _require_result_envelope(
            action,
            coefficient_count=0,
            constraints=(),
            source_bytes=source_bytes,
        )
        return _ConstraintPlan(positions=(), constraints=())
    cell_count = constraint_coefficient_count(dimension, len(action.generators), kind)
    if cell_count > MAX_CONSTRAINT_CELLS:
        raise _validation_error(
            "budget_exceeded",
            "the congruence expansion exceeds the structural bound of "
            f"{MAX_CONSTRAINT_CELLS} coefficients",
        )
    _require_constraint_expansion_envelope(
        action,
        kind=kind,
        coefficient_count=len(positions),
        coefficient_cells=cell_count,
    )
    constraints: set[IntegerConstraint] = set()
    stored_digits = 0
    for generator in action.generators:
        matrix = tuple(
            tuple(value.as_fraction() for value in row)
            for row in generator.matrix.entries
        )
        for equation_row in range(dimension):
            for equation_column in range(dimension):
                rational_row = tuple(
                    _constraint_coefficient(
                        matrix,
                        equation_row=equation_row,
                        equation_column=equation_column,
                        position=position,
                        kind=kind,
                    )
                    for position in positions
                )
                constraint, row_digits = _normalize_constraint(rational_row)
                if constraint is None or constraint in constraints:
                    continue
                stored_digits += row_digits
                if stored_digits > MAX_STORED_CONSTRAINT_DIGITS:
                    raise _validation_error(
                        "budget_exceeded",
                        "the normalized congruence system exceeds the exact "
                        f"{MAX_STORED_CONSTRAINT_DIGITS}-digit intermediate bound",
                    )
                constraints.add(constraint)
    ordered_constraints = tuple(sorted(constraints))
    _require_result_envelope(
        action,
        coefficient_count=len(positions),
        constraints=ordered_constraints,
        source_bytes=source_bytes,
    )
    plan = _ConstraintPlan(positions=positions, constraints=ordered_constraints)
    _require_nullspace_work_envelope(plan)
    return plan


def _integer_kernel_basis(plan: _ConstraintPlan) -> tuple[list[list[int]], int]:
    coefficient_count = len(plan.positions)
    if coefficient_count == 0:
        return [], 0
    if not plan.constraints:
        return (
            [
                [int(row == column) for column in range(coefficient_count)]
                for row in range(coefficient_count)
            ],
            0,
        )

    from flint import fmpz_mat

    constraint_matrix = fmpz_mat([list(row) for row in plan.constraints])
    raw_kernel, nullity = constraint_matrix.nullspace()
    integer_nullity = int(nullity)
    constraint_rank = coefficient_count - integer_nullity
    if integer_nullity == 0:
        return [], constraint_rank
    raw_basis = [
        [int(raw_kernel[row, column]) for row in range(coefficient_count)]
        for column in range(integer_nullity)
    ]
    _require_normal_form_work_envelope(plan, raw_basis)
    saturated_basis = saturated_lattice_basis(raw_basis)
    row_hnf, _ = hermite_basis(saturated_basis)
    return row_hnf, constraint_rank


def _form_entries(
    coefficient_vector: list[int],
    *,
    positions: tuple[CoefficientPosition, ...],
    dimension: int,
    kind: FormKind,
) -> tuple[tuple[str, ...], ...]:
    entries = [[0] * dimension for _ in range(dimension)]
    for value, (row, column) in zip(coefficient_vector, positions, strict=True):
        entries[row][column] = value
        if kind == "SYMMETRIC":
            entries[column][row] = value
        elif kind == "ALTERNATING":
            entries[column][row] = -value
    return tuple(
        tuple(format_canonical_integer(value) for value in row) for row in entries
    )


def invariant_bilinear_form_lattice_kernel(
    action: RationalMatrixAction, kind: FormKind
) -> InvariantBilinearFormLattice:
    """Return the saturated integer lattice of forms fixed by every generator."""

    plan = _build_constraint_plan(action, kind)
    basis, constraint_rank = _integer_kernel_basis(plan)
    dimension = len(action.coordinate_axis)
    basis_forms = tuple(
        IntegralBilinearForm._from_kernel(
            coordinate_axis=action.coordinate_axis,
            kind=kind,
            entries=_form_entries(
                vector,
                positions=plan.positions,
                dimension=dimension,
                kind=kind,
            ),
        )
        for vector in basis
    )
    return InvariantBilinearFormLattice._from_kernel(
        action=action,
        kind=kind,
        coefficient_dimension=len(plan.positions),
        constraint_rank=constraint_rank,
        basis_forms=basis_forms,
    )


__all__ = ["invariant_bilinear_form_lattice_kernel"]
