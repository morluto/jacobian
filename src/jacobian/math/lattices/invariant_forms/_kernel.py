"""Exact kernel for integral forms fixed by rational congruence actions."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import ceil, gcd, lcm, log10
from unicodedata import normalize

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    current_request_execution,
    request_cancelled,
)
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
)
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

_INVARIANT_FORM_WALL_SECONDS = 3600.0


def _require_active_request(stage: str) -> None:
    """Raise if the request deadline expired or was cancelled during *stage*."""

    if request_cancelled():
        raise OperationExecutionCancelledError(
            f"invariant-form lattice cancelled {stage}"
        )
    execution = current_request_execution()
    if execution is not None and execution.deadline is not None:
        import time

        if time.monotonic() >= execution.deadline:
            raise OperationExecutionTimeoutError(
                f"invariant-form lattice deadline expired {stage}"
            )


def _bind_request_deadline() -> float:
    """Bind a conservative owner deadline if the request lacks one."""

    import time

    from jacobian._execution import bind_request_deadline

    execution = current_request_execution()
    if execution is not None and execution.deadline is not None:
        return execution.deadline
    started_at = execution.started_at if execution is not None else time.monotonic()
    deadline = started_at + _INVARIANT_FORM_WALL_SECONDS
    bind_request_deadline(deadline)
    return deadline


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


def _normalize_retained_strings(value: object) -> object:
    """Normalize retained JSON strings without changing JSON container types."""

    if isinstance(value, str):
        return normalize("NFC", value)
    if isinstance(value, list):
        return [_normalize_retained_strings(item) for item in value]
    if isinstance(value, dict):
        return {
            _normalize_retained_strings(key): _normalize_retained_strings(item)
            for key, item in value.items()
        }
    return value


def _retained_action_bytes(action: RationalMatrixAction) -> int:
    """Measure the exact retained source before expanding any constraints."""

    # Preflight the expanded source-byte budget before model_dump and
    # encode_strict_json materialize the full canonical representation.
    # Accumulate a conservative size over every generator, not just the
    # first, since later generators may carry much larger rationals.
    dimension = len(action.coordinate_axis)
    generator_count = len(action.generators)
    if generator_count > 0:
        max_row_bytes = 0
        for generator in action.generators:
            for row in generator.matrix.entries:
                row_bytes = sum(len(str(value)) + 4 for value in row)
                if row_bytes > max_row_bytes:
                    max_row_bytes = row_bytes
        axis_label_bytes = sum(
            len(encode_strict_json(label)) + 1 for label in action.coordinate_axis
        )
        generator_label_bytes = sum(
            len(encode_strict_json(generator.label)) + 1
            for generator in action.generators
        )
        estimated_bytes = (
            4_096
            + axis_label_bytes
            + generator_label_bytes
            + generator_count * (dimension * max_row_bytes + 32)
        )
        if estimated_bytes + 4_096 > MAX_INVARIANT_FORM_RESULT_BYTES:
            raise _validation_error(
                "budget_exceeded",
                "the retained rational matrix action leaves no room for the canonical result",
            )
    try:
        retained_payload: object = action.model_dump(mode="json")
        retained_payload = _normalize_retained_strings(retained_payload)
        source_bytes = len(encode_strict_json(retained_payload))
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


def _minor_digit_bound(*, rank: int, component_digits: int) -> int:
    """Bound decimal digits in a rank minor by Hadamard's inequality."""

    if rank == 0:
        return 1
    return max(1, ceil(rank * (component_digits + 0.5 * log10(rank))))


def _kernel_entry_digit_bound(
    *, coefficient_count: int, constraint_count: int, constraint_digits: int
) -> int:
    """Bound canonical integer-kernel row-HNF coefficients before backend work.

    For constraint rank ``r`` in ``ZZ^m``, the primitive Pluecker coordinates
    of the saturated kernel are the complementary rank-``r`` minors of an
    independent constraint-row basis, divided by their common determinantal
    divisor. Hadamard bounds each minor by ``r^(r/2) H^r``. Every entry of the
    full-row-rank row HNF is bounded by its largest maximal minor. Since the
    actual rank is not known before backend work, admission uses the largest
    possible non-full rank; full rank has no basis entries to serialize.
    """

    if coefficient_count <= 1 or constraint_count == 0:
        return 1
    rank_ceiling = min(constraint_count, coefficient_count - 1)
    return _minor_digit_bound(
        rank=rank_ceiling,
        component_digits=constraint_digits,
    )


def _require_result_envelope(
    action: RationalMatrixAction,
    *,
    coefficient_count: int,
    constraints: tuple[IntegerConstraint, ...],
    source_bytes: int,
) -> None:
    """Prove a conservative exact-output envelope before graph-HNF work."""

    if constraints:
        constraint_digits = max(
            _integer_digit_count(value)
            for constraint in constraints
            for value in constraint
        )
        possible_nonfull_ranks = range(
            1, min(len(constraints), coefficient_count - 1) + 1
        )
    else:
        constraint_digits = 1
        possible_nonfull_ranks = range(1)
    maximum_entry_digits = _kernel_entry_digit_bound(
        coefficient_count=coefficient_count,
        constraint_count=len(constraints),
        constraint_digits=constraint_digits,
    )
    if possible_nonfull_ranks and maximum_entry_digits > MAX_MATRIX_SCALAR_DIGITS:
        raise _validation_error(
            "budget_exceeded",
            "the exact invariant-form basis can exceed the canonical integer "
            f"component bound of {MAX_MATRIX_SCALAR_DIGITS} decimal digits",
        )
    axis_bytes = len(encode_strict_json(list(action.coordinate_axis)))
    dimension = len(action.coordinate_axis)
    maximum_basis_bytes = max(
        (
            (coefficient_count - rank)
            * (
                axis_bytes
                + 1_024
                + dimension
                * dimension
                * (
                    _minor_digit_bound(
                        rank=rank,
                        component_digits=constraint_digits,
                    )
                    + 5
                )
            )
            for rank in possible_nonfull_ranks
        ),
        default=0,
    )
    predicted_result_bytes = source_bytes + 4_096 + maximum_basis_bytes
    if predicted_result_bytes > MAX_INVARIANT_FORM_RESULT_BYTES:
        raise _validation_error(
            "budget_exceeded",
            "the conservative exact invariant-form basis bound exceeds the "
            f"{MAX_INVARIANT_FORM_RESULT_BYTES}-byte canonical output limit",
        )


def _require_integer_kernel_work_envelope(plan: _ConstraintPlan) -> None:
    """Admit canonical graph-lattice HNF from the realized constraint plan."""

    if not plan.constraints or not plan.positions:
        return
    constraint_count = len(plan.constraints)
    coefficient_count = len(plan.positions)
    constraint_digits = max(
        _integer_digit_count(value)
        for constraint in plan.constraints
        for value in constraint
    )
    graph_entry_digits = _minor_digit_bound(
        rank=min(constraint_count, coefficient_count),
        component_digits=constraint_digits,
    )
    # The first FLINT HNF is taken on the full-row-rank graph matrix
    # [C^T | I_m], of shape m x (q+m); its canonical output contains the
    # primitive kernel in row HNF directly, without requesting a backend
    # transformation matrix.
    digit_work = (
        coefficient_count**2
        * (constraint_count + coefficient_count)
        * max(constraint_digits, graph_entry_digits)
    )
    if digit_work > MAX_INTEGER_KERNEL_DIGIT_WORK:
        raise _validation_error(
            "budget_exceeded",
            "the realized congruence matrix exceeds the exact primitive-kernel "
            f"digit-work bound of {MAX_INTEGER_KERNEL_DIGIT_WORK:,} units",
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
        _require_active_request("during constraint expansion")
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
    _require_integer_kernel_work_envelope(plan)
    return plan


def _integer_kernel_basis(plan: _ConstraintPlan) -> tuple[list[list[int]], int]:
    """Extract the primitive kernel from a canonical graph-lattice HNF.

    Rows of ``[C^T | I_m]`` form the graph of ``x -> x C^T`` inside
    ``ZZ^q + ZZ^m``. Its row HNF spans the same graph. Because the constraint
    columns come first, the rows whose first ``q`` entries vanish are exactly
    ``(0, x)`` for a row-HNF basis of ``ker_ZZ(C)``. The identity block makes
    the graph full-row-rank, so no arbitrary rational-nullspace scaling or
    separate saturation transform is materialized.
    """

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

    _bind_request_deadline()
    _require_active_request("before the graph-lattice HNF")

    from flint import fmpz_mat

    constraint_count = len(plan.constraints)
    graph = fmpz_mat(
        [
            [
                *(
                    plan.constraints[constraint][coordinate]
                    for constraint in range(constraint_count)
                ),
                *(int(coordinate == column) for column in range(coefficient_count)),
            ]
            for coordinate in range(coefficient_count)
        ]
    )
    graph_hnf = graph.hnf()
    primitive_kernel = [
        [
            int(graph_hnf[row, constraint_count + column])
            for column in range(coefficient_count)
        ]
        for row in range(coefficient_count)
        if all(
            graph_hnf[row, constraint] == 0 for constraint in range(constraint_count)
        )
    ]
    constraint_rank = coefficient_count - len(primitive_kernel)
    if not primitive_kernel:
        return [], constraint_rank
    return primitive_kernel, constraint_rank


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

    _bind_request_deadline()
    _require_active_request("before constraint expansion")
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
