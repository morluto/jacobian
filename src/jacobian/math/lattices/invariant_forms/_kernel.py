"""Exact kernel for integral forms fixed by rational congruence actions."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from math import ceil, gcd, lcm, log10
from time import monotonic
from typing import Any
from unicodedata import normalize

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    current_request_execution,
    request_cancelled,
)
from jacobian.canonical import (
    CanonicalizationError,
    format_canonical_integer,
    loads_strict_json,
    parse_canonical_integer,
)
from jacobian.math.lattices.invariant_forms._hnf_process import run_hnf_worker
from jacobian.math.lattices.invariant_forms._models import (
    MAX_CONSTRAINT_CELLS,
    MAX_CONSTRAINT_DIGIT_WORK,
    MAX_INTEGER_KERNEL_DIGIT_WORK,
    EmbeddedRealNumberFieldMatrixAction,
    FormKind,
    IntegralBilinearForm,
    InvariantBilinearFormLattice,
    MatrixAction,
    _validation_error,
    constraint_coefficient_count,
)
from jacobian.math.matrices._number_field import (
    EmbeddedNumberFieldRecognitionError,
    RecognizedRealSimpleNumberField,
    field_element_coordinates,
    field_element_from_value,
    recognize_real_simple_number_field,
)
from jacobian.math.matrices.values import MAX_MATRIX_SCALAR_DIGITS
from jacobian.math.number_theory.number_fields.values import (
    MAX_NUMBER_FIELD_EMBEDDING_DEGREE,
)

MAX_CONSTRAINT_COMPONENT_DIGITS = 65_536
MAX_STORED_CONSTRAINT_DIGITS = 2_000_000
MAX_INVARIANT_FORM_BASIS_CELLS = 2_000_000

_INVARIANT_FORM_WALL_SECONDS = 3600.0


def _require_active_request(stage: str, *, deadline: float | None = None) -> None:
    """Raise if the request deadline expired or was cancelled during *stage*."""

    if request_cancelled():
        raise OperationExecutionCancelledError(
            f"invariant-form lattice cancelled {stage}"
        )
    execution = current_request_execution()
    active_deadline = deadline
    if active_deadline is None and execution is not None:
        active_deadline = execution.deadline
    if active_deadline is not None and monotonic() >= active_deadline:
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
    expansion_digit_work: int = 0
    kernel_digit_work: int = 0


@dataclass(frozen=True, slots=True)
class _InvariantFormExecutionPlan:
    """One admitted invariant-form envelope with digit-work estimates."""

    expansion_digit_work: int
    kernel_digit_work: int
    constraint_plan: _ConstraintPlan
    recognized_field: RecognizedRealSimpleNumberField | None = None


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
    matrix: tuple[tuple[Any, ...], ...],
    *,
    equation_row: int,
    equation_column: int,
    position: CoefficientPosition,
    kind: FormKind,
) -> Any:
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
    action: MatrixAction,
    *,
    kind: FormKind,
    coefficient_count: int,
    coefficient_cells: int,
) -> int:
    """Bound Fraction construction, products, and row normalization by height."""

    if not action.generators or coefficient_count == 0:
        return 0
    if isinstance(action, EmbeddedRealNumberFieldMatrixAction):
        values = tuple(
            coordinate
            for generator in action.generators
            for row in generator.matrix.entries
            for value in row
            for coordinate in value.coefficients_ascending
        )
        field_coefficients = action.generators[
            0
        ].matrix.embedding.presentation.coefficients_descending
        field_digits = max(
            len(coefficient.lstrip("-")) for coefficient in field_coefficients
        )
        leading_coefficient = abs(int(field_coefficients[0]))
        leading_denominator_digits = (
            0 if leading_coefficient == 1 else len(str(leading_coefficient))
        )
        degree = action.generators[0].matrix.embedding.presentation.degree
    else:
        values = tuple(
            value
            for generator in action.generators
            for row in generator.matrix.entries
            for value in row
        )
        field_digits = 1
        leading_denominator_digits = 0
        degree = 1
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
    if isinstance(action, EmbeddedRealNumberFieldMatrixAction):
        product_denominator_digits = (
            2 * degree * denominator_growth_digits
            + max(degree - 1, 0) * leading_denominator_digits
        )
        product_numerator_digits = (
            2 * numerator_digits
            + 2 * max(degree - 1, 0) * denominator_growth_digits
            + max(degree - 1, 0) * field_digits
            + ceil(log10(max(degree, 1)))
            + degree * ceil(log10(degree + 1))
            + 2
        )
        coefficient_denominator_digits = transformed_terms * product_denominator_digits
        coefficient_numerator_digits = (
            max(
                product_numerator_digits
                + max(transformed_terms - 1, 0) * product_denominator_digits
                + int(transformed_terms == 2),
                coefficient_denominator_digits,
            )
            + 1
        )
    else:
        coefficient_denominator_digits = (
            2 * transformed_terms * denominator_growth_digits
        )
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
    digit_work = matrix_cells * degree * source_component_digits + coefficient_cells * (
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
    return digit_work


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
    *,
    dimension: int,
    coefficient_count: int,
    constraints: tuple[IntegerConstraint, ...],
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
    maximum_basis_count = coefficient_count - min(possible_nonfull_ranks, default=0)
    if maximum_basis_count * dimension * dimension > MAX_INVARIANT_FORM_BASIS_CELLS:
        raise _validation_error(
            "budget_exceeded",
            "the exact invariant-form basis exceeds the "
            f"{MAX_INVARIANT_FORM_BASIS_CELLS:,}-cell result bound",
        )


def _require_integer_kernel_work_envelope(plan: _ConstraintPlan) -> int:
    """Admit canonical graph-lattice HNF from the realized constraint plan."""

    if not plan.constraints or not plan.positions:
        return 0
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
    return digit_work


def _recognize_action_field(
    action: EmbeddedRealNumberFieldMatrixAction,
) -> RecognizedRealSimpleNumberField:
    if not action.generators:
        raise _validation_error(
            "empty_embedded_action",
            "an embedded action with no generators has no field to recognize",
        )
    try:
        return recognize_real_simple_number_field(action.generators[0].matrix.embedding)
    except EmbeddedNumberFieldRecognitionError as exc:
        raise _validation_error(exc.reason, str(exc)) from exc


def _build_constraint_plan(
    action: MatrixAction,
    kind: FormKind,
    *,
    recognized_field: RecognizedRealSimpleNumberField | None = None,
    deadline: float | None = None,
) -> _ConstraintPlan:
    if deadline is not None:
        _require_active_request("before constraint expansion", deadline=deadline)
    dimension = len(action.coordinate_axis)
    if isinstance(action, EmbeddedRealNumberFieldMatrixAction):
        field_degree = (
            action.generators[0].matrix.embedding.presentation.degree
            if action.generators
            else 1
        )
        if field_degree > MAX_NUMBER_FIELD_EMBEDDING_DEGREE:
            raise _validation_error(
                "budget_exceeded",
                "embedded invariant-form actions support field degree at most "
                f"{MAX_NUMBER_FIELD_EMBEDDING_DEGREE}",
            )
    else:
        field_degree = 1
    positions = _coefficient_positions(dimension, kind)
    cell_count = constraint_coefficient_count(dimension, len(action.generators), kind)
    if cell_count > MAX_CONSTRAINT_CELLS:
        raise _validation_error(
            "budget_exceeded",
            "the congruence expansion exceeds the structural bound of "
            f"{MAX_CONSTRAINT_CELLS} coefficients",
        )
    if not positions:
        _require_result_envelope(
            dimension=dimension,
            coefficient_count=0,
            constraints=(),
        )
        return _ConstraintPlan(
            positions=(),
            constraints=(),
            expansion_digit_work=0,
            kernel_digit_work=0,
        )
    cell_count = (
        constraint_coefficient_count(dimension, len(action.generators), kind)
        * field_degree
    )
    if cell_count > MAX_CONSTRAINT_CELLS:
        raise _validation_error(
            "budget_exceeded",
            "the congruence expansion exceeds the structural bound of "
            f"{MAX_CONSTRAINT_CELLS} coefficients",
        )
    expansion_digit_work = _require_constraint_expansion_envelope(
        action,
        kind=kind,
        coefficient_count=len(positions),
        coefficient_cells=cell_count,
    )
    constraints: set[IntegerConstraint] = set()
    stored_digits = 0
    recognized = (
        recognized_field or _recognize_action_field(action)
        if isinstance(action, EmbeddedRealNumberFieldMatrixAction) and action.generators
        else None
    )
    generator_matrices: tuple[tuple[tuple[Any, ...], ...], ...]
    if isinstance(action, EmbeddedRealNumberFieldMatrixAction):
        if action.generators:
            assert recognized is not None
            generator_matrices = tuple(
                tuple(
                    tuple(field_element_from_value(value, recognized) for value in row)
                    for row in generator.matrix.entries
                )
                for generator in action.generators
            )
        else:
            generator_matrices = ()
    else:
        generator_matrices = tuple(
            tuple(
                tuple(value.as_fraction() for value in row)
                for row in generator.matrix.entries
            )
            for generator in action.generators
        )
    for matrix in generator_matrices:
        _require_active_request(
            "during exact invariant-form constraint expansion", deadline=deadline
        )
        for equation_row in range(dimension):
            for equation_column in range(dimension):
                _require_active_request(
                    "during exact invariant-form constraint expansion",
                    deadline=deadline,
                )
                exact_row = tuple(
                    _constraint_coefficient(
                        matrix,
                        equation_row=equation_row,
                        equation_column=equation_column,
                        position=position,
                        kind=kind,
                    )
                    for position in positions
                )
                rational_rows = (
                    tuple(
                        tuple(
                            field_element_coordinates(value, recognized)[coordinate]
                            for value in exact_row
                        )
                        for coordinate in range(recognized.degree)
                    )
                    if recognized is not None
                    else (exact_row,)
                )
                for rational_row in rational_rows:
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
    _require_active_request(
        "after exact invariant-form constraint expansion", deadline=deadline
    )
    ordered_constraints = tuple(sorted(constraints))
    _require_result_envelope(
        dimension=dimension,
        coefficient_count=len(positions),
        constraints=ordered_constraints,
    )
    plan = _ConstraintPlan(
        positions=positions,
        constraints=ordered_constraints,
        expansion_digit_work=expansion_digit_work,
    )
    return replace(
        plan,
        kernel_digit_work=_require_integer_kernel_work_envelope(plan),
    )


def _integer_kernel_basis(
    plan: _ConstraintPlan,
    *,
    deadline: float | None = None,
) -> tuple[list[list[int]], int]:
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

    if deadline is None:
        deadline = _bind_request_deadline()
    _require_active_request("before the graph-lattice HNF")
    payload = json.dumps(
        {
            "coefficient_count": coefficient_count,
            "constraints": [
                [format_canonical_integer(value) for value in row]
                for row in plan.constraints
            ],
        },
        separators=(",", ":"),
    ).encode("utf-8")
    completed = run_hnf_worker(payload, deadline=deadline)
    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "invariant-form lattice cancelled during graph-lattice HNF"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "invariant-form lattice deadline expired during graph-lattice HNF"
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError("bounded invariant-form HNF worker did not return a basis")
    try:
        response = loads_strict_json(completed.stdout)
        primitive_kernel = []
        for row in response["primitive_kernel"]:
            decoded_row = []
            for value in row:
                if not isinstance(value, str):
                    raise ValueError("worker integer must be a canonical string")
                decoded = parse_canonical_integer(value)
                if format_canonical_integer(decoded) != value:
                    raise ValueError("worker integer is not canonical")
                decoded_row.append(decoded)
            primitive_kernel.append(decoded_row)
        constraint_rank = response["constraint_rank"]
    except (KeyError, TypeError, ValueError, CanonicalizationError) as exc:
        raise RuntimeError(
            "bounded invariant-form HNF worker returned malformed data"
        ) from exc
    if (
        not isinstance(primitive_kernel, list)
        or not isinstance(constraint_rank, int)
        or constraint_rank != coefficient_count - len(primitive_kernel)
        or any(
            not isinstance(row, list)
            or len(row) != coefficient_count
            or any(not isinstance(value, int) for value in row)
            for row in primitive_kernel
        )
    ):
        raise RuntimeError(
            "bounded invariant-form HNF worker returned invalid dimensions"
        )
    _require_active_request("after graph-lattice HNF")
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


def _admit_invariant_bilinear_form_lattice(
    action: MatrixAction,
    kind: FormKind,
    *,
    deadline: float | None = None,
) -> _InvariantFormExecutionPlan:
    """Validate the action envelope and return work estimates.

    Ensures the action is a structurally valid rational matrix action
    before the kernel builds and solves the congruence system.
    """

    if kind not in ("BILINEAR", "SYMMETRIC", "ALTERNATING"):
        raise _validation_error(
            "invalid_kind",
            "kind must be BILINEAR, SYMMETRIC, or ALTERNATING",
        )
    # Build the constraint plan to derive work estimates.  The plan
    # construction itself enforces the expansion and kernel digit-work
    # bounds via the existing _require_*_envelope helpers.
    recognized_field = (
        _recognize_action_field(action)
        if isinstance(action, EmbeddedRealNumberFieldMatrixAction) and action.generators
        else None
    )
    plan = _build_constraint_plan(
        action,
        kind,
        recognized_field=recognized_field,
        deadline=deadline,
    )
    return _InvariantFormExecutionPlan(
        expansion_digit_work=plan.expansion_digit_work,
        kernel_digit_work=plan.kernel_digit_work,
        constraint_plan=plan,
        recognized_field=recognized_field,
    )


def invariant_bilinear_form_lattice_kernel(
    action: MatrixAction,
    kind: FormKind,
    *,
    admission: _InvariantFormExecutionPlan | None = None,
    execution_checkpoint: Callable[[str], None] | None = None,
    recognized_field: Any = None,
    deadline: float | None = None,
) -> InvariantBilinearFormLattice:
    """Return the saturated integer lattice of forms fixed by every generator."""

    if deadline is None:
        deadline = _bind_request_deadline()
    checkpoint = execution_checkpoint or _require_active_request
    checkpoint("before constraint expansion")
    plan = (
        admission.constraint_plan
        if admission is not None
        else _build_constraint_plan(
            action,
            kind,
            recognized_field=recognized_field,
            deadline=deadline,
        )
    )
    basis, constraint_rank = _integer_kernel_basis(plan, deadline=deadline)
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


__all__ = [
    "_InvariantFormExecutionPlan",
    "_admit_invariant_bilinear_form_lattice",
    "invariant_bilinear_form_lattice_kernel",
]
