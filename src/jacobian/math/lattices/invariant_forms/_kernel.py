"""Exact kernel for integral forms fixed by rational congruence actions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, gcd, lcm, log10
from typing import Any

from jacobian._execution import (
    OperationExecutionCancelledError,
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
MAX_INVARIANT_FORM_RESULT_BYTES = CanonicalLimits().max_output_bytes

type CoefficientPosition = tuple[int, int]
type IntegerConstraint = tuple[int, ...]
type _ExecutionCheckpoint = Callable[[str], None]


def _require_execution_active(phase: str) -> None:
    if request_cancelled():
        raise OperationExecutionCancelledError(f"request cancelled {phase}")


@dataclass(frozen=True, slots=True)
class _ConstraintExpansionEnvelope:
    """Source-derived height and work bounds for congruence expansion."""

    maximum_component_digits: int
    digit_work: int


@dataclass(frozen=True, slots=True)
class _ConstraintPlan:
    """One admitted exact constraint matrix and its coefficient coordinates."""

    positions: tuple[CoefficientPosition, ...]
    constraints: tuple[IntegerConstraint, ...]


@dataclass(frozen=True, slots=True)
class _InvariantFormExecutionPlan:
    """One source admission reused by constraint construction and graph HNF."""

    positions: tuple[CoefficientPosition, ...]
    maximum_constraint_count: int
    maximum_constraint_digits: int
    expansion_digit_work: int
    kernel_digit_work: int


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
) -> _ConstraintExpansionEnvelope:
    """Bound Fraction construction, products, and row normalization by height."""

    if not action.generators or coefficient_count == 0:
        return _ConstraintExpansionEnvelope(
            maximum_component_digits=1,
            digit_work=0,
        )
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
        # Put each degree-d source element over one common denominator, multiply
        # two such polynomials, then reduce at most d-1 times by the primitive
        # defining polynomial. A nonmonic leading coefficient contributes the
        # only new reduction denominator. Combining the at-most-two transformed
        # products may multiply their independently bounded denominators.
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
    return _ConstraintExpansionEnvelope(
        maximum_component_digits=cleared_component_digits,
        digit_work=digit_work,
    )


def _retained_action_bytes(action: MatrixAction) -> int:
    """Measure the exact retained source before expanding any constraints."""

    try:
        source_bytes = len(encode_strict_json(action.model_dump(mode="json")))
    except CanonicalizationError:
        raise _validation_error(
            "budget_exceeded",
            "the retained exact matrix action exceeds the canonical output limit",
        ) from None
    if source_bytes + 4_096 > MAX_INVARIANT_FORM_RESULT_BYTES:
        raise _validation_error(
            "budget_exceeded",
            "the retained exact matrix action leaves no room for the canonical result",
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
    action: MatrixAction,
    *,
    coefficient_count: int,
    constraint_count: int,
    constraint_digits: int,
    source_bytes: int,
) -> None:
    """Prove a conservative exact-output envelope before graph-HNF work."""

    rank_ceiling = min(constraint_count, max(coefficient_count - 1, 0))
    # A source-derived constraint-count ceiling may be positive even when every
    # realized equation vanishes, so rank zero must always remain reachable.
    possible_nonfull_ranks = range(rank_ceiling + 1)
    maximum_entry_digits = _kernel_entry_digit_bound(
        coefficient_count=coefficient_count,
        constraint_count=constraint_count,
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


def _require_integer_kernel_work_envelope(
    *,
    coefficient_count: int,
    constraint_count: int,
    constraint_digits: int,
) -> int:
    """Admit canonical graph-lattice HNF from bounded matrix dimensions."""

    if constraint_count == 0 or coefficient_count == 0:
        return 0
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


def _all_generators_are_identity(action: MatrixAction) -> bool:
    """Recognize the exact identity case without multiplying source scalars."""

    if isinstance(action, EmbeddedRealNumberFieldMatrixAction):
        return all(
            all(
                (
                    value.coefficients_ascending[0].num == "1"
                    and value.coefficients_ascending[0].den == "1"
                    and all(
                        coordinate.num == "0"
                        for coordinate in value.coefficients_ascending[1:]
                    )
                    if row == column
                    else all(
                        coordinate.num == "0"
                        for coordinate in value.coefficients_ascending
                    )
                )
                for row, entries in enumerate(generator.matrix.entries)
                for column, value in enumerate(entries)
            )
            for generator in action.generators
        )
    return all(
        all(
            value.num == str(int(row == column)) and value.den == "1"
            for row, entries in enumerate(generator.matrix.entries)
            for column, value in enumerate(entries)
        )
        for generator in action.generators
    )


def _admit_invariant_bilinear_form_lattice(
    action: MatrixAction,
    kind: FormKind,
) -> _InvariantFormExecutionPlan:
    """Derive one reusable source plan without constructing a constraint."""

    dimension = len(action.coordinate_axis)
    if (
        isinstance(action, EmbeddedRealNumberFieldMatrixAction)
        and action.generators[0].matrix.embedding.presentation.degree
        > MAX_NUMBER_FIELD_EMBEDDING_DEGREE
    ):
        raise _validation_error(
            "budget_exceeded",
            "embedded invariant-form actions support field degree at most "
            f"{MAX_NUMBER_FIELD_EMBEDDING_DEGREE}",
        )
    positions = _coefficient_positions(dimension, kind)
    source_bytes = _retained_action_bytes(action)
    field_degree = (
        action.generators[0].matrix.embedding.presentation.degree
        if isinstance(action, EmbeddedRealNumberFieldMatrixAction)
        else 1
    )
    coefficient_cells = (
        constraint_coefficient_count(dimension, len(action.generators), kind)
        * field_degree
    )
    if coefficient_cells > MAX_CONSTRAINT_CELLS:
        raise _validation_error(
            "budget_exceeded",
            "the congruence expansion exceeds the structural bound of "
            f"{MAX_CONSTRAINT_CELLS} coefficients",
        )
    expansion = _require_constraint_expansion_envelope(
        action,
        kind=kind,
        coefficient_count=len(positions),
        coefficient_cells=coefficient_cells,
    )
    maximum_constraint_count = (
        0
        if _all_generators_are_identity(action)
        else len(action.generators) * dimension * dimension * field_degree
    )
    maximum_stored_digits = (
        maximum_constraint_count * len(positions) * expansion.maximum_component_digits
    )
    if maximum_stored_digits > MAX_STORED_CONSTRAINT_DIGITS:
        raise _validation_error(
            "budget_exceeded",
            "the source congruence system can exceed the exact "
            f"{MAX_STORED_CONSTRAINT_DIGITS}-digit intermediate bound",
        )
    _require_result_envelope(
        action,
        coefficient_count=len(positions),
        constraint_count=maximum_constraint_count,
        constraint_digits=expansion.maximum_component_digits,
        source_bytes=source_bytes,
    )
    kernel_digit_work = _require_integer_kernel_work_envelope(
        coefficient_count=len(positions),
        constraint_count=maximum_constraint_count,
        constraint_digits=expansion.maximum_component_digits,
    )
    return _InvariantFormExecutionPlan(
        positions=positions,
        maximum_constraint_count=maximum_constraint_count,
        maximum_constraint_digits=expansion.maximum_component_digits,
        expansion_digit_work=expansion.digit_work,
        kernel_digit_work=kernel_digit_work,
    )


def _recognize_action_field(
    action: EmbeddedRealNumberFieldMatrixAction,
) -> RecognizedRealSimpleNumberField:
    try:
        return recognize_real_simple_number_field(action.generators[0].matrix.embedding)
    except EmbeddedNumberFieldRecognitionError as exc:
        raise _validation_error(exc.reason, str(exc)) from exc


def _build_constraint_plan(
    action: MatrixAction,
    kind: FormKind,
    *,
    admission: _InvariantFormExecutionPlan,
    recognized_field: RecognizedRealSimpleNumberField | None = None,
    execution_checkpoint: _ExecutionCheckpoint = _require_execution_active,
) -> _ConstraintPlan:
    dimension = len(action.coordinate_axis)
    positions = admission.positions
    if not positions:
        return _ConstraintPlan(positions=(), constraints=())
    constraints: set[IntegerConstraint] = set()
    stored_digits = 0
    recognized: RecognizedRealSimpleNumberField | None
    generator_matrices: tuple[tuple[tuple[Any, ...], ...], ...]
    if isinstance(action, EmbeddedRealNumberFieldMatrixAction):
        execution_checkpoint("before invariant-form number-field recognition")
        recognized = recognized_field or _recognize_action_field(action)
        if recognized.embedding != action.generators[0].matrix.embedding:
            raise RuntimeError("recognized field does not match the action embedding")
        generator_matrices = tuple(
            tuple(
                tuple(field_element_from_value(value, recognized) for value in row)
                for row in generator.matrix.entries
            )
            for generator in action.generators
        )
        execution_checkpoint("after invariant-form number-field recognition")
    else:
        recognized = None
        generator_matrices = tuple(
            tuple(
                tuple(value.as_fraction() for value in row)
                for row in generator.matrix.entries
            )
            for generator in action.generators
        )
    for matrix in generator_matrices:
        for equation_row in range(dimension):
            execution_checkpoint("during exact invariant-form constraint expansion")
            for equation_column in range(dimension):
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
                rational_rows: tuple[tuple[Fraction, ...], ...] = (
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
                        raise RuntimeError(
                            "the admitted invariant-form storage bound was exceeded"
                        )
                    constraints.add(constraint)
    ordered_constraints = tuple(sorted(constraints))
    constraint_digits = (
        max(
            _integer_digit_count(value)
            for constraint in ordered_constraints
            for value in constraint
        )
        if ordered_constraints
        else 1
    )
    if (
        len(ordered_constraints) > admission.maximum_constraint_count
        or constraint_digits > admission.maximum_constraint_digits
    ):
        raise RuntimeError("the invariant-form source admission was not conservative")
    plan = _ConstraintPlan(positions=positions, constraints=ordered_constraints)
    return plan


def _integer_kernel_basis(
    plan: _ConstraintPlan,
    *,
    execution_checkpoint: _ExecutionCheckpoint,
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

    from flint import fmpz_mat

    execution_checkpoint("before the primitive FLINT graph-HNF kernel")
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
    execution_checkpoint("after the primitive FLINT graph-HNF kernel")
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
    action: MatrixAction,
    kind: FormKind,
    *,
    admission: _InvariantFormExecutionPlan,
    recognized_field: RecognizedRealSimpleNumberField | None = None,
    execution_checkpoint: _ExecutionCheckpoint = _require_execution_active,
) -> InvariantBilinearFormLattice:
    """Return the saturated integer lattice of forms fixed by every generator."""

    plan = _build_constraint_plan(
        action,
        kind,
        admission=admission,
        recognized_field=recognized_field,
        execution_checkpoint=execution_checkpoint,
    )
    basis, constraint_rank = _integer_kernel_basis(
        plan,
        execution_checkpoint=execution_checkpoint,
    )
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
    result = InvariantBilinearFormLattice._from_kernel(
        action=action,
        kind=kind,
        coefficient_dimension=len(plan.positions),
        constraint_rank=constraint_rank,
        basis_forms=basis_forms,
    )
    execution_checkpoint("after invariant-form result construction")
    return result


__all__ = ["invariant_bilinear_form_lattice_kernel"]
