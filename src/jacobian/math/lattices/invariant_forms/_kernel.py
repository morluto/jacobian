"""Exact kernel for integral forms fixed by rational congruence actions."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, gcd, lcm, log10
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
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
    loads_strict_json,
    parse_canonical_integer,
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
_HNF_WORKER = Path(__file__).with_name("_hnf_worker.py")
_HNF_STDERR_LIMIT = 64 * 1024


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


def _retained_action_bytes(
    action: RationalMatrixAction, *, deadline: float | None = None
) -> int:
    """Measure the exact retained source before expanding any constraints."""

    # Preflight the expanded source-byte budget before model_dump and
    # encode_strict_json materialize the full canonical representation.
    # Accumulate a conservative size over every generator, not just the
    # first, since later generators may carry much larger rationals.
    generator_count = len(action.generators)
    if generator_count > 0:
        estimated_bytes = 4_096 + sum(
            len(encode_strict_json(label)) + 1 for label in action.coordinate_axis
        )
        for generator in action.generators:
            _require_active_request("during retained action sizing", deadline=deadline)
            estimated_bytes += len(encode_strict_json(generator.label)) + 64
            for row in generator.matrix.entries:
                estimated_bytes += 32
                for value in row:
                    estimated_bytes += len(value.num) + len(value.den) + 20
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
    action: RationalMatrixAction,
    kind: FormKind,
    *,
    deadline: float | None = None,
) -> _ConstraintPlan:
    dimension = len(action.coordinate_axis)
    positions = _coefficient_positions(dimension, kind)
    cell_count = constraint_coefficient_count(dimension, len(action.generators), kind)
    if cell_count > MAX_CONSTRAINT_CELLS:
        raise _validation_error(
            "budget_exceeded",
            "the congruence expansion exceeds the structural bound of "
            f"{MAX_CONSTRAINT_CELLS} coefficients",
        )
    source_bytes = _retained_action_bytes(action, deadline=deadline)
    if not positions:
        _require_result_envelope(
            action,
            coefficient_count=0,
            constraints=(),
            source_bytes=source_bytes,
        )
        return _ConstraintPlan(positions=(), constraints=())
    _require_constraint_expansion_envelope(
        action,
        kind=kind,
        coefficient_count=len(positions),
        coefficient_cells=cell_count,
    )
    constraints: set[IntegerConstraint] = set()
    stored_digits = 0
    for generator in action.generators:
        _require_active_request("during constraint expansion", deadline=deadline)
        matrix = tuple(
            tuple(value.as_fraction() for value in row)
            for row in generator.matrix.entries
        )
        for equation_row in range(dimension):
            for equation_column in range(dimension):
                _require_active_request(
                    "during constraint expansion", deadline=deadline
                )
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
    _require_active_request("after constraint expansion", deadline=deadline)
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


def _integer_kernel_basis(  # noqa: C901
    plan: _ConstraintPlan, *, deadline: float | None = None
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
    _require_active_request("before the graph-lattice HNF", deadline=deadline)
    from jacobian.process import (
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
    )

    remaining = deadline - monotonic()
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            "invariant-form lattice deadline expired before graph-lattice HNF"
        )
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
    request_digest = hashlib.sha256(payload).hexdigest()
    with TemporaryDirectory(prefix="jacobian-invariant-form-hnf-") as directory:
        completed = run_bounded_process(
            [sys.executable, str(_HNF_WORKER)],
            input_bytes=payload,
            timeout_seconds=remaining,
            environment=worker_environment(locale="C.UTF-8"),
            stdout_limit=MAX_INVARIANT_FORM_RESULT_BYTES,
            stderr_limit=_HNF_STDERR_LIMIT,
            resource_limits=ProcessResourceLimits(
                cpu_seconds=max(1, ceil(_INVARIANT_FORM_WALL_SECONDS)),
                address_space_bytes=1024 * 1024 * 1024,
                file_size_bytes=1024 * 1024,
            ),
            cwd=directory,
        )
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
        if response.get("request_digest") != request_digest:
            raise ValueError("worker result is not bound to the admitted constraints")
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
    _require_active_request("after graph-lattice HNF", deadline=deadline)
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

    deadline = _bind_request_deadline()
    _require_active_request("before constraint expansion", deadline=deadline)
    plan = _build_constraint_plan(action, kind, deadline=deadline)
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
    _require_active_request("after result construction", deadline=deadline)
    return InvariantBilinearFormLattice._from_kernel(
        action=action,
        kind=kind,
        coefficient_dimension=len(plan.positions),
        constraint_rank=constraint_rank,
        basis_forms=basis_forms,
    )


__all__ = ["invariant_bilinear_form_lattice_kernel"]
