"""Exact native operations for lattice-presented complex tori."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any

from pydantic_core import PydanticCustomError
from sympy import QQ
from sympy.polys.matrices import DomainMatrix

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
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
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.complex_tori._models import (
    HermitianDefiniteness,
    HermitianInertia,
    LatticeComplexStructure,
    RiemannFormHodgeNonPositive,
    RiemannFormNotHodge,
    RiemannFormPositive,
    RiemannFormProfile,
    _validation_error,
)
from jacobian.math.lattices.invariant_forms._kernel import (
    _admit_invariant_bilinear_form_lattice,
    _InvariantFormExecutionPlan,
    invariant_bilinear_form_lattice_kernel,
)
from jacobian.math.lattices.invariant_forms._models import (
    EmbeddedRealNumberFieldActionGenerator,
    EmbeddedRealNumberFieldMatrixAction,
    IntegralBilinearForm,
    InvariantBilinearFormLattice,
    MatrixAction,
    RationalActionGenerator,
    RationalMatrixAction,
)
from jacobian.math.matrices._number_field import (
    EmbeddedNumberFieldRecognitionError,
    RecognizedRealSimpleNumberField,
    domain_matrix_from_embedded,
    embedded_matrix_from_domain,
    recognize_real_simple_number_field,
)
from jacobian.math.matrices.analysis._models import InertiaResult
from jacobian.math.matrices.analysis.operations import (
    _admit_inertia_from_bounds,
    _compute_inertia,
    _InertiaExecutionPlan,
)
from jacobian.math.matrices.operations import (
    _admit_exact_linear_matrix,
)
from jacobian.math.matrices.values import (
    EmbeddedRealSimpleNumberFieldMatrix,
    IntegerMatrix,
    RationalMatrix,
    SmithNormalForm,
    rational_matrix_from_fractions,
)
from jacobian.math.number_theory.number_fields.values import (
    MAX_NUMBER_FIELD_EMBEDDING_DEGREE,
    MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS,
)

MAX_COMPLEX_TORUS_SCALAR_WORK = 500_000_000
# The exact work ledgers remain the mathematical admission evidence. This
# generous wall limit is only a cooperative request-occupancy backstop around
# bounded in-process phases; it does not claim that SymPy or FLINT is preemptible.
_COMPLEX_TORUS_WALL_SECONDS = 3600.0
_SMITH_WORKER = Path(__file__).with_name("_smith_worker.py")
_SMITH_STDERR_LIMIT = 64 * 1024
type _ExecutionCheckpoint = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class _ScalarProductAdmission:
    """Source-derived height and work bounds for dense exact products."""

    algebraic: bool
    degree: int
    numerator_digits: int
    denominator_digits: int
    field_digits: int
    leading_digits: int
    digit_work: int


@dataclass(frozen=True, slots=True)
class _NeronSeveriExecutionPlan:
    """Complete pre-product envelope for one Neron-Severi computation."""

    invariant_forms: _InvariantFormExecutionPlan


@dataclass(frozen=True, slots=True)
class _RiemannFormExecutionPlan:
    """Complete pre-product envelope for one selected Riemann-form profile."""

    associated_inertia: _InertiaExecutionPlan


def _execution_deadline() -> float:
    """Bind one owner deadline measured from the original request start."""

    execution = current_request_execution()
    started_at = execution.started_at if execution is not None else monotonic()
    owner_deadline = started_at + _COMPLEX_TORUS_WALL_SECONDS
    deadline = (
        min(owner_deadline, execution.deadline)
        if execution is not None and execution.deadline is not None
        else owner_deadline
    )
    bind_request_deadline(deadline)
    return deadline


def _require_execution_active(deadline: float, phase: str) -> None:
    if request_cancelled():
        raise OperationExecutionCancelledError(f"request cancelled {phase}")
    if monotonic() >= deadline:
        raise OperationExecutionTimeoutError(f"complex-torus deadline expired {phase}")


def _smith_normal_form_killable(
    matrix: IntegerMatrix,
    *,
    deadline: float,
) -> SmithNormalForm:
    """Compute Riemann-form Smith data in a deadline-bound child process."""

    from jacobian.process import (
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
    )

    _require_execution_active(deadline, "before alternating Smith kernel")
    payload = json.dumps(
        {
            "entries": [
                [format_canonical_integer(parse_canonical_integer(value)) for value in row]
                for row in matrix.entries
            ]
        },
        separators=(",", ":"),
    ).encode("utf-8")
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise OperationExecutionTimeoutError(
            "complex-torus deadline expired before alternating Smith kernel"
        )
    with TemporaryDirectory(prefix="jacobian-riemann-smith-") as directory:
        completed = run_bounded_process(
            [sys.executable, str(_SMITH_WORKER)],
            input_bytes=payload,
            timeout_seconds=remaining,
            environment=worker_environment(locale="C.UTF-8"),
            stdout_limit=CanonicalLimits().max_output_bytes,
            stderr_limit=_SMITH_STDERR_LIMIT,
            resource_limits=ProcessResourceLimits(
                cpu_seconds=max(1, ceil(remaining)),
                address_space_bytes=1024 * 1024 * 1024,
                file_size_bytes=1024 * 1024,
            ),
            cwd=directory,
        )
    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "complex-torus request cancelled during alternating Smith kernel"
        )
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "complex-torus deadline expired during alternating Smith kernel"
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError("bounded alternating Smith worker did not return a form")
    try:
        response = loads_strict_json(completed.stdout)
        if response["request_digest"] != hashlib.sha256(payload).hexdigest():
            raise ValueError("worker request digest mismatch")
        raw_normal_form = response["normal_form"]
        if not isinstance(raw_normal_form, list):
            raise ValueError("worker Smith form must be a list")
        normal_form = []
        for row in raw_normal_form:
            if not isinstance(row, list):
                raise ValueError("worker Smith form rows must be lists")
            decoded_row = []
            for value in row:
                if not isinstance(value, str):
                    raise ValueError("worker integer must be a canonical string")
                decoded = parse_canonical_integer(value)
                if format_canonical_integer(decoded) != value:
                    raise ValueError("worker integer is not canonical")
                decoded_row.append(decoded)
            normal_form.append(decoded_row)
    except (KeyError, TypeError, ValueError, CanonicalizationError) as exc:
        raise RuntimeError(
            "bounded alternating Smith worker returned malformed data"
        ) from exc
    dimension = len(matrix.entries)
    if (
        len(normal_form) != dimension
        or any(len(row) != dimension for row in normal_form)
    ):
        raise RuntimeError(
            "bounded alternating Smith worker returned invalid dimensions"
        )
    diagonal = tuple(normal_form[index][index] for index in range(dimension))
    rank = next(
        (index for index, value in enumerate(diagonal) if value == 0), dimension
    )
    if any(value == 0 for value in diagonal[:rank]) or any(
        value != 0 for value in diagonal[rank:]
    ):
        raise RuntimeError(
            "bounded alternating Smith worker returned an invalid diagonal"
        )
    factors = tuple(abs(value) for value in diagonal[:rank])
    canonical_normal_form = IntegerMatrix(
        entries=tuple(
            tuple(
                format_canonical_integer(factors[row])
                if row == column and row < rank
                else "0"
                for column in range(dimension)
            )
            for row in range(dimension)
        )
    )
    _require_execution_active(deadline, "after alternating Smith kernel")
    return SmithNormalForm(
        normal_form=canonical_normal_form,
        rank=rank,
        invariant_factors=tuple(format_canonical_integer(value) for value in factors),
    )


def _scalar_height_ledger(
    torus: LatticeComplexStructure,
) -> tuple[bool, int, int, int, int, int]:
    matrix = torus.complex_structure
    if isinstance(matrix, EmbeddedRealSimpleNumberFieldMatrix):
        degree = matrix.embedding.presentation.degree
        if degree > MAX_NUMBER_FIELD_EMBEDDING_DEGREE:
            raise _validation_error(
                "budget_exceeded",
                "exact complex tori support field degree at most "
                f"{MAX_NUMBER_FIELD_EMBEDDING_DEGREE}",
            )
        coordinates = tuple(
            coordinate
            for row in matrix.entries
            for value in row
            for coordinate in value.coefficients_ascending
        )
        field_digits = max(
            len(coefficient.lstrip("-"))
            for coefficient in matrix.embedding.presentation.coefficients_descending
        )
        leading_digits = len(
            matrix.embedding.presentation.coefficients_descending[0].lstrip("-")
        )
        algebraic = True
    else:
        degree = 1
        coordinates = tuple(value for row in matrix.entries for value in row)
        field_digits = 1
        leading_digits = 1
        algebraic = False
    numerator_digits = max(len(value.num.lstrip("-")) for value in coordinates)
    denominator_digits = (
        0
        if all(value.den == "1" for value in coordinates)
        else max(len(value.den) for value in coordinates)
    )
    return (
        algebraic,
        degree,
        numerator_digits,
        denominator_digits,
        field_digits,
        leading_digits,
    )


def _complex_structure_scalar_work(
    torus: LatticeComplexStructure,
    *,
    matrix_products: int,
) -> _ScalarProductAdmission:
    """Admit exact scalar growth for a fixed number of dense products."""

    dimension = len(torus.coordinate_axis)
    (
        algebraic,
        degree,
        numerator_digits,
        denominator_digits,
        field_digits,
        leading_digits,
    ) = _scalar_height_ledger(torus)
    exact_scalar_work = (
        matrix_products
        * dimension**3
        * degree**2
        * (2 * max(numerator_digits, denominator_digits) + degree * field_digits + 8)
    )
    if exact_scalar_work > MAX_COMPLEX_TORUS_SCALAR_WORK:
        raise _validation_error(
            "budget_exceeded",
            "exact complex-structure products exceed the "
            f"{MAX_COMPLEX_TORUS_SCALAR_WORK:,}-unit scalar-work bound",
        )
    return _ScalarProductAdmission(
        algebraic=algebraic,
        degree=degree,
        numerator_digits=numerator_digits,
        denominator_digits=denominator_digits,
        field_digits=field_digits,
        leading_digits=leading_digits,
        digit_work=exact_scalar_work,
    )


def _require_result_envelope(*, predicted_result_bytes: int) -> None:
    if predicted_result_bytes > CanonicalLimits().max_output_bytes:
        raise _validation_error(
            "budget_exceeded",
            "the exact complex-torus result exceeds the canonical output envelope",
        )


def _complex_structure_action(torus: LatticeComplexStructure) -> MatrixAction:
    matrix = torus.complex_structure
    if isinstance(matrix, EmbeddedRealSimpleNumberFieldMatrix):
        return EmbeddedRealNumberFieldMatrixAction(
            coordinate_axis=torus.coordinate_axis,
            generators=(
                EmbeddedRealNumberFieldActionGenerator(
                    label="complex_structure",
                    matrix=matrix,
                ),
            ),
        )
    return RationalMatrixAction(
        coordinate_axis=torus.coordinate_axis,
        generators=(
            RationalActionGenerator(
                label="complex_structure",
                matrix=matrix,
            ),
        ),
    )


def _as_complex_torus_admission_error(exc: PydanticCustomError) -> PydanticCustomError:
    return _validation_error(exc.type.rsplit(".", 1)[-1], exc.message())


def _admit_neron_severi_execution(
    torus: LatticeComplexStructure,
) -> tuple[MatrixAction, _NeronSeveriExecutionPlan]:
    """Admit J^2, constraint expansion, graph HNF, and exact output."""

    products = _complex_structure_scalar_work(torus, matrix_products=1)
    action = _complex_structure_action(torus)
    try:
        invariant_forms = _admit_invariant_bilinear_form_lattice(
            action,
            "ALTERNATING",
        )
    except PydanticCustomError as exc:
        raise _as_complex_torus_admission_error(exc) from exc
    total_digit_work = (
        products.digit_work
        + invariant_forms.expansion_digit_work
        + invariant_forms.kernel_digit_work
    )
    if total_digit_work > MAX_COMPLEX_TORUS_SCALAR_WORK:
        raise _validation_error(
            "budget_exceeded",
            "the complete Neron-Severi computation exceeds the "
            f"{MAX_COMPLEX_TORUS_SCALAR_WORK:,}-unit exact-work bound",
        )
    return action, _NeronSeveriExecutionPlan(invariant_forms=invariant_forms)


def _admit_riemann_form_execution(
    torus: LatticeComplexStructure,
    form: IntegralBilinearForm,
) -> _RiemannFormExecutionPlan:
    """Admit every exact product, Smith input, and retained profile value."""

    dimension = len(torus.coordinate_axis)
    products = _complex_structure_scalar_work(torus, matrix_products=4)
    try:
        _admit_exact_linear_matrix(form.matrix.entries)
    except PydanticCustomError as exc:
        raise _as_complex_torus_admission_error(exc) from exc
    form_digits = max(
        len(value.lstrip("-")) for row in form.matrix.entries for value in row
    )
    associated_numerator_digits = (
        products.numerator_digits
        + form_digits
        + max(dimension - 1, 0) * products.denominator_digits
        + len(str(dimension))
        + 2
    )
    associated_denominator_digits = dimension * products.denominator_digits
    associated_digits = max(
        associated_numerator_digits,
        max(1, associated_denominator_digits),
    )
    output_component_limit = (
        MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS
        if isinstance(torus.complex_structure, EmbeddedRealSimpleNumberFieldMatrix)
        else 4_096
    )
    if associated_digits > output_component_limit:
        raise _validation_error(
            "budget_exceeded",
            "the associated symmetric form can exceed the canonical exact-real "
            "matrix component bound",
        )
    presentation_bytes = (
        len(
            encode_strict_json(
                torus.complex_structure.embedding.presentation.model_dump(mode="json")
            )
        )
        if isinstance(torus.complex_structure, EmbeddedRealSimpleNumberFieldMatrix)
        else 0
    )
    per_associated_entry = (
        presentation_bytes + 256 + products.degree * (2 * associated_digits + 32)
    )
    predicted_result_bytes = (
        len(
            encode_strict_json(
                {
                    "torus": torus.model_dump(mode="json"),
                    "form": form.model_dump(mode="json"),
                }
            )
        )
        + 8_192
        + dimension**2 * (per_associated_entry + form_digits + 16)
    )
    _require_result_envelope(
        predicted_result_bytes=predicted_result_bytes,
    )
    try:
        associated_inertia = _admit_inertia_from_bounds(
            order=dimension,
            algebraic=products.algebraic,
            algebraic_degree=products.degree,
            numerator_digits=associated_numerator_digits,
            denominator_digits=associated_denominator_digits,
            field_digits=products.field_digits,
            leading_digits=products.leading_digits,
            zero_matrix=all(
                value == "0" for row in form.matrix.entries for value in row
            ),
        )
    except PydanticCustomError as exc:
        raise _as_complex_torus_admission_error(exc) from exc
    smith_digit_work = dimension**3 * form_digits
    if (
        products.digit_work + associated_inertia.digit_work + smith_digit_work
        > MAX_COMPLEX_TORUS_SCALAR_WORK
    ):
        raise _validation_error(
            "budget_exceeded",
            "the complete Riemann-form profile exceeds the "
            f"{MAX_COMPLEX_TORUS_SCALAR_WORK:,}-unit exact-work bound",
        )
    return _RiemannFormExecutionPlan(associated_inertia=associated_inertia)


def _rational_domain_matrix(matrix: RationalMatrix) -> DomainMatrix:
    rows = [
        [
            QQ(parse_canonical_integer(value.num), parse_canonical_integer(value.den))
            for value in row
        ]
        for row in matrix.entries
    ]
    return DomainMatrix(rows, (len(rows), len(rows[0])), QQ)


def _require_complex_structure(
    torus: LatticeComplexStructure,
    *,
    execution_checkpoint: _ExecutionCheckpoint,
    recognized_field: RecognizedRealSimpleNumberField | None = None,
) -> RecognizedRealSimpleNumberField | None:
    """Recognize the scalar domain and establish ``J^2 = -I`` exactly."""

    execution_checkpoint("before exact complex-structure recognition")
    matrix = torus.complex_structure
    try:
        if isinstance(matrix, EmbeddedRealSimpleNumberFieldMatrix):
            recognized = recognized_field or recognize_real_simple_number_field(
                matrix.embedding
            )
            domain_matrix = domain_matrix_from_embedded(matrix, recognized)
        else:
            recognized = None
            domain_matrix = _rational_domain_matrix(matrix)
    except EmbeddedNumberFieldRecognitionError as exc:
        raise _validation_error(exc.reason, str(exc)) from exc

    dimension = len(torus.coordinate_axis)
    expected = -DomainMatrix.eye(
        (dimension, dimension), domain_matrix.domain
    ).to_dense()
    if domain_matrix.matmul(domain_matrix) != expected:
        raise _validation_error(
            "not_complex_structure",
            "the exact complex-structure matrix must satisfy J^2 = -I",
        )
    execution_checkpoint("after exact complex-structure recognition")
    return recognized


def compute_neron_severi_lattice(
    torus: LatticeComplexStructure,
) -> InvariantBilinearFormLattice:
    """Compute every integral alternating Hodge ``(1,1)`` form exactly."""

    deadline = _execution_deadline()
    execution_checkpoint = partial(_require_execution_active, deadline)
    try:
        execution_checkpoint("before Neron-Severi semantic admission")
        action, plan = _admit_neron_severi_execution(torus)
        execution_checkpoint("after Neron-Severi semantic admission")
        _require_complex_structure(
            torus,
            execution_checkpoint=execution_checkpoint,
            recognized_field=plan.invariant_forms.recognized_field,
        )
        execution_checkpoint("before the integral Hodge-lattice kernel")
        result = invariant_bilinear_form_lattice_kernel(
            action,
            "ALTERNATING",
            admission=plan.invariant_forms,
            execution_checkpoint=execution_checkpoint,
            deadline=deadline,
        )
        execution_checkpoint("after the integral Hodge-lattice kernel")
        return result
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("torus",),
            code=exc.type,
            message=exc.message(),
        ) from exc


def _integer_form_domain_matrix(
    form: IntegralBilinearForm,
    domain: Any,
) -> DomainMatrix:
    entries = [
        [domain.convert(int(value)) for value in row] for row in form.matrix.entries
    ]
    return DomainMatrix(entries, (len(entries), len(entries[0])), domain)


def _rational_matrix_from_domain(matrix: DomainMatrix) -> RationalMatrix:
    dense = matrix.to_dense().rep.to_ddm()
    return rational_matrix_from_fractions(
        [
            [Fraction(int(value.numerator), int(value.denominator)) for value in row]
            for row in dense
        ]
    )


def _hermitian_inertia(real_inertia: InertiaResult) -> HermitianInertia:
    positive = real_inertia.n_positive // 2
    negative = real_inertia.n_negative // 2
    zero = real_inertia.n_zero // 2
    if positive == 0 and negative == 0:
        definiteness: HermitianDefiniteness = "zero"
    elif zero == 0:
        if negative == 0:
            definiteness = "positive_definite"
        elif positive == 0:
            definiteness = "negative_definite"
        else:
            definiteness = "indefinite"
    elif negative == 0:
        definiteness = "positive_semidefinite"
    elif positive == 0:
        definiteness = "negative_semidefinite"
    else:
        definiteness = "indefinite"
    return HermitianInertia(
        n_positive=positive,
        n_negative=negative,
        n_zero=zero,
        definiteness=definiteness,
    )


def compute_riemann_form_profile(
    torus: LatticeComplexStructure,
    form: IntegralBilinearForm,
) -> RiemannFormProfile:
    """Classify one integral alternating form under the standard Riemann sign."""

    deadline = _execution_deadline()
    execution_checkpoint = partial(_require_execution_active, deadline)
    try:
        execution_checkpoint("before Riemann-form semantic admission")
        if form.coordinate_axis != torus.coordinate_axis:
            raise _validation_error(
                "form_axis",
                "the selected form must use the complex torus coordinate axis",
            )
        if form.kind != "ALTERNATING":
            raise _validation_error(
                "form_kind", "a Riemann-form profile requires an alternating form"
            )
        plan = _admit_riemann_form_execution(torus, form)
        execution_checkpoint("after Riemann-form semantic admission")
        recognized = _require_complex_structure(
            torus,
            execution_checkpoint=execution_checkpoint,
        )
        matrix = torus.complex_structure
        if isinstance(matrix, EmbeddedRealSimpleNumberFieldMatrix):
            if recognized is None:
                raise RuntimeError("an algebraic torus lost its recognized field")
            complex_structure = domain_matrix_from_embedded(matrix, recognized)
            form_matrix = _integer_form_domain_matrix(form, recognized.field)
        else:
            complex_structure = _rational_domain_matrix(matrix)
            form_matrix = _integer_form_domain_matrix(form, QQ)

        execution_checkpoint("before the Hodge compatibility products")
        transformed = (
            complex_structure.transpose()
            .matmul(form_matrix)
            .matmul(complex_structure)
            .to_dense()
        )
        execution_checkpoint("after the Hodge compatibility products")
        hodge_type_11 = transformed == form_matrix.to_dense()
        inertia: InertiaResult | None = None
        if hodge_type_11:
            associated = complex_structure.transpose().matmul(form_matrix).to_dense()
            if associated != associated.transpose().to_dense():
                raise RuntimeError("a compatible associated Riemann form lost symmetry")
            associated_matrix = (
                embedded_matrix_from_domain(associated, recognized)
                if recognized is not None
                else _rational_matrix_from_domain(associated)
            )
            execution_checkpoint("before exact associated-form inertia")
            inertia = _compute_inertia(
                associated_matrix,
                admission=plan.associated_inertia,
                recognized_field=recognized,
                execution_checkpoint=execution_checkpoint,
            )
            execution_checkpoint("after exact associated-form inertia")
        execution_checkpoint("before the alternating Smith kernel")
        smith = _smith_normal_form_killable(form.matrix, deadline=deadline)
        execution_checkpoint("after the alternating Smith kernel")
        factors = smith.invariant_factors
        if len(factors) % 2 or any(
            factors[index] != factors[index + 1] for index in range(0, len(factors), 2)
        ):
            raise RuntimeError("Smith factors of an alternating form were not paired")
        elementary_divisors = factors[::2]
        degenerate = smith.rank < len(torus.coordinate_axis)
        if not hodge_type_11:
            result = RiemannFormProfile(
                torus=torus,
                form=form,
                smith_normal_form=smith,
                alternating_elementary_divisors=elementary_divisors,
                is_degenerate=degenerate,
                outcome=RiemannFormNotHodge(status="NOT_HODGE"),
            )
            execution_checkpoint("after Riemann-form result construction")
            return result

        if inertia is None:
            raise RuntimeError("a Hodge form lost its admitted inertia result")
        if any(
            count % 2
            for count in (inertia.n_positive, inertia.n_negative, inertia.n_zero)
        ):
            raise RuntimeError("a Hermitian realification had odd inertia")
        is_riemann = inertia.n_positive == len(torus.coordinate_axis)
        outcome = (
            RiemannFormPositive(
                status="RIEMANN_FORM",
                associated_form_inertia=inertia,
                hermitian_inertia=_hermitian_inertia(inertia),
                polarization_type=elementary_divisors,
            )
            if is_riemann
            else RiemannFormHodgeNonPositive(
                status="HODGE_NON_POSITIVE",
                associated_form_inertia=inertia,
                hermitian_inertia=_hermitian_inertia(inertia),
            )
        )
        result = RiemannFormProfile(
            torus=torus,
            form=form,
            smith_normal_form=smith,
            alternating_elementary_divisors=elementary_divisors,
            is_degenerate=degenerate,
            outcome=outcome,
        )
        execution_checkpoint("after Riemann-form result construction")
        return result
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("torus", "form"),
            code=exc.type,
            message=exc.message(),
        ) from exc


__all__ = [
    "compute_neron_severi_lattice",
    "compute_riemann_form_profile",
]
