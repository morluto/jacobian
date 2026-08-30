"""Exact native operations for lattice-presented complex tori."""

from __future__ import annotations

import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from pydantic_core import PydanticCustomError
from sympy import QQ
from sympy.polys.matrices import DomainMatrix

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    current_request_execution,
    request_cancelled,
)
from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.complex_tori._models import (
    HermitianDefiniteness,
    HermitianInertia,
    LatticeComplexStructure,
    RiemannFormHodgeType11,
    RiemannFormNotHodgeType11,
    RiemannFormProfile,
    _validation_error,
)
from jacobian.math.lattices.invariant_forms._kernel import (
    invariant_bilinear_form_lattice_kernel,
)
from jacobian.math.lattices.invariant_forms._models import (
    MAX_CONSTRAINT_CELLS,
    EmbeddedRealNumberFieldActionGenerator,
    EmbeddedRealNumberFieldMatrixAction,
    IntegralBilinearForm,
    InvariantBilinearFormLattice,
    MatrixAction,
    RationalActionGenerator,
    RationalMatrixAction,
    constraint_coefficient_count,
)
from jacobian.math.matrices._number_field import (
    EmbeddedNumberFieldRecognitionError,
    RecognizedRealSimpleNumberField,
    domain_matrix_from_embedded,
    embedded_matrix_from_domain,
    recognize_real_simple_number_field,
)
from jacobian.math.matrices.analysis._models import InertiaResult
from jacobian.math.matrices.analysis.operations import _compute_inertia
from jacobian.math.matrices.operations import (
    _admit_exact_linear_matrix,
    _smith_normal_form_kernel,
)
from jacobian.math.matrices.values import (
    EmbeddedRealSimpleNumberFieldMatrix,
    RationalMatrix,
    rational_matrix_from_fractions,
)
from jacobian.math.number_theory.number_fields.values import (
    MAX_NUMBER_FIELD_EMBEDDING_DEGREE,
    MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS,
)

MAX_COMPLEX_TORUS_SCALAR_WORK = 500_000_000


@dataclass(frozen=True, slots=True)
class _ComplexTorusExecutionPlan:
    """One pre-backend work ledger bound to the ambient request deadline."""

    deadline: float | None
    field_degree: int
    exact_scalar_work: int
    hodge_constraint_cells: int
    predicted_result_bytes: int


def _require_execution_active(
    plan: _ComplexTorusExecutionPlan,
    phase: str,
) -> None:
    if request_cancelled():
        raise OperationExecutionCancelledError(f"request cancelled {phase}")
    if plan.deadline is not None and plan.deadline <= time.monotonic():
        raise OperationExecutionTimeoutError(f"request deadline expired {phase}")


def _scalar_height_ledger(
    torus: LatticeComplexStructure,
) -> tuple[int, int, int, int]:
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
    else:
        degree = 1
        coordinates = tuple(value for row in matrix.entries for value in row)
        field_digits = 1
    numerator_digits = max(len(value.num.lstrip("-")) for value in coordinates)
    denominator_digits = max(len(value.den) for value in coordinates)
    return degree, numerator_digits, denominator_digits, field_digits


def _complex_structure_scalar_work(
    torus: LatticeComplexStructure,
    *,
    matrix_products: int,
) -> tuple[int, int, int, int]:
    """Admit exact scalar growth for a fixed number of dense products."""

    dimension = len(torus.coordinate_axis)
    degree, numerator_digits, denominator_digits, field_digits = _scalar_height_ledger(
        torus
    )
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
    return degree, numerator_digits, denominator_digits, exact_scalar_work


def _execution_plan(
    *,
    field_degree: int,
    exact_scalar_work: int,
    hodge_constraint_cells: int,
    predicted_result_bytes: int,
) -> _ComplexTorusExecutionPlan:
    if predicted_result_bytes > CanonicalLimits().max_output_bytes:
        raise _validation_error(
            "budget_exceeded",
            "the exact complex-torus result exceeds the canonical output envelope",
        )
    execution = current_request_execution()
    return _ComplexTorusExecutionPlan(
        deadline=execution.deadline if execution is not None else None,
        field_degree=field_degree,
        exact_scalar_work=exact_scalar_work,
        hodge_constraint_cells=hodge_constraint_cells,
        predicted_result_bytes=predicted_result_bytes,
    )


def _admit_neron_severi_execution(
    torus: LatticeComplexStructure,
) -> _ComplexTorusExecutionPlan:
    """Admit J^2 and the degree-expanded alternating constraint system."""

    dimension = len(torus.coordinate_axis)
    degree, _, _, exact_scalar_work = _complex_structure_scalar_work(
        torus,
        matrix_products=1,
    )
    constraint_cells = (
        constraint_coefficient_count(dimension, 1, "ALTERNATING") * degree
    )
    if constraint_cells > MAX_CONSTRAINT_CELLS:
        raise _validation_error(
            "budget_exceeded",
            "the algebraic Hodge constraint expansion exceeds the structural "
            f"bound of {MAX_CONSTRAINT_CELLS} coefficients",
        )
    return _execution_plan(
        field_degree=degree,
        exact_scalar_work=exact_scalar_work,
        hodge_constraint_cells=constraint_cells,
        predicted_result_bytes=(
            len(encode_strict_json({"torus": torus.model_dump(mode="json")})) + 4_096
        ),
    )


def _admit_riemann_form_execution(
    torus: LatticeComplexStructure,
    form: IntegralBilinearForm,
) -> _ComplexTorusExecutionPlan:
    """Admit every exact product, Smith input, and retained profile value."""

    dimension = len(torus.coordinate_axis)
    degree, numerator_digits, denominator_digits, exact_scalar_work = (
        _complex_structure_scalar_work(torus, matrix_products=4)
    )
    _admit_exact_linear_matrix(form.matrix.entries)
    form_digits = max(
        len(value.lstrip("-")) for row in form.matrix.entries for value in row
    )
    associated_numerator_digits = (
        numerator_digits
        + form_digits
        + max(dimension - 1, 0) * denominator_digits
        + len(str(dimension))
        + 2
    )
    associated_denominator_digits = dimension * denominator_digits
    associated_digits = max(
        associated_numerator_digits,
        associated_denominator_digits,
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
        presentation_bytes + 256 + degree * (2 * associated_digits + 32)
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
    return _execution_plan(
        field_degree=degree,
        exact_scalar_work=exact_scalar_work,
        hodge_constraint_cells=0,
        predicted_result_bytes=predicted_result_bytes,
    )


def _rational_domain_matrix(matrix: RationalMatrix) -> DomainMatrix:
    rows = [
        [QQ(int(value.num), int(value.den)) for value in row] for row in matrix.entries
    ]
    return DomainMatrix(rows, (len(rows), len(rows[0])), QQ)


def _require_complex_structure(
    torus: LatticeComplexStructure,
    plan: _ComplexTorusExecutionPlan,
) -> RecognizedRealSimpleNumberField | None:
    """Recognize the scalar domain and establish ``J^2 = -I`` exactly."""

    _require_execution_active(plan, "before exact complex-structure recognition")
    matrix = torus.complex_structure
    try:
        if isinstance(matrix, EmbeddedRealSimpleNumberFieldMatrix):
            recognized = recognize_real_simple_number_field(matrix.embedding)
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
    _require_execution_active(plan, "after exact complex-structure recognition")
    return recognized


def compute_neron_severi_lattice(
    torus: LatticeComplexStructure,
) -> InvariantBilinearFormLattice:
    """Compute every integral alternating Hodge ``(1,1)`` form exactly."""

    try:
        plan = _admit_neron_severi_execution(torus)
        recognized = _require_complex_structure(torus, plan)
        matrix = torus.complex_structure
        action: MatrixAction
        if isinstance(matrix, EmbeddedRealSimpleNumberFieldMatrix):
            action = EmbeddedRealNumberFieldMatrixAction(
                coordinate_axis=torus.coordinate_axis,
                generators=(
                    EmbeddedRealNumberFieldActionGenerator(
                        label="complex_structure",
                        matrix=matrix,
                    ),
                ),
            )
        else:
            action = RationalMatrixAction(
                coordinate_axis=torus.coordinate_axis,
                generators=(
                    RationalActionGenerator(
                        label="complex_structure",
                        matrix=matrix,
                    ),
                ),
            )
        _require_execution_active(plan, "before the integral Hodge-lattice kernel")
        result = invariant_bilinear_form_lattice_kernel(
            action,
            "ALTERNATING",
            recognized_field=recognized,
        )
        _require_execution_active(plan, "after the integral Hodge-lattice kernel")
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

    try:
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
        recognized = _require_complex_structure(torus, plan)
        matrix = torus.complex_structure
        if isinstance(matrix, EmbeddedRealSimpleNumberFieldMatrix):
            assert recognized is not None
            complex_structure = domain_matrix_from_embedded(matrix, recognized)
            form_matrix = _integer_form_domain_matrix(form, recognized.field)
        else:
            complex_structure = _rational_domain_matrix(matrix)
            form_matrix = _integer_form_domain_matrix(form, QQ)

        _require_execution_active(plan, "before the alternating Smith kernel")
        smith = _smith_normal_form_kernel(form.matrix)
        _require_execution_active(plan, "after the alternating Smith kernel")
        factors = smith.invariant_factors
        if len(factors) % 2 or any(
            factors[index] != factors[index + 1] for index in range(0, len(factors), 2)
        ):
            raise RuntimeError("Smith factors of an alternating form were not paired")
        elementary_divisors = factors[::2]
        degenerate = smith.rank < len(torus.coordinate_axis)

        _require_execution_active(plan, "before the Hodge compatibility products")
        transformed = (
            complex_structure.transpose()
            .matmul(form_matrix)
            .matmul(complex_structure)
            .to_dense()
        )
        if transformed != form_matrix.to_dense():
            _require_execution_active(plan, "after the Hodge compatibility products")
            return RiemannFormProfile(
                torus=torus,
                form=form,
                smith_normal_form=smith,
                alternating_elementary_divisors=elementary_divisors,
                is_degenerate=degenerate,
                outcome=RiemannFormNotHodgeType11(),
            )

        associated = complex_structure.transpose().matmul(form_matrix).to_dense()
        if associated != associated.transpose().to_dense():
            raise RuntimeError("a compatible associated Riemann form lost symmetry")
        associated_matrix = (
            embedded_matrix_from_domain(associated, recognized)
            if recognized is not None
            else _rational_matrix_from_domain(associated)
        )
        _require_execution_active(plan, "before exact associated-form inertia")
        inertia = _compute_inertia(
            associated_matrix,
            recognized_field=recognized,
        )
        _require_execution_active(plan, "after exact associated-form inertia")
        if any(
            count % 2
            for count in (inertia.n_positive, inertia.n_negative, inertia.n_zero)
        ):
            raise RuntimeError("a Hermitian realification had odd inertia")
        is_riemann = inertia.n_positive == len(torus.coordinate_axis)
        return RiemannFormProfile(
            torus=torus,
            form=form,
            smith_normal_form=smith,
            alternating_elementary_divisors=elementary_divisors,
            is_degenerate=degenerate,
            outcome=RiemannFormHodgeType11(
                associated_form_inertia=inertia,
                hermitian_inertia=_hermitian_inertia(inertia),
                is_riemann_form=is_riemann,
                polarization_type=elementary_divisors if is_riemann else None,
            ),
        )
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
