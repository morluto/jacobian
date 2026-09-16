"""Number field operations backed by SymPy."""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import factorial

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
)
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.number_theory.algebraic_numbers.complex import (
    ComplexAlgebraicValue,
    algebraic_root_separation_denominator_bound,
    complex_isolator_component_digit_bound,
)
from jacobian.math.number_theory.algebraic_numbers.real import (
    RealAlgebraicValue,
)
from jacobian.math.number_theory.number_fields._binary_power_sum import (
    BinaryPowerSumGapProfile,
    binary_power_sum_gap_profile,
)
from jacobian.math.number_theory.number_fields._embedding_limits import (
    MAX_NUMBER_FIELD_REAL_PART_RESULTANT_STORAGE_BITS,
    MAX_NUMBER_FIELD_ROOT_REFINEMENT_BITS,
)
from jacobian.math.number_theory.number_fields._embedding_protocol import (
    NumberFieldEmbeddingWorkerInvalid,
    NumberFieldEmbeddingWorkerRejected,
)
from jacobian.math.number_theory.number_fields._embeddings_process import (
    EMBEDDINGS_WORKER_WALL_SECONDS,
    embeddings_worker_cancelled,
    run_embeddings_worker,
)
from jacobian.math.number_theory.number_fields._models import (
    NumberFieldClassGroupResult,
    NumberFieldDiscriminantResult,
    NumberFieldUnitGroupResult,
)
from jacobian.math.number_theory.number_fields._real_embedding_order import (
    compare_real_embedding_elements,
)
from jacobian.math.number_theory.number_fields._relative_trace_norm import (
    relative_trace_norm,
)
from jacobian.math.number_theory.number_fields.values import (
    MAX_NUMBER_FIELD_EMBEDDING_DEGREE,
    ComplexNumberFieldEmbedding,
    ComplexNumberFieldEmbeddingRecord,
    NumberFieldConjugatePair,
    NumberFieldDiscriminantInteger,
    NumberFieldEmbeddingProfile,
    NumberFieldSignature,
    RealNumberFieldEmbedding,
    RealNumberFieldEmbeddingRecord,
    SimpleNumberFieldPresentation,
)

# The native integral-basis consumers retain their prior degree-31 envelope.
# This is intentionally independent of the smaller degree bound for the
# isolated all-embedding worker and the widened shared field carrier.
_MAX_NATIVE_INTEGRAL_BASIS_DEGREE = 31
# The PARI class/unit-group worker retains its own, smaller degree ceiling; the
# native entry keeps it as an explicit admission decision rather than letting a
# pydantic request-model error escape.
_MAX_NATIVE_CLASS_GROUP_DEGREE = 16


class NumberFieldEmbeddingAdmissionError(ValueError):
    """A proved owner-local resource rejection for embedding enumeration."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class NumberFieldEmbeddingAdmission:
    root_isolation_bits: int
    evidence_grid_bits: int
    predicted_worker_output_bytes: int


def _decimal_digits_from_bits(bits: int) -> int:
    return (max(bits, 1) * 30_103) // 100_000 + 1


def _require_embedding_execution_active(deadline: float, phase: str) -> None:
    if embeddings_worker_cancelled():
        raise OperationExecutionCancelledError(f"request cancelled {phase}")
    if deadline <= time.monotonic():
        raise OperationExecutionTimeoutError(f"request deadline expired {phase}")


def _admit_number_field_embeddings(
    field: SimpleNumberFieldPresentation,
) -> NumberFieldEmbeddingAdmission:
    """Preflight root work, intermediates, and the worker projection."""

    coefficients = tuple(coefficient for coefficient in field.coefficients_descending)
    degree = field.degree
    if degree > MAX_NUMBER_FIELD_EMBEDDING_DEGREE:
        raise NumberFieldEmbeddingAdmissionError(
            "degree_bound",
            "number-field embeddings are limited to degree "
            f"{MAX_NUMBER_FIELD_EMBEDDING_DEGREE}",
        )
    height = max(abs(coefficient) for coefficient in coefficients)
    separation_denominator = algebraic_root_separation_denominator_bound(
        field.coefficients_descending
    )
    separation_bits = separation_denominator.bit_length()
    # Public evidence uses a dyadic grid more than 2^4 finer than Mignotte
    # separation.  The worker isolates on a grid another 2^4 finer before
    # outward normalization; the public rectangle spans at most four evidence
    # cells, whose diagonal remains below the separation bound.
    isolation_bits = separation_bits + 8
    isolator_digits = complex_isolator_component_digit_bound(
        field.coefficients_descending
    )
    if degree == 1:
        # The exact singleton root -b/a retains a reduced divisor of the
        # source coefficients rather than a dyadic denominator.
        isolator_digits = max(
            isolator_digits,
            *(
                len(format_canonical_integer(abs(coefficient)))
                for coefficient in field.coefficients_descending
            ),
        )

    if isolator_digits > 4_096:
        raise NumberFieldEmbeddingAdmissionError(
            "isolation_intermediate_bound",
            "the Mignotte-derived root-isolation envelope exceeds the "
            "4,096-digit rational component bound",
        )
    if isolation_bits > MAX_NUMBER_FIELD_ROOT_REFINEMENT_BITS:
        raise NumberFieldEmbeddingAdmissionError(
            "root_isolation_precision_bound",
            "exact root isolation exceeds the "
            f"{MAX_NUMBER_FIELD_ROOT_REFINEMENT_BITS:,}-bit refinement bound",
        )

    # Hadamard on the (2n-1)-square Sylvester matrix gives
    # |disc(f)| <= |Res(f,f')|
    # <= n^n (n+1)^((2n-1)/2) H^(2n-1).
    # Replacing the half power by (n+1)^(2n-1) is an integer upper bound.
    discriminant_bound = (
        1
        if degree == 1
        else degree**degree
        * (degree + 1) ** (2 * degree - 1)
        * height ** (2 * degree - 1)
    )
    discriminant_digits = _decimal_digits_from_bits(discriminant_bound.bit_length())

    # In f(u + i*v), every coefficient of Re(f) and Im(f) has magnitude at
    # most C = (n+1) 2^n H.  A Leibniz expansion of the at-most-2n Sylvester
    # determinant therefore bounds every resultant coefficient by
    # (2n)! (n+1)^(2n-1) C^(2n).  Its u-degree is at most 2n^2.  The final
    # Landau-Mignotte factor covers coefficient growth when taking the
    # primitive square-free part used for real-coordinate separation.
    sylvester_size = 2 * degree
    resultant_degree = 2 * degree * degree
    expanded_coefficient_bound = (degree + 1) * 2**degree * height
    expanded_resultant_coefficient_bound = (
        factorial(sylvester_size)
        * (degree + 1) ** max(sylvester_size - 1, 0)
        * expanded_coefficient_bound**sylvester_size
    )
    resultant_coefficient_bound = (
        (resultant_degree + 1)
        * 2**resultant_degree
        * expanded_resultant_coefficient_bound
    )
    resultant_coefficient_bits = max(resultant_coefficient_bound.bit_length(), 1)
    resultant_storage_bits = (resultant_degree + 1) * resultant_coefficient_bits
    if resultant_storage_bits > MAX_NUMBER_FIELD_REAL_PART_RESULTANT_STORAGE_BITS:
        raise NumberFieldEmbeddingAdmissionError(
            "pair_ordering_resultant_bound",
            "the exact real-coordinate resultant exceeds the "
            f"{MAX_NUMBER_FIELD_REAL_PART_RESULTANT_STORAGE_BITS:,}-bit "
            "intermediate storage bound",
        )

    rational_bytes = 2 * isolator_digits + 32
    # The worker projection carries either one two-rational interval per real
    # root or one four-rational rectangle per conjugate pair.  Both cases use
    # at most two bounded rationals per degree unit.  It does not echo the
    # retained presentation or construct public embedding records.
    predicted_worker_output_bytes = (
        degree * (2 * rational_bytes + 256) + discriminant_digits + 1_024
    )
    return NumberFieldEmbeddingAdmission(
        root_isolation_bits=isolation_bits,
        evidence_grid_bits=separation_bits + 4,
        predicted_worker_output_bytes=predicted_worker_output_bytes,
    )


def embeddings(
    field: SimpleNumberFieldPresentation,
) -> NumberFieldEmbeddingProfile:
    """Return every exact embedding of one bounded presented field.

    Root identity uses increasing real roots followed by conjugate pairs sorted
    by the positive representative's exact ``(Re, Im)`` coordinates, with the
    negative root first in each pair.  Exact real-coordinate elimination and
    Mignotte separation establish this order inside one killable worker.  Only
    indexed canonical roots and rational isolation evidence cross the boundary.
    """

    execution = current_request_execution()
    started = execution.started_at if execution is not None else time.monotonic()
    owner_deadline = started + EMBEDDINGS_WORKER_WALL_SECONDS
    deadline = (
        min(execution.deadline, owner_deadline)
        if execution is not None and execution.deadline is not None
        else owner_deadline
    )
    bind_request_deadline(deadline)
    _require_embedding_execution_active(deadline, "before embedding admission")
    admission = _admit_number_field_embeddings(field)
    _require_embedding_execution_active(deadline, "after embedding admission")
    worker_response = run_embeddings_worker(
        field,
        root_isolation_bits=admission.root_isolation_bits,
        evidence_grid_bits=admission.evidence_grid_bits,
        deadline=deadline,
        stdout_limit=admission.predicted_worker_output_bytes,
    )
    _require_embedding_execution_active(deadline, "after embedding worker execution")
    if isinstance(worker_response, NumberFieldEmbeddingWorkerInvalid):
        raise NumberFieldEmbeddingAdmissionError(
            "not_irreducible",
            "simple number-field polynomial must be irreducible over QQ",
        )
    if isinstance(worker_response, NumberFieldEmbeddingWorkerRejected):
        raise NumberFieldEmbeddingAdmissionError(
            worker_response.reason,
            "exact conjugate-pair ordering exceeds the "
            f"{MAX_NUMBER_FIELD_ROOT_REFINEMENT_BITS:,}-bit refinement bound",
        )
    real_count = len(worker_response.real_intervals)
    pair_count = len(worker_response.negative_complex_rectangles)
    if real_count + 2 * pair_count != field.degree:
        raise RuntimeError(
            "number-field embedding worker returned an inconsistent root count"
        )

    records: list[
        RealNumberFieldEmbeddingRecord | ComplexNumberFieldEmbeddingRecord
    ] = []
    for root_index in range(real_count):
        real_root = RealAlgebraicValue._from_admitted_polynomial(
            polynomial=field.coefficients_descending,
            real_root_index=root_index,
        )
        real_embedding = RealNumberFieldEmbedding(
            kind="REAL",
            presentation=field,
            root=real_root,
        )
        records.append(
            RealNumberFieldEmbeddingRecord._from_kernel(
                embedding=real_embedding,
                isolating_interval=worker_response.real_intervals[root_index],
            )
        )

    conjugate_pairs: list[NumberFieldConjugatePair] = []
    for pair_offset in range(pair_count):
        negative_index = real_count + 2 * pair_offset
        positive_index = negative_index + 1
        negative_root = ComplexAlgebraicValue._from_admitted_polynomial(
            polynomial=field.coefficients_descending,
            root_index=negative_index,
        )
        positive_root = ComplexAlgebraicValue._from_admitted_polynomial(
            polynomial=field.coefficients_descending,
            root_index=positive_index,
        )
        negative_embedding = ComplexNumberFieldEmbedding(
            kind="COMPLEX",
            presentation=field,
            root=negative_root,
        )
        positive_embedding = ComplexNumberFieldEmbedding(
            kind="COMPLEX",
            presentation=field,
            root=positive_root,
        )
        negative_rectangle = worker_response.negative_complex_rectangles[pair_offset]
        positive_rectangle = negative_rectangle.conjugate()
        records.extend(
            (
                ComplexNumberFieldEmbeddingRecord._from_kernel(
                    embedding=negative_embedding,
                    isolating_rectangle=negative_rectangle,
                    half_plane="NEGATIVE_IMAGINARY",
                ),
                ComplexNumberFieldEmbeddingRecord._from_kernel(
                    embedding=positive_embedding,
                    isolating_rectangle=positive_rectangle,
                    half_plane="POSITIVE_IMAGINARY",
                ),
            )
        )
        conjugate_pairs.append(
            NumberFieldConjugatePair(
                negative_embedding_index=negative_index,
                positive_embedding_index=positive_index,
            )
        )

    result = NumberFieldEmbeddingProfile._from_kernel(
        field=field,
        records=tuple(records),
        signature=NumberFieldSignature(
            real_embedding_count=real_count,
            complex_conjugate_pair_count=pair_count,
        ),
        complex_conjugate_pairs=tuple(conjugate_pairs),
        defining_polynomial_discriminant=(
            worker_response.defining_polynomial_discriminant
        ),
    )
    _require_embedding_execution_active(deadline, "after embedding result construction")
    return result


def discriminant(
    field: SimpleNumberFieldPresentation,
) -> NumberFieldDiscriminantInteger:
    """Return the field discriminant through the shared admitted worker.

    This is a package-exported native entry, so it is an admission boundary in
    its own right: it must not take a cheaper route than ``math.run``. The
    irreducibility, discriminant, and ``round_two`` work - including the
    factorization-work admission that bounds ``round_two`` - belongs to the
    request-owned killable worker, which this call now enters exactly like
    ``number_field.discriminant.compute`` and ``verify_discriminant``. Previously
    it called ``recognized_integral_basis`` synchronously with no
    ``require_factorizable_discriminant``, so the same field was admitted on the
    catalog path and refused - or run unbounded - on the native one.
    """

    from jacobian.math.number_theory.number_fields._discriminant_process import (
        compute_nf_discriminant,
    )
    from jacobian.math.number_theory.number_fields._models import NumberFieldRequest

    if not isinstance(field, SimpleNumberFieldPresentation):
        raise OperationDomainValidationError(
            location=("field",),
            code="number_field.discriminant_field_type",
            message=(
                "discriminant requires a SimpleNumberFieldPresentation field argument"
            ),
        )
    if field.degree > _MAX_NATIVE_INTEGRAL_BASIS_DEGREE:
        # The published native entry keeps its own degree ceiling. That ceiling
        # is part of this callable's contract and is independent of the worker
        # admission, which is what the shared route now supplies.
        raise OperationDomainValidationError(
            location=("field",),
            code="number_field.native_integral_basis_degree_bound",
            message=(
                "native number-field integral-basis operations are limited to "
                f"degree {_MAX_NATIVE_INTEGRAL_BASIS_DEGREE}"
            ),
        )
    return int(compute_nf_discriminant(NumberFieldRequest(field=field)).discriminant)


def verify_discriminant(claim: NumberFieldDiscriminantResult) -> bool:
    """Check a field-discriminant claim under the bounded worker's admission."""
    from jacobian.math.number_theory.number_fields._discriminant_process import (
        compute_nf_discriminant,
    )
    from jacobian.math.number_theory.number_fields._models import NumberFieldRequest

    try:
        expected = compute_nf_discriminant(NumberFieldRequest(field=claim.field))
    except OperationResourceAdmissionError:
        # Resource non-completion is not a negative mathematical conclusion:
        # a discriminant outside the factorization envelope is unknown here,
        # not false.
        raise
    except (ArithmeticError, TypeError, ValueError):
        return False
    return expected.discriminant == claim.discriminant


def verify_binary_power_sum_gap_profile(claim: BinaryPowerSumGapProfile) -> bool:
    """Check every retained binary power-sum relation against its source."""
    try:
        expected = binary_power_sum_gap_profile(claim.base, claim.exponent_count)
    except (TypeError, ValueError):
        return False
    return expected == claim


def class_group(
    field: SimpleNumberFieldPresentation,
) -> NumberFieldClassGroupResult:
    """Return the class group of one presented field through the PARI worker.

    This native entry is an admission boundary in its own right: it validates
    the field type and degree, then enters the shared request-owned killable
    worker exactly like ``number_field.class_group.compute``.
    """

    from jacobian.math.number_theory.number_fields._bnf_process import run_bnf_worker
    from jacobian.math.number_theory.number_fields._models import (
        NumberFieldClassGroupRequest,
        NumberFieldClassGroupResult,
    )

    if not isinstance(field, SimpleNumberFieldPresentation):
        raise OperationDomainValidationError(
            location=("field",),
            code="number_field.class_group_field_type",
            message="class_group requires a SimpleNumberFieldPresentation field",
        )
    if field.degree > _MAX_NATIVE_CLASS_GROUP_DEGREE:
        raise OperationDomainValidationError(
            location=("field",),
            code="number_field.class_group_degree_bound",
            message=(
                "class-group computation admits degree at most "
                f"{_MAX_NATIVE_CLASS_GROUP_DEGREE}"
            ),
        )
    request = NumberFieldClassGroupRequest(field=field)
    result = run_bnf_worker(request)
    return NumberFieldClassGroupResult._from_kernel(
        field=field,
        class_number=result.class_number,
        abelian_invariants=result.abelian_invariants,
        field_discriminant=result.field_discriminant,
        real_embedding_count=result.real_embedding_count,
        complex_embedding_pair_count=result.complex_embedding_pair_count,
        ideal_representatives=tuple(
            IntegerMatrix(
                row_count=len(matrix),
                column_count=len(matrix[0]) if matrix else 0,
                entries=matrix,
            )
            for matrix in result.ideal_representatives
        ),
    )


def unit_group(
    field: SimpleNumberFieldPresentation,
) -> NumberFieldUnitGroupResult:
    """Return the unit group of one presented field through the PARI worker."""

    from jacobian.math.number_theory.number_fields._bnf_process import (
        bind_element,
        run_bnf_worker,
    )
    from jacobian.math.number_theory.number_fields._models import (
        NumberFieldUnitGroupRequest,
        NumberFieldUnitGroupResult,
    )

    if not isinstance(field, SimpleNumberFieldPresentation):
        raise OperationDomainValidationError(
            location=("field",),
            code="number_field.unit_group_field_type",
            message="unit_group requires a SimpleNumberFieldPresentation field",
        )
    if field.degree > _MAX_NATIVE_CLASS_GROUP_DEGREE:
        raise OperationDomainValidationError(
            location=("field",),
            code="number_field.unit_group_degree_bound",
            message=(
                "unit-group computation admits degree at most "
                f"{_MAX_NATIVE_CLASS_GROUP_DEGREE}"
            ),
        )
    request = NumberFieldUnitGroupRequest(field=field)
    result = run_bnf_worker(request)
    return NumberFieldUnitGroupResult._from_kernel(
        field=field,
        rank=result.rank,
        torsion_order=result.torsion_order,
        torsion_generator=bind_element(field, result.torsion_generator),
        fundamental_units=tuple(
            bind_element(field, unit) for unit in result.fundamental_units
        ),
        field_discriminant=result.field_discriminant,
        real_embedding_count=result.real_embedding_count,
        complex_embedding_pair_count=result.complex_embedding_pair_count,
    )


__all__ = [
    "binary_power_sum_gap_profile",
    "class_group",
    "compare_real_embedding_elements",
    "discriminant",
    "embeddings",
    "relative_trace_norm",
    "unit_group",
    "verify_binary_power_sum_gap_profile",
    "verify_discriminant",
]
