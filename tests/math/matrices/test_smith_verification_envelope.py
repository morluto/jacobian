"""Resource non-conclusions and independently authored Smith replay controls."""

from time import monotonic

import pytest

from jacobian._execution import OperationExecutionTimeoutError, request_execution
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.matrices.certified_snf import _verification as bounds
from jacobian.math.matrices.certified_snf.operations import (
    verify_smith_normal_form_certificate,
)
from jacobian.math.matrices.certified_snf.values import SmithNormalFormCertificate
from jacobian.math.matrices.values import IntegerMatrix


def _certificate(
    size: int, *, scalar: int = 1, shear: int = 0
) -> SmithNormalFormCertificate:
    diagonal = IntegerMatrix(
        entries=tuple(
            tuple(scalar if i == j else 0 for j in range(size)) for i in range(size)
        )
    )

    # U=I+uv^T, V=I-uv^T, with u=(1,...,1), v=(t,-t,0,...).
    # v^T u=0 implies UV=I and det(U)=det(V)=1 (matrix determinant lemma).
    def transform(sign: int) -> IntegerMatrix:
        return IntegerMatrix(
            entries=tuple(
                tuple(
                    int(i == j) + sign * (shear if j == 0 else -shear if j == 1 else 0)
                    for j in range(size)
                )
                for i in range(size)
            )
        )

    claim = SmithNormalFormCertificate(
        source=diagonal,
        diagonal=diagonal,
        left_transformation=transform(1),
        right_transformation=transform(-1),
        rank=size if scalar else 0,
        invariant_factors=(scalar,) * size if scalar else (),
        left_determinant=1,
        right_determinant=1,
    )
    return SmithNormalFormCertificate.model_validate_json(claim.model_dump_json())


@pytest.mark.parametrize("size,power,shear", ((1, 5000, 0), (2, 64, 7)))
def test_replay_admits_source_height_beyond_standalone_producer(
    size: int, power: int, shear: int
) -> None:
    assert verify_smith_normal_form_certificate(
        _certificate(size, scalar=10**power, shear=shear)
    )


def test_zero_relation_does_not_hide_nonunimodular_transform() -> None:
    claim = _certificate(2, scalar=0)
    bad_left = IntegerMatrix(entries=((2, 0), (0, 1)))
    forged = SmithNormalFormCertificate.model_validate_json(
        claim.model_copy(update={"left_transformation": bad_left}).model_dump_json()
    )
    assert not verify_smith_normal_form_certificate(forged)


def test_replay_refusal_preserves_a_true_dense_claim() -> None:
    claim = _certificate(32, shear=10**256)
    with pytest.raises(OperationResourceAdmissionError) as error:
        verify_smith_normal_form_certificate(claim)
    assert error.value.errors()[0]["type"] == "matrix.smith_verification.allocation"


@pytest.mark.parametrize(
    "limit_name,plan_field,reason",
    (
        ("MAX_SMITH_VERIFICATION_WORK", "work", "work"),
        (
            "MAX_SMITH_VERIFICATION_INTERMEDIATE_BITS",
            "intermediate_bits",
            "intermediate",
        ),
        ("MAX_SMITH_VERIFICATION_ALLOCATION_BITS", "allocation_bits", "allocation"),
    ),
)
def test_replay_budget_boundaries_are_inclusive(
    monkeypatch: pytest.MonkeyPatch, limit_name: str, plan_field: str, reason: str
) -> None:
    claim = _certificate(3, scalar=6, shear=7)
    limit = getattr(bounds.admit_smith_verification(claim), plan_field)
    monkeypatch.setattr(bounds, limit_name, limit)
    assert verify_smith_normal_form_certificate(claim)
    monkeypatch.setattr(bounds, limit_name, limit - 1)
    with pytest.raises(OperationResourceAdmissionError) as error:
        verify_smith_normal_form_certificate(claim)
    assert error.value.errors()[0]["type"] == f"matrix.smith_verification.{reason}"


def test_replay_respects_callers_deadline() -> None:
    claim = _certificate(17)
    with (
        request_execution(monotonic() - 2, outer_deadline=monotonic() - 1),
        pytest.raises(OperationExecutionTimeoutError),
    ):
        verify_smith_normal_form_certificate(claim)


def test_all_claim_matrix_axes_are_admitted_before_replay() -> None:
    claim = _certificate(32)
    # Canonical matrix shape can be bigger than the replay envelope; its
    # source stays within that envelope, so a source-only gate misses it.
    oversized = IntegerMatrix(
        row_count=33,
        column_count=33,
        entries=tuple(tuple(int(i == j) for j in range(33)) for i in range(33)),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        verify_smith_normal_form_certificate(
            claim.model_copy(update={"left_transformation": oversized})
        )
    assert error.value.errors()[0]["type"] == "matrix.smith_verification.dimension"


def test_full_axis_replay_preserves_negative_determinant_sign() -> None:
    claim = _certificate(32)
    permutation = IntegerMatrix(
        entries=tuple(
            tuple(int(j == (1 if i == 0 else 0 if i == 1 else i)) for j in range(32))
            for i in range(32)
        )
    )
    claim = SmithNormalFormCertificate.model_validate_json(
        claim.model_copy(
            update={
                "source": permutation,
                "left_transformation": permutation,
                "left_determinant": -1,
            }
        ).model_dump_json()
    )
    assert verify_smith_normal_form_certificate(claim)
