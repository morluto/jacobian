"""Code certificates retain empty axes and distinguish failed verification."""

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.codes.general import _models, _tools, operations
from jacobian.math.combinatorics.codes.linear.values import PrimeFieldLinearEncoder


def _encoder(*, empty: bool = False) -> PrimeFieldLinearEncoder:
    return PrimeFieldLinearEncoder(
        field_order=2,
        message_axis=() if empty else ("m",),
        coordinate_axis=() if empty else ("x", "y"),
        generator_matrix=() if empty else ((1, 1),),
    )


def _claim(operation: str, *, empty: bool = False) -> object:
    request_type = (
        _models.CoveringRadiusRequest
        if operation == "covering_radius"
        else _models.LinearCodeRequest
    )
    return getattr(_tools, "_" + operation)(request_type(encoder=_encoder(empty=empty)))


@pytest.mark.parametrize(
    "operation", ["minimum_distance", "weight_distribution", "covering_radius"]
)
@pytest.mark.parametrize("error_type", [ArithmeticError, ValueError, TypeError])
def test_code_verifier_preserves_failed_backend(
    monkeypatch: pytest.MonkeyPatch, operation: str, error_type: type[Exception]
) -> None:
    claim = _claim(operation)
    verify = getattr(operations, "verify_" + operation)
    assert verify(claim)

    def fail(*args: object, **kwargs: object) -> None:
        raise error_type("backend failure")

    monkeypatch.setattr(operations, operation, fail)
    with pytest.raises(error_type, match="backend failure"):
        verify(claim)


@pytest.mark.parametrize(
    "operation", ["minimum_distance", "weight_distribution", "covering_radius"]
)
def test_code_verifier_preserves_actual_resource_refusal(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    claim = _claim(operation)
    budget = (
        "MAX_COVERING_RADIUS_STATES_PER_PASS"
        if operation == "covering_radius"
        else "MAX_EXACT_CODEWORD_EVALUATIONS"
    )
    monkeypatch.setattr(operations, budget, 0)
    with pytest.raises(OperationResourceAdmissionError):
        getattr(operations, "verify_" + operation)(claim)


def test_zero_length_code_retains_its_canonical_empty_axes() -> None:
    encoder = _encoder(empty=True)
    assert operations.minimum_distance(encoder) == 0
    assert operations.weight_distribution(encoder) == [(0, 1)]
    assert operations.covering_radius(encoder) == 0
    for operation in ("minimum_distance", "weight_distribution", "covering_radius"):
        claim = _claim(operation, empty=True)
        assert getattr(operations, "verify_" + operation)(claim)
