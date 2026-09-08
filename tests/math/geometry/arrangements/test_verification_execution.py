"""Native arrangement verifiers preserve operational failures."""

import pytest

from jacobian.math.geometry.arrangements import operations


@pytest.mark.parametrize("operation", ["characteristic_polynomial", "chamber_count"])
@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("backend"),
        ValueError("backend"),
        TypeError("backend"),
        MemoryError("backend"),
        TimeoutError("backend"),
    ],
)
def test_formula_verifier_propagates_computation_failure(
    monkeypatch: pytest.MonkeyPatch, operation: str, error: Exception
) -> None:
    claim = getattr(operations, operation)(2, 3)
    verifier = getattr(operations, "verify_" + operation)
    assert verifier(claim)

    def fail(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(operations, operation, fail)
    with pytest.raises(type(error), match="backend"):
        verifier(claim)


@pytest.mark.parametrize("operation", ["characteristic_polynomial", "chamber_count"])
def test_formula_verifier_preserves_resource_refusal(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    from jacobian.catalog.models import OperationResourceAdmissionError

    claim = getattr(operations, operation)(2, 3)
    monkeypatch.setattr(operations, "MAX_GENERIC_FORMULA_WORK", 0)
    with pytest.raises(OperationResourceAdmissionError):
        getattr(operations, "verify_" + operation)(claim)


def test_arrangement_verifier_preserves_execution_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian._exact import CanonicalRational
    from jacobian.math.geometry.arrangements._models import RationalHyperplane

    claim = operations.arrangement(
        1,
        (
            RationalHyperplane(
                coefficients=(CanonicalRational(num=1, den=1),),
                constant=CanonicalRational(num=0, den=1),
            ),
        ),
    )
    assert operations.verify_arrangement(claim)

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("backend failure")

    monkeypatch.setattr(operations, "arrangement", fail)
    with pytest.raises(RuntimeError, match="backend failure"):
        operations.verify_arrangement(claim)
