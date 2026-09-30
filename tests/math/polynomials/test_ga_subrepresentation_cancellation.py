"""Admitted Ga work observes cancellation inside its mandatory phases."""

from threading import Event

import pytest

from jacobian._execution import (
    OperationExecutionCancelledError,
    request_cancellation,
    request_checkpoint,
)
from jacobian.math.polynomials.derivations import _stable_kernels
from jacobian.math.polynomials.derivations._models import PolynomialGaAction
from jacobian.math.polynomials.values import RationalPolynomial


def _poly(variables: tuple[str, ...], *powers: tuple[int, ...]) -> RationalPolynomial:
    return RationalPolynomial.model_validate(
        {
            "variables": variables,
            "polynomial": {
                "terms": [
                    {"coefficient": {"num": 1, "den": 1}, "exponents": exponent}
                    for exponent in powers
                ]
            },
        }
    )


def _shear_action() -> PolynomialGaAction:
    return PolynomialGaAction(
        source_variables=("x", "y"),
        parameter="t",
        generator_images=(
            _poly(("x", "y", "t"), (1, 0, 0), (0, 16, 1)),
            _poly(("x", "y", "t"), (0, 1, 0)),
        ),
    )


@pytest.mark.parametrize(
    ("phase", "cancel_at"),
    [
        ("support multiplication", 2),
        ("coefficient grouping", 2),
        ("coefficient height", 2),
        ("action-law multiplication", 2),
        ("action-law accumulation", 2),
        ("basis multiplication", 2),
        ("basis accumulation", 1),
    ],
)
def test_ga_cancels_inside_later_action_and_basis_phases(
    monkeypatch: pytest.MonkeyPatch, phase: str, cancel_at: int
) -> None:
    action = _shear_action()
    # A single high-degree invariant monomial is admitted and has identity action.
    basis = (_poly(("x", "y"), (0, 16)),)
    cancelled = Event()
    observed: list[str] = []

    def cancel_during_work(stage: str) -> None:
        if phase in stage:
            observed.append(stage)
            if len(observed) == cancel_at:
                cancelled.set()
        request_checkpoint(stage)

    monkeypatch.setattr(_stable_kernels, "request_checkpoint", cancel_during_work)
    with (
        request_cancellation(cancelled),
        pytest.raises(OperationExecutionCancelledError, match=phase),
    ):
        _stable_kernels.ga_stable_subrepresentation(action, basis)
    assert len(observed) == cancel_at
    assert cancelled.is_set()


def test_admitted_high_degree_invariant_retains_exact_identity_action() -> None:
    basis = (_poly(("x", "y"), (0, 16)),)
    result = _stable_kernels.ga_stable_subrepresentation(_shear_action(), basis)
    assert result.basis == basis
    assert result.action_matrix == ((_poly(("t",), (0,)),),)


def test_derivation_product_cancels_within_one_multiplication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.polynomials.derivations import operations
    from jacobian.math.polynomials.derivations._models import PolynomialDerivation

    variables = ("x",)
    image = _poly(variables, *((degree,) for degree in reversed(range(16))))
    source = _poly(variables, *((degree,) for degree in reversed(range(1, 17))))
    derivation = PolynomialDerivation(variables=variables, images=(image,))
    cancelled = Event()
    observed: list[str] = []

    def cancel_after_first_batch(stage: str) -> None:
        observed.append(stage)
        if len(observed) == 2:
            cancelled.set()
        request_checkpoint(stage)

    monkeypatch.setattr(operations, "request_checkpoint", cancel_after_first_batch)
    with (
        request_cancellation(cancelled),
        pytest.raises(OperationExecutionCancelledError, match="multiplication"),
    ):
        operations.apply_derivation(derivation, source)
    assert len(observed) == 2
    assert all("multiplication" in stage for stage in observed)
