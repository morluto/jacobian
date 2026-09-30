"""Admitted Ga work observes cancellation inside its mandatory phases."""

from fractions import Fraction
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


def _linear_shear_and_basis() -> tuple[
    PolynomialGaAction, tuple[RationalPolynomial, ...]
]:
    action = PolynomialGaAction(
        source_variables=("x", "y"),
        parameter="t",
        generator_images=(
            _poly(("x", "y", "t"), (1, 0, 0), (0, 1, 1)),
            _poly(("x", "y", "t"), (0, 1, 0)),
        ),
    )
    basis = tuple(_poly(("x", "y"), powers) for powers in ((0, 0), (1, 0), (0, 1)))
    return action, basis


@pytest.mark.parametrize(
    ("fixed", "phase"),
    [
        (False, "coordinate-frame support"),
        (False, "coordinate-frame source rows"),
        (False, "coordinate-frame echelon"),
        (False, "coordinate-frame inversion"),
        (False, "coordinate dot products"),
        (False, "parameter grouping"),
        (False, "matrix encoding"),
        (True, "claimed matrix admission"),
        (True, "infinitesimal matrix"),
        (True, "denominator admission"),
        (True, "integerization"),
        (True, "fixed-space elimination"),
        (True, "kernel assembly"),
        (True, "fixed-polynomial accumulation"),
        (True, "fixed-polynomial coefficient grouping"),
        (True, "fixed result encoding"),
    ],
)
def test_stable_and_fixed_operations_cancel_inside_remaining_phases(
    monkeypatch: pytest.MonkeyPatch, fixed: bool, phase: str
) -> None:
    action, basis = _linear_shear_and_basis()
    stable = _stable_kernels.ga_stable_subrepresentation(action, basis)
    cancelled = Event()
    observed: list[str] = []

    def cancel_after_first_batch(stage: str) -> None:
        if phase in stage:
            observed.append(stage)
            if len(observed) == 2:
                cancelled.set()
        request_checkpoint(stage)

    monkeypatch.setattr(_stable_kernels, "request_checkpoint", cancel_after_first_batch)
    with (
        request_cancellation(cancelled),
        pytest.raises(OperationExecutionCancelledError, match=phase),
    ):
        if fixed:
            _stable_kernels.ga_fixed_subspace(stable)
        else:
            _stable_kernels.ga_stable_subrepresentation(action, basis)
    assert len(observed) == 2
    assert cancelled.is_set()


def test_linear_shear_fixed_space_retains_constant_and_y() -> None:
    action, basis = _linear_shear_and_basis()
    result = _stable_kernels.ga_fixed_subspace(
        _stable_kernels.ga_stable_subrepresentation(action, basis)
    )
    assert result.basis == (basis[0], basis[2])
    assert tuple(
        tuple(value.as_fraction() for value in row) for row in result.coordinates
    ) == (
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )


def test_empty_kernel_elimination_still_observes_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A full-rank matrix has no fixed-vector reconstruction phase to catch a
    # late cancellation, so elimination itself must observe the request signal.
    matrix = [[Fraction(int(i == j)) for j in range(32)] for i in range(32)]
    assert _stable_kernels._rational_kernel_basis([row.copy() for row in matrix]) == []
    cancelled = Event()
    observed: list[str] = []

    def cancel_during_elimination(stage: str) -> None:
        if "fixed-space elimination" in stage:
            observed.append(stage)
            if len(observed) == 33:
                cancelled.set()
        request_checkpoint(stage)

    monkeypatch.setattr(
        _stable_kernels, "request_checkpoint", cancel_during_elimination
    )
    with (
        request_cancellation(cancelled),
        pytest.raises(
            OperationExecutionCancelledError, match="fixed-space elimination"
        ),
    ):
        _stable_kernels._rational_kernel_basis(matrix)
    assert len(observed) == 33
    assert cancelled.is_set()
