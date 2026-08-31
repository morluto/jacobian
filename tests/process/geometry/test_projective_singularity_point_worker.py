"""Process-boundary tests for exact projective singular-point construction."""

from __future__ import annotations

import threading
from fractions import Fraction
from time import monotonic

import pytest

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)
from jacobian.math.geometry.algebraic_curves._singularity_point_process import (
    run_point_construction_worker,
)
from jacobian.math.geometry.algebraic_curves._singularity_point_worker import (
    ProjectiveSingularityPointWorkerRequest,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)
from jacobian.process import bounded_process_cancellation


def _polynomial(
    variables: tuple[str, ...],
    *terms: tuple[int, tuple[int, ...]],
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=str(coefficient), den="1"),
                    exponents=exponents,
                )
                for coefficient, exponents in terms
            )
        ),
    )


def _conjugate_point_request() -> ProjectiveSingularityPointWorkerRequest:
    axis = ("X", "Y", "Z")
    component = RationalPolynomialIdeal(
        variables=("Y", "Z"),
        generators=(
            _polynomial(("Y", "Z"), (1, (2, 0)), (1, (0, 0))),
            _polynomial(("Y", "Z"), (1, (0, 1))),
        ),
    )
    return ProjectiveSingularityPointWorkerRequest(
        variables=axis,
        chart_zero_components=(component,),
        chart_one_components=(),
        chart_two_present=False,
    )


def test_point_worker_returns_one_exact_quadratic_residue_field_seed() -> None:
    result = run_point_construction_worker(
        _conjugate_point_request(),
        deadline=monotonic() + 30,
    )

    assert result.kind == "complete"
    assert len(result.seeds) == 1
    seed = result.seeds[0]
    assert seed.presentation.coefficients_descending == ("1", "0", "1")
    assert tuple(
        tuple(
            coefficient.as_fraction()
            for coefficient in coordinate.coefficients_ascending
        )
        for coordinate in seed.coordinates
    ) == (
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
        (Fraction(0), Fraction(0)),
    )


def test_expired_point_worker_deadline_fails_before_launch() -> None:
    with pytest.raises(OperationExecutionTimeoutError):
        run_point_construction_worker(
            _conjugate_point_request(),
            deadline=monotonic() - 1,
        )


def test_point_worker_preserves_request_cancellation() -> None:
    cancellation = threading.Event()
    cancellation.set()

    with (
        bounded_process_cancellation(cancellation),
        pytest.raises(OperationExecutionCancelledError),
    ):
        run_point_construction_worker(
            _conjugate_point_request(),
            deadline=monotonic() + 30,
        )
