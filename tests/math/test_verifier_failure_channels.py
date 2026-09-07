from __future__ import annotations

from types import SimpleNamespace

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.geometry.differential import operations as differential
from jacobian.math.geometry.exact import operations as exact_geometry
from jacobian.math.matrices.quadratic_spectral import operations as quadratic_spectral
from jacobian.math.number_theory.sequences.recurrence_solving import (
    operations as recurrence,
)
from jacobian.math.polynomials.real_algebra import operations as real_algebra


@pytest.mark.parametrize(
    ("module", "recompute", "claim"),
    [
        (
            differential,
            "lie_derivative",
            SimpleNamespace(vector_field=None, source=None, lie_derivative=None),
        ),
        (
            real_algebra,
            "common_interlacing_profile",
            SimpleNamespace(family=None),
        ),
        (
            quadratic_spectral,
            "inertia",
            SimpleNamespace(matrix=None),
        ),
        (exact_geometry, "distance_profile", SimpleNamespace(configuration=None)),
        (
            recurrence,
            "closed_form",
            SimpleNamespace(characteristic_coefficients=None, initial_values=None),
        ),
    ],
)
def test_verifiers_propagate_operational_runtime_failures(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    recompute: str,
    claim: object,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("backend failed")

    monkeypatch.setattr(module, recompute, fail)
    verifier = getattr(module, f"verify_{recompute}")

    with pytest.raises(RuntimeError, match="backend failed"):
        verifier(claim)


@pytest.mark.parametrize(
    ("module", "recompute", "claim"),
    [
        (
            differential,
            "lie_derivative",
            SimpleNamespace(vector_field=None, source=None, lie_derivative=None),
        ),
        (real_algebra, "common_interlacing_profile", SimpleNamespace(family=None)),
        (quadratic_spectral, "inertia", SimpleNamespace(matrix=None)),
        (exact_geometry, "distance_profile", SimpleNamespace(configuration=None)),
        (
            recurrence,
            "closed_form",
            SimpleNamespace(characteristic_coefficients=None, initial_values=None),
        ),
    ],
)
def test_verifiers_propagate_resource_admission_failures(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    recompute: str,
    claim: object,
) -> None:
    def reject(*_args: object, **_kwargs: object) -> object:
        raise OperationResourceAdmissionError(
            location=("source",),
            code="test.resource_bound",
            message="resource bound exceeded",
        )

    monkeypatch.setattr(module, recompute, reject)
    verifier = getattr(module, f"verify_{recompute}")

    with pytest.raises(OperationResourceAdmissionError, match="resource bound"):
        verifier(claim)
