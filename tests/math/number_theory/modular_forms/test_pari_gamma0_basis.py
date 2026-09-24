"""Independent checks for exact PARI-backed rational Gamma0 bases."""

from __future__ import annotations

import json
import time
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian._execution import request_execution
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.modular_forms import basis
from jacobian.math.number_theory.modular_forms._models import (
    ModularFormBasisFrameRequest,
)
from jacobian.math.number_theory.modular_forms._tools import TOOLS
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
)

PARI_BASIS_ID = "gamma0-rational-gamma0-sturm-rref-v1"


def _space(kind: str, weight: int = 4) -> ModularFormSpace:
    return ModularFormSpace(level=5, weight=weight, kind=kind)  # type: ignore[arg-type]


def _coefficients(vector: object) -> tuple[Fraction, ...]:
    return tuple(coefficient.as_fraction() for coefficient in vector)  # type: ignore[attr-defined]


def test_gamma0_five_m4_basis_is_exact_sturm_frame_and_composes() -> None:
    space = _space("M")
    basis_value = basis.modular_form_basis_q_expansions(space, 3)

    assert basis_value.space == space
    assert basis_value.basis_id == PARI_BASIS_ID
    assert basis_value.precision == 3
    assert tuple(element.label for element in basis_value.elements) == (
        "q^0",
        "q^1",
        "q^2",
    )
    assert tuple(
        _coefficients(element.expansion.q_expansion.coefficients)
        for element in basis_value.elements
    ) == (
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )

    coordinate_form = ModularFormCoordinates(
        space=space,
        basis_id=basis_value.basis_id,
        coordinates=(
            CanonicalRational(num=3, den=1),
            CanonicalRational(num=-1, den=1),
            CanonicalRational(num=2, den=1),
        ),
    )
    expansion = basis.modular_form_coordinates_q_expansion(coordinate_form, 3)
    assert _coefficients(expansion.q_expansion.coefficients) == (
        Fraction(3),
        Fraction(-1),
        Fraction(2),
    )


def test_gamma0_five_s4_matches_independent_eta_product_oracle() -> None:
    space = _space("S")
    basis_value = basis.modular_form_basis_q_expansions(space, 4)

    # eta(tau)^4 eta(5 tau)^4 = q * product_n (1-q^n)^4 (1-q^(5n))^4.
    # Through q^3, only (1-q)^4 and (1-q^2)^4 contribute.
    assert len(basis_value.elements) == 1
    assert basis_value.elements[0].label == "q^1"
    assert _coefficients(
        basis_value.elements[0].expansion.q_expansion.coefficients
    ) == (
        Fraction(0),
        Fraction(1),
        Fraction(-4),
        Fraction(2),
    )

    # The q^1 pivot proves rank one through the exact Sturm determining prefix.
    assert basis_value.precision >= 3
    assert basis_value.elements[0].expansion.q_expansion.coefficients[1] == (
        CanonicalRational(num=1, den=1)
    )


def test_basis_identity_and_prefix_are_stable_when_precision_grows() -> None:
    space = _space("S")
    shorter = basis.modular_form_basis_q_expansions(space, 3)
    longer = basis.modular_form_basis_q_expansions(space, 4)

    assert shorter.basis_id == longer.basis_id == PARI_BASIS_ID
    assert (
        _coefficients(shorter.elements[0].expansion.q_expansion.coefficients)
        == (_coefficients(longer.elements[0].expansion.q_expansion.coefficients)[:3])
    )


def test_catalog_example_exercises_new_basis_representation() -> None:
    operation = next(
        item
        for item in TOOLS
        if item.operation_id == "modular_form.space.basis_q_expansions.compute"
    )
    example = next(
        item for item in operation.examples if item.name == "gamma0_five_m4_basis"
    )
    request = operation.request_type.model_validate_json(json.dumps(example.input))
    value = operation.run(request)
    assert value.basis_id == PARI_BASIS_ID
    assert len(value.elements) == 3


def test_sturm_precision_rejection_precedes_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_backend(*args: object, **kwargs: object) -> object:
        pytest.fail("backend ran before precision admission")

    monkeypatch.setattr(basis, "pari_gamma0_rational_basis", unexpected_backend)
    with pytest.raises(OperationDomainValidationError):
        basis.modular_form_basis_q_expansions(_space("M"), 2)


def test_dimension_rejection_precedes_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_backend(*args: object, **kwargs: object) -> object:
        pytest.fail("backend ran before dimension admission")

    monkeypatch.setattr(basis, "pari_gamma0_rational_basis", unexpected_backend)
    with pytest.raises(OperationResourceAdmissionError):
        basis.modular_form_basis_q_expansions(_space("M", weight=120), 128)


def test_excess_coordinate_precision_rejects_rref_growth_before_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = _space("M")
    form = ModularFormCoordinates(
        space=space,
        basis_id=PARI_BASIS_ID,
        coordinates=tuple(CanonicalRational(num=0, den=1) for _ in range(3)),
    )

    def unexpected_backend(*args: object, **kwargs: object) -> object:
        pytest.fail("PARI ran before coordinate output-growth admission")

    monkeypatch.setattr(basis, "pari_gamma0_rational_basis", unexpected_backend)
    with pytest.raises(OperationResourceAdmissionError):
        basis.modular_form_coordinates_q_expansion(form, 4)


def test_gamma0_five_hecke_matrix_uses_pari_basis_and_exact_t2_formula() -> None:
    space = _space("M")
    matrix = basis.modular_form_hecke_matrix(space, 2)
    basis_value = basis.modular_form_basis_q_expansions(space, 5)

    assert matrix.space == space
    assert matrix.basis_id == PARI_BASIS_ID
    assert matrix.index == 2
    assert matrix.row_labels == matrix.column_labels
    assert len(matrix.entries) == 3

    # The Γ0(5), weight-four Sturm prefix is the identity frame at q^0..q^2.
    # Independently apply a_2(m) = a_(2m) + 2^3 a_(m/2) when 2 divides m.
    expected_columns = []
    for element in basis_value.elements:
        coefficients = _coefficients(element.expansion.q_expansion.coefficients)
        expected_columns.append(
            tuple(
                coefficients[2 * m] + (8 * coefficients[m // 2] if m % 2 == 0 else 0)
                for m in range(3)
            )
        )
    actual_columns = tuple(
        tuple(matrix.entries[row][column].as_fraction() for row in range(3))
        for column in range(3)
    )
    assert actual_columns == tuple(expected_columns)


def test_gamma0_five_hecke_matrix_rejects_non_coprime_index() -> None:
    with pytest.raises(OperationDomainValidationError):
        basis.modular_form_hecke_matrix(_space("M"), 5)


def _rat(value: int) -> CanonicalRational:
    return CanonicalRational(num=value, den=1)


def test_basis_worker_retains_a_longer_caller_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.process as process

    captured: dict[str, object] = {}

    def failing_launch(_argv: object, **kwargs: object) -> None:
        captured.update(kwargs)
        raise OSError("fake launch barrier")

    monkeypatch.setattr(process, "run_checked_worker_process", failing_launch)
    started = time.monotonic()
    with (
        request_execution(started, outer_deadline=started + 600.0),
        pytest.raises(RuntimeError, match="could not start"),
    ):
        basis.modular_form_basis_q_expansions(_space("M"), 3)

    remaining = captured["timeout_seconds"]
    limits = captured["resource_limits"]
    assert isinstance(remaining, float)
    assert remaining > 30.0
    assert limits.cpu_seconds >= 600


def test_basis_worker_caps_only_without_a_caller_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.process as process

    captured: dict[str, object] = {}

    def failing_launch(_argv: object, **kwargs: object) -> None:
        captured.update(kwargs)
        raise OSError("fake launch barrier")

    monkeypatch.setattr(process, "run_checked_worker_process", failing_launch)
    with (
        request_execution(time.monotonic()),
        pytest.raises(RuntimeError, match="could not start"),
    ):
        basis.modular_form_basis_q_expansions(_space("M"), 3)

    remaining = captured["timeout_seconds"]
    limits = captured["resource_limits"]
    assert isinstance(remaining, float)
    assert remaining <= 30.0
    assert limits.cpu_seconds == 30


def test_identity_frame_at_pari_space_uses_the_sturm_determining_precision() -> None:
    space = _space("M")
    frame = basis.modular_form_basis_frame(
        ModularFormBasisFrameRequest(
            space=space,
            source_basis_id=PARI_BASIS_ID,
            source_labels=("q^0", "q^1", "q^2"),
            labels=("f0", "f1", "f2"),
            entries=(
                (_rat(1), _rat(0), _rat(0)),
                (_rat(0), _rat(1), _rat(0)),
                (_rat(0), _rat(0), _rat(1)),
            ),
        )
    )

    framed = basis.modular_form_hecke_matrix_in_frame(frame, 2)
    canonical = basis.modular_form_hecke_matrix(space, 2)

    assert framed.row_labels == framed.column_labels == ("f0", "f1", "f2")
    assert tuple(
        tuple(value.as_fraction() for value in row) for row in framed.entries
    ) == tuple(tuple(value.as_fraction() for value in row) for row in canonical.entries)

    form = ModularFormCoordinates(
        space=space,
        basis_id=PARI_BASIS_ID,
        coordinates=(_rat(3), _rat(-1), _rat(2)),
    )
    converted = basis.modular_form_coordinates_to_frame(frame, form)
    assert tuple(value.as_fraction() for value in converted.coordinates) == (
        Fraction(3),
        Fraction(-1),
        Fraction(2),
    )
