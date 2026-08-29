"""Exact contract tests for the reviewed level-one named q-expansion leaf."""

from __future__ import annotations

from typing import cast

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.modular_forms import kernel as level_one_kernel
from jacobian.math.number_theory.modular_forms._models import (
    LevelOneNamedQExpansionRequest,
)
from jacobian.math.number_theory.modular_forms.kernel import (
    NAMED_LEVEL_ONE_FORMS,
    NamedLevelOneModularForm,
    divisor_power_sum,
    eisenstein_coefficients,
    metadata,
    require_level_one_admission,
)
from jacobian.math.number_theory.modular_forms.operations import (
    _series,
    level_one_named_q_expansion,
)
from jacobian.math.number_theory.modular_forms.values import LevelOneModularQExpansion
from jacobian.math.polynomials.series.operations import (
    power,
    scalar_multiply,
    subtract,
)


def _integers(expansion: LevelOneModularQExpansion) -> tuple[int, ...]:
    return tuple(
        int(coefficient.as_fraction())
        for coefficient in expansion.q_expansion.coefficients
    )


def _validation_type(error: pytest.ExceptionInfo[ValidationError]) -> str:
    return str(error.value.errors()[0]["type"])


def test_e4_and_e6_known_normalized_prefixes() -> None:
    assert _integers(level_one_named_q_expansion("E4", 6)) == (
        1,
        240,
        2160,
        6720,
        17520,
        30240,
    )
    assert _integers(level_one_named_q_expansion("E6", 6)) == (
        1,
        -504,
        -16632,
        -122976,
        -532728,
        -1575504,
    )


def test_delta_known_prefix_and_defining_e4_e6_identity() -> None:
    delta = level_one_named_q_expansion("DELTA", 7)
    assert _integers(delta) == (0, 1, -24, 252, -1472, 4830, -6048)
    e4 = level_one_named_q_expansion("E4", 7).q_expansion
    e6 = level_one_named_q_expansion("E6", 7).q_expansion
    derived = scalar_multiply(
        subtract(power(e4, 3).result, power(e6, 2).result).result,
        CanonicalRational(num="1", den="1728"),
    ).result
    assert derived == delta.q_expansion


def test_full_public_precision_is_complete_and_carries_parent_metadata() -> None:
    result = level_one_named_q_expansion("DELTA", 844)
    assert result.q_expansion.truncation_order == 844
    assert len(result.q_expansion.coefficients) == 844
    assert result.congruence_subgroup == "SL2Z"
    assert result.level == 1
    assert result.weight == 12
    assert result.space_kind == "CUSP"


def test_e4_and_e6_report_the_standard_holomorphic_space_kind() -> None:
    assert level_one_named_q_expansion("E4", 3).space_kind == "HOLOMORPHIC"
    assert level_one_named_q_expansion("E6", 3).space_kind == "HOLOMORPHIC"


def test_value_rejects_the_misspelled_holomorphic_space_kind() -> None:
    result = level_one_named_q_expansion("E4", 3)
    payload = result.model_dump()
    payload["space_kind"] = "HOLMORPHIC"
    with pytest.raises(ValidationError) as error:
        LevelOneModularQExpansion.model_validate(payload)
    assert _validation_type(error) == "literal_error"


def test_order_one_retains_the_known_constant_coefficient() -> None:
    assert _integers(level_one_named_q_expansion("E4", 1)) == (1,)
    assert _integers(level_one_named_q_expansion("DELTA", 1)) == (0,)


def test_eisenstein_prefixes_beyond_the_former_carrier_ceiling_are_admitted() -> None:
    e6 = level_one_named_q_expansion("E6", 1000)
    assert len(e6.q_expansion.coefficients) == 1000
    assert all(coefficient.den == "1" for coefficient in e6.q_expansion.coefficients)
    assert int(e6.q_expansion.coefficients[-1].as_fraction()) == -504 * (
        divisor_power_sum(999, 5)
    )
    e4 = level_one_named_q_expansion("E4", 1477)
    assert len(e4.q_expansion.coefficients) == 1477


def test_requests_above_the_serialized_budget_name_the_controlling_quantity() -> None:
    request = LevelOneNamedQExpansionRequest(form="E6", truncation_order=1301)
    with pytest.raises(OperationDomainValidationError, match="serialized result bound"):
        level_one_named_q_expansion(request.form, request.truncation_order)
    with pytest.raises(OperationDomainValidationError, match="serialized result bound"):
        require_level_one_admission("E4", 1478)


def test_delta_above_the_serialized_budget_names_the_controlling_quantity() -> None:
    request = LevelOneNamedQExpansionRequest(form="DELTA", truncation_order=845)
    with pytest.raises(OperationDomainValidationError, match="serialized result bound"):
        level_one_named_q_expansion(request.form, request.truncation_order)
    with pytest.raises(OperationDomainValidationError, match="serialized result bound"):
        require_level_one_admission("DELTA", 845)


def test_wire_request_rejects_boolean_truncation_orders() -> None:
    with pytest.raises(ValidationError):
        LevelOneNamedQExpansionRequest(form="E4", truncation_order=True)


def test_native_admission_rejects_boolean_truncation_orders() -> None:
    with pytest.raises(OperationDomainValidationError, match="plain integer"):
        require_level_one_admission("DELTA", True)
    with pytest.raises(OperationDomainValidationError, match="plain integer"):
        level_one_named_q_expansion("E4", True)


@pytest.mark.parametrize("form", ["E5", "delta", "e4", "SIGMA", ""])
def test_native_admission_rejects_unknown_forms_before_any_scan(
    monkeypatch: pytest.MonkeyPatch, form: str
) -> None:
    def fail(_index: int, _exponent: int) -> int:
        raise AssertionError("an unknown form must never reach a divisor scan")

    monkeypatch.setattr(level_one_kernel, "divisor_power_sum", fail)
    with pytest.raises(
        OperationDomainValidationError,
        match="form must be one of 'E4', 'E6', or 'DELTA'",
    ):
        level_one_named_q_expansion(cast(NamedLevelOneModularForm, form), 8)
    with pytest.raises(
        OperationDomainValidationError,
        match="form must be one of 'E4', 'E6', or 'DELTA'",
    ):
        require_level_one_admission(cast(NamedLevelOneModularForm, form), 8)


@pytest.mark.parametrize("form", ["E4", "E6", "DELTA"])
def test_every_closed_family_member_is_admitted_at_order_one(
    form: NamedLevelOneModularForm,
) -> None:
    assert form in NAMED_LEVEL_ONE_FORMS
    require_level_one_admission(form, 1)


def test_wire_request_rejects_unknown_form_names() -> None:
    with pytest.raises(ValidationError) as error:
        LevelOneNamedQExpansionRequest(
            form=cast(NamedLevelOneModularForm, "E5"), truncation_order=8
        )
    assert _validation_type(error) == "literal_error"


def _beyond_budget_e4_payload() -> dict[str, object]:
    weight, space_kind, normalization = metadata("E4")
    return {
        "form": "E4",
        "weight": weight,
        "space_kind": space_kind,
        "normalization": normalization,
        "q_expansion": _series(eisenstein_coefficients("E4", 1478)).model_dump(),
    }


def test_exact_expansions_beyond_the_producer_envelope_remain_canonical() -> None:
    with pytest.raises(OperationDomainValidationError, match="serialized result bound"):
        require_level_one_admission("E4", 1478)

    value = LevelOneModularQExpansion.model_validate(_beyond_budget_e4_payload())
    assert value.q_expansion.truncation_order == 1478
    assert value.q_expansion.coefficients[-1].as_fraction() == 240 * divisor_power_sum(
        1477, 3
    )
    assert LevelOneModularQExpansion.model_validate(value.model_dump()) == value
