"""Exact and contract tests for modular-form space dimensions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.modular_forms import (
    level_one_named_q_expansion,
    space_dimension,
)
from jacobian.math.number_theory.modular_forms._models import (
    SpaceDimensionRequest,
    SpaceDimensionResult,
)
from jacobian.math.number_theory.modular_forms._tools import compute_space_dimension
from jacobian.math.number_theory.modular_forms.values import ModularFormSpace


def _space(weight: int, kind: str = "M", level: int = 1) -> ModularFormSpace:
    return ModularFormSpace(
        group="GAMMA0",
        level=level,
        weight=weight,
        kind=kind,  # type: ignore[arg-type]
        character="TRIVIAL",
        coefficient_domain="QQ",
    )


# (weight, dim M_k, dim S_k) for level one.
_KNOWN = [
    (0, 1, 0),
    (1, 0, 0),
    (2, 0, 0),
    (3, 0, 0),
    (4, 1, 0),
    (6, 1, 0),
    (8, 1, 0),
    (10, 1, 0),
    (12, 2, 1),
    (14, 1, 0),
    (16, 2, 1),
    (18, 2, 1),
    (20, 2, 1),
    (22, 2, 1),
    (24, 3, 2),
    (26, 2, 1),
    (36, 4, 3),
]


@pytest.mark.parametrize("weight, holomorphic, cusp", _KNOWN)
def test_known_level_one_dimensions(weight: int, holomorphic: int, cusp: int) -> None:
    full = space_dimension(_space(weight, "M"))
    cuspidal = space_dimension(_space(weight, "S"))

    assert full.dimension == holomorphic
    assert cuspidal.dimension == cusp
    assert full.floor_term == weight // 12
    assert full.congruence_class == weight % 12
    assert cuspidal.floor_term == weight // 12


def test_m12_decomposition_and_ingredients() -> None:
    result = compute_space_dimension(SpaceDimensionRequest(space=_space(12, "M")))

    assert result.dimension == 2
    assert result.eisenstein_dimension == 1
    assert result.cusp_dimension == 1
    assert result.floor_term == 1
    assert result.congruence_class == 0


def test_defining_invariant_holomorphic_is_eisenstein_plus_cusp() -> None:
    for weight in range(60):
        full = space_dimension(_space(weight, "M"))
        cuspidal = space_dimension(_space(weight, "S"))

        assert full.dimension == full.eisenstein_dimension + full.cusp_dimension
        assert cuspidal.dimension == cuspidal.cusp_dimension
        assert full.cusp_dimension == cuspidal.cusp_dimension
        assert full.eisenstein_dimension == (
            1 if (weight == 0 or (weight >= 4 and weight % 2 == 0)) else 0
        )


def test_dimension_agrees_with_named_form_spans() -> None:
    # S_12 is spanned by Delta; M_12 adds the E_12 Eisenstein line.
    assert space_dimension(_space(12, "S")).dimension == 1
    delta = level_one_named_q_expansion("DELTA", 3)
    assert delta.weight == 12
    assert delta.space_kind == "CUSP"
    e4 = level_one_named_q_expansion("E4", 2)
    assert e4.weight == 4
    assert space_dimension(_space(4, "M")).dimension == 1


def test_no_q_coefficients_in_result() -> None:
    result = space_dimension(_space(12, "M"))
    payload = result.model_dump(mode="json")

    assert set(payload) == {
        "space",
        "dimension",
        "floor_term",
        "congruence_class",
        "eisenstein_dimension",
        "cusp_dimension",
    }


def test_native_and_catalog_results_agree() -> None:
    space = _space(24, "S")
    native = space_dimension(space)
    catalog = compute_space_dimension(SpaceDimensionRequest(space=space))

    assert catalog == native
    assert catalog.dimension == 2
    assert (
        SpaceDimensionResult.model_validate_json(catalog.model_dump_json()) == catalog
    )


def test_higher_level_rejected_as_unsupported() -> None:
    with pytest.raises(OperationDomainValidationError):
        space_dimension(_space(12, "M", level=2))
    with pytest.raises(OperationDomainValidationError):
        compute_space_dimension(SpaceDimensionRequest(space=_space(4, "S", level=11)))


def test_negative_weight_rejected_structurally() -> None:
    with pytest.raises(ValidationError):
        _space(-2)


def test_forged_decomposition_rejected() -> None:
    space = _space(12, "M")

    with pytest.raises(ValidationError):
        SpaceDimensionResult(
            space=space,
            dimension=5,
            floor_term=1,
            congruence_class=0,
            eisenstein_dimension=1,
            cusp_dimension=1,
        )


def test_native_rejects_a_non_space_value() -> None:
    with pytest.raises(OperationDomainValidationError):
        space_dimension("not-a-space")  # type: ignore[arg-type]
