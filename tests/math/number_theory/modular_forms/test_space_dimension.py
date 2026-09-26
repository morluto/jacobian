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
    assert (full.index, full.genus, full.cusp_count) == (1, 0, 1)
    assert (full.elliptic_points_order_2, full.elliptic_points_order_3) == (1, 1)


def test_m12_decomposition_and_ingredients() -> None:
    result = compute_space_dimension(
        SpaceDimensionRequest(space=_space(12, "M").model_dump())
    )

    assert result.dimension == 2
    assert result.eisenstein_dimension == 1
    assert result.cusp_dimension == 1
    assert (result.index, result.genus, result.cusp_count) == (1, 0, 1)
    assert (result.elliptic_points_order_2, result.elliptic_points_order_3) == (1, 1)


@pytest.mark.parametrize(
    "level, weight, index, genus, cusps, e2, e3, holomorphic, cusp",
    [
        # Small composite levels independently evaluated from the Gamma0
        # index, cusp, elliptic-point, and dimension formulas.
        (6, 2, 12, 0, 4, 0, 0, 3, 0),
        (6, 4, 12, 0, 4, 0, 0, 5, 1),
        (6, 6, 12, 0, 4, 0, 0, 7, 3),
        (12, 2, 24, 0, 6, 0, 0, 5, 0),
        (12, 4, 24, 0, 6, 0, 0, 9, 3),
        (12, 6, 24, 0, 6, 0, 0, 13, 7),
        # Stein's published dimension table includes these Gamma0(10) cusp
        # dimensions for k=2,4,6,24.
        (10, 2, 18, 0, 4, 2, 0, 3, 0),
        (10, 4, 18, 0, 4, 2, 0, 7, 3),
        (10, 6, 18, 0, 4, 2, 0, 9, 5),
        (10, 24, 18, 0, 4, 2, 0, 37, 33),
    ],
)
def test_composite_gamma0_dimensions(
    level: int,
    weight: int,
    index: int,
    genus: int,
    cusps: int,
    e2: int,
    e3: int,
    holomorphic: int,
    cusp: int,
) -> None:
    full = space_dimension(_space(weight, "M", level))
    cuspidal = space_dimension(_space(weight, "S", level))

    assert full.dimension == holomorphic
    assert full.cusp_dimension == cusp
    assert full.eisenstein_dimension == holomorphic - cusp
    assert cuspidal.dimension == cusp
    assert (
        full.index,
        full.genus,
        full.cusp_count,
        full.elliptic_points_order_2,
        full.elliptic_points_order_3,
    ) == (index, genus, cusps, e2, e3)
    assert full.dimension == full.eisenstein_dimension + full.cusp_dimension


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
        "index",
        "genus",
        "cusp_count",
        "elliptic_points_order_2",
        "elliptic_points_order_3",
        "eisenstein_dimension",
        "cusp_dimension",
    }


def test_native_and_catalog_results_agree() -> None:
    space = _space(4, "M", level=10)
    native = space_dimension(space)
    catalog = compute_space_dimension(SpaceDimensionRequest(space=space.model_dump()))

    assert catalog == native
    assert catalog.dimension == 7
    assert (
        SpaceDimensionResult.model_validate_json(catalog.model_dump_json()) == catalog
    )


def test_level_above_operation_envelope_rejected() -> None:
    with pytest.raises(OperationDomainValidationError):
        space_dimension(_space(12, "M", level=10_001))
    with pytest.raises(ValidationError):
        SpaceDimensionRequest(
            space={
                **_space(4, "S", level=10_000).model_dump(),
                "level": 10_001,
            }
        )


@pytest.mark.parametrize(
    "field, value",
    [
        ("level", 0),
        ("level", -1),
        ("level", 10_001),
        ("weight", -1),
        ("weight", 1_000_001),
    ],
)
def test_forged_space_scalars_are_admitted_before_factorization(
    field: str, value: int
) -> None:
    fields = _space(4, "M").model_dump()
    fields[field] = value
    forged = ModularFormSpace.model_construct(**fields)

    with pytest.raises(OperationDomainValidationError):
        space_dimension(forged)


def test_request_schema_advertises_the_operation_level_envelope() -> None:
    schema = SpaceDimensionRequest.model_json_schema()
    space_schema = schema["$defs"]["Gamma0DimensionSpaceInput"]

    assert space_schema["properties"]["level"]["minimum"] == 1
    assert space_schema["properties"]["level"]["maximum"] == 10_000


def test_upper_admitted_level_is_computable() -> None:
    result = space_dimension(_space(1_000_000, "M", level=10_000))

    assert (result.index, result.genus, result.cusp_count) == (18_000, 1_411, 180)
    assert result.dimension == 1_499_998_590
    assert result.cusp_dimension == 1_499_998_410
    assert result.eisenstein_dimension == 180


def test_negative_weight_rejected_structurally() -> None:
    with pytest.raises(ValidationError):
        _space(-2)


def test_forged_decomposition_rejected() -> None:
    space = _space(12, "M")

    with pytest.raises(ValidationError):
        SpaceDimensionResult(
            space=space,
            dimension=5,
            index=1,
            genus=0,
            cusp_count=1,
            elliptic_points_order_2=1,
            elliptic_points_order_3=1,
            eisenstein_dimension=1,
            cusp_dimension=1,
        )


def test_native_rejects_a_non_space_value() -> None:
    with pytest.raises(OperationDomainValidationError):
        space_dimension("not-a-space")  # type: ignore[arg-type]
