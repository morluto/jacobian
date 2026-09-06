"""Empty values compose unchanged across the shared group contracts."""

import pytest
from pydantic import ValidationError

from jacobian.canonical import encode_strict_json
from jacobian.math.groups.abelian import (
    FiniteAbelianElement,
    FiniteAbelianProductGroup,
    FiniteAbelianSubgroup,
    element_order,
    elements_equal,
    generated_subgroup,
    normalize_presentation,
    quotient_group,
    reduce_element,
)
from jacobian.math.groups.abelian._tools import TOOLS
from jacobian.math.groups.finite_abelian import (
    FiniteAbelianCharacterSumIntervalProfileSource,
    FiniteAbelianSpectralPairSource,
    compute_finite_abelian_character_sum_interval_profile,
    decide_finite_abelian_spectral_pair,
    finite_abelian_group_factorization,
)


def test_trivial_group_composes_through_element_operations() -> None:
    parent = normalize_presentation(FiniteAbelianProductGroup(moduli=())).group
    element = reduce_element(parent, ())
    payload = element.model_dump(mode="json")
    for name, request in (
        ("abelian_group.element.order.compute", {"element": payload}),
        ("abelian_group.element.equal.decide", {"left": payload, "right": payload}),
    ):
        operation = next(tool for tool in TOOLS if tool.operation_id == name)
        result = operation.run(
            operation.request_type.model_validate_json(encode_strict_json(request))
        )
        assert (
            operation.result_type.model_validate_json(result.model_dump_json())
            == result
        )
    assert element_order(element).order == 1
    assert elements_equal(element, element).equal
    assert element.group.order == element.group.exponent == 1


@pytest.mark.parametrize("moduli", [(), (2, 6)])
@pytest.mark.parametrize("include_identity", [False, True])
def test_empty_generators_preserve_ambient_group(
    moduli: tuple[int, ...], include_identity: bool
) -> None:
    group = FiniteAbelianProductGroup(moduli=moduli)
    subgroup = FiniteAbelianSubgroup(
        group=group, generators=((0,) * len(moduli),) if include_identity else ()
    )
    source = FiniteAbelianSubgroup.model_validate_json(subgroup.model_dump_json())
    assert generated_subgroup(source).index == group.order
    quotient = quotient_group(source)
    assert quotient.quotient == group
    assert quotient.subgroup == source


def test_existing_factorization_consumer_accepts_shared_trivial_parent() -> None:
    group = normalize_presentation(FiniteAbelianProductGroup(moduli=())).group
    result = finite_abelian_group_factorization(group, ((),), ((),))
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_shared_trivial_parent_composes_with_character_operations() -> None:
    group = FiniteAbelianProductGroup(moduli=())
    spectral = decide_finite_abelian_spectral_pair(
        FiniteAbelianSpectralPairSource(group=group, points=((),), frequencies=((),))
    )
    assert type(spectral).model_validate_json(spectral.model_dump_json()) == spectral
    profile = compute_finite_abelian_character_sum_interval_profile(
        FiniteAbelianCharacterSumIntervalProfileSource(
            group=group,
            sequence=((), ()),
            frequencies=((),),
            intervals=((0, 0), (0, 2)),
        )
    )
    assert profile.group_exponent == 1
    assert tuple(cell.remainder_coefficients for cell in profile.sums) == (
        (0,),
        (2,),
    )
    assert type(profile).model_validate_json(profile.model_dump_json()) == profile


@pytest.mark.parametrize("moduli,coordinates", [((), (1,)), ((2,), ())])
def test_empty_support_does_not_relax_coordinate_shape(
    moduli: tuple[int, ...], coordinates: tuple[int, ...]
) -> None:
    with pytest.raises(ValidationError, match="coordinates"):
        FiniteAbelianElement(
            group=FiniteAbelianProductGroup(moduli=moduli),
            coordinates=coordinates,
        )
