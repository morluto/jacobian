"""Native model bypasses cannot evade retained-source admission."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import ValidationError
from tests.math.geometry.crystallographic.extensions.test_validator_repair_boundaries import (
    _real_complex,
)
from tests.math.geometry.crystallographic.extensions.translation_tori.test_translation_torus_chains import (
    _checked_translation_torus,
)

from jacobian.math.geometry.crystallographic.extensions._models import (
    BieberbachFaceOrbitComplex,
    CrystallographicFundamentalDomainResult,
)
from jacobian.math.geometry.crystallographic.extensions.translation_tori._models import (
    BieberbachTranslationTorusChains,
)
from jacobian.math.geometry.crystallographic.extensions.translation_tori.operations import (
    translation_torus_quotient_chains,
)


@pytest.fixture(scope="module", params=["face", "torus"])
def result(
    request: pytest.FixtureRequest,
) -> BieberbachFaceOrbitComplex | BieberbachTranslationTorusChains:
    if request.param == "face":
        return _real_complex()
    return translation_torus_quotient_chains(_checked_translation_torus(((1,),)))


def _forged_source(
    source: CrystallographicFundamentalDomainResult, defect: str
) -> CrystallographicFundamentalDomainResult:
    if defect == "missing_source":
        return type(source).model_construct()
    pairing = source.source
    realization = pairing.affine_realization
    if defect == "wrong_realization":
        pairing = pairing.model_copy(update={"affine_realization": object()})
    elif defect == "empty_actions":
        extension = realization.source.model_copy(update={"action_matrices": ()})
        pairing = pairing.model_copy(
            update={
                "affine_realization": realization.model_copy(
                    update={"source": extension}
                )
            }
        )
    elif defect == "empty_sections":
        pairing = pairing.model_copy(
            update={
                "affine_realization": realization.model_copy(
                    update={"section_maps": ()}
                )
            }
        )
    elif defect == "vertex_index":
        profile = pairing.facet_profile
        facet = profile.facets[0].model_copy(update={"source_vertex_indices": (99,)})
        pairing = pairing.model_copy(
            update={
                "facet_profile": profile.model_copy(
                    update={"facets": (facet, *profile.facets[1:])}
                )
            }
        )
    else:
        field, value = {
            "source_index": ("source_facet_index", 99),
            "target_index": ("target_facet_index", 99),
            "translation_rank": ("lattice_translation", ()),
        }[defect]
        side = pairing.pairings[0].model_copy(update={field: value})
        pairing = pairing.model_copy(update={"pairings": (side, *pairing.pairings[1:])})
    return source.model_copy(update={"source": pairing})


@pytest.mark.parametrize(
    "defect",
    [
        "missing_source",
        "wrong_realization",
        "empty_actions",
        "empty_sections",
        "vertex_index",
        "source_index",
        "target_index",
        "translation_rank",
    ],
)
@pytest.mark.parametrize("whole_instance", [False, True])
def test_native_source_is_admitted_before_parent_dereferences(
    result: BieberbachFaceOrbitComplex | BieberbachTranslationTorusChains,
    defect: str,
    whole_instance: bool,
) -> None:
    source = _forged_source(result.source, defect)
    candidate = result.model_copy(update={"source": source})
    # Whole instances skip ordinary nested field validation unless the result
    # explicitly revalidates them. A raw source field must be equally safe.
    payload = candidate if whole_instance else {**vars(result), "source": source}
    with pytest.raises(ValidationError):
        type(result).model_validate(payload)


class _UninspectedTuple(tuple[object, ...]):
    def __len__(self) -> int:
        pytest.fail("unadmitted native container length hook ran")

    def __iter__(self) -> Iterator[object]:
        pytest.fail("unadmitted native container iteration hook ran")


def test_native_preflight_does_not_serialize_or_inspect_subclass_containers(
    result: BieberbachFaceOrbitComplex | BieberbachTranslationTorusChains,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = result.source
    pairing = source.source
    realization = pairing.affine_realization
    bad_extension = realization.source.model_copy(
        update={"action_matrices": _UninspectedTuple()}
    )
    bad_pairing = pairing.model_copy(
        update={
            "affine_realization": realization.model_copy(
                update={"source": bad_extension}
            )
        }
    )
    bad_source = source.model_copy(update={"source": bad_pairing})

    def unexpected_dump(*args: object, **kwargs: object) -> None:
        pytest.fail("native preflight invoked serialization")

    monkeypatch.setattr(
        CrystallographicFundamentalDomainResult, "model_dump", unexpected_dump
    )
    with pytest.raises(ValidationError):
        type(result).model_validate(result.model_copy(update={"source": bad_source}))


def test_valid_native_and_json_results_retain_the_same_source(
    result: BieberbachFaceOrbitComplex | BieberbachTranslationTorusChains,
) -> None:
    assert type(result).model_validate(result) == result
    assert type(result).model_validate_json(result.model_dump_json()) == result
