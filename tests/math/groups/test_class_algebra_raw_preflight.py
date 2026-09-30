"""Malformed native class partitions are bounded before recursive serialization."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups._models import GroupConjugacyClassesResult, PermutationGroup
from jacobian.math.groups.characters._models import (
    ClassMultiplicationConstantsRequest,
    ClassMultiplicationConstantsResult,
)
from jacobian.math.groups.characters.class_algebra import class_multiplication_constants


def _request(case: str) -> ClassMultiplicationConstantsRequest:
    source = PermutationGroup.model_construct(degree=1, generators=((0,),))
    class_cases: dict[str, object] = {
        "outer-count-before-child": ([],) * 65,
        "classes-list": [((0,),)] * 65,
        "classes-mapping": {"nested": [0] * 65},
        "empty-classes": (),
        "member-count": (((0,),) * 257,),
        "aggregate-member-count": (((0,),) * 128, ((0,),) * 129),
        "class-list": ([(0,)],),
        "empty-class": ((),),
        "member-list": (([0],),),
        "member-width": (((0,) * 65,),),
        "member-scalar": ((([0] * 65,),),),
        "member-bool": (((False,),),),
    }
    source_cases: dict[str, dict[str, object]] = {
        "degree-container": {"degree": [0] * 65},
        "degree-bool": {"degree": True},
        "generator-count": {"generators": ((0,),) * 65},
        "generators-list": {"generators": [[0]]},
        "generator-width": {"generators": ((0,) * 65,)},
        "generator-scalar": {"generators": (([0] * 65,),)},
    }
    classes = class_cases.get(case, (((0,),),))
    source = source.model_copy(update=source_cases.get(case, {}))
    if case == "missing-degree":
        source = PermutationGroup.model_construct(generators=((0,),))
    partition = GroupConjugacyClassesResult.model_construct(
        source=source, classes=classes
    )
    if case == "missing-classes":
        partition = GroupConjugacyClassesResult.model_construct(source=source)
    elif case == "missing-source":
        partition = GroupConjugacyClassesResult.model_construct(classes=classes)
    elif case == "mapping-source":
        partition = partition.model_copy(
            update={"source": {"degree": 1, "generators": [[0]]}}
        )
    if case == "missing-partition":
        return ClassMultiplicationConstantsRequest.model_construct()
    if case == "mapping-partition":
        return ClassMultiplicationConstantsRequest.model_construct(
            partition={"source": source, "classes": classes}
        )
    return ClassMultiplicationConstantsRequest.model_construct(partition=partition)


@pytest.mark.parametrize(
    "case",
    (
        "outer-count-before-child",
        "classes-list",
        "classes-mapping",
        "empty-classes",
        "degree-container",
        "degree-bool",
        "missing-degree",
        "generator-count",
        "generators-list",
        "generator-width",
        "generator-scalar",
        "member-count",
        "aggregate-member-count",
        "class-list",
        "empty-class",
        "member-list",
        "member-width",
        "member-scalar",
        "member-bool",
        "missing-classes",
        "missing-source",
        "mapping-source",
        "missing-partition",
        "mapping-partition",
    ),
)
def test_malformed_native_partition_is_refused_before_dump(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    request = _request(case)
    dumps: list[object] = []

    def unexpected_dump(self: ClassMultiplicationConstantsRequest) -> dict[str, object]:
        dumps.append(self)
        raise AssertionError("raw malformed input reached recursive dump")

    monkeypatch.setattr(
        ClassMultiplicationConstantsRequest, "model_dump", unexpected_dump
    )
    with pytest.raises(
        (OperationDomainValidationError, OperationResourceAdmissionError)
    ) as error:
        class_multiplication_constants(request)
    expected = {
        "outer-count-before-child": "class_algebra_class_count_exceeds_envelope",
        "member-count": "class_algebra_group_order_exceeds_envelope",
        "aggregate-member-count": "class_algebra_group_order_exceeds_envelope",
    }.get(case, "class_algebra_request")
    assert error.value.errors()[0]["type"] == f"groups.characters.{expected}"
    assert dumps == []


@pytest.mark.parametrize("decode_json", (False, True))
def test_bounded_native_partition_preserves_cyclic_products_and_roundtrip(
    decode_json: bool,
) -> None:
    source = PermutationGroup(degree=3, generators=((1, 2, 0),))
    partition = GroupConjugacyClassesResult(
        source=source, classes=(((0, 1, 2),), ((1, 2, 0),), ((2, 0, 1),))
    )
    request = ClassMultiplicationConstantsRequest(partition=partition)
    if decode_json:
        request = ClassMultiplicationConstantsRequest.model_validate_json(
            request.model_dump_json()
        )
    result = class_multiplication_constants(request)
    assert result.constants == tuple(
        tuple(tuple(int((i + j) % 3 == k) for k in range(3)) for j in range(3))
        for i in range(3)
    )
    assert (
        ClassMultiplicationConstantsResult.model_validate_json(result.model_dump_json())
        == result
    )
