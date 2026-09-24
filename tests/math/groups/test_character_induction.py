import json
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups._models import PermutationGroup
from jacobian.math.groups.characters._models import (
    ClassAxis,
    ClassFunctionInductionRequest,
    CyclotomicValue,
    FiniteClassFunction,
)
from jacobian.math.groups.characters.operations import (
    class_function_induce_from_subgroup,
)
from jacobian.math.groups.operations import group_conjugacy_classes

_C2 = PermutationGroup(degree=3, generators=((1, 0, 2),))
_C3 = PermutationGroup(degree=3, generators=((1, 2, 0),))
_S3 = PermutationGroup(degree=3, generators=((1, 2, 0), (1, 0, 2)))


def _class_function(
    group: PermutationGroup, values: tuple[int, ...]
) -> FiniteClassFunction:
    classes = group_conjugacy_classes(
        group.degree, [list(generator) for generator in group.generators]
    )
    axis = ClassAxis._from_kernel(
        class_sizes=tuple(len(conjugacy_class) for conjugacy_class in classes),
        cyclotomic_order=1,
        group=group,
        class_representatives=tuple(tuple(cls[0]) for cls in classes),
    )
    return FiniteClassFunction(
        axis=axis,
        values=tuple(
            CyclotomicValue(
                order=1,
                coefficients=(CanonicalRational.from_fraction(Fraction(value)),),
            )
            for value in values
        ),
    )


@pytest.mark.parametrize(
    ("source_values", "expected_induced", "expected_multiplicities"),
    (
        ((1, 1), (3, 1, 0), (1, 0, 1)),
        ((1, -1), (3, -1, 0), (0, 1, 1)),
    ),
)
def test_s3_induction_from_transposition_subgroup(
    source_values: tuple[int, ...],
    expected_induced: tuple[int, ...],
    expected_multiplicities: tuple[int, ...],
):
    request = ClassFunctionInductionRequest(
        class_function=_class_function(_C2, source_values), parent_group=_S3
    )

    result = class_function_induce_from_subgroup(request)

    assert result.subgroup_class_to_parent_class == (0, 1)
    assert tuple(
        value.coefficients[0].as_fraction() for value in result.induced.values
    ) == tuple(Fraction(value) for value in expected_induced)
    assert type(result).model_validate_json(result.model_dump_json()) == result

    # Independent character-inner-product oracle against the three known
    # irreducible S3 characters on classes of sizes 1, 3, 2.
    irreducibles = ((1, 1, 1), (1, -1, 1), (2, 0, -1))
    multiplicities = tuple(
        sum(
            class_size * induced_value * character_value
            for class_size, induced_value, character_value in zip(
                (1, 3, 2), expected_induced, character, strict=True
            )
        )
        // 6
        for character in irreducibles
    )
    assert multiplicities == expected_multiplicities


def test_induction_rejects_source_group_not_in_parent():
    request = ClassFunctionInductionRequest(
        class_function=_class_function(_C2, (1, 1)),
        parent_group=PermutationGroup(degree=3, generators=((1, 2, 0),)),
    )

    with pytest.raises(OperationDomainValidationError) as exc_info:
        class_function_induce_from_subgroup(request)

    assert exc_info.value.errors()[0]["type"] == (
        "groups.characters.induction_not_subgroup"
    )


def test_deserializing_induction_result_does_not_authenticate_class_map():
    result = class_function_induce_from_subgroup(
        ClassFunctionInductionRequest(
            class_function=_class_function(_C2, (1, 1)), parent_group=_S3
        )
    )
    payload = json.loads(result.model_dump_json())
    payload["subgroup_class_to_parent_class"][1] = 2

    restored = type(result).model_validate_json(json.dumps(payload))

    assert restored.subgroup_class_to_parent_class == (0, 2)


def test_s3_induction_preserves_the_cyclotomic_field():
    classes = group_conjugacy_classes(
        _C3.degree, [list(generator) for generator in _C3.generators]
    )
    axis = ClassAxis._from_kernel(
        class_sizes=tuple(len(conjugacy_class) for conjugacy_class in classes),
        cyclotomic_order=3,
        group=_C3,
        class_representatives=tuple(tuple(cls[0]) for cls in classes),
    )
    source_function = FiniteClassFunction(
        axis=axis,
        values=tuple(
            CyclotomicValue(
                order=3,
                coefficients=tuple(
                    CanonicalRational.from_fraction(Fraction(coefficient))
                    for coefficient in coefficients
                ),
            )
            for coefficients in ((1, 0), (0, 1), (-1, -1))
        ),
    )

    result = class_function_induce_from_subgroup(
        ClassFunctionInductionRequest(class_function=source_function, parent_group=_S3)
    )

    assert result.subgroup_class_to_parent_class == (0, 2, 2)
    assert tuple(value.order for value in result.induced.values) == (3, 3, 3)
    assert tuple(
        tuple(coefficient.as_fraction() for coefficient in value.coefficients)
        for value in result.induced.values
    ) == (
        (Fraction(2), Fraction(0)),
        (Fraction(0), Fraction(0)),
        (Fraction(-1), Fraction(0)),
    )
