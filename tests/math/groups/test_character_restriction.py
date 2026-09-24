from fractions import Fraction

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups._models import PermutationGroup
from jacobian.math.groups.characters._models import (
    FiniteClassFunction,
)
from jacobian.math.groups.characters.operations import (
    character_table,
    class_function_restrict_to_subgroup,
)
from jacobian.math.groups.operations import group_conjugacy_classes


def _s3_standard_class_function() -> FiniteClassFunction:
    source = PermutationGroup(degree=3, generators=((1, 2, 0), (1, 0, 2)))
    classes = group_conjugacy_classes(
        source.degree, [list(generator) for generator in source.generators]
    )
    from jacobian.math.groups._models import GroupConjugacyClassesResult
    from jacobian.math.groups.characters._models import ConjugacyClassPartition

    partition = ConjugacyClassPartition._from_group_result(
        GroupConjugacyClassesResult._from_kernel(
            source,
            tuple(
                tuple(tuple(element) for element in conjugacy_class)
                for conjugacy_class in classes
            ),
        )
    )
    table = character_table(partition)
    standard = next(row for row in table.rows if row.label == "standard")
    return FiniteClassFunction._from_kernel(axis=table.axis, values=standard.values)


def test_s3_standard_restricts_to_transposition_subgroup_with_class_map():
    function = _s3_standard_class_function()
    result = class_function_restrict_to_subgroup(
        function, PermutationGroup(degree=3, generators=((1, 0, 2),))
    )

    assert result.target_class_to_source_class == (0, 1)
    assert result.subgroup_partition.classes == (
        ((0, 1, 2),),
        ((1, 0, 2),),
    )
    assert type(result).model_validate_json(result.model_dump_json()) == result
    assert tuple(
        tuple(coefficient.as_fraction() for coefficient in value.coefficients)
        for value in result.restricted.values
    ) == ((Fraction(2), Fraction(0)), (Fraction(0), Fraction(0)))

    # Independently check the expected multiplicities against the two
    # irreducible C2 characters: (1/2)(2 + 0)=1 for each sign choice.
    values = tuple(
        value.coefficients[0].as_fraction() for value in result.restricted.values
    )
    assert (values[0] + values[1]) / 2 == 1
    assert (values[0] - values[1]) / 2 == 1


def test_s3_standard_restricts_to_three_cycle_subgroup():
    function = _s3_standard_class_function()
    result = class_function_restrict_to_subgroup(
        function, PermutationGroup(degree=3, generators=((1, 2, 0),))
    )

    assert result.target_class_to_source_class == (0, 2, 2)
    assert tuple(
        tuple(coefficient.as_fraction() for coefficient in value.coefficients)
        for value in result.restricted.values
    ) == (
        (Fraction(2), Fraction(0)),
        (Fraction(-1), Fraction(0)),
        (Fraction(-1), Fraction(0)),
    )


def test_restriction_rejects_a_generator_outside_the_source_group():
    source = PermutationGroup(degree=3, generators=((1, 2, 0),))
    classes = group_conjugacy_classes(
        source.degree, [list(generator) for generator in source.generators]
    )
    from jacobian._exact import CanonicalRational
    from jacobian.math.groups.characters._models import (
        ClassAxis,
        CyclotomicValue,
    )

    axis = ClassAxis._from_kernel(
        class_sizes=tuple(len(conjugacy_class) for conjugacy_class in classes),
        cyclotomic_order=1,
        group=source,
        class_representatives=tuple(tuple(cls[0]) for cls in classes),
    )
    function = FiniteClassFunction(
        axis=axis,
        values=tuple(
            CyclotomicValue(
                order=1,
                coefficients=(CanonicalRational.from_fraction(Fraction(1)),),
            )
            for _ in classes
        ),
    )

    with pytest.raises(OperationDomainValidationError) as exc_info:
        class_function_restrict_to_subgroup(
            function, PermutationGroup(degree=3, generators=((1, 0, 2),))
        )
    assert exc_info.value.errors()[0]["type"] == (
        "groups.characters.restriction_not_subgroup"
    )


def test_native_restriction_composes_from_canonical_values():
    # Thread follow-up: the native API composes canonical mathematical values
    # directly, without constructing the wire-only request model.
    function = _s3_standard_class_function()
    result = class_function_restrict_to_subgroup(
        function, PermutationGroup(degree=3, generators=((1, 0, 2),))
    )
    assert result.target_class_to_source_class == (0, 1)


def test_native_restriction_rejects_forged_class_function():
    from jacobian.math.groups.characters._models import ClassAxis

    forged = FiniteClassFunction.model_construct(
        axis=ClassAxis(class_sizes=(1, 3, 2), group_order=6, cyclotomic_order=1),
        values=(),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        class_function_restrict_to_subgroup(
            forged, PermutationGroup(degree=3, generators=((1, 0, 2),))
        )
    assert error.value.errors()[0]["type"] == (
        "groups.characters.invalid_class_function"
    )


def test_native_restriction_rejects_malformed_subgroup_without_pydantic_leak():
    function = _s3_standard_class_function()
    with pytest.raises(OperationDomainValidationError) as error:
        class_function_restrict_to_subgroup(
            function, {"degree": 3, "generators": [(1, 0, 2)]}
        )
    assert error.value.errors()[0]["type"] == (
        "groups.characters.permutation_group_type"
    )

    forged_group = PermutationGroup.model_construct(degree=3, generators=((4, 5, 6),))
    with pytest.raises(OperationDomainValidationError) as error:
        class_function_restrict_to_subgroup(function, forged_group)
    assert error.value.errors()[0]["type"] == (
        "groups.characters.permutation_group_shape"
    )
