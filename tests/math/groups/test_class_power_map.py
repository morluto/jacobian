"""Independent finite checks for exact conjugacy-class power maps."""

import pytest
from pydantic import ValidationError

from jacobian.math.groups._models import GroupConjugacyClassesRequest
from jacobian.math.groups._tools import compute_group_conjugacy_classes
from jacobian.math.groups.characters._models import ClassPowerMapRequest
from jacobian.math.groups.characters.operations import class_power_map


def _compose(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(right[left[index]] for index in range(len(left)))


def _power(value: tuple[int, ...], exponent: int) -> tuple[int, ...]:
    result = tuple(range(len(value)))
    for _ in range(exponent):
        result = _compose(result, value)
    return result


@pytest.mark.parametrize(
    ("degree", "generators", "exponent"),
    [
        (3, ((1, 2, 0), (1, 0, 2)), 2),  # S3
        (5, ((1, 2, 3, 4, 0),), 3),  # C5
    ],
)
def test_class_power_map_matches_elementwise_oracle(
    degree: int, generators: tuple[tuple[int, ...], ...], exponent: int
) -> None:
    partition = compute_group_conjugacy_classes(
        GroupConjugacyClassesRequest(degree=degree, generators=generators)
    )
    result = class_power_map(
        ClassPowerMapRequest(partition=partition, exponent=exponent)
    )

    element_to_class = {
        tuple(element): index
        for index, conjugacy_class in enumerate(partition.classes)
        for element in conjugacy_class
    }
    expected: list[int] = []
    for conjugacy_class in partition.classes:
        targets = {
            element_to_class[_power(tuple(element), exponent)]
            for element in conjugacy_class
        }
        assert len(targets) == 1
        expected.append(targets.pop())
    assert result.image_class_indices == tuple(expected)


def test_class_power_map_reestablishes_input_partition() -> None:
    group = {"degree": 3, "generators": [[1, 2, 0], [1, 0, 2]]}
    request = GroupConjugacyClassesRequest.model_validate(group)
    partition = compute_group_conjugacy_classes(request)
    altered = partition.model_copy(update={"classes": partition.classes[:-1]})
    with pytest.raises(Exception, match="partition"):
        class_power_map(ClassPowerMapRequest(partition=altered, exponent=2))


def test_power_exponent_is_bounded_before_execution() -> None:
    with pytest.raises(ValidationError):
        ClassPowerMapRequest.model_validate({"partition": {}, "exponent": 1_000_001})
