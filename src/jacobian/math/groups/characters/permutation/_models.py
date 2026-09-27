"""Typed values for finite permutation characters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from pydantic import model_validator

from jacobian.math.groups._models import PermutationGroup
from jacobian.math.groups.actions._models import (
    MAX_DOMAIN_SIZE,
    MAX_GENERATORS,
    FinitePermutationAction,
)
from jacobian.math.groups.characters._models import (
    ClassAxis,
    ConjugacyClassPartition,
    CyclotomicValue,
    FiniteClassFunction,
)

MAX_PERMUTATION_CHARACTER_GROUP_ORDER = 64
MAX_PERMUTATION_CHARACTER_OUTPUT_BYTES = 1_000_000
MAX_ACTION_LABEL_UTF8_BYTES = 16_384


def _error(reason: str, message: str) -> ValueError:
    from pydantic_core import PydanticCustomError

    return PydanticCustomError(f"groups.characters.permutation.{reason}", message)


def _preflight_action(value: object) -> FinitePermutationAction:
    if not isinstance(value, FinitePermutationAction):
        raise _error(
            "action_type", "character source must be a finite permutation action"
        )
    domain = getattr(value, "domain", None)
    generators = getattr(value, "generators", None)
    if (
        not isinstance(domain, tuple)
        or not 1 <= len(domain) <= MAX_DOMAIN_SIZE
        or not isinstance(generators, tuple)
        or not 1 <= len(generators) <= MAX_GENERATORS
    ):
        raise _error("action_shape", "permutation action exceeds its structural bounds")
    labels_bytes = 0
    for label in domain:
        if (
            type(label) is not str
            or not label
            or len(label) > 64
            or any(0xD800 <= ord(character) <= 0xDFFF for character in label)
        ):
            raise _error(
                "action_label", "action labels must be bounded Unicode scalars"
            )
        labels_bytes += len(label.encode("utf-8"))
        if labels_bytes > MAX_ACTION_LABEL_UTF8_BYTES:
            raise _error(
                "action_labels_bytes", "action labels exceed their UTF-8 byte bound"
            )
    if len(set(domain)) != len(domain):
        raise _error("action_labels_unique", "action domain labels must be distinct")
    degree = len(domain)
    for generator in generators:
        if (
            not isinstance(generator, tuple)
            or len(generator) != degree
            or any(type(point) is not int for point in generator)
            or sorted(generator) != list(range(degree))
        ):
            raise _error(
                "action_generator", "each action generator must permute its domain"
            )
    return value


def _validate_action_partition(
    action: FinitePermutationAction, partition: ConjugacyClassPartition
) -> None:
    classes = getattr(partition, "classes", None)
    if (
        not isinstance(classes, tuple)
        or not 1 <= len(classes) <= MAX_PERMUTATION_CHARACTER_GROUP_ORDER
    ):
        raise _error("partition_shape", "character partition exceeds its class bound")
    total_members = 0
    for class_index, conjugacy_class in enumerate(classes):
        if not isinstance(conjugacy_class, tuple) or not conjugacy_class:
            raise _error("partition_shape", "character classes must be nonempty tuples")
        total_members += len(conjugacy_class)
        if total_members > MAX_PERMUTATION_CHARACTER_GROUP_ORDER:
            raise _error(
                "partition_size_bound",
                "character partition exceeds its group-order bound",
            )
        for element in conjugacy_class:
            if (
                not isinstance(element, tuple)
                or len(element) != len(action.domain)
                or any(type(point) is not int for point in element)
                or sorted(element) != list(range(len(action.domain)))
            ):
                raise _error(
                    "partition_element",
                    f"class {class_index} contains an invalid action permutation",
                )

    expected_group = PermutationGroup(
        degree=len(action.domain), generators=action.generators
    )
    if partition.source != expected_group:
        raise _error(
            "partition_group",
            "class partition must retain the action's generated permutation group",
        )
    # Model validation is structural only. Completeness is established by the
    # admitted producer and checked by consumers only when they rely on it.
    actual = tuple(tuple(tuple(element) for element in cls) for cls in classes)
    flattened = tuple(element for cls in actual for element in cls)
    if (
        len(set(flattened)) != len(flattened)
        or any(tuple(sorted(cls)) != cls for cls in actual)
        or tuple(sorted(actual, key=lambda row: row[0])) != actual
    ):
        raise _error(
            "partition_structure",
            "class partition must be canonically ordered without duplicate elements",
        )


class FiniteCharacter(FiniteClassFunction):
    """A permutation character with its exact action and class partition.

    This value is a true character because its retained permutation action is
    the representation: each class value is exactly the number of fixed
    domain points of a representative. It is a subclass of
    :class:`FiniteClassFunction`, so class-function consumers can use it
    without unpacking or translating the exact values.
    """

    action: FinitePermutationAction
    partition: ConjugacyClassPartition

    @model_validator(mode="before")
    @classmethod
    def preflight_source(cls, data: object) -> object:
        if isinstance(data, Mapping):
            action = data.get("action")
            if isinstance(action, Mapping):
                domain = action.get("domain")
                generators = action.get("generators")
                if isinstance(domain, (list, tuple)) and len(domain) > MAX_DOMAIN_SIZE:
                    raise _error(
                        "action_domain_bound", "action domain exceeds its bound"
                    )
                if (
                    isinstance(generators, (list, tuple))
                    and len(generators) > MAX_GENERATORS
                ):
                    raise _error(
                        "action_generator_bound",
                        "action generator count exceeds its bound",
                    )
                if (
                    isinstance(domain, (list, tuple))
                    and isinstance(generators, (list, tuple))
                    and any(
                        not isinstance(row, (list, tuple)) or len(row) != len(domain)
                        for row in generators
                    )
                ):
                    raise _error(
                        "action_generator_shape",
                        "raw action generator dimensions are invalid",
                    )
            partition = data.get("partition")
            if isinstance(partition, Mapping):
                classes = partition.get("classes")
                if isinstance(classes, (list, tuple)):
                    if len(classes) > MAX_PERMUTATION_CHARACTER_GROUP_ORDER:
                        raise _error(
                            "partition_class_bound",
                            "character partition exceeds its class-count bound",
                        )
                    element_count = 0
                    for conjugacy_class in classes:
                        if not isinstance(conjugacy_class, (list, tuple)):
                            continue
                        element_count += len(conjugacy_class)
                        if element_count > MAX_PERMUTATION_CHARACTER_GROUP_ORDER:
                            raise _error(
                                "partition_size_bound",
                                "character partition exceeds its order bound",
                            )
        return data

    @model_validator(mode="after")
    def require_action_class_values(self) -> Self:
        action = _preflight_action(self.action)
        partition = self.partition
        _validate_action_partition(action, partition)
        classes = partition.classes
        axis = self.axis
        if (
            axis.group != partition.source
            or axis.class_sizes != tuple(len(cls) for cls in classes)
            or axis.class_representatives != tuple(cls[0] for cls in classes)
            or axis.group_order != sum(len(cls) for cls in classes)
        ):
            raise _error(
                "axis_partition",
                "character axis must match its retained class partition",
            )
        if len(self.values) != len(classes) or axis.cyclotomic_order != 1:
            raise _error(
                "value_axis",
                "permutation character values must use the complete rational class axis",
            )
        expected = tuple(
            sum(point == representative[point] for point in range(len(action.domain)))
            for representative in (cls[0] for cls in classes)
        )
        actual: list[int] = []
        for value in self.values:
            if (
                value.order != 1
                or len(value.coefficients) != 1
                or value.coefficients[0].den != 1
            ):
                raise _error(
                    "value_integrality",
                    "permutation character values must be exact nonnegative integers",
                )
            actual.append(value.coefficients[0].num)
        if tuple(actual) != expected:
            raise _error(
                "fixed_point_values",
                "character values must equal the action's class fixed-point counts",
            )
        return self

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        if not update:
            return super().model_copy(deep=deep)
        payload = self.model_dump(mode="python")
        payload.update(update)
        return type(self).model_validate(payload)

    @classmethod
    def _from_permutation_kernel(
        cls,
        *,
        action: FinitePermutationAction,
        partition: ConjugacyClassPartition,
        axis: ClassAxis,
        values: tuple[CyclotomicValue, ...],
    ) -> Self:
        # The operation has already bounded and established the source group,
        # its complete class partition, and each fixed-point value.
        return cls.model_construct(
            action=action, partition=partition, axis=axis, values=values
        )


__all__ = [
    "MAX_ACTION_LABEL_UTF8_BYTES",
    "MAX_PERMUTATION_CHARACTER_GROUP_ORDER",
    "MAX_PERMUTATION_CHARACTER_OUTPUT_BYTES",
    "FiniteCharacter",
]
