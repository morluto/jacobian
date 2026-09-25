"""Bounded exact permutation-character construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NoReturn

from pydantic import ConfigDict, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups._models import (
    GroupConjugacyClassesResult,
    PermutationGroup,
)
from jacobian.math.groups.actions._models import (
    MAX_DOMAIN_SIZE,
    MAX_GENERATORS,
    FinitePermutationAction,
)
from jacobian.math.groups.characters._models import (
    ClassAxis,
    ConjugacyClassPartition,
    CyclotomicValue,
)
from jacobian.math.groups.characters.permutation._models import (
    MAX_ACTION_LABEL_UTF8_BYTES,
    MAX_PERMUTATION_CHARACTER_GROUP_ORDER,
    MAX_PERMUTATION_CHARACTER_OUTPUT_BYTES,
    FiniteCharacter,
    _preflight_action,
)

MAX_PERMUTATION_CHARACTER_WORK = 1_000_000


def _domain_error(location: tuple[str | int, ...], code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(location=location, code=code, message=message)


def _resource_error(
    location: tuple[str | int, ...], code: str, message: str
) -> NoReturn:
    raise OperationResourceAdmissionError(location=location, code=code, message=message)


def _compose(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    """Return left after right for image-array permutations."""
    return tuple(left[right[index]] for index in range(len(left)))


def _inverse(permutation: tuple[int, ...]) -> tuple[int, ...]:
    inverse = [0] * len(permutation)
    for point, image in enumerate(permutation):
        inverse[image] = point
    return tuple(inverse)


def _enumerate_group(action: FinitePermutationAction) -> tuple[tuple[int, ...], ...]:
    """Enumerate at most 65 generated permutations, then fail closed."""
    degree = len(action.domain)
    identity = tuple(range(degree))
    generators = action.generators
    elements = {identity}
    queue = [identity]
    cursor = 0
    while cursor < len(queue):
        current = queue[cursor]
        cursor += 1
        for generator in generators:
            candidate = _compose(generator, current)
            if candidate in elements:
                continue
            elements.add(candidate)
            if len(elements) > MAX_PERMUTATION_CHARACTER_GROUP_ORDER:
                _resource_error(
                    ("action", "generators"),
                    "groups.characters.permutation.group_order_bound",
                    "permutation-character computation admits group order at most "
                    f"{MAX_PERMUTATION_CHARACTER_GROUP_ORDER}",
                )
            queue.append(candidate)
    return tuple(sorted(elements))


def _enumerate_group_classes(
    action: FinitePermutationAction,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Return the complete canonically ordered class partition of the action group."""
    return _classes_from_elements(_enumerate_group(action))


def _classes_from_elements(
    elements: tuple[tuple[int, ...], ...],
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Partition one already admitted complete finite group into classes."""
    inverses = {element: _inverse(element) for element in elements}
    unseen = set(elements)
    classes: list[tuple[tuple[int, ...], ...]] = []
    for representative in elements:
        if representative not in unseen:
            continue
        conjugates = {
            _compose(_compose(conjugator, representative), inverses[conjugator])
            for conjugator in elements
        }
        if not conjugates <= unseen:
            raise RuntimeError(
                "finite permutation conjugacy classes did not partition the group"
            )
        unseen.difference_update(conjugates)
        classes.append(tuple(sorted(conjugates)))
    classes.sort(key=lambda conjugacy_class: conjugacy_class[0])
    return tuple(classes)


def _admit_action(action: object) -> FinitePermutationAction:
    try:
        action = _preflight_action(action)
    except PydanticCustomError as exc:
        _domain_error(("action",), exc.type, exc.message())
    assert isinstance(action, FinitePermutationAction)
    return action


class PermutationCharacterRequest(StrictModel):
    """One bounded finite permutation action."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Compute the exact permutation character of one finite left action. "
                "The action group is enumerated only after bounded generator and "
                "domain admission; groups of order above 64 are refused."
            ),
            "admission_limits": {
                "max_domain_points": MAX_DOMAIN_SIZE,
                "max_action_generators": MAX_GENERATORS,
                "max_action_label_utf8_bytes": MAX_ACTION_LABEL_UTF8_BYTES,
                "max_group_order": MAX_PERMUTATION_CHARACTER_GROUP_ORDER,
                "max_exact_work": MAX_PERMUTATION_CHARACTER_WORK,
                "max_output_bytes": MAX_PERMUTATION_CHARACTER_OUTPUT_BYTES,
            },
        }
    )

    action: FinitePermutationAction

    @model_validator(mode="before")
    @classmethod
    def preflight_raw_action(cls, data: object) -> object:
        if isinstance(data, Mapping):
            action = data.get("action")
            if isinstance(action, Mapping):
                domain = action.get("domain")
                generators = action.get("generators")
                if isinstance(domain, (list, tuple)):
                    if not 1 <= len(domain) <= MAX_DOMAIN_SIZE:
                        raise PydanticCustomError(
                            "groups.characters.permutation.action_domain_bound",
                            "action domain must contain between 1 and "
                            f"{MAX_DOMAIN_SIZE} points",
                        )
                    byte_count = 0
                    for label in domain:
                        if isinstance(label, str) and len(label) <= 64:
                            try:
                                byte_count += len(label.encode("utf-8"))
                            except UnicodeEncodeError:
                                continue
                            if byte_count > MAX_ACTION_LABEL_UTF8_BYTES:
                                raise PydanticCustomError(
                                    "groups.characters.permutation.action_labels_bytes",
                                    "action labels exceed their UTF-8 byte bound",
                                )
                if isinstance(generators, (list, tuple)):
                    if not 1 <= len(generators) <= MAX_GENERATORS:
                        raise PydanticCustomError(
                            "groups.characters.permutation.action_generator_bound",
                            "action generator count is outside its supported bound",
                        )
                    if isinstance(domain, (list, tuple)) and any(
                        not isinstance(generator, (list, tuple))
                        or len(generator) != len(domain)
                        for generator in generators
                    ):
                        raise PydanticCustomError(
                            "groups.characters.permutation.action_generator_shape",
                            "action generators must match the admitted domain size",
                        )
        return data

    @model_validator(mode="after")
    def require_admitted_action(self) -> PermutationCharacterRequest:
        _admit_action(self.action)
        return self


def permutation_character(request: PermutationCharacterRequest) -> FiniteCharacter:
    """Return the complete fixed-point character of an admitted finite action."""
    action = _admit_action(request.action)
    degree = len(action.domain)
    generator_count = len(action.generators)

    # The worst-case work and output bounds use the public order cap before
    # even the bounded closure enumeration begins.
    work_bound = (
        (MAX_PERMUTATION_CHARACTER_GROUP_ORDER + 1) * generator_count * degree
        + MAX_PERMUTATION_CHARACTER_GROUP_ORDER**2 * degree
        + MAX_PERMUTATION_CHARACTER_GROUP_ORDER * degree
    )
    output_bound = (
        64 * MAX_PERMUTATION_CHARACTER_GROUP_ORDER * degree
        + 128 * MAX_PERMUTATION_CHARACTER_GROUP_ORDER * degree
        + 2048 * MAX_PERMUTATION_CHARACTER_GROUP_ORDER
        + 4096
    )
    if work_bound > MAX_PERMUTATION_CHARACTER_WORK:
        _resource_error(
            ("action",),
            "groups.characters.permutation.work_over_envelope",
            "permutation-character group/class work exceeds its exact envelope",
        )
    if output_bound > MAX_PERMUTATION_CHARACTER_OUTPUT_BYTES:
        _resource_error(
            ("action",),
            "groups.characters.permutation.output_over_envelope",
            "permutation-character output exceeds its exact byte envelope",
        )

    # Group closure is capped at 65 elements; no over-order group is ever
    # expanded into a complete class partition.
    elements = _enumerate_group(action)
    class_rows = _classes_from_elements(elements)

    group = PermutationGroup(degree=degree, generators=action.generators)
    group_partition = GroupConjugacyClassesResult._from_kernel(group, class_rows)
    partition = ConjugacyClassPartition._from_group_result(group_partition)
    class_sizes = tuple(len(conjugacy_class) for conjugacy_class in class_rows)
    representatives = tuple(conjugacy_class[0] for conjugacy_class in class_rows)
    axis = ClassAxis._from_kernel(
        class_sizes=class_sizes,
        cyclotomic_order=1,
        group=group,
        class_representatives=representatives,
    )
    values = tuple(
        CyclotomicValue._from_kernel(
            order=1,
            coefficients=(
                CanonicalRational.from_integer_ratio(
                    sum(point == representative[point] for point in range(degree)),
                    1,
                ),
            ),
        )
        for representative in representatives
    )
    return FiniteCharacter._from_permutation_kernel(
        action=action,
        partition=partition,
        axis=axis,
        values=values,
    )


__all__ = ["PermutationCharacterRequest", "permutation_character"]
