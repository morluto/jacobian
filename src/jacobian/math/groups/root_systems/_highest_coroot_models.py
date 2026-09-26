"""Typed componentwise highest-coroot values."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.groups.root_systems._models import (
    MAX_POSITIVE_ROOTS,
    MAX_RANK,
    PositiveCorootsResult,
    _cartan_components,
    _validation_error,
)


class HighestCorootComponent(StrictModel):
    """The maximum positive-coroot element for one connected factor."""

    component_index: StrictInt = Field(ge=0, le=MAX_RANK - 1)
    simple_root_indices: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_RANK
    )
    highest_positive_root_coroot_pair_index: StrictInt = Field(
        ge=0, le=MAX_POSITIVE_ROOTS - 1
    )
    coroot_preimage_root_coefficients: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_RANK
    )
    highest_coroot_coefficients: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_RANK
    )
    comarks: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_RANK)

    @model_validator(mode="after")
    def require_positive_comarks(self) -> Self:
        if (
            self.simple_root_indices != tuple(sorted(set(self.simple_root_indices)))
            or len(self.comarks) != len(self.simple_root_indices)
            or any(value <= 0 for value in self.comarks)
            or not any(self.coroot_preimage_root_coefficients)
            or not any(self.highest_coroot_coefficients)
            or any(value < 0 for value in self.coroot_preimage_root_coefficients)
            or any(value < 0 for value in self.highest_coroot_coefficients)
        ):
            raise _validation_error(
                "highest_coroot_component",
                "comarks must be positive and agree in length with the ordered component axis",
            )
        return self


class HighestCorootsResult(StrictModel):
    """Highest positive coroots mapped into the canonical positive-root table."""

    positive_coroots: PositiveCorootsResult
    components: tuple[HighestCorootComponent, ...] = Field(
        min_length=1, max_length=MAX_RANK
    )

    @model_validator(mode="after")
    def require_source_map(self) -> Self:
        pairs = self.positive_coroots.positive_root_coroot_pairs
        matrix = self.positive_coroots.datum.cartan_matrix.entries
        rank = len(matrix)
        expected_factors = _cartan_components(matrix)
        actual_factors = tuple(
            component.simple_root_indices for component in self.components
        )
        simple_indices = tuple(
            index
            for component in self.components
            for index in component.simple_root_indices
        )
        first_indices = tuple(
            component.simple_root_indices[0] for component in self.components
        )
        if (
            tuple(sorted(simple_indices)) != tuple(range(rank))
            or len(set(simple_indices)) != rank
            or actual_factors != expected_factors
            or first_indices != tuple(sorted(first_indices))
        ):
            raise _validation_error(
                "highest_coroot_components",
                "component axes must partition and canonically order the Cartan simple-root axis",
            )

        for component_index, component in enumerate(self.components):
            if component.component_index != component_index:
                raise _validation_error(
                    "highest_coroot_component_index",
                    "component indices must follow the canonical component order",
                )
            simple_indices = component.simple_root_indices
            factor = set(simple_indices)
            pair_index = component.highest_positive_root_coroot_pair_index
            if pair_index >= len(pairs):
                raise _validation_error(
                    "highest_coroot_pair_index",
                    "selected highest-coroot pair index must lie in the source table",
                )
            highest = pairs[pair_index]
            support = {
                index
                for index, coefficient in enumerate(highest.root_coefficients)
                if coefficient
            }
            if (
                len(highest.root_coefficients) != rank
                or not support
                or not support <= factor
                or component.coroot_preimage_root_coefficients
                != highest.root_coefficients
                or component.highest_coroot_coefficients != highest.coroot_coefficients
                or component.comarks
                != tuple(highest.coroot_coefficients[index] for index in simple_indices)
                or any(
                    value != 0
                    for index, value in enumerate(highest.coroot_coefficients)
                    if index not in factor
                )
            ):
                raise _validation_error(
                    "highest_coroot_source_map",
                    "highest root, coroot, and comarks must map to the selected source pair",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        positive_coroots: PositiveCorootsResult,
        components: tuple[HighestCorootComponent, ...],
    ) -> Self:
        return cls.model_construct(
            positive_coroots=positive_coroots,
            components=components,
        )
