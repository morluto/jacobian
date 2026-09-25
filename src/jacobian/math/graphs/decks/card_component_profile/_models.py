"""Contracts for connected-component profiles of anonymous deck cards."""

from typing import Annotated, Self

from pydantic import Field, model_validator

from jacobian._exact import DecimalIntegerEncoding
from jacobian._models import StrictModel
from jacobian.math.graphs.decks._models import (
    MAX_UNLABELLED_DECK_VERTICES,
    AnonymousGraphCardMultiset,
)

MAX_PROFILE_COUNT_DIGITS = 17
MAX_COMPONENT_SIZE_PROFILES = 42


class AnonymousDeckComponentProfileRequest(StrictModel):
    """Compute a cardwise component-size profile for an anonymous multiset."""

    deck: AnonymousGraphCardMultiset


class CardComponentSizeProfile(StrictModel):
    """One card invariant: its component orders, sorted increasingly."""

    component_orders: tuple[int, ...] = Field(
        max_length=MAX_UNLABELLED_DECK_VERTICES,
        description=(
            "Positive component vertex counts in nondecreasing order; their sum "
            "is the declared card order. The empty tuple denotes the zero-vertex graph."
        ),
    )
    multiplicity: Annotated[
        int, DecimalIntegerEncoding(max_digits=MAX_PROFILE_COUNT_DIGITS)
    ] = Field(ge=1)


class AnonymousDeckComponentProfile(StrictModel):
    """Histogram of card component-size multisets, weighted by deck multiplicity."""

    card_order: int = Field(ge=0, le=MAX_UNLABELLED_DECK_VERTICES)
    card_count: Annotated[
        int, DecimalIntegerEncoding(max_digits=MAX_PROFILE_COUNT_DIGITS)
    ] = Field(ge=0)
    profiles: tuple[CardComponentSizeProfile, ...] = Field(
        max_length=MAX_COMPONENT_SIZE_PROFILES
    )

    @model_validator(mode="after")
    def require_canonical_profile(self) -> Self:
        if type(self.profiles) is not tuple:
            raise ValueError("profiles must be an immutable tuple")
        keys: list[tuple[int, ...]] = []
        total = 0
        for item in self.profiles:
            orders = item.component_orders
            if (
                type(orders) is not tuple
                or any(type(order) is not int or order < 1 for order in orders)
                or tuple(sorted(orders)) != orders
                or sum(orders) != self.card_order
                or (self.card_order == 0 and orders != ())
                or (self.card_order > 0 and not orders)
            ):
                raise ValueError(
                    "component orders must partition the declared card order"
                )
            keys.append(orders)
            total += item.multiplicity
        if keys != sorted(set(keys)):
            raise ValueError("component profiles must be unique and ordered")
        if total != self.card_count:
            raise ValueError("profile multiplicities must sum to card_count")
        return self

    @classmethod
    def _from_kernel(
        cls,
        card_order: int,
        card_count: int,
        profiles: tuple[CardComponentSizeProfile, ...],
    ) -> Self:
        return cls.model_construct(
            card_order=card_order, card_count=card_count, profiles=profiles
        )


__all__ = [
    "MAX_COMPONENT_SIZE_PROFILES",
    "MAX_PROFILE_COUNT_DIGITS",
    "AnonymousDeckComponentProfile",
    "AnonymousDeckComponentProfileRequest",
    "CardComponentSizeProfile",
]
