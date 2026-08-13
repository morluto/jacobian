"""Exact finite factorization in products of cyclic groups."""

from __future__ import annotations

from collections import Counter
from itertools import product
from math import prod
from typing import Literal, Self

from pydantic import Field, StrictBool, StrictInt, model_validator

from jacobian.contracts.base import ContractModel

MAX_FINITE_GROUP_ORDER = 4_096
MAX_FINITE_GROUP_RANK = 6
MAX_FINITE_GROUP_FACTOR_SIZE = 256
MAX_FINITE_GROUP_MODULUS = 1_000_000
MAX_FINITE_GROUP_COORDINATE = 1_000_000

GroupElement = tuple[StrictInt, ...]


class FiniteAbelianGroupFactorizationRequest(ContractModel):
    """Two bounded integer-vector factors in a product of cyclic groups."""

    moduli: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_GROUP_RANK,
    )
    left: tuple[GroupElement, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_GROUP_FACTOR_SIZE,
    )
    right: tuple[GroupElement, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_GROUP_FACTOR_SIZE,
    )

    @model_validator(mode="after")
    def require_bounded_product_group(self) -> Self:
        if any(
            modulus < 2 or modulus > MAX_FINITE_GROUP_MODULUS for modulus in self.moduli
        ):
            raise ValueError("cyclic moduli must be between 2 and 1,000,000")
        if prod(self.moduli) > MAX_FINITE_GROUP_ORDER:
            raise ValueError("finite abelian group exceeds the 4,096-element bound")
        if len(self.left) * len(self.right) > MAX_FINITE_GROUP_ORDER:
            raise ValueError("factor Cartesian product exceeds the 4,096-pair bound")
        if any(
            len(element) != len(self.moduli)
            for factor in (self.left, self.right)
            for element in factor
        ):
            raise ValueError("every factor element must match the group rank")
        if any(
            abs(coordinate) > MAX_FINITE_GROUP_COORDINATE
            for factor in (self.left, self.right)
            for element in factor
            for coordinate in element
        ):
            raise ValueError("factor coordinates exceed the input bound")
        for factor in (self.left, self.right):
            normalized = {
                tuple(
                    coordinate % modulus
                    for coordinate, modulus in zip(element, self.moduli, strict=True)
                )
                for element in factor
            }
            if len(normalized) != len(factor):
                raise ValueError("factor elements must be distinct after normalization")
        return self


class FiniteAbelianRepresentationCount(ContractModel):
    """Number of group elements having one representation count."""

    representation_count: StrictInt = Field(ge=0, le=MAX_FINITE_GROUP_ORDER)
    element_count: StrictInt = Field(ge=1, le=MAX_FINITE_GROUP_ORDER)


class FiniteAbelianRepresentationWitness(ContractModel):
    """The first element with two distinct displayed representations."""

    element: GroupElement = Field(min_length=1, max_length=MAX_FINITE_GROUP_RANK)
    left: GroupElement = Field(min_length=1, max_length=MAX_FINITE_GROUP_RANK)
    right: GroupElement = Field(min_length=1, max_length=MAX_FINITE_GROUP_RANK)
    other_left: GroupElement = Field(min_length=1, max_length=MAX_FINITE_GROUP_RANK)
    other_right: GroupElement = Field(min_length=1, max_length=MAX_FINITE_GROUP_RANK)


class FiniteAbelianGroupFactorizationResult(ContractModel):
    """Complete unique-representation summary for ``G = left + right``."""

    moduli: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_GROUP_RANK,
    )
    normalized_left: tuple[GroupElement, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_GROUP_FACTOR_SIZE,
    )
    normalized_right: tuple[GroupElement, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_GROUP_FACTOR_SIZE,
    )
    group_order: StrictInt = Field(ge=2, le=MAX_FINITE_GROUP_ORDER)
    pair_count: StrictInt = Field(ge=1, le=MAX_FINITE_GROUP_ORDER)
    distinct_sum_count: StrictInt = Field(ge=1, le=MAX_FINITE_GROUP_ORDER)
    representation_histogram: tuple[FiniteAbelianRepresentationCount, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_GROUP_ORDER,
    )
    is_exact_factorization: StrictBool
    first_missing: GroupElement | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_FINITE_GROUP_RANK,
    )
    first_duplicate: FiniteAbelianRepresentationWitness | None = None
    convention: Literal["UNIQUE_SUM_REPRESENTATION_IN_PRODUCT_OF_CYCLIC_GROUPS"] = (
        "UNIQUE_SUM_REPRESENTATION_IN_PRODUCT_OF_CYCLIC_GROUPS"
    )

    @model_validator(mode="after")
    def bind_structural_summary(self) -> Self:
        self._validate_group_structure()
        self._validate_factor_sets()
        self._validate_histogram()
        self._validate_decision_witness_presence()
        self._validate_witnesses()
        return self

    def _validate_group_structure(self) -> None:
        if self.group_order != prod(self.moduli):
            raise ValueError("group order must equal the product of cyclic moduli")

    def _validate_factor_sets(self) -> None:
        factors = (self.normalized_left, self.normalized_right)
        for factor in factors:
            if factor != tuple(sorted(set(factor))):
                raise ValueError("normalized factors must be unique and sorted")
            if any(len(element) != len(self.moduli) for element in factor):
                raise ValueError("normalized factor elements must match the group rank")
            if any(
                coordinate < 0 or coordinate >= modulus
                for element in factor
                for coordinate, modulus in zip(element, self.moduli, strict=True)
            ):
                raise ValueError("normalized factor coordinates must be residues")
        if self.pair_count != len(self.normalized_left) * len(self.normalized_right):
            raise ValueError("pair count must equal the factor product size")

    def _validate_histogram(self) -> None:
        counts = tuple(
            item.representation_count for item in self.representation_histogram
        )
        if counts != tuple(sorted(set(counts))):
            raise ValueError("histogram representation counts must be increasing")
        if (
            sum(item.element_count for item in self.representation_histogram)
            != self.group_order
        ):
            raise ValueError("representation histogram must cover the group")
        if (
            sum(
                item.representation_count * item.element_count
                for item in self.representation_histogram
            )
            != self.pair_count
        ):
            raise ValueError("representation histogram must cover every factor pair")
        positive_count = sum(
            item.element_count
            for item in self.representation_histogram
            if item.representation_count > 0
        )
        if positive_count != self.distinct_sum_count:
            raise ValueError("distinct sum count must match the histogram")

    def _validate_decision_witness_presence(self) -> None:
        exact = (
            self.pair_count == self.group_order
            and self.representation_histogram
            == (
                FiniteAbelianRepresentationCount(
                    representation_count=1,
                    element_count=self.group_order,
                ),
            )
        )
        if self.is_exact_factorization != exact:
            raise ValueError("factorization decision must match the complete histogram")
        has_missing = any(
            item.representation_count == 0 for item in self.representation_histogram
        )
        has_duplicate = any(
            item.representation_count > 1 for item in self.representation_histogram
        )
        if (self.first_missing is not None) != has_missing:
            raise ValueError("missing witness presence must match the histogram")
        if (self.first_duplicate is not None) != has_duplicate:
            raise ValueError("duplicate witness presence must match the histogram")
        if self.is_exact_factorization and (
            self.first_missing is not None or self.first_duplicate is not None
        ):
            raise ValueError("exact factorizations cannot carry failure witnesses")

    def _validate_witnesses(self) -> None:
        def canonical(element: tuple[int, ...]) -> bool:
            return len(element) == len(self.moduli) and all(
                0 <= coordinate < modulus
                for coordinate, modulus in zip(element, self.moduli, strict=True)
            )

        if self.first_missing is not None and not canonical(self.first_missing):
            raise ValueError("missing witness must be a canonical group element")
        duplicate = self.first_duplicate
        if duplicate is None:
            return
        elements = (
            duplicate.element,
            duplicate.left,
            duplicate.right,
            duplicate.other_left,
            duplicate.other_right,
        )
        if not all(canonical(element) for element in elements):
            raise ValueError("duplicate witness elements must be canonical")
        first_pair = (duplicate.left, duplicate.right)
        second_pair = (duplicate.other_left, duplicate.other_right)
        if first_pair == second_pair:
            raise ValueError("duplicate witness representations must be distinct")
        for left, right in (first_pair, second_pair):
            total = tuple(
                (left_coordinate + right_coordinate) % modulus
                for left_coordinate, right_coordinate, modulus in zip(
                    left, right, self.moduli, strict=True
                )
            )
            if total != duplicate.element:
                raise ValueError("duplicate witness representations must sum correctly")


def finite_abelian_group_factorization(
    request: FiniteAbelianGroupFactorizationRequest,
) -> FiniteAbelianGroupFactorizationResult:
    """Exhaustively test unique representation in a product of cyclic groups."""

    moduli = request.moduli

    def normalize(element: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            coordinate % modulus
            for coordinate, modulus in zip(element, moduli, strict=True)
        )

    left = tuple(sorted(normalize(element) for element in request.left))
    right = tuple(sorted(normalize(element) for element in request.right))
    representations: dict[
        tuple[int, ...], list[tuple[tuple[int, ...], tuple[int, ...]]]
    ] = {}
    for left_element in left:
        for right_element in right:
            total = tuple(
                (left_coordinate + right_coordinate) % modulus
                for left_coordinate, right_coordinate, modulus in zip(
                    left_element, right_element, moduli, strict=True
                )
            )
            representations.setdefault(total, []).append((left_element, right_element))
    group = tuple(product(*(range(modulus) for modulus in moduli)))
    histogram = Counter(len(representations.get(element, ())) for element in group)
    first_missing = next(
        (element for element in group if element not in representations),
        None,
    )
    duplicate_element = next(
        (element for element in group if len(representations.get(element, ())) > 1),
        None,
    )
    duplicate = None
    if duplicate_element is not None:
        first, second = representations[duplicate_element][:2]
        duplicate = FiniteAbelianRepresentationWitness(
            element=duplicate_element,
            left=first[0],
            right=first[1],
            other_left=second[0],
            other_right=second[1],
        )
    group_order = prod(moduli)
    exact = len(left) * len(right) == group_order and histogram == {1: group_order}
    return FiniteAbelianGroupFactorizationResult(
        moduli=moduli,
        normalized_left=left,
        normalized_right=right,
        group_order=group_order,
        pair_count=len(left) * len(right),
        distinct_sum_count=len(representations),
        representation_histogram=tuple(
            FiniteAbelianRepresentationCount(
                representation_count=count,
                element_count=histogram[count],
            )
            for count in sorted(histogram)
        ),
        is_exact_factorization=exact,
        first_missing=None if exact else first_missing,
        first_duplicate=None if exact else duplicate,
    )


__all__ = [
    "FiniteAbelianGroupFactorizationRequest",
    "FiniteAbelianGroupFactorizationResult",
    "FiniteAbelianRepresentationCount",
    "FiniteAbelianRepresentationWitness",
    "finite_abelian_group_factorization",
]
