"""Typed contracts for cyclic prefix-sum operations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, StringConstraints, model_validator
from pydantic.json_schema import WithJsonSchema
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.combinatorics.additive.values import (
    IndexedIntegerSequence,
    indexed_sequence_item_ceiling,
)
from jacobian.math.groups.finite_abelian import (
    BoundedGroupElement,
    CanonicalGroupElement,
    FiniteAbelianProductGroup,
)

MAX_SEQUENCE_LENGTH = 10_000
MAX_MODULUS_DIGITS = 100
MAX_SEQUENCING_SOURCE_ITEMS = 8
MAX_SEQUENCING_GROUP_ORDER = 4_096
MAX_SEQUENCING_PERMUTATION_NODES = 109_601
MAX_SEQUENCING_FORBIDDEN_VALUES = 64


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"additive_combinatorics.{reason}", message)


BoundedModulus = Annotated[
    CanonicalInteger,
    StringConstraints(max_length=MAX_MODULUS_DIGITS, strict=True),
]


class CyclicPrefixSumResidueProfileRequest(StrictModel):
    """Request for the cyclic prefix-sum residue profile."""

    sequence: Annotated[
        IndexedIntegerSequence,
        WithJsonSchema(indexed_sequence_item_ceiling(MAX_SEQUENCE_LENGTH)),
    ]
    modulus: BoundedModulus


class PrefixSumResidueRow(StrictModel):
    """One row of the residue profile."""

    residue: CanonicalInteger
    positions: tuple[StrictInt, ...] = Field(min_length=1)


class CyclicPrefixSumResidueProfileResult(StrictModel):
    """The complete cyclic prefix-sum residue profile."""

    modulus: BoundedModulus
    rows: tuple[PrefixSumResidueRow, ...] = Field(max_length=MAX_SEQUENCE_LENGTH)

    @model_validator(mode="after")
    def require_sorted_unique_rows(self) -> Self:
        residues = tuple(int(row.residue) for row in self.rows)
        if residues != tuple(sorted(residues)) or len(set(residues)) != len(residues):
            raise ValueError("residue rows must be sorted and unique")
        positions = [position for row in self.rows for position in row.positions]
        if len(positions) > MAX_SEQUENCE_LENGTH:
            raise ValueError("residue rows contain too many prefix positions")
        if any(
            position < 1 or position > MAX_SEQUENCE_LENGTH for position in positions
        ):
            raise ValueError("prefix positions must be within the sequence bound")
        return self


class FiniteAbelianSequencingSource(StrictModel):
    """A distinct finite subset in one explicit product of cyclic groups.

    Coordinates are reduced on their declared axes and rows are sorted
    lexicographically. Distinctness is checked after reduction, so this value
    has one canonical residue-tuple representation.
    """

    group: FiniteAbelianProductGroup
    elements: tuple[BoundedGroupElement, ...] = Field(
        default=(),
        max_length=MAX_SEQUENCING_SOURCE_ITEMS,
        description=(
            "Distinct group elements as integer coordinate tuples. Each row "
            "is reduced modulo its cyclic factor; reduced rows must be "
            "distinct and sorted lexicographically."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def bound_raw_source(cls, value: object) -> object:
        value = canonicalize_json_containers(value)
        if not isinstance(value, Mapping):
            return value
        prepared: dict[str, object] = dict(value)
        raw_group = prepared.get("group")
        if isinstance(raw_group, Mapping):
            group: dict[str, object] = dict(raw_group)
            raw_moduli = group.get("moduli")
            if isinstance(raw_moduli, list):
                group["moduli"] = tuple(raw_moduli)
            prepared["group"] = group
        raw_elements = prepared.get("elements")
        if isinstance(raw_elements, list):
            prepared["elements"] = tuple(raw_elements)
        return prepared

    @model_validator(mode="after")
    def require_canonical_source(self) -> Self:
        rank = len(self.group.moduli)
        if len(self.elements) > MAX_SEQUENCING_SOURCE_ITEMS:
            raise _validation_error(
                "sequencing_source_cardinality",
                "sequencing source exceeds the "
                f"{MAX_SEQUENCING_SOURCE_ITEMS}-element bound",
            )
        if any(len(element) != rank for element in self.elements):
            raise _validation_error(
                "sequencing_source_rank",
                "every sequencing element must match the group rank",
            )
        normalized = [
            tuple(
                coordinate % modulus
                for coordinate, modulus in zip(element, self.group.moduli, strict=True)
            )
            for element in self.elements
        ]
        if normalized != sorted(normalized) or len(set(normalized)) != len(normalized):
            raise _validation_error(
                "sequencing_source_canonical",
                "reduced sequencing elements must be distinct and sorted",
            )
        if normalized:
            object.__setattr__(self, "elements", tuple(normalized))
        return self


class ForbiddenPrefixSequencingRequest(StrictModel):
    """Search one finite-Abelian cyclic ordering under forbidden prefixes.

    All proper prefix sums (positions ``1`` through ``|A|-1``) must be
    pairwise distinct, nonzero, and outside ``forbidden_values``. The terminal
    sum may be zero, following the standard cyclic-sequencing convention.
    The admitted exhaustive search returns FOUND, EXHAUSTED, or UNKNOWN.
    """

    source: FiniteAbelianSequencingSource
    first_element: BoundedGroupElement | None = Field(
        default=None,
        description=(
            "Optional prescribed first element, reduced in the source group; "
            "it must be one of the source's canonical elements when supplied."
        ),
    )
    forbidden_values: tuple[BoundedGroupElement, ...] = Field(
        default=(),
        max_length=MAX_SEQUENCING_FORBIDDEN_VALUES,
        description=(
            "Group elements excluded from every proper prefix sum. Rows are "
            "reduced, sorted, and distinct; the terminal sum is never tested, "
            "so forbidden zero does not reject a zero total."
        ),
    )
    search_node_limit: StrictInt = Field(
        default=MAX_SEQUENCING_PERMUTATION_NODES,
        ge=1,
        le=MAX_SEQUENCING_PERMUTATION_NODES,
        description=(
            "Maximum partial-ordering states visited in one deterministic "
            "pass. A stop at this limit returns UNKNOWN and never establishes "
            "nonexistence."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def bound_raw_request(cls, value: object) -> object:
        value = canonicalize_json_containers(value)
        if not isinstance(value, Mapping):
            return value
        prepared: dict[str, object] = dict(value)
        raw_first = prepared.get("first_element")
        if isinstance(raw_first, list):
            prepared["first_element"] = tuple(raw_first)
        raw_forbidden = prepared.get("forbidden_values")
        if isinstance(raw_forbidden, list):
            prepared["forbidden_values"] = tuple(raw_forbidden)
        return prepared

    @model_validator(mode="after")
    def require_canonical_request(self) -> Self:
        if self.first_element is not None:
            if len(self.first_element) != len(self.source.group.moduli):
                raise _validation_error(
                    "sequencing_first_element_rank",
                    "first_element must match the source group rank",
                )
            reduced_first = tuple(
                coordinate % modulus
                for coordinate, modulus in zip(
                    self.first_element, self.source.group.moduli, strict=True
                )
            )
            if reduced_first not in self.source.elements:
                raise _validation_error(
                    "sequencing_first_element_membership",
                    "first_element must reduce to a source element",
                )
            object.__setattr__(self, "first_element", reduced_first)

        rank = len(self.source.group.moduli)
        if any(len(value) != rank for value in self.forbidden_values):
            raise _validation_error(
                "sequencing_forbidden_rank",
                "every forbidden value must match the source group rank",
            )
        normalized_forbidden = [
            tuple(
                coordinate % modulus
                for coordinate, modulus in zip(
                    value, self.source.group.moduli, strict=True
                )
            )
            for value in self.forbidden_values
        ]
        canonical_forbidden = tuple(sorted(normalized_forbidden))
        if len(set(canonical_forbidden)) != len(canonical_forbidden):
            raise _validation_error(
                "sequencing_forbidden_canonical",
                "reduced forbidden values must be distinct",
            )
        object.__setattr__(self, "forbidden_values", canonical_forbidden)
        return self


class SequencingPrefixSum(StrictModel):
    """One source index, its reduced element, and its proper prefix sum."""

    source_index: StrictInt = Field(ge=0, lt=MAX_SEQUENCING_SOURCE_ITEMS)
    element: CanonicalGroupElement
    prefix_sum: CanonicalGroupElement


class ForbiddenPrefixSequencingResult(StrictModel):
    """One sequencing witness, exact exhaustion, or a non-conclusion.

    ``FOUND`` is the only status carrying an ordering. ``EXHAUSTED`` is
    produced only after every admitted ordering was searched. ``UNKNOWN``
    means the node budget stopped before a mathematical conclusion; it never
    establishes nonexistence.
    """

    source: FiniteAbelianSequencingSource
    first_element: CanonicalGroupElement | None
    forbidden_values: tuple[CanonicalGroupElement, ...]
    search_node_limit: StrictInt = Field(ge=1, le=MAX_SEQUENCING_PERMUTATION_NODES)
    status: Literal["FOUND", "EXHAUSTED", "UNKNOWN"]
    ordering: tuple[SequencingPrefixSum, ...] | None = Field(
        default=None,
        max_length=MAX_SEQUENCING_SOURCE_ITEMS,
    )
    states_explored: StrictInt = Field(ge=0, le=MAX_SEQUENCING_PERMUTATION_NODES)

    @model_validator(mode="before")
    @classmethod
    def bound_raw_result(cls, value: object) -> object:
        value = canonicalize_json_containers(value)
        if not isinstance(value, Mapping):
            return value
        prepared: dict[str, object] = dict(value)
        raw_first = prepared.get("first_element")
        if isinstance(raw_first, list):
            prepared["first_element"] = tuple(raw_first)
        raw_forbidden = prepared.get("forbidden_values")
        if isinstance(raw_forbidden, list):
            prepared["forbidden_values"] = tuple(raw_forbidden)
        raw_ordering = prepared.get("ordering")
        if isinstance(raw_ordering, list):
            prepared["ordering"] = tuple(raw_ordering)
        return prepared

    @model_validator(mode="after")
    def require_result_branch_shape(self) -> Self:
        if self.status == "FOUND":
            if self.ordering is None or len(self.ordering) != len(self.source.elements):
                raise _validation_error(
                    "sequencing_found_shape",
                    "a FOUND result must contain one row per source element",
                )
            if sorted(row.source_index for row in self.ordering) != list(
                range(len(self.source.elements))
            ):
                raise _validation_error(
                    "sequencing_found_indices",
                    "a FOUND ordering must be a source-index permutation",
                )
            if any(
                row.element != self.source.elements[row.source_index]
                for row in self.ordering
            ):
                raise _validation_error(
                    "sequencing_found_elements",
                    "each ordering row element must equal its indexed source element",
                )
        elif self.ordering is not None:
            raise _validation_error(
                "sequencing_nonfound_shape",
                "only a FOUND result may carry an ordering",
            )
        return self


__all__ = [
    "MAX_MODULUS_DIGITS",
    "MAX_SEQUENCE_LENGTH",
    "MAX_SEQUENCING_FORBIDDEN_VALUES",
    "MAX_SEQUENCING_GROUP_ORDER",
    "MAX_SEQUENCING_PERMUTATION_NODES",
    "MAX_SEQUENCING_SOURCE_ITEMS",
    "BoundedModulus",
    "CyclicPrefixSumResidueProfileRequest",
    "CyclicPrefixSumResidueProfileResult",
    "FiniteAbelianSequencingSource",
    "ForbiddenPrefixSequencingRequest",
    "ForbiddenPrefixSequencingResult",
    "PrefixSumResidueRow",
    "SequencingPrefixSum",
]
