"""Typed diagonal-action tuple-family orbit profiles.

Tuple coordinates are positions on one finite permutation action. Keeping the
action with the source family makes coordinate axes explicit across a
serialized producer/consumer boundary; tuple coordinate order is never
normalized.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.groups._models import MAX_GROUP_DEGREE
from jacobian.math.groups.actions._models import (
    MAX_DOMAIN_SIZE,
    MAX_FAMILY_MEMBERS,
    MAX_GENERATORS,
    MAX_GROUP_ORDER,
    FinitePermutationAction,
)

MAX_TUPLE_ARITY = MAX_GROUP_DEGREE


def _tuple_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"finite_group_action.tuple_family_{reason}", message)


class TupleFamilyOrbitSource(StrictModel):
    """One explicit family of tuples on a finite permutation action.

    ``family`` is source ordered and preserves repeated source rows for
    compatibility with materialized research families. Repetition *within* a
    tuple is meaningful, so ``(a, b)`` and ``(b, a)`` remain distinct and
    ``(a, a)`` is valid.
    """

    action: FinitePermutationAction
    arity: StrictInt = Field(
        ge=0,
        le=MAX_TUPLE_ARITY,
        description=(
            "Fixed tuple arity. Coordinates retain this order, including "
            "repeated positions; arity zero is the singleton empty tuple."
        ),
    )
    family: tuple[tuple[StrictInt, ...], ...] = Field(
        max_length=MAX_FAMILY_MEMBERS,
        description=(
            "Source-ordered tuple rows on action.domain. Rows may repeat; "
            "the action orbit profile retains every source index."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_json_containers(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        action = data.get("action")
        if isinstance(action, Mapping):
            domain = action.get("domain")
            generators = action.get("generators")
            if isinstance(domain, (list, tuple)) and len(domain) > MAX_DOMAIN_SIZE:
                raise _tuple_error(
                    "action_domain_bound",
                    f"action domain admits at most {MAX_DOMAIN_SIZE} labels",
                )
            degree = (
                len(domain) if isinstance(domain, (list, tuple)) else MAX_DOMAIN_SIZE
            )
            if isinstance(generators, (list, tuple)):
                if len(generators) > MAX_GENERATORS:
                    raise _tuple_error(
                        "action_generator_bound",
                        f"actions admit at most {MAX_GENERATORS} generators",
                    )
                for generator in generators:
                    if isinstance(generator, (list, tuple)) and len(generator) > degree:
                        raise _tuple_error(
                            "generator_length_mismatch",
                            "every generator must be a permutation of the domain",
                        )
        family = data.get("family")
        if isinstance(family, (list, tuple)):
            if len(family) > MAX_FAMILY_MEMBERS:
                raise _tuple_error(
                    "input_bound",
                    f"at most {MAX_FAMILY_MEMBERS} tuple rows are admitted",
                )
            raw_arity = data.get("arity")
            arity_is_int = isinstance(raw_arity, int) and not isinstance(
                raw_arity, bool
            )
            for member in family:
                if not isinstance(member, (list, tuple)):
                    continue
                if arity_is_int and len(member) != raw_arity:
                    raise _tuple_error(
                        "arity_mismatch",
                        "every family member must have the declared arity",
                    )
                if len(member) > MAX_TUPLE_ARITY:
                    raise _tuple_error(
                        "arity_out_of_range",
                        "tuple arity must be a non-negative action-domain-sized integer",
                    )
        return canonicalize_json_containers(dict(data))

    @model_validator(mode="after")
    def bind_family_axis(self) -> Self:
        degree = len(self.action.domain)
        if any(len(member) != self.arity for member in self.family):
            raise _tuple_error(
                "arity_mismatch", "every family member must have the declared arity"
            )
        if any(
            not 0 <= coordinate < degree
            for member in self.family
            for coordinate in member
        ):
            raise _tuple_error(
                "coordinate_out_of_range",
                "tuple coordinates must lie on the action domain axis",
            )
        return self


class TupleOrbitRow(StrictModel):
    """One ambient orbit represented in the supplied family.

    ``least_transporter`` maps the first source row listed in
    ``source_indices`` to ``representative``. The result owner checks that
    source-to-codomain map structurally without replaying group enumeration.
    """

    representative: tuple[StrictInt, ...] = Field(
        max_length=MAX_TUPLE_ARITY,
        description="Lexicographically least ambient tuple in this orbit.",
    )
    source_indices: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_FAMILY_MEMBERS,
        description="Increasing indices of all supplied rows in this orbit.",
    )
    orbit_size: ExactInteger = Field(
        ge=1,
        le=MAX_GROUP_ORDER,
        description="Full ambient orbit cardinality, not only supplied rows.",
    )
    stabilizer_size: ExactInteger = Field(
        ge=1,
        le=MAX_GROUP_ORDER,
        description="Ambient stabilizer cardinality of the representative.",
    )
    least_transporter: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_GROUP_DEGREE,
        description=(
            "Lexicographically least permutation map from the first supplied "
            "source tuple to the representative."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_json_containers(cls, data: Any) -> Any:
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def bind_row_shape(self) -> Self:
        if tuple(sorted(set(self.source_indices))) != self.source_indices:
            raise _tuple_error(
                "source_indices_not_canonical",
                "source indices must be distinct and increasing",
            )
        return self


class TupleFamilyOrbitResult(StrictModel):
    """Exact source-indexed diagonal-action orbit profile.

    Result decoding checks only source axes, canonical ordering, permutation
    map shape, and partition structure. Group membership, orbit generation,
    and completeness are established once by the producing operation; a
    consumer that relies on an authored profile must run its own operation
    admission rather than relying on result parsing.
    """

    source: TupleFamilyOrbitSource
    rows: tuple[TupleOrbitRow, ...] = Field(max_length=MAX_FAMILY_MEMBERS)
    is_union_of_complete_ambient_orbits: bool

    @model_validator(mode="before")
    @classmethod
    def normalize_json_containers(cls, data: Any) -> Any:
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def bind_result_axes(self) -> Self:
        action = self.source.action
        degree = len(action.domain)
        family = self.source.family
        if self.rows != tuple(sorted(self.rows, key=lambda row: row.representative)):
            raise _tuple_error(
                "rows_not_canonical", "rows must be ordered by representative"
            )
        covered: list[int] = []
        for row in self.rows:
            if len(row.representative) != self.source.arity or any(
                not 0 <= coordinate < degree for coordinate in row.representative
            ):
                raise _tuple_error(
                    "representative_axis",
                    "representatives must use the source arity and action axis",
                )
            if any(index < 0 or index >= len(family) for index in row.source_indices):
                raise _tuple_error(
                    "source_index_out_of_range",
                    "source indices must index the retained family",
                )
            transporter = row.least_transporter
            if len(transporter) != degree or sorted(transporter) != list(range(degree)):
                raise _tuple_error(
                    "transporter_axis",
                    "transporters must be permutations of the action axis",
                )
            first_source = family[row.source_indices[0]]
            if (
                tuple(transporter[position] for position in first_source)
                != row.representative
            ):
                raise _tuple_error(
                    "transporter_mismatch",
                    "the transporter must map the first source tuple to the representative",
                )
            covered.extend(row.source_indices)
        if tuple(sorted(covered)) != tuple(range(len(family))):
            raise _tuple_error(
                "source_partition",
                "rows must partition every source index exactly once",
            )
        return self


__all__ = [
    "TupleFamilyOrbitResult",
    "TupleFamilyOrbitSource",
    "TupleOrbitRow",
]
