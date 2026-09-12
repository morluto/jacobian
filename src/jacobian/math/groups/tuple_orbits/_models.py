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


def _preflight_action_dimensions(*, domain: object, generators: object) -> None:
    if isinstance(domain, (list, tuple)) and len(domain) > MAX_DOMAIN_SIZE:
        raise _tuple_error(
            "action_domain_bound",
            f"action domain admits at most {MAX_DOMAIN_SIZE} labels",
        )
    degree = len(domain) if isinstance(domain, (list, tuple)) else MAX_DOMAIN_SIZE
    if not isinstance(generators, (list, tuple)):
        return
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


def _preflight_action_payload(action: object) -> None:
    if isinstance(action, Mapping):
        _preflight_action_dimensions(
            domain=action.get("domain"), generators=action.get("generators")
        )
        return
    if action is None:
        return
    _preflight_action_dimensions(
        domain=getattr(action, "domain", None),
        generators=getattr(action, "generators", None),
    )


def _row_field(row: object, name: str) -> object:
    if isinstance(row, Mapping):
        return row.get(name)
    return getattr(row, name, None)


def _preflight_row_payload(row: object) -> None:
    representative = _row_field(row, "representative")
    if (
        isinstance(representative, (list, tuple))
        and len(representative) > MAX_TUPLE_ARITY
    ):
        raise _tuple_error(
            "arity_out_of_range",
            "tuple arity must be a non-negative action-domain-sized integer",
        )
    source_indices = _row_field(row, "source_indices")
    if (
        isinstance(source_indices, (list, tuple))
        and len(source_indices) > MAX_FAMILY_MEMBERS
    ):
        raise _tuple_error(
            "input_bound",
            f"at most {MAX_FAMILY_MEMBERS} tuple rows are admitted",
        )
    transporter = _row_field(row, "least_transporter")
    if isinstance(transporter, (list, tuple)) and len(transporter) > MAX_GROUP_DEGREE:
        raise _tuple_error(
            "transporter_axis",
            "transporters must be permutations of the action axis",
        )


def _preflight_source_payload(data: object) -> None:
    if isinstance(data, Mapping):
        action = data.get("action")
        family = data.get("family")
        raw_arity = data.get("arity")
    else:
        action = getattr(data, "action", None)
        family = getattr(data, "family", None)
        raw_arity = getattr(data, "arity", None)
    _preflight_action_payload(action)
    if not isinstance(family, (list, tuple)):
        return
    if len(family) > MAX_FAMILY_MEMBERS:
        raise _tuple_error(
            "input_bound",
            f"at most {MAX_FAMILY_MEMBERS} tuple rows are admitted",
        )
    arity_is_int = isinstance(raw_arity, int) and not isinstance(raw_arity, bool)
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
            _preflight_source_payload(data)
            return data
        _preflight_source_payload(data)
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
        if isinstance(data, Mapping):
            _preflight_row_payload(data)
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
        if isinstance(data, Mapping):
            rows = data.get("rows")
            if isinstance(rows, (list, tuple)):
                if len(rows) > MAX_FAMILY_MEMBERS:
                    raise _tuple_error(
                        "input_bound",
                        f"at most {MAX_FAMILY_MEMBERS} tuple rows are admitted",
                    )
                total_source_indices = 0
                for row in rows:
                    _preflight_row_payload(row)
                    source_indices = _row_field(row, "source_indices")
                    if isinstance(source_indices, (list, tuple)):
                        total_source_indices += len(source_indices)
                        if total_source_indices > MAX_FAMILY_MEMBERS:
                            raise _tuple_error(
                                "input_bound",
                                f"at most {MAX_FAMILY_MEMBERS} tuple rows are admitted",
                            )
            source = data.get("source")
            if source is not None:
                _preflight_source_payload(source)
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def bind_result_axes(self) -> Self:
        action = self.source.action
        degree = len(action.domain)
        family = self.source.family
        representatives = tuple(row.representative for row in self.rows)
        if representatives != tuple(sorted(set(representatives))):
            raise _tuple_error(
                "rows_not_canonical",
                "rows must be ordered by unique representatives",
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
