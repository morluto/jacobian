"""Typed wire contracts for the relational homomorphism check."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.logic.relational_structures.values import (
    MAX_RELATIONAL_ARITY,
    MAX_RELATIONAL_CARRIER,
    FiniteRelationalStructure,
    RelationSymbolId,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"relational.homomorphism.{reason}", message)


class HomomorphismStatus(StrEnum):
    """Closed outcome of one exhaustive preservation replay."""

    HOMOMORPHISM = "HOMOMORPHISM"
    NOT_HOMOMORPHISM = "NOT_HOMOMORPHISM"


class HomomorphismViolationWitness(StrictModel):
    """The first source tuple whose image is absent from the target relation.

    Witnesses are found in deterministic replay order: signature order of the
    relation symbols, then increasing row order of each complete source
    table. The witness is exact nonmembership data, not a search artifact.
    """

    symbol_id: RelationSymbolId
    arity: StrictInt = Field(ge=0, le=MAX_RELATIONAL_ARITY)
    source_tuple: tuple[StrictInt, ...] = Field(
        description="The violating tuple t in R^A, in source carrier labels."
    )
    image_tuple: tuple[StrictInt, ...] = Field(
        description=(
            "The coordinatewise image h(t), which is exactly NOT in the "
            "target relation R^B."
        )
    )

    @model_validator(mode="after")
    def require_witness_shape(self) -> Self:
        if len(self.source_tuple) != self.arity or len(self.image_tuple) != self.arity:
            raise _validation_error(
                "witness_shape",
                "both witness tuples must have exactly the declared arity",
            )
        return self


class SymbolTransportProfile(StrictModel):
    """Complete per-symbol transport counts of one exhaustive replay."""

    symbol_id: RelationSymbolId
    arity: StrictInt = Field(ge=0, le=MAX_RELATIONAL_ARITY)
    source_tuples: StrictInt = Field(
        ge=0, description="The complete cardinality |R^A| of the source table."
    )
    preserved_tuples: StrictInt = Field(
        ge=0,
        description=(
            "The number of source tuples whose image lies in the target "
            "relation. Equal to source_tuples exactly when this symbol is "
            "preserved; the replay is exhaustive, never sampled."
        ),
    )

    @model_validator(mode="after")
    def require_bounded_counts(self) -> Self:
        if self.preserved_tuples > self.source_tuples:
            raise _validation_error(
                "transport_count",
                "preserved tuples cannot exceed the complete source table",
            )
        return self


class HomomorphismCheckResult(StrictModel):
    """A source-bound exhaustive homomorphism decision.

    The result retains both structures and the complete candidate map, the
    closed status, the first violating witness when the map fails, and the
    complete per-symbol transport counts. A ``NOT_HOMOMORPHISM`` status is a
    mathematical negative about this one map; it is never a claim that no
    homomorphism exists — bounded homomorphism search is deferred.
    """

    status: HomomorphismStatus
    source: FiniteRelationalStructure
    target: FiniteRelationalStructure
    carrier_map: tuple[StrictInt, ...] = Field(
        max_length=MAX_RELATIONAL_CARRIER,
        description=(
            "The checked total function h: source carrier -> target carrier, "
            "as one target label per source label in increasing source order."
        ),
    )
    witness: HomomorphismViolationWitness | None = None
    symbol_profiles: tuple[SymbolTransportProfile, ...]
    preservation_invariant: Literal[
        "EVERY_SOURCE_RELATION_TUPLE_TRANSPORTS_INTO_THE_TARGET_RELATION"
    ] = "EVERY_SOURCE_RELATION_TUPLE_TRANSPORTS_INTO_THE_TARGET_RELATION"

    @model_validator(mode="after")
    def require_structural_consistency(self) -> Self:
        if self.source.signature != self.target.signature:
            raise _validation_error(
                "signature_mismatch",
                "source and target structures must share one signature",
            )
        if len(self.carrier_map) != self.source.carrier_size:
            raise _validation_error(
                "carrier_map_axis",
                "the retained carrier map must be total on the source carrier",
            )
        if any(not 0 <= image < self.target.carrier_size for image in self.carrier_map):
            raise _validation_error(
                "carrier_map_value",
                "every retained carrier map image must lie in the target carrier",
            )
        expected = tuple(
            (symbol.symbol_id, symbol.arity) for symbol in self.source.signature
        )
        if (
            tuple(
                (profile.symbol_id, profile.arity) for profile in self.symbol_profiles
            )
            != expected
        ):
            raise _validation_error(
                "symbol_profile_axis",
                "symbol transport profiles must cover the shared signature "
                "exactly once, in signature order",
            )
        failed = self.status is HomomorphismStatus.NOT_HOMOMORPHISM
        if (self.witness is None) == failed:
            raise _validation_error(
                "status_witness_binding",
                "a NOT_HOMOMORPHISM result must retain its first violating "
                "witness and a HOMOMORPHISM result must retain none",
            )
        if self.witness is not None and not any(
            profile.symbol_id == self.witness.symbol_id
            and profile.arity == self.witness.arity
            and profile.preserved_tuples < profile.source_tuples
            for profile in self.symbol_profiles
        ):
            raise _validation_error(
                "witness_symbol_binding",
                "the witness symbol must show a strict transport deficit in "
                "the retained per-symbol counts",
            )
        if not failed and any(
            profile.preserved_tuples != profile.source_tuples
            for profile in self.symbol_profiles
        ):
            raise _validation_error(
                "transport_identity",
                "a homomorphism preserves every source tuple of every symbol",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        status: HomomorphismStatus,
        source: FiniteRelationalStructure,
        target: FiniteRelationalStructure,
        carrier_map: tuple[int, ...],
        witness: HomomorphismViolationWitness | None,
        symbol_profiles: tuple[SymbolTransportProfile, ...],
    ) -> Self:
        """Build the result after the admitted kernel replayed the invariant.

        The exhaustive preservation replay ran in the kernel; result
        construction does not transport any tuple again.
        """

        return cls.model_construct(
            status=status,
            source=source,
            target=target,
            carrier_map=carrier_map,
            witness=witness,
            symbol_profiles=symbol_profiles,
            preservation_invariant=(
                "EVERY_SOURCE_RELATION_TUPLE_TRANSPORTS_INTO_THE_TARGET_RELATION"
            ),
        )


class HomomorphismCheckRequest(StrictModel):
    """Check one candidate carrier map between two same-signature structures.

    The defining invariant ``t in R^A => h(t) in R^B`` is replayed
    exhaustively over every symbol and every source tuple. A malformed or
    incomplete carrier map and a signature mismatch are boundary-invalid;
    a total map violating one relation is the mathematical negative
    ``NOT_HOMOMORPHISM`` with its first witness.
    """

    source: FiniteRelationalStructure
    target: FiniteRelationalStructure
    carrier_map: tuple[StrictInt, ...] = Field(
        max_length=MAX_RELATIONAL_CARRIER,
        description=(
            "One target carrier label per source carrier label, in increasing "
            "source label order; exactly source.carrier_size entries, each in "
            "0..target.carrier_size-1."
        ),
    )

    @model_validator(mode="after")
    def require_shared_signature_and_total_map(self) -> Self:
        if self.source.signature != self.target.signature:
            raise _validation_error(
                "signature_mismatch",
                "source and target structures must be declared over one "
                "shared signature",
            )
        if len(self.carrier_map) != self.source.carrier_size:
            raise _validation_error(
                "carrier_map_axis",
                "a candidate homomorphism must be a total function on the "
                "complete source carrier",
            )
        if any(not 0 <= image < self.target.carrier_size for image in self.carrier_map):
            raise _validation_error(
                "carrier_map_value",
                "every carrier map image must belong to the target carrier",
            )
        return self


__all__ = [
    "HomomorphismCheckRequest",
    "HomomorphismCheckResult",
    "HomomorphismStatus",
    "HomomorphismViolationWitness",
    "SymbolTransportProfile",
]
