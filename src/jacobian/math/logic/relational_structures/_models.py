"""Typed wire contracts for the relational homomorphism check."""

from __future__ import annotations

from enum import StrEnum
from itertools import product
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
    mathematical negative about this one map; existence of some other
    homomorphism is decided by the bounded search operation.
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
    witness_kind: Literal["PRESERVATION", "REFLECTION"] = "PRESERVATION"
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
        if (
            self.witness is not None
            and self.witness_kind == "PRESERVATION"
            and not any(
                profile.symbol_id == self.witness.symbol_id
                and profile.arity == self.witness.arity
                and profile.preserved_tuples < profile.source_tuples
                for profile in self.symbol_profiles
            )
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
        witness_kind: Literal["PRESERVATION", "REFLECTION"] = "PRESERVATION",
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
            witness_kind=witness_kind,
            symbol_profiles=symbol_profiles,
            preservation_invariant=(
                "EVERY_SOURCE_RELATION_TUPLE_TRANSPORTS_INTO_THE_TARGET_RELATION"
            ),
        )


class InducedRelationProfile(StrictModel):
    symbol_id: RelationSymbolId
    arity: StrictInt = Field(ge=0, le=MAX_RELATIONAL_ARITY)
    relation_cells: StrictInt = Field(ge=0)
    matching_cells: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def require_bounded_counts(self) -> Self:
        if self.matching_cells > self.relation_cells:
            raise _validation_error(
                "reflection_count",
                "matching cells cannot exceed relation domain",
            )
        return self


class InducedEmbeddingCheckResult(HomomorphismCheckResult):
    reflection_profiles: tuple[InducedRelationProfile, ...]
    induced_invariant: Literal[
        "EVERY_SOURCE_RELATION_TUPLE_REFLECTS_INTO_THE_TARGET"
    ] = "EVERY_SOURCE_RELATION_TUPLE_REFLECTS_INTO_THE_TARGET"

    @model_validator(mode="after")
    def require_injective_carrier_map(self) -> Self:
        if len(set(self.carrier_map)) != len(self.carrier_map):
            raise _validation_error(
                "carrier_map_not_injective",
                "an induced embedding check requires distinct target images",
            )
        return self

    @model_validator(mode="after")
    def require_reflection_invariant(self) -> Self:
        expected = tuple((s.symbol_id, s.arity) for s in self.source.signature)
        if tuple((p.symbol_id, p.arity) for p in self.reflection_profiles) != expected:
            raise _validation_error(
                "reflection_profile_axis",
                "reflection profiles must cover the signature",
            )
        for profile in self.reflection_profiles:
            cells = 1 if profile.arity == 0 else self.source.carrier_size**profile.arity
            if profile.relation_cells != cells:
                raise _validation_error(
                    "reflection_domain",
                    "reflection profile must cover every Cartesian cell",
                )
        if self.status is HomomorphismStatus.HOMOMORPHISM and any(
            p.matching_cells != p.relation_cells for p in self.reflection_profiles
        ):
            raise _validation_error(
                "reflection_invariant",
                "an induced embedding must reflect every relation cell",
            )
        target_tables = tuple(set(table) for table in self.target.relation_tables)
        source_tables = tuple(set(table) for table in self.source.relation_tables)
        for symbol_index, profile in enumerate(self.reflection_profiles):
            matching = 0
            for coordinates in product(
                range(self.source.carrier_size), repeat=profile.arity
            ):
                image = tuple(self.carrier_map[index] for index in coordinates)
                if (coordinates in source_tables[symbol_index]) == (
                    image in target_tables[symbol_index]
                ):
                    matching += 1
            if matching != profile.matching_cells:
                raise _validation_error(
                    "reflection_truth",
                    "reflection profiles must match the retained structures and map",
                )
        return self

    @classmethod
    def _from_induced_kernel(
        cls,
        *,
        status: HomomorphismStatus,
        source: FiniteRelationalStructure,
        target: FiniteRelationalStructure,
        carrier_map: tuple[int, ...],
        witness: HomomorphismViolationWitness | None,
        symbol_profiles: tuple[SymbolTransportProfile, ...],
        reflection_profiles: tuple[InducedRelationProfile, ...],
        witness_kind: Literal["PRESERVATION", "REFLECTION"] = "PRESERVATION",
    ) -> Self:
        return cls.model_construct(
            status=status,
            source=source,
            target=target,
            carrier_map=carrier_map,
            witness=witness,
            witness_kind=witness_kind,
            symbol_profiles=symbol_profiles,
            preservation_invariant=(
                "EVERY_SOURCE_RELATION_TUPLE_TRANSPORTS_INTO_THE_TARGET_RELATION"
            ),
            reflection_profiles=reflection_profiles,
            induced_invariant="EVERY_SOURCE_RELATION_TUPLE_REFLECTS_INTO_THE_TARGET",
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


class HomomorphismSearchStatus(StrEnum):
    """Closed outcome of one exhaustive homomorphism search."""

    FOUND = "FOUND"
    EXHAUSTED = "EXHAUSTED"


class HomomorphismSearchResult(StrictModel):
    """A source-bound exhaustive homomorphism search outcome.

    ``FOUND`` retains the first homomorphism in lexicographic carrier-map
    order as a complete ``HomomorphismCheckResult``; ``EXHAUSTED`` retains
    the receipt that every one of the ``total_candidates`` carrier maps
    was examined and none transports. An exhausted search is a proved
    mathematical negative about homomorphism existence inside the
    admitted envelope, never a truncated or unknown outcome.
    """

    status: HomomorphismSearchStatus
    source: FiniteRelationalStructure
    target: FiniteRelationalStructure
    check: HomomorphismCheckResult | None = None
    candidates_examined: StrictInt = Field(
        ge=0,
        description="Carrier maps replayed before the outcome was reached.",
    )
    total_candidates: StrictInt = Field(
        ge=0,
        description="The complete |B|^|A| carrier-map count (1 for the empty "
        "source, 0 for a nonempty source into the empty carrier).",
    )

    @model_validator(mode="after")
    def require_search_consistency(self) -> Self:
        source_size = self.source.carrier_size
        target_size = self.target.carrier_size
        expected_total = (
            1
            if source_size == 0
            else (0 if target_size == 0 else target_size**source_size)
        )
        if self.total_candidates != expected_total:
            raise _validation_error(
                "search_receipt",
                "total_candidates must be the complete carrier-map count",
            )
        if not 0 <= self.candidates_examined <= self.total_candidates:
            raise _validation_error(
                "search_receipt",
                "candidates_examined must lie within the complete space",
            )
        if self.status is HomomorphismSearchStatus.FOUND:
            if (
                self.check is None
                or self.check.status is not HomomorphismStatus.HOMOMORPHISM
                or self.check.source != self.source
                or self.check.target != self.target
                or self.candidates_examined < 1
            ):
                raise _validation_error(
                    "found_witness_binding",
                    "a FOUND search must retain its homomorphism check over "
                    "the searched structures after examining at least one map",
                )
        elif (
            self.check is not None or self.candidates_examined != self.total_candidates
        ):
            raise _validation_error(
                "exhaustion_receipt",
                "an EXHAUSTED search retains no map and must have examined "
                "every carrier map",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        status: HomomorphismSearchStatus,
        source: FiniteRelationalStructure,
        target: FiniteRelationalStructure,
        check: HomomorphismCheckResult | None,
        candidates_examined: int,
        total_candidates: int,
    ) -> Self:
        """Build the result after the admitted kernel scanned its receipt.

        A found map was established by the reused homomorphism replay;
        an exhausted scan visited every carrier map. Result construction
        replays nothing.
        """

        return cls.model_construct(
            status=status,
            source=source,
            target=target,
            check=check,
            candidates_examined=candidates_examined,
            total_candidates=total_candidates,
        )


class HomomorphismSearchRequest(StrictModel):
    """Search two same-signature structures for a homomorphism.

    Candidate carrier maps are replayed in lexicographic order (source
    label 0 varying slowest); the first transporting map is FOUND, and
    a complete scan without one is the proved negative EXHAUSTED. A
    signature mismatch is boundary-invalid; a candidate space or joint
    replay work beyond the published envelope is a resource refusal.
    """

    source: FiniteRelationalStructure
    target: FiniteRelationalStructure

    @model_validator(mode="after")
    def require_shared_signature(self) -> Self:
        if self.source.signature != self.target.signature:
            raise _validation_error(
                "signature_mismatch",
                "source and target structures must be declared over one "
                "shared signature",
            )
        return self


class HomomorphismCountRequest(StrictModel):
    """Count every homomorphism between two same-signature structures.

    The complete carrier-map space admitted up front is scanned without
    early stopping; the count is exact. Admission, boundaries, and
    refusal semantics match the search request.
    """

    source: FiniteRelationalStructure
    target: FiniteRelationalStructure

    @model_validator(mode="after")
    def require_shared_signature(self) -> Self:
        if self.source.signature != self.target.signature:
            raise _validation_error(
                "signature_mismatch",
                "source and target structures must be declared over one "
                "shared signature",
            )
        return self


class HomomorphismCountResult(StrictModel):
    """The exact homomorphism count with its source-bound structures.

    The admitted complete space was scanned without early stopping, so
    the count is complete and exact: it is neither a sample, a bound,
    nor a truncated profile.
    """

    source: FiniteRelationalStructure
    target: FiniteRelationalStructure
    count: StrictInt = Field(
        ge=0,
        description="The exact number of transporting carrier maps.",
    )
    total_candidates: StrictInt = Field(
        ge=0,
        description="The complete |B|^|A| carrier-map count scanned.",
    )

    @model_validator(mode="after")
    def require_count_consistency(self) -> Self:
        source_size = self.source.carrier_size
        target_size = self.target.carrier_size
        expected_total = (
            1
            if source_size == 0
            else (0 if target_size == 0 else target_size**source_size)
        )
        if self.total_candidates != expected_total:
            raise _validation_error(
                "count_receipt",
                "total_candidates must be the complete carrier-map count",
            )
        if not 0 <= self.count <= self.total_candidates:
            raise _validation_error(
                "count_range",
                "the exact count must lie within the complete space",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source: FiniteRelationalStructure,
        target: FiniteRelationalStructure,
        count: int,
        total_candidates: int,
    ) -> Self:
        """Build the result after the admitted kernel scanned its receipt.

        Every carrier map was decided by the reused homomorphism replay;
        result construction replays nothing.
        """

        return cls.model_construct(
            source=source,
            target=target,
            count=count,
            total_candidates=total_candidates,
        )


class HomomorphismCoreRequest(StrictModel):
    """Compute the core of one finite relational structure.

    The core is the minimal retract, reached by restricting along the
    first non-surjective endomorphism in lexicographic order until no
    smaller image remains. The empty structure is already a core.
    """

    source: FiniteRelationalStructure


class HomomorphismCoreResult(StrictModel):
    """A minimal retract with its witnessing maps.

    ``core`` is the induced substructure on the retraction image with
    canonical carrier labels; ``inclusion`` embeds core labels back
    into source labels in increasing order; ``retraction`` maps every
    source label into a core label. The section identity
    ``retraction[inclusion[c]] == c`` holds, and the composed
    ``inclusion ∘ retraction`` is an idempotent endomorphism of the
    source. Minimality — no endomorphism of the core has a smaller
    image — was established by the final exhaustive level scan.
    """

    source: FiniteRelationalStructure
    core: FiniteRelationalStructure
    inclusion: tuple[StrictInt, ...] = Field(
        description=(
            "The retraction image in the source carrier, strictly "
            "increasing; core label c embeds as inclusion[c]."
        )
    )
    retraction: tuple[StrictInt, ...] = Field(
        description=(
            "One core label per source label: the idempotent retraction "
            "onto the core image."
        )
    )

    @model_validator(mode="after")
    def require_core_consistency(self) -> Self:
        if self.core.signature != self.source.signature:
            raise _validation_error(
                "core_signature",
                "the core shares the source signature",
            )
        core_size = self.core.carrier_size
        if len(self.inclusion) != core_size or tuple(sorted(set(self.inclusion))) != (
            self.inclusion
        ):
            raise _validation_error(
                "inclusion_shape",
                "the inclusion must list the strictly increasing image labels",
            )
        if any(not 0 <= label < self.source.carrier_size for label in self.inclusion):
            raise _validation_error(
                "inclusion_axis",
                "every inclusion label must lie in the source carrier",
            )
        if len(self.retraction) != self.source.carrier_size or any(
            not 0 <= label < core_size for label in self.retraction
        ):
            raise _validation_error(
                "retraction_shape",
                "the retraction must map every source label into the core carrier",
            )
        if core_size == 0:
            if self.retraction != ():
                raise _validation_error(
                    "retraction_shape",
                    "the empty core retains the empty retraction",
                )
        elif any(
            self.retraction[source_label] != core_label
            for core_label, source_label in enumerate(self.inclusion)
        ):
            raise _validation_error(
                "section_identity",
                "the retraction must fix its inclusion image pointwise",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source: FiniteRelationalStructure,
        core: FiniteRelationalStructure,
        inclusion: tuple[int, ...],
        retraction: tuple[int, ...],
    ) -> Self:
        """Build the result after the admitted kernel iterated its receipt.

        Every level scan was decided by the reused homomorphism replay
        and the composed retraction was replayed once; result
        construction replays nothing.
        """

        return cls.model_construct(
            source=source,
            core=core,
            inclusion=inclusion,
            retraction=retraction,
        )


class EmbeddingSearchRequest(StrictModel):
    """Search two same-signature structures for an injective homomorphism.

    Candidate carrier maps with distinct images are replayed in
    lexicographic order; the first transporting map is FOUND, and a
    complete scan without one is the proved negative EXHAUSTED.
    Admission, boundaries, and refusal semantics match the homomorphism
    search request.
    """

    source: FiniteRelationalStructure
    target: FiniteRelationalStructure

    @model_validator(mode="after")
    def require_shared_signature(self) -> Self:
        if self.source.signature != self.target.signature:
            raise _validation_error(
                "signature_mismatch",
                "source and target structures must be declared over one "
                "shared signature",
            )
        return self


class EmbeddingSearchResult(StrictModel):
    """A source-bound exhaustive embedding search outcome.

    ``FOUND`` retains the first injective transporting map in
    lexicographic carrier-map order as a complete
    ``InducedEmbeddingCheckResult``; ``EXHAUSTED`` retains the receipt that
    every one of the ``total_candidates`` carrier maps was examined.
    """

    status: HomomorphismSearchStatus
    source: FiniteRelationalStructure
    target: FiniteRelationalStructure
    check: InducedEmbeddingCheckResult | None = None
    candidates_examined: StrictInt = Field(
        ge=0,
        description="Carrier maps replayed before the outcome was reached.",
    )
    total_candidates: StrictInt = Field(
        ge=0,
        description="The complete |B|^|A| carrier-map count scanned.",
    )

    @model_validator(mode="after")
    def require_embedding_consistency(self) -> Self:
        source_size = self.source.carrier_size
        target_size = self.target.carrier_size
        expected_total = (
            1
            if source_size == 0
            else (0 if target_size == 0 else target_size**source_size)
        )
        if self.total_candidates != expected_total:
            raise _validation_error(
                "search_receipt",
                "total_candidates must be the complete carrier-map count",
            )
        if not 0 <= self.candidates_examined <= self.total_candidates:
            raise _validation_error(
                "search_receipt",
                "candidates_examined must lie within the complete space",
            )
        if self.source.signature != self.target.signature:
            raise _validation_error(
                "signature_mismatch",
                "source and target structures must share one signature",
            )
        if self.status is HomomorphismSearchStatus.FOUND:
            if (
                self.check is None
                or not isinstance(self.check, InducedEmbeddingCheckResult)
                or self.check.status is not HomomorphismStatus.HOMOMORPHISM
                or self.check.source != self.source
                or self.check.target != self.target
                or len(set(self.check.carrier_map)) != source_size
                or self.candidates_examined < 1
            ):
                raise _validation_error(
                    "found_witness_binding",
                    "a FOUND embedding search must retain an injective "
                    "homomorphism check over the searched structures",
                )
        elif (
            self.check is not None or self.candidates_examined != self.total_candidates
        ):
            raise _validation_error(
                "exhaustion_receipt",
                "an EXHAUSTED search retains no map and must have examined "
                "every carrier map",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        status: HomomorphismSearchStatus,
        source: FiniteRelationalStructure,
        target: FiniteRelationalStructure,
        check: InducedEmbeddingCheckResult | None,
        candidates_examined: int,
        total_candidates: int,
    ) -> Self:
        """Build the result after the admitted kernel scanned its receipt.

        A found embedding was established by the reused homomorphism
        replay on a distinct-image map; an exhausted scan visited every
        carrier map. Result construction replays nothing.
        """

        return cls.model_construct(
            status=status,
            source=source,
            target=target,
            check=check,
            candidates_examined=candidates_examined,
            total_candidates=total_candidates,
        )


__all__ = [
    "EmbeddingSearchRequest",
    "EmbeddingSearchResult",
    "HomomorphismCheckRequest",
    "HomomorphismCheckResult",
    "HomomorphismCoreRequest",
    "HomomorphismCoreResult",
    "HomomorphismCountRequest",
    "HomomorphismCountResult",
    "HomomorphismSearchRequest",
    "HomomorphismSearchResult",
    "HomomorphismSearchStatus",
    "HomomorphismStatus",
    "HomomorphismViolationWitness",
    "InducedEmbeddingCheckResult",
    "InducedRelationProfile",
    "SymbolTransportProfile",
]
