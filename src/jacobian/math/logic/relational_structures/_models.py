"""Typed wire contracts for the relational homomorphism check."""

from __future__ import annotations

from enum import StrEnum
from itertools import product
from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.logic.relational_structures.values import (
    MAX_RELATIONAL_ARITY,
    MAX_RELATIONAL_CARRIER,
    MAX_RELATIONAL_OPERATION_TABLE_CELLS,
    MAX_RELATIONAL_POLYMORPHISM_ARITY,
    MAX_RELATIONAL_SYMBOLS,
    FiniteRelationalStructure,
    RelationalHomomorphism,
    RelationSymbolId,
)

MAX_CSP_CONSTRAINTS = 4_096
MAX_CSP_SCOPE_ENTRIES = 16_384
MAX_CSP_SOLUTION_ASSIGNMENTS = 65_536
MAX_CSP_SOLUTION_LABELS = 1_048_576


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"relational.homomorphism.{reason}", message)


def _polymorphism_validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"relational.polymorphism.{reason}", message)


class HomomorphismStatus(StrEnum):
    """Closed outcome of one exhaustive preservation replay."""

    HOMOMORPHISM = "HOMOMORPHISM"
    NOT_HOMOMORPHISM = "NOT_HOMOMORPHISM"


class InducedSubstructureRequest(StrictModel):
    """Select an ordered carrier subset and take its induced structure.

    ``inclusion[i]`` is the source label represented by the canonical label
    ``i`` in the induced carrier. Its order is significant and is retained by
    the result as the source-axis transport.
    """

    source: FiniteRelationalStructure
    inclusion: tuple[StrictInt, ...] = Field(
        max_length=MAX_RELATIONAL_CARRIER,
        description=(
            "An ordered list of distinct source labels. Position i becomes "
            "label i in the induced carrier; any ordering of the selected "
            "subset is supported."
        ),
    )


class InducedSubstructureResult(StrictModel):
    """An induced finite structure with its exact source and carrier map.

    ``inclusion[i]`` embeds induced-carrier label i into the retained source.
    The substructure carries one complete induced relation table per source
    signature symbol, including exact nullary truth values.
    """

    source: FiniteRelationalStructure
    substructure: FiniteRelationalStructure
    inclusion: tuple[StrictInt, ...] = Field(
        max_length=MAX_RELATIONAL_CARRIER,
        description=(
            "The source label represented by each induced-carrier label, in "
            "the caller-selected order."
        ),
    )

    @model_validator(mode="after")
    def require_source_binding(self) -> Self:
        if self.source.signature != self.substructure.signature:
            raise _validation_error(
                "induced_signature",
                "an induced substructure retains the exact source signature",
            )
        if len(self.inclusion) != self.substructure.carrier_size:
            raise _validation_error(
                "induced_inclusion_axis",
                "the inclusion has one source label per induced-carrier label",
            )
        if len(set(self.inclusion)) != len(self.inclusion):
            raise _validation_error(
                "induced_inclusion_injective",
                "the induced-carrier inclusion must be injective",
            )
        if any(not 0 <= label < self.source.carrier_size for label in self.inclusion):
            raise _validation_error(
                "induced_inclusion_range",
                "every inclusion label must belong to the exact source carrier",
            )
        return self


class RelationalReductRequest(StrictModel):
    """Restrict a structure to a selected sub-signature."""

    source: FiniteRelationalStructure
    symbol_ids: tuple[RelationSymbolId, ...] = Field(
        max_length=MAX_RELATIONAL_SYMBOLS,
        description=(
            "Relation symbols to retain. The returned sub-signature follows "
            "source signature order, independently of this selection order."
        ),
    )

    @model_validator(mode="after")
    def require_known_distinct_symbols(self) -> Self:
        if len(set(self.symbol_ids)) != len(self.symbol_ids):
            raise _validation_error(
                "reduct.symbol_ids_not_unique",
                "selected relation symbol IDs must be unique",
            )
        source_ids = {symbol.symbol_id for symbol in self.source.signature}
        if any(symbol_id not in source_ids for symbol_id in self.symbol_ids):
            raise _validation_error(
                "reduct.symbol_id_unknown",
                "every selected relation symbol must belong to the source signature",
            )
        return self


class RelationalProductRequest(StrictModel):
    """Form the direct product of two structures with identical signatures."""

    left: FiniteRelationalStructure
    right: FiniteRelationalStructure


class RelationalReductResult(StrictModel):
    """A reduct together with its exact relation-symbol inclusion map."""

    source: FiniteRelationalStructure
    reduct: FiniteRelationalStructure
    source_symbol_indices: tuple[StrictInt, ...] = Field(
        max_length=MAX_RELATIONAL_SYMBOLS,
        description=(
            "For each reduct signature position, the corresponding position "
            "in the source signature."
        ),
    )

    @model_validator(mode="after")
    def require_signature_transport(self) -> Self:
        if self.source.carrier_size != self.reduct.carrier_size:
            raise _validation_error(
                "reduct.carrier", "a signature reduct preserves the carrier"
            )
        if len(self.source_symbol_indices) != len(self.reduct.signature):
            raise _validation_error(
                "reduct.symbol_map_axis",
                "the symbol map must cover every reduct signature position",
            )
        if len(set(self.source_symbol_indices)) != len(self.source_symbol_indices):
            raise _validation_error(
                "reduct.symbol_map_injective",
                "distinct reduct symbols must map to distinct source symbols",
            )
        if any(
            not 0 <= index < len(self.source.signature)
            for index in self.source_symbol_indices
        ):
            raise _validation_error(
                "reduct.symbol_map_range",
                "every mapped source symbol must belong to the source signature",
            )
        if (
            tuple(self.source.signature[index] for index in self.source_symbol_indices)
            != self.reduct.signature
        ):
            raise _validation_error(
                "reduct.signature_transport",
                "the map must identify each reduct symbol with its source symbol",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source: FiniteRelationalStructure,
        reduct: FiniteRelationalStructure,
        source_symbol_indices: tuple[int, ...],
    ) -> Self:
        """Construct the source-bound reduct emitted by the admitted kernel."""

        return cls.model_construct(
            source=source,
            reduct=reduct,
            source_symbol_indices=source_symbol_indices,
        )


class RelationalProductResult(StrictModel):
    """A direct product with canonical coordinate projections.

    Product carrier label ``i * right.carrier_size + j`` denotes ``(i, j)``.
    Relation tables use that encoding coordinatewise; signatures are retained
    verbatim from both (necessarily equal) factors.
    """

    left: FiniteRelationalStructure
    right: FiniteRelationalStructure
    product: FiniteRelationalStructure
    left_projection: tuple[StrictInt, ...]
    right_projection: tuple[StrictInt, ...]

    @model_validator(mode="after")
    def require_canonical_product_axis(self) -> Self:
        if self.left.signature != self.right.signature:
            raise _validation_error(
                "product_signature", "product factors need the same ranked signature"
            )
        if self.product.signature != self.left.signature:
            raise _validation_error(
                "product_signature", "product retains the factor signature"
            )
        size = self.left.carrier_size * self.right.carrier_size
        if self.product.carrier_size != size:
            raise _validation_error(
                "product_carrier", "product carrier is the Cartesian carrier"
            )
        expected_left = (
            tuple(index // self.right.carrier_size for index in range(size))
            if self.right.carrier_size
            else ()
        )
        expected_right = (
            tuple(index % self.right.carrier_size for index in range(size))
            if self.right.carrier_size
            else ()
        )
        if (
            self.left_projection != expected_left
            or self.right_projection != expected_right
        ):
            raise _validation_error(
                "product_projections",
                "projections must follow canonical lexicographic pair labels",
            )
        return self


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

    def to_homomorphism(self) -> RelationalHomomorphism:
        """Return the canonical composable map for a successful check.

        The kernel already replayed preservation; construction is trusted
        and replays nothing. The serialized check round-trips through
        ``model_validate_json`` before this conversion in the covered
        producer-to-consumer path.
        """

        if self.status is not HomomorphismStatus.HOMOMORPHISM:
            raise ValueError(
                "a NOT_HOMOMORPHISM check retains no composable homomorphism"
            )
        return RelationalHomomorphism._from_kernel(
            source=self.source,
            target=self.target,
            mapping=tuple(self.carrier_map),
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


class RelationalHomomorphismIdentityRequest(StrictModel):
    """Construct the identity homomorphism of one exact structure."""

    structure: FiniteRelationalStructure


class RelationalHomomorphismCompositionRequest(StrictModel):
    """Compose ``second`` after ``first`` over an exact shared structure."""

    first: RelationalHomomorphism
    second: RelationalHomomorphism


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

    def to_homomorphism(self) -> RelationalHomomorphism:
        """Return the canonical composable map for a FOUND search."""

        if self.status is not HomomorphismSearchStatus.FOUND or self.check is None:
            raise ValueError("an EXHAUSTED search retains no composable homomorphism")
        return self.check.to_homomorphism()


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


class HomomorphismEnumerationRequest(StrictModel):
    """Enumerate all maps preserving every relation over a shared signature."""

    source: FiniteRelationalStructure
    target: FiniteRelationalStructure

    @model_validator(mode="after")
    def require_shared_signature(self) -> Self:
        if self.source.signature != self.target.signature:
            raise _validation_error(
                "signature_mismatch",
                "source and target structures must be declared over one shared signature",
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


class HomomorphismEnumerationResult(StrictModel):
    """Complete, source-bound lexicographic list of all homomorphisms."""

    source: FiniteRelationalStructure
    target: FiniteRelationalStructure
    carrier_maps: tuple[tuple[StrictInt, ...], ...] = Field(
        max_length=65_536,
        description=(
            "Every relation-preserving total carrier map, in lexicographic "
            "order with source label 0 varying slowest."
        ),
    )
    total_candidates: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def require_enumeration_shape(self) -> Self:
        if self.source.signature != self.target.signature:
            raise _validation_error(
                "signature_mismatch", "structures must share a signature"
            )
        expected = (
            1
            if self.source.carrier_size == 0
            else (
                0
                if self.target.carrier_size == 0
                else self.target.carrier_size**self.source.carrier_size
            )
        )
        if self.total_candidates != expected:
            raise _validation_error(
                "enumeration_receipt",
                "total_candidates must be the complete map-space size",
            )
        previous = None
        for carrier_map in self.carrier_maps:
            if len(carrier_map) != self.source.carrier_size or any(
                not 0 <= image < self.target.carrier_size for image in carrier_map
            ):
                raise _validation_error(
                    "carrier_map", "each map must be total and target-valued"
                )
            if previous is not None and carrier_map <= previous:
                raise _validation_error(
                    "ordering",
                    "carrier maps must be unique and lexicographically ordered",
                )
            previous = carrier_map
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source: FiniteRelationalStructure,
        target: FiniteRelationalStructure,
        carrier_maps: tuple[tuple[int, ...], ...],
        total_candidates: int,
    ) -> Self:
        """Build after the admitted exhaustive scan; do not replay maps here."""
        return cls.model_construct(
            source=source,
            target=target,
            carrier_maps=carrier_maps,
            total_candidates=total_candidates,
        )

    def to_homomorphisms(self) -> tuple[RelationalHomomorphism, ...]:
        """Return every enumerated map as a canonical composable value."""

        return tuple(
            RelationalHomomorphism._from_kernel(
                source=self.source,
                target=self.target,
                mapping=tuple(carrier_map),
            )
            for carrier_map in self.carrier_maps
        )


class FiniteCspConstraint(StrictModel):
    """One named constraint occurrence over a finite template."""

    constraint_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
    symbol_id: RelationSymbolId
    scope: tuple[StrictInt, ...] = Field(max_length=MAX_RELATIONAL_ARITY)


class FiniteCspInstance(StrictModel):
    """A finite CSP instance with canonical variable labels ``0..n-1``.

    Each constraint occurrence names one template relation and an ordered
    scope. Repeated variables and repeated scopes are significant and retained.
    """

    template: FiniteRelationalStructure
    variable_count: StrictInt = Field(ge=0, le=MAX_RELATIONAL_CARRIER)
    constraints: tuple[FiniteCspConstraint, ...] = Field(max_length=MAX_CSP_CONSTRAINTS)

    @model_validator(mode="after")
    def require_valid_constraints(self) -> Self:
        ids = tuple(constraint.constraint_id for constraint in self.constraints)
        if len(set(ids)) != len(ids):
            raise _validation_error(
                "constraint_identity", "constraint IDs must be unique"
            )
        symbols = {symbol.symbol_id: symbol for symbol in self.template.signature}
        for constraint in self.constraints:
            symbol = symbols.get(constraint.symbol_id)
            if symbol is None:
                raise _validation_error(
                    "constraint_symbol",
                    "every constraint must name a template relation",
                )
            if len(constraint.scope) != symbol.arity:
                raise _validation_error(
                    "constraint_arity",
                    "constraint scope length must match relation arity",
                )
            if any(
                not 0 <= variable < self.variable_count for variable in constraint.scope
            ):
                raise _validation_error(
                    "constraint_variable",
                    "constraint variables must lie on the declared axis",
                )
        if (
            sum(len(constraint.scope) for constraint in self.constraints)
            > MAX_CSP_SCOPE_ENTRIES
        ):
            raise _validation_error(
                "scope_bound", "aggregate CSP scope entries exceed the bound"
            )
        return self


class CspAssignmentRequest(StrictModel):
    """One complete assignment of the variables of a finite CSP instance."""

    instance: FiniteCspInstance
    assignment: tuple[StrictInt, ...] = Field(
        max_length=MAX_RELATIONAL_CARRIER,
        description="One template-carrier label for each variable in increasing order.",
    )

    @model_validator(mode="after")
    def require_complete_assignment(self) -> Self:
        if len(self.assignment) != self.instance.variable_count:
            raise _validation_error(
                "assignment_axis",
                "assignment must have one value per instance variable",
            )
        if any(
            not 0 <= value < self.instance.template.carrier_size
            for value in self.assignment
        ):
            raise _validation_error(
                "assignment_value", "assignment values must lie in the template carrier"
            )
        return self


class CspConstraintEvaluation(StrictModel):
    """The exact target tuple induced by an assignment at one occurrence."""

    constraint_id: str
    symbol_id: RelationSymbolId
    scope: tuple[StrictInt, ...]
    target_tuple: tuple[StrictInt, ...]
    allowed: bool


class CspAssignmentProfile(StrictModel):
    """Complete per-occurrence evaluation of one CSP assignment.

    Repeated constraint occurrences remain visible even when they have the
    same relation symbol and scope. ``first_violation`` is the first rejected
    occurrence in the instance's declared order.
    """

    status: Literal["SOLUTION", "NOT_A_SOLUTION"]
    instance: FiniteCspInstance
    assignment: tuple[StrictInt, ...]
    evaluations: tuple[CspConstraintEvaluation, ...] = Field(
        max_length=MAX_CSP_CONSTRAINTS
    )
    first_violation: CspConstraintEvaluation | None

    @model_validator(mode="after")
    def require_complete_profile(self) -> Self:
        if len(self.assignment) != self.instance.variable_count:
            raise _validation_error(
                "assignment_axis", "profile assignment is not total"
            )
        if any(
            not 0 <= value < self.instance.template.carrier_size
            for value in self.assignment
        ):
            raise _validation_error(
                "assignment_value",
                "profile assignment values must lie in the template carrier",
            )
        if len(self.evaluations) != len(self.instance.constraints):
            raise _validation_error(
                "assignment_profile", "every constraint occurrence must be evaluated"
            )
        for constraint, evaluation in zip(
            self.instance.constraints, self.evaluations, strict=True
        ):
            if (
                evaluation.constraint_id != constraint.constraint_id
                or evaluation.symbol_id != constraint.symbol_id
                or evaluation.scope != constraint.scope
                or evaluation.target_tuple
                != tuple(self.assignment[variable] for variable in constraint.scope)
            ):
                raise _validation_error(
                    "assignment_binding",
                    "evaluation must retain its exact constraint occurrence and assigned tuple",
                )
        failed = next((item for item in self.evaluations if not item.allowed), None)
        if self.first_violation != failed:
            raise _validation_error(
                "assignment_witness",
                "first violation must be the first rejected occurrence",
            )
        if (self.status == "SOLUTION") != (failed is None):
            raise _validation_error(
                "assignment_status", "status must agree with all constraint evaluations"
            )
        return self


class CspSolutions(StrictModel):
    """The complete lexicographically ordered solution family of one CSP.

    Assignments use the variable axis and labels of the retained template.
    The instance keeps named constraint occurrences and their provenance,
    including occurrences that deduplicate in its canonical source structure.
    """

    instance: FiniteCspInstance
    assignments: tuple[
        Annotated[tuple[StrictInt, ...], Field(max_length=MAX_RELATIONAL_CARRIER)], ...
    ] = Field(max_length=MAX_CSP_SOLUTION_ASSIGNMENTS)
    total_candidates: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def require_assignment_family_shape(self) -> Self:
        expected = (
            1
            if self.instance.variable_count == 0
            else (
                0
                if self.instance.template.carrier_size == 0
                else self.instance.template.carrier_size**self.instance.variable_count
            )
        )
        if self.total_candidates != expected:
            raise _validation_error(
                "solutions.candidate_count",
                "total_candidates must be the complete assignment-space size",
            )
        if len(self.assignments) > expected:
            raise _validation_error(
                "solutions.assignment_count",
                "the solution family cannot exceed the complete assignment space",
            )
        if sum(map(len, self.assignments)) > MAX_CSP_SOLUTION_LABELS:
            raise _validation_error(
                "solutions.output_bound",
                "the retained assignment labels exceed the output envelope",
            )
        previous: tuple[int, ...] | None = None
        for assignment in self.assignments:
            if len(assignment) != self.instance.variable_count or any(
                not 0 <= value < self.instance.template.carrier_size
                for value in assignment
            ):
                raise _validation_error(
                    "solutions.assignment_shape",
                    "each assignment must be total and template-valued",
                )
            if previous is not None and assignment <= previous:
                raise _validation_error(
                    "solutions.order",
                    "assignments must be unique and lexicographically ordered",
                )
            previous = assignment
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        instance: FiniteCspInstance,
        assignments: tuple[tuple[int, ...], ...],
        total_candidates: int,
    ) -> Self:
        """Build after the admitted exhaustive search without replaying it."""

        return cls.model_construct(
            instance=instance,
            assignments=assignments,
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


class RelationalQuotientRequest(StrictModel):
    """Form a quotient by a partition of the finite carrier."""

    source: FiniteRelationalStructure
    classes: tuple[StrictInt, ...] = Field(
        max_length=MAX_RELATIONAL_CARRIER,
        description=(
            "One equivalence-class label per source carrier element. Labels "
            "may be arbitrary integers; output classes are canonically ordered "
            "by their least source representative."
        ),
    )

    @model_validator(mode="after")
    def require_total_partition(self) -> Self:
        if len(self.classes) != self.source.carrier_size:
            raise _validation_error(
                "quotient.partition_axis",
                "classes must give exactly one label for each source element",
            )
        return self


class RelationalQuotient(StrictModel):
    """A quotient structure and its canonical surjective projection."""

    source: FiniteRelationalStructure
    quotient: FiniteRelationalStructure
    quotient_map: tuple[StrictInt, ...] = Field(max_length=MAX_RELATIONAL_CARRIER)

    @model_validator(mode="after")
    def require_projection_shape(self) -> Self:
        if self.source.signature != self.quotient.signature:
            raise _validation_error(
                "quotient.signature", "quotient must retain the source signature"
            )
        if len(self.quotient_map) != self.source.carrier_size:
            raise _validation_error(
                "quotient.map_axis", "quotient map must be total on the source"
            )
        if any(
            not 0 <= value < self.quotient.carrier_size for value in self.quotient_map
        ):
            raise _validation_error(
                "quotient.map_value", "quotient map values must lie in the quotient"
            )
        if set(self.quotient_map) != set(range(self.quotient.carrier_size)):
            raise _validation_error(
                "quotient.map_surjective", "quotient map must be surjective"
            )
        return self


class RelationalPolymorphismRequest(StrictModel):
    """Check one complete finite operation table against a structure."""

    source: FiniteRelationalStructure
    arity: StrictInt = Field(ge=1, le=MAX_RELATIONAL_POLYMORPHISM_ARITY)
    operation_table: tuple[StrictInt, ...] = Field(
        max_length=MAX_RELATIONAL_OPERATION_TABLE_CELLS,
        description=(
            "Complete table of f:A^m→A in lexicographic input-tuple order; "
            "the exact number of entries is |A|^m. This is an operation "
            "table, not a carrier map A→B."
        ),
    )

    @model_validator(mode="after")
    def require_complete_table(self) -> Self:
        expected = self.source.carrier_size**self.arity
        if len(self.operation_table) != expected:
            raise _polymorphism_validation_error(
                "polymorphism.table_axis",
                "operation_table must contain one value for every tuple in A^m",
            )
        if any(
            not 0 <= value < self.source.carrier_size for value in self.operation_table
        ):
            raise _polymorphism_validation_error(
                "polymorphism.table_value",
                "every operation table value must lie in the exact source carrier",
            )
        return self


class RelationalPolymorphism(StrictModel):
    """A complete operation table established to preserve one exact structure."""

    source: FiniteRelationalStructure
    arity: StrictInt = Field(ge=1, le=MAX_RELATIONAL_POLYMORPHISM_ARITY)
    operation_table: tuple[StrictInt, ...] = Field(
        max_length=MAX_RELATIONAL_OPERATION_TABLE_CELLS
    )

    @model_validator(mode="after")
    def require_complete_table(self) -> Self:
        expected = self.source.carrier_size**self.arity
        if len(self.operation_table) != expected:
            raise _polymorphism_validation_error(
                "polymorphism.table_axis",
                "operation_table must contain one value for every tuple in A^m",
            )
        if any(
            not 0 <= value < self.source.carrier_size for value in self.operation_table
        ):
            raise _polymorphism_validation_error(
                "polymorphism.table_value",
                "every operation table value must lie in the exact source carrier",
            )
        return self


class RelationalPolymorphismStatus(StrEnum):
    POLYMORPHISM = "POLYMORPHISM"
    NOT_POLYMORPHISM = "NOT_POLYMORPHISM"


class RelationalPolymorphismWitness(StrictModel):
    """Relation rows whose coordinatewise operation image is absent."""

    symbol_id: RelationSymbolId
    relation_arity: StrictInt = Field(ge=0, le=MAX_RELATIONAL_ARITY)
    input_rows: tuple[tuple[StrictInt, ...], ...] = Field(
        max_length=MAX_RELATIONAL_POLYMORPHISM_ARITY
    )
    output_row: tuple[StrictInt, ...]

    @model_validator(mode="after")
    def require_witness_axes(self) -> Self:
        if any(len(row) != self.relation_arity for row in self.input_rows):
            raise _polymorphism_validation_error(
                "polymorphism.witness_rows",
                "each witness input row must have the declared relation arity",
            )
        if len(self.output_row) != self.relation_arity:
            raise _polymorphism_validation_error(
                "polymorphism.witness_output",
                "the coordinatewise output must have the declared relation arity",
            )
        return self


class RelationalPolymorphismRelationProfile(StrictModel):
    symbol_id: RelationSymbolId
    arity: StrictInt = Field(ge=0, le=MAX_RELATIONAL_ARITY)
    input_combinations: StrictInt = Field(ge=0)
    preserved_combinations: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def require_profile_bounds(self) -> Self:
        if self.preserved_combinations > self.input_combinations:
            raise _polymorphism_validation_error(
                "polymorphism.profile_bounds",
                "preserved combinations cannot exceed the complete relation product",
            )
        return self


class RelationalPolymorphismCheckResult(StrictModel):
    """Complete preservation profile for one caller-supplied operation table.

    Validation is structural: the witness must reference the named source
    relation and its rows, with carrier-valued coordinates, and the relation
    profiles must agree with the source signature and counts. The admitted
    ``relational.polymorphism.check`` kernel performs the exhaustive
    coordinatewise image replay; re-checking an externally authored claim
    means running that operation again on its admitted source, arity, and
    operation table, not replaying it inside deserialization.
    """

    source: FiniteRelationalStructure
    arity: StrictInt = Field(ge=1, le=MAX_RELATIONAL_POLYMORPHISM_ARITY)
    operation_table: tuple[StrictInt, ...] = Field(
        max_length=MAX_RELATIONAL_OPERATION_TABLE_CELLS
    )
    status: RelationalPolymorphismStatus
    polymorphism: RelationalPolymorphism | None
    witness: RelationalPolymorphismWitness | None
    relation_profiles: tuple[RelationalPolymorphismRelationProfile, ...]

    @model_validator(mode="after")
    def require_result_binding(self) -> Self:
        expected_rows = self.source.carrier_size**self.arity
        if len(self.operation_table) != expected_rows or any(
            not 0 <= value < self.source.carrier_size for value in self.operation_table
        ):
            raise _polymorphism_validation_error(
                "polymorphism.result_table",
                "the retained operation table must be total and source-bound",
            )
        expected_signature = tuple(
            (symbol.symbol_id, symbol.arity) for symbol in self.source.signature
        )
        if (
            tuple((p.symbol_id, p.arity) for p in self.relation_profiles)
            != expected_signature
        ):
            raise _polymorphism_validation_error(
                "polymorphism.result_profiles",
                "profiles must cover the complete signature in source order",
            )
        success = self.status is RelationalPolymorphismStatus.POLYMORPHISM
        if success and (self.polymorphism is None or self.witness is not None):
            raise _polymorphism_validation_error(
                "polymorphism.result_status",
                "successful checks retain a polymorphism and no failure witness",
            )
        if not success and (self.polymorphism is not None or self.witness is None):
            raise _polymorphism_validation_error(
                "polymorphism.result_status",
                "failed checks retain an exact failure witness and no polymorphism",
            )
        if self.polymorphism is not None and (
            self.polymorphism.source != self.source
            or self.polymorphism.arity != self.arity
            or self.polymorphism.operation_table != self.operation_table
        ):
            raise _polymorphism_validation_error(
                "polymorphism.result_value",
                "the checked polymorphism must retain the exact source and operation table",
            )
        if success and any(
            p.preserved_combinations != p.input_combinations
            for p in self.relation_profiles
        ):
            raise _polymorphism_validation_error(
                "polymorphism.result_profile",
                "a polymorphism preserves every relation-row combination",
            )
        for profile, table in zip(
            self.relation_profiles,
            self.source.relation_tables,
            strict=True,
        ):
            if profile.input_combinations != len(table) ** self.arity:
                raise _polymorphism_validation_error(
                    "polymorphism.result_profile_axis",
                    "profile input combinations must equal the complete relation power",
                )
        if self.witness is not None:
            witness = self.witness
            if len(witness.input_rows) != self.arity:
                raise _polymorphism_validation_error(
                    "polymorphism.witness_operation_arity",
                    "witness must contain one source relation row per operation input",
                )
            try:
                symbol_index = next(
                    i
                    for i, symbol in enumerate(self.source.signature)
                    if symbol.symbol_id == witness.symbol_id
                )
            except StopIteration as exc:
                raise _polymorphism_validation_error(
                    "polymorphism.witness_symbol",
                    "witness symbol must belong to the exact source signature",
                ) from exc
            symbol = self.source.signature[symbol_index]
            relation = self.source.relation_tables[symbol_index]
            if symbol.arity != witness.relation_arity or any(
                row not in relation for row in witness.input_rows
            ):
                raise _polymorphism_validation_error(
                    "polymorphism.witness_source",
                    "witness inputs must be rows of the named source relation",
                )
            if any(
                not 0 <= value < self.source.carrier_size
                for value in witness.output_row
            ):
                raise _polymorphism_validation_error(
                    "polymorphism.witness_output",
                    "witness output coordinates must be source carrier labels",
                )
            profile = self.relation_profiles[symbol_index]
            if profile.preserved_combinations >= profile.input_combinations:
                raise _polymorphism_validation_error(
                    "polymorphism.witness_profile",
                    "the witnessed relation must record a non-preserved combination",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source: FiniteRelationalStructure,
        arity: int,
        operation_table: tuple[int, ...],
        status: RelationalPolymorphismStatus,
        polymorphism: RelationalPolymorphism | None,
        witness: RelationalPolymorphismWitness | None,
        relation_profiles: tuple[RelationalPolymorphismRelationProfile, ...],
    ) -> Self:
        """Construct after exhaustive, pre-admitted relation-product replay."""

        return cls.model_construct(
            source=source,
            arity=arity,
            operation_table=operation_table,
            status=status,
            polymorphism=polymorphism,
            witness=witness,
            relation_profiles=relation_profiles,
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
    "InducedSubstructureRequest",
    "InducedSubstructureResult",
    "RelationalHomomorphismCompositionRequest",
    "RelationalHomomorphismIdentityRequest",
    "RelationalPolymorphism",
    "RelationalPolymorphismCheckResult",
    "RelationalPolymorphismRelationProfile",
    "RelationalPolymorphismRequest",
    "RelationalPolymorphismStatus",
    "RelationalPolymorphismWitness",
    "RelationalQuotient",
    "RelationalQuotientRequest",
    "SymbolTransportProfile",
]
