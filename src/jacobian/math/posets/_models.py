"""Bounded contracts for exact finite partially ordered sets."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._digest import Sha256Digest
from jacobian._models import StrictModel
from jacobian.canonical import canonicalize_json


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"poset.{reason}", message)


MAX_POSET_ELEMENTS = 64
MAX_POSET_RELATIONS = MAX_POSET_ELEMENTS * MAX_POSET_ELEMENTS
MAX_LINEAR_EXTENSION_ELEMENTS = 20
MAX_ANTICHAIN_PROFILE_ELEMENTS = 14
MAX_ANTICHAIN_PROFILE_CANDIDATES = 1 << MAX_ANTICHAIN_PROFILE_ELEMENTS

ElementLabel = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,31}$",
        strict=True,
    ),
]


class RelationInterpretation(StrEnum):
    COVER_EDGES = "COVER_EDGES"
    COMPARABLE_PAIRS = "COMPARABLE_PAIRS"


class ReflexivePairPolicy(StrEnum):
    FORBIDDEN = "FORBIDDEN"
    REQUIRED = "REQUIRED"


class MobiusScope(StrEnum):
    COMPLETE_MATRIX = "COMPLETE_MATRIX"
    SELECTED_INTERVALS = "SELECTED_INTERVALS"


class OrderedPair(StrictModel):
    lower: ElementLabel
    upper: ElementLabel

    @model_validator(mode="after")
    def require_distinct_endpoints(self) -> Self:
        if self.lower == self.upper:
            raise _validation_error(
                "ordered_pair_distinct", "strict order pairs require distinct endpoints"
            )
        return self


class PresentationPair(StrictModel):
    lower: ElementLabel
    upper: ElementLabel


class IncomparablePair(StrictModel):
    left: ElementLabel
    right: ElementLabel

    @model_validator(mode="after")
    def require_canonical_distinct_endpoints(self) -> Self:
        if self.left >= self.right:
            raise _validation_error(
                "incomparable_pair_canonical",
                "incomparable pairs must use canonical distinct order",
            )
        return self


class ElementRank(StrictModel):
    element: ElementLabel
    rank: StrictInt = Field(ge=0, le=MAX_POSET_ELEMENTS - 1)


def _strict_closure(
    elements: tuple[str, ...],
    pairs: set[tuple[str, str]],
) -> set[tuple[str, str]]:
    successors: dict[str, set[str]] = {element: set() for element in elements}
    for lower, upper in pairs:
        successors[lower].add(upper)
    for pivot in elements:
        reaching_pivot = {
            element for element in elements if pivot in successors[element]
        }
        for lower in reaching_pivot:
            successors[lower].update(successors[pivot])
    return {(lower, upper) for lower in elements for upper in successors[lower]}


def _transitive_reduction(
    elements: tuple[str, ...],
    closure: set[tuple[str, str]],
) -> set[tuple[str, str]]:
    return {
        (lower, upper)
        for lower, upper in closure
        if not any(
            (lower, middle) in closure and (middle, upper) in closure
            for middle in elements
            if middle not in {lower, upper}
        )
    }


def canonical_poset_ranks(
    elements: tuple[str, ...],
    covers: set[tuple[str, str]],
) -> tuple[ElementRank, ...] | None:
    predecessors: dict[str, set[str]] = {element: set() for element in elements}
    successors: dict[str, set[str]] = {element: set() for element in elements}
    for lower, upper in covers:
        predecessors[upper].add(lower)
        successors[lower].add(upper)
    ranks: dict[str, int] = {}
    remaining = set(elements)
    while remaining:
        ready = sorted(
            element for element in remaining if predecessors[element].issubset(ranks)
        )
        if not ready:  # pragma: no cover - validated presentations are acyclic
            raise _validation_error(
                "cover_relation_acyclic", "poset cover relation is cyclic"
            )
        for element in ready:
            parent_ranks = {ranks[parent] for parent in predecessors[element]}
            if len(parent_ranks) > 1:
                return None
            ranks[element] = 0 if not parent_ranks else next(iter(parent_ranks)) + 1
            remaining.remove(element)
    maximal_ranks = {ranks[element] for element in elements if not successors[element]}
    if len(maximal_ranks) > 1:
        return None
    return tuple(
        ElementRank(element=element, rank=ranks[element]) for element in elements
    )


def _validate_presentation_elements_and_pairs(
    elements: tuple[str, ...],
    relation: tuple[PresentationPair, ...],
) -> tuple[tuple[str, str], ...]:
    if len(elements) != len(set(elements)):
        raise _validation_error("elements_unique", "poset elements must be unique")
    carrier = set(elements)
    pairs = tuple((pair.lower, pair.upper) for pair in relation)
    if len(pairs) != len(set(pairs)):
        raise _validation_error(
            "relation_pairs_unique", "relation pairs must be unique"
        )
    if any(lower not in carrier or upper not in carrier for lower, upper in pairs):
        raise _validation_error(
            "relation_endpoints_declared",
            "relation endpoints must be declared elements",
        )
    return pairs


def _resolve_strict_pairs(
    elements: tuple[str, ...],
    pairs: tuple[tuple[str, str], ...],
    interpretation: RelationInterpretation,
    reflexive_pairs: ReflexivePairPolicy,
) -> set[tuple[str, str]]:
    diagonal = {(element, element) for element in elements}
    supplied = set(pairs)
    if interpretation is RelationInterpretation.COVER_EDGES:
        if reflexive_pairs is not ReflexivePairPolicy.FORBIDDEN:
            raise _validation_error(
                "cover_edges_reflexive_policy",
                "cover edges require reflexive pairs to be forbidden",
            )
        if supplied & diagonal:
            raise _validation_error(
                "cover_edges_irreflexive", "cover edges must be irreflexive"
            )
        return supplied
    present_diagonal = supplied & diagonal
    if reflexive_pairs is ReflexivePairPolicy.FORBIDDEN:
        if present_diagonal:
            raise _validation_error(
                "comparable_reflexive_forbidden",
                "reflexive comparable pairs are forbidden",
            )
    elif present_diagonal != diagonal:
        raise _validation_error(
            "required_reflexive_full_carrier",
            "required reflexive pairs must cover the full carrier",
        )
    return supplied - diagonal


def _validate_strict_order_shape(
    elements: tuple[str, ...],
    strict: set[tuple[str, str]],
    interpretation: RelationInterpretation,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    if any((upper, lower) in strict for lower, upper in strict):
        raise _validation_error(
            "relation_antisymmetric", "relation must be antisymmetric"
        )
    closure = _strict_closure(elements, strict)
    if any(lower == upper for lower, upper in closure):
        raise _validation_error("relation_acyclic", "relation must be acyclic")
    reduction = _transitive_reduction(elements, closure)
    if interpretation is RelationInterpretation.COVER_EDGES:
        if strict != reduction:
            raise _validation_error(
                "cover_edges_transitive_redundancy",
                "cover-edge input contains a transitively redundant edge",
            )
    elif strict != closure:
        raise _validation_error(
            "comparable_pairs_complete",
            "comparable-pair input must contain the complete strict order",
        )
    return closure, reduction


def _validated_presentation(
    elements: tuple[str, ...],
    relation: tuple[PresentationPair, ...],
    interpretation: RelationInterpretation,
    reflexive_pairs: ReflexivePairPolicy,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    pairs = _validate_presentation_elements_and_pairs(elements, relation)
    strict = _resolve_strict_pairs(elements, pairs, interpretation, reflexive_pairs)
    return _validate_strict_order_shape(elements, strict, interpretation)


class FinitePosetRequest(StrictModel):
    elements: tuple[ElementLabel, ...] = Field(
        default=(),
        max_length=MAX_POSET_ELEMENTS,
    )
    relation: tuple[PresentationPair, ...] = Field(
        default=(),
        max_length=MAX_POSET_RELATIONS,
    )
    interpretation: RelationInterpretation
    reflexive_pairs: ReflexivePairPolicy = ReflexivePairPolicy.FORBIDDEN

    @model_validator(mode="after")
    def require_finite_partial_order(self) -> Self:
        _validated_presentation(
            self.elements,
            self.relation,
            self.interpretation,
            self.reflexive_pairs,
        )
        return self


def finite_poset_digest(
    *,
    elements: tuple[str, ...],
    strict_order_pairs: tuple[OrderedPair, ...],
    cover_relations: tuple[OrderedPair, ...],
    incomparable_pairs: tuple[IncomparablePair, ...],
    minimal_elements: tuple[str, ...],
    maximal_elements: tuple[str, ...],
    graded: bool,
    ranks: tuple[ElementRank, ...] | None,
) -> str:
    payload = {
        "elements": list(elements),
        "strict_order_pairs": [
            pair.model_dump(mode="json") for pair in strict_order_pairs
        ],
        "cover_relations": [pair.model_dump(mode="json") for pair in cover_relations],
        "incomparable_pairs": [
            pair.model_dump(mode="json") for pair in incomparable_pairs
        ],
        "minimal_elements": list(minimal_elements),
        "maximal_elements": list(maximal_elements),
        "graded": graded,
        "ranks": (
            None if ranks is None else [rank.model_dump(mode="json") for rank in ranks]
        ),
    }
    return "sha256:" + hashlib.sha256(canonicalize_json(payload)).hexdigest()


def _validate_canonical_poset_elements_and_pairs(
    elements: tuple[str, ...],
    strict_order_pairs: tuple[OrderedPair, ...],
    cover_relations: tuple[OrderedPair, ...],
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    if tuple(sorted(set(elements))) != elements:
        raise _validation_error(
            "canonical_elements", "poset elements must be unique and canonical"
        )
    strict = tuple((pair.lower, pair.upper) for pair in strict_order_pairs)
    covers = tuple((pair.lower, pair.upper) for pair in cover_relations)
    if strict != tuple(sorted(set(strict))) or covers != tuple(sorted(set(covers))):
        raise _validation_error(
            "canonical_order_pairs",
            "order and cover pairs must be unique and canonical",
        )
    return strict, covers


def _validate_poset_incomparable_pairs(
    elements: tuple[str, ...],
    incomparable_pairs: tuple[IncomparablePair, ...],
    closure: set[tuple[str, str]],
) -> None:
    expected_incomparable = tuple(
        (left, right)
        for index, left in enumerate(elements)
        for right in elements[index + 1 :]
        if (left, right) not in closure and (right, left) not in closure
    )
    actual_incomparable = tuple((pair.left, pair.right) for pair in incomparable_pairs)
    if actual_incomparable != expected_incomparable:
        raise _validation_error(
            "canonical_incomparable_pairs",
            "incomparable_pairs is not complete and canonical",
        )


def _compute_poset_extremal_elements(
    elements: tuple[str, ...],
    closure: set[tuple[str, str]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    expected_minimal = tuple(
        element
        for element in elements
        if not any(upper == element for _, upper in closure)
    )
    expected_maximal = tuple(
        element
        for element in elements
        if not any(lower == element for lower, _ in closure)
    )
    return expected_minimal, expected_maximal


def _validate_poset_extremal_elements(
    minimal_elements: tuple[str, ...],
    maximal_elements: tuple[str, ...],
    expected_minimal: tuple[str, ...],
    expected_maximal: tuple[str, ...],
) -> None:
    if minimal_elements != expected_minimal or maximal_elements != expected_maximal:
        raise _validation_error(
            "extremal_elements_complete", "minimal or maximal elements are incomplete"
        )


def _validate_poset_rank_structure(
    elements: tuple[str, ...],
    graded: bool,
    ranks: tuple[ElementRank, ...] | None,
    expected_ranks: tuple[ElementRank, ...] | None,
    expected_minimal: tuple[str, ...],
    expected_maximal: tuple[str, ...],
    reduction: set[tuple[str, str]],
) -> None:
    if graded != (expected_ranks is not None):
        raise _validation_error(
            "graded_metadata_matches",
            "graded metadata does not match the canonical poset",
        )
    if ranks != expected_ranks:
        raise _validation_error("ranks_match", "ranks do not match the canonical poset")
    if expected_ranks is not None:
        if tuple(rank.element for rank in expected_ranks) != elements:
            raise _validation_error(
                "ranks_cover_carrier", "rank entries must cover the canonical carrier"
            )
        rank_for = {rank.element: rank.rank for rank in expected_ranks}
        if any(rank_for[element] != 0 for element in expected_minimal):
            raise _validation_error(
                "minimal_ranks_zero", "minimal elements must have rank zero"
            )
        if any(rank_for[upper] != rank_for[lower] + 1 for lower, upper in reduction):
            raise _validation_error(
                "cover_rank_increment", "every cover must increase rank by one"
            )
        maximal_ranks = {rank_for[element] for element in expected_maximal}
        if len(maximal_ranks) > 1:
            raise _validation_error(
                "maximal_ranks_equal", "graded maximal elements must share one rank"
            )


class FinitePoset(StrictModel):
    elements: tuple[ElementLabel, ...] = Field(
        default=(),
        max_length=MAX_POSET_ELEMENTS,
    )
    strict_order_pairs: tuple[OrderedPair, ...] = Field(
        default=(),
        max_length=MAX_POSET_RELATIONS,
    )
    cover_relations: tuple[OrderedPair, ...] = Field(
        default=(),
        max_length=MAX_POSET_RELATIONS,
    )
    incomparable_pairs: tuple[IncomparablePair, ...] = Field(
        default=(),
        max_length=MAX_POSET_RELATIONS,
    )
    minimal_elements: tuple[ElementLabel, ...] = ()
    maximal_elements: tuple[ElementLabel, ...] = ()
    graded: bool
    ranks: tuple[ElementRank, ...] | None = None
    poset_digest: Sha256Digest

    @model_validator(mode="after")
    def require_complete_canonical_poset(self) -> Self:
        strict, covers = _validate_canonical_poset_elements_and_pairs(
            self.elements, self.strict_order_pairs, self.cover_relations
        )
        closure, reduction = _validated_presentation(
            self.elements,
            tuple(
                PresentationPair(lower=lower, upper=upper) for lower, upper in strict
            ),
            RelationInterpretation.COMPARABLE_PAIRS,
            ReflexivePairPolicy.FORBIDDEN,
        )
        if set(covers) != reduction:
            raise _validation_error(
                "cover_relations_reduction",
                "cover_relations is not the transitive reduction",
            )
        _validate_poset_incomparable_pairs(
            self.elements, self.incomparable_pairs, closure
        )
        expected_minimal, expected_maximal = _compute_poset_extremal_elements(
            self.elements, closure
        )
        _validate_poset_extremal_elements(
            self.minimal_elements,
            self.maximal_elements,
            expected_minimal,
            expected_maximal,
        )
        expected_ranks = canonical_poset_ranks(self.elements, reduction)
        _validate_poset_rank_structure(
            self.elements,
            self.graded,
            self.ranks,
            expected_ranks,
            expected_minimal,
            expected_maximal,
            reduction,
        )
        expected_digest = finite_poset_digest(
            elements=self.elements,
            strict_order_pairs=self.strict_order_pairs,
            cover_relations=self.cover_relations,
            incomparable_pairs=self.incomparable_pairs,
            minimal_elements=self.minimal_elements,
            maximal_elements=self.maximal_elements,
            graded=self.graded,
            ranks=self.ranks,
        )
        if self.poset_digest != expected_digest:
            raise _validation_error(
                "digest_binds_poset", "poset_digest does not bind the canonical poset"
            )
        return self


class PosetExactResult(StrictModel):
    exactness: Literal["EXACT_FINITE"] = "EXACT_FINITE"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"


class FinitePosetMaterializationResult(PosetExactResult):
    poset: FinitePoset
    completeness: Literal["COMPLETE_CLOSURE_AND_REDUCTION"] = (
        "COMPLETE_CLOSURE_AND_REDUCTION"
    )


class PosetRequest(StrictModel):
    """A canonical finite poset consumed directly by bounded operations."""

    poset: FinitePoset


class PosetChain(StrictModel):
    elements: tuple[ElementLabel, ...] = Field(
        min_length=1,
        max_length=MAX_POSET_ELEMENTS,
    )


class MatchingEdge(StrictModel):
    left: ElementLabel
    right: ElementLabel


class PosetWidthResult(PosetExactResult):
    poset_digest: Sha256Digest
    width: StrictInt = Field(ge=0, le=MAX_POSET_ELEMENTS)
    maximum_antichain: tuple[ElementLabel, ...] = Field(
        default=(),
        max_length=MAX_POSET_ELEMENTS,
    )
    minimum_chain_cover: tuple[PosetChain, ...] = Field(
        default=(),
        max_length=MAX_POSET_ELEMENTS,
    )
    matching: tuple[MatchingEdge, ...] = Field(
        default=(),
        max_length=MAX_POSET_ELEMENTS,
    )
    matching_size: StrictInt = Field(ge=0, le=MAX_POSET_ELEMENTS)
    certificate: Literal["DILWORTH_ANTICHAIN_CHAIN_COVER"] = (
        "DILWORTH_ANTICHAIN_CHAIN_COVER"
    )

    @model_validator(mode="after")
    def require_certificate_dimensions(self) -> Self:
        if (
            self.width != len(self.maximum_antichain)
            or self.width != len(self.minimum_chain_cover)
            or self.matching_size != len(self.matching)
            or self.matching_size + self.width
            != sum(len(chain.elements) for chain in self.minimum_chain_cover)
        ):
            raise _validation_error(
                "width_certificate_dimensions",
                "width certificate dimensions are inconsistent",
            )
        return self


class LinearExtensionRequest(StrictModel):
    poset: FinitePoset

    @model_validator(mode="after")
    def require_subset_dp_bound(self) -> Self:
        if len(self.poset.elements) > MAX_LINEAR_EXTENSION_ELEMENTS:
            raise _validation_error(
                "linear_extension_size_bound",
                "linear-extension counting supports at most "
                f"{MAX_LINEAR_EXTENSION_ELEMENTS} elements",
            )
        return self


class LinearExtensionCountResult(PosetExactResult):
    count: StrictInt = Field(ge=1)
    completeness: Literal["COMPLETE"] = "COMPLETE"


class PosetInterval(StrictModel):
    lower: ElementLabel
    upper: ElementLabel


class MobiusFunctionRequest(StrictModel):
    poset: FinitePoset
    scope: MobiusScope = MobiusScope.COMPLETE_MATRIX
    intervals: tuple[PosetInterval, ...] = Field(
        default=(),
        max_length=MAX_POSET_RELATIONS,
    )

    @model_validator(mode="after")
    def require_explicit_interval_scope(self) -> Self:
        pairs = tuple((item.lower, item.upper) for item in self.intervals)
        if len(pairs) != len(set(pairs)):
            raise _validation_error(
                "selected_intervals_unique", "selected intervals must be unique"
            )
        if self.scope is MobiusScope.COMPLETE_MATRIX:
            if self.intervals:
                raise _validation_error(
                    "complete_scope_no_intervals",
                    "complete Möbius scope must not list intervals",
                )
        elif not self.intervals:
            raise _validation_error(
                "selected_scope_nonempty",
                "selected Möbius scope requires at least one interval",
            )
        carrier = set(self.poset.elements)
        comparable = {
            (pair.lower, pair.upper) for pair in self.poset.strict_order_pairs
        }
        for lower, upper in pairs:
            if lower not in carrier or upper not in carrier:
                raise _validation_error(
                    "interval_endpoint_in_carrier",
                    "selected interval endpoint is outside the poset",
                )
            if lower != upper and (lower, upper) not in comparable:
                raise _validation_error(
                    "interval_is_comparable",
                    "selected interval must satisfy lower <= upper",
                )
        return self


class MobiusContribution(StrictModel):
    intermediate: ElementLabel
    value: StrictInt


class MobiusValue(StrictModel):
    lower: ElementLabel
    upper: ElementLabel
    value: StrictInt
    recurrence_contributions: tuple[MobiusContribution, ...] | None = Field(
        default=None,
        max_length=MAX_POSET_ELEMENTS,
    )


class MobiusFunctionResult(PosetExactResult):
    poset_digest: Sha256Digest
    element_order: tuple[ElementLabel, ...] = Field(
        default=(),
        max_length=MAX_POSET_ELEMENTS,
    )
    scope: MobiusScope
    intervals: tuple[PosetInterval, ...] = Field(
        default=(),
        max_length=MAX_POSET_RELATIONS,
    )
    values: tuple[MobiusValue, ...] = Field(
        default=(),
        max_length=MAX_POSET_RELATIONS,
    )
    completeness: Literal["COMPLETE_MATRIX", "SELECTED_INTERVALS"]
    recurrence_identity: Literal["SUM_LOWER_TO_UPPER_EQUALS_DELTA"] = (
        "SUM_LOWER_TO_UPPER_EQUALS_DELTA"
    )

    @model_validator(mode="after")
    def require_scope_metadata(self) -> Self:
        expected = (
            "COMPLETE_MATRIX"
            if self.scope is MobiusScope.COMPLETE_MATRIX
            else "SELECTED_INTERVALS"
        )
        if self.completeness != expected:
            raise _validation_error(
                "mobius_completeness_matches",
                "Möbius completeness does not match the requested scope",
            )
        keys = tuple((item.lower, item.upper) for item in self.values)
        if keys != tuple(sorted(set(keys))):
            raise _validation_error(
                "canonical_mobius_values", "Möbius values must be unique and canonical"
            )
        return self


# ---------------------------------------------------------------------------
# Issue #1746: Closures, duals, zeta transforms, antichain profiles
# ---------------------------------------------------------------------------


class PosetSubset(StrictModel):
    """A subset of poset elements for closure operations."""

    elements: tuple[ElementLabel, ...] = Field(
        default=(), max_length=MAX_POSET_ELEMENTS
    )


class PosetClosureRequest(StrictModel):
    """Request lower/upper closure of a subset in a poset."""

    poset: FinitePoset
    subset: PosetSubset
    closure_type: Literal["LOWER", "UPPER"] = "LOWER"

    @model_validator(mode="after")
    def require_subset_in_carrier(self) -> Self:
        carrier = set(self.poset.elements)
        for element in self.subset.elements:
            if element not in carrier:
                raise _validation_error(
                    "subset_elements_in_carrier",
                    "subset elements must be poset elements",
                )
        return self


class PosetClosureResult(PosetExactResult):
    poset_digest: Sha256Digest
    closure_type: Literal["LOWER", "UPPER"]
    closure: tuple[ElementLabel, ...] = Field(default=(), max_length=MAX_POSET_ELEMENTS)
    generated_element: tuple[ElementLabel, ...] = Field(
        default=(), max_length=MAX_POSET_ELEMENTS
    )


class PosetDualRequest(StrictModel):
    """Request the dual (opposite) of a poset."""

    poset: FinitePoset


class PosetDualResult(PosetExactResult):
    poset: FinitePoset


class ZetaTransformRequest(StrictModel):
    """Compute the zeta transform of a function on the poset."""

    poset: FinitePoset
    function_values: tuple[MobiusValue, ...] = Field(
        default=(), max_length=MAX_POSET_RELATIONS
    )

    @model_validator(mode="after")
    def require_function_domain_covers(self) -> Self:
        pairs = {v.lower + "|" + v.upper for v in self.function_values}
        if len(pairs) != len(self.function_values):
            raise _validation_error(
                "function_intervals_unique",
                "function values must have unique intervals",
            )
        carrier = set(self.poset.elements)
        comparable = {(p.lower, p.upper) for p in self.poset.strict_order_pairs}
        for v in self.function_values:
            if v.lower not in carrier or v.upper not in carrier:
                raise _validation_error(
                    "function_endpoints_in_carrier",
                    "function value endpoints must be poset elements",
                )
            if v.lower != v.upper and (v.lower, v.upper) not in comparable:
                raise _validation_error(
                    "function_interval_comparable",
                    "function value must satisfy lower <= upper",
                )
        return self


class ZetaTransformResult(PosetExactResult):
    poset_digest: Sha256Digest
    transform: Literal["ZETA"] = "ZETA"
    values: tuple[MobiusValue, ...] = Field(default=(), max_length=MAX_POSET_RELATIONS)


class IncidenceConvolutionRequest(StrictModel):
    """Incidence-algebra convolution of two functions on the poset."""

    poset: FinitePoset
    first: tuple[MobiusValue, ...] = Field(default=(), max_length=MAX_POSET_RELATIONS)
    second: tuple[MobiusValue, ...] = Field(default=(), max_length=MAX_POSET_RELATIONS)

    @model_validator(mode="after")
    def require_unique_values(self) -> Self:
        for label, values in (("first", self.first), ("second", self.second)):
            pairs = {v.lower + "|" + v.upper for v in values}
            if len(pairs) != len(values):
                raise _validation_error(
                    "function_intervals_unique",
                    f"{label} values must have unique intervals",
                )
            carrier = set(self.poset.elements)
            comparable = {(p.lower, p.upper) for p in self.poset.strict_order_pairs}
            for v in values:
                if v.lower not in carrier or v.upper not in carrier:
                    raise _validation_error(
                        "function_endpoints_in_carrier",
                        f"{label} value endpoints must be poset elements",
                    )
                if v.lower != v.upper and (v.lower, v.upper) not in comparable:
                    raise _validation_error(
                        "function_interval_comparable",
                        f"{label} value must satisfy lower <= upper",
                    )
        return self


class IncidenceConvolutionResult(PosetExactResult):
    poset_digest: Sha256Digest
    values: tuple[MobiusValue, ...] = Field(default=(), max_length=MAX_POSET_RELATIONS)


class AntichainProfileRequest(StrictModel):
    """Request an exact profile within its bounded subset-enumeration envelope."""

    poset: FinitePoset

    @model_validator(mode="after")
    def require_bounded_subset_enumeration(self) -> Self:
        if len(self.poset.elements) > MAX_ANTICHAIN_PROFILE_ELEMENTS:
            raise _validation_error(
                "antichain_profile_elements",
                "antichain profiles enumerate every subset and accept at most "
                f"{MAX_ANTICHAIN_PROFILE_ELEMENTS} elements "
                f"({MAX_ANTICHAIN_PROFILE_CANDIDATES} candidate subsets)",
            )
        return self


class AntichainProfileResult(PosetExactResult):
    poset_digest: Sha256Digest
    maximum_antichain_size: StrictInt = Field(ge=0, le=MAX_POSET_ELEMENTS)
    antichain_count: StrictInt = Field(ge=1)
    maximum_antichains: tuple[tuple[ElementLabel, ...], ...] = Field(
        default=(), max_length=MAX_ANTICHAIN_PROFILE_CANDIDATES
    )


__all__ = [
    "MAX_ANTICHAIN_PROFILE_CANDIDATES",
    "MAX_ANTICHAIN_PROFILE_ELEMENTS",
    "MAX_LINEAR_EXTENSION_ELEMENTS",
    "MAX_POSET_ELEMENTS",
    "MAX_POSET_RELATIONS",
    "AntichainProfileRequest",
    "AntichainProfileResult",
    "ElementLabel",
    "ElementRank",
    "FinitePoset",
    "FinitePosetMaterializationResult",
    "FinitePosetRequest",
    "IncidenceConvolutionRequest",
    "IncidenceConvolutionResult",
    "IncomparablePair",
    "LinearExtensionCountResult",
    "LinearExtensionRequest",
    "MatchingEdge",
    "MobiusContribution",
    "MobiusFunctionRequest",
    "MobiusFunctionResult",
    "MobiusScope",
    "MobiusValue",
    "OrderedPair",
    "PosetChain",
    "PosetClosureRequest",
    "PosetClosureResult",
    "PosetDualRequest",
    "PosetDualResult",
    "PosetInterval",
    "PosetRequest",
    "PosetSubset",
    "PosetWidthResult",
    "PresentationPair",
    "ReflexivePairPolicy",
    "RelationInterpretation",
    "ZetaTransformRequest",
    "ZetaTransformResult",
    "canonical_poset_ranks",
    "finite_poset_digest",
]
