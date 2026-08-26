"""Typed exact certificate check for a claimed graph chromatic number."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, lcm
from typing import Annotated, Literal, Self

from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    StrictInt,
    WithJsonSchema,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_CHROMATIC_CERTIFICATE_VERTICES = 20
MAX_CHROMATIC_CERTIFICATE_EDGES = (
    MAX_CHROMATIC_CERTIFICATE_VERTICES * (MAX_CHROMATIC_CERTIFICATE_VERTICES - 1) // 2
)
MAX_CHROMATIC_CERTIFICATE_RATIONAL_DIGITS = 64

# A common denominator has at most n times the source component height.  Scaling
# can add one further source numerator height, and summing n signed scaled
# weights adds at most digits(n) plus one sign/carry digit.  Fraction reduction
# cannot increase either component.
MAX_CHROMATIC_CERTIFICATE_DERIVED_RATIONAL_DIGITS = (
    (MAX_CHROMATIC_CERTIFICATE_VERTICES + 1) * MAX_CHROMATIC_CERTIFICATE_RATIONAL_DIGITS
    + len(str(MAX_CHROMATIC_CERTIFICATE_VERTICES))
    + 1
)

# One byte records each subset's independence during exhaustive replay.  The
# checker visits every subset twice overall: once for the operation and once
# when the source-bound result validator replays the defining relation.
MAX_CHROMATIC_CERTIFICATE_SUBSET_STATES = 1 << MAX_CHROMATIC_CERTIFICATE_VERTICES
CHROMATIC_CERTIFICATE_REPLAY_PASSES = 2

# Work is charged in conservative elementary decimal-digit operations.  For
# each pass, admission includes common-denominator construction and scaling,
# one independence-state update per subset, and one bounded-height integer
# update and comparison per Gray-code step.  This result-sensitive cap admits
# order 20 with ordinary small weights and order 19 with distinct 40-digit
# denominators, while rejecting the corresponding order-20 adversarial case.
MAX_CHROMATIC_CERTIFICATE_DIGIT_WORK = 3_000_000_000

# Authored results may carry two derived rationals at the static height above.
# Charge their canonical reduction and exact replay comparisons independently
# of the source-sensitive producer/replay estimate.
_RESULT_RATIONAL_VALIDATION_DIGIT_WORK = (
    8 * MAX_CHROMATIC_CERTIFICATE_DERIVED_RATIONAL_DIGITS**2
)

_RESULT_ENVELOPE_RESERVE_BYTES = 4_096

CertificateColor = Annotated[
    StrictInt,
    Field(ge=0, le=MAX_CHROMATIC_CERTIFICATE_VERTICES),
]


def _preflight_bounded_rational(
    value: object,
    *,
    max_digits: int,
    label: str,
) -> object:
    """Reject oversized wire components before canonical integer parsing."""
    if isinstance(value, CanonicalRational):
        require_bounded_rational(
            value,
            max_digits=max_digits,
            label=label,
        )
        return value
    if isinstance(value, Mapping):
        for component in ("num", "den"):
            raw_component = value.get(component)
            if (
                isinstance(raw_component, str)
                and len(raw_component.lstrip("-")) > max_digits
            ):
                raise PydanticCustomError(
                    "graph.label_exceeds_max_digits_digit_bound_return",
                    f"{label} exceeds the {max_digits}-digit bound",
                )
    return value


def _preflight_certificate_weight(value: object) -> object:
    return _preflight_bounded_rational(
        value,
        max_digits=MAX_CHROMATIC_CERTIFICATE_RATIONAL_DIGITS,
        label="chromatic-number certificate weight",
    )


def _preflight_derived_rational(value: object) -> object:
    return _preflight_bounded_rational(
        value,
        max_digits=MAX_CHROMATIC_CERTIFICATE_DERIVED_RATIONAL_DIGITS,
        label="chromatic-number certificate derived rational",
    )


CertificateWeight = Annotated[
    CanonicalRational,
    BeforeValidator(_preflight_certificate_weight),
]
CertificateDerivedRational = Annotated[
    CanonicalRational,
    BeforeValidator(_preflight_derived_rational),
]
CertificateReason = Literal[
    "ACCEPTED",
    "CLAIM_OUT_OF_RANGE",
    "COLOR_OUT_OF_PALETTE",
    "MONOCHROMATIC_EDGE",
    "NEGATIVE_WEIGHT",
    "INDEPENDENT_SET_OVERWEIGHT",
    "LOWER_BOUND_BELOW_CLAIM",
]


def _chromatic_certificate_graph_schema() -> JsonSchemaValue:
    schema = SimpleUndirectedGraph.model_json_schema()
    schema["description"] = (
        "A canonical simple undirected graph with at most "
        f"{MAX_CHROMATIC_CERTIFICATE_VERTICES} vertices. Vertex order is the "
        "authoritative axis for the coloring and rational weights."
    )
    schema["properties"]["vertices"].update(maxItems=MAX_CHROMATIC_CERTIFICATE_VERTICES)
    schema["properties"]["edges"].update(maxItems=MAX_CHROMATIC_CERTIFICATE_EDGES)
    return schema


ChromaticCertificateGraph = Annotated[
    SimpleUndirectedGraph,
    WithJsonSchema(_chromatic_certificate_graph_schema()),
]


@dataclass(frozen=True, slots=True)
class _CertificateEvaluation:
    verdict: Literal["ACCEPTED", "REJECTED"]
    reason: CertificateReason
    weight_sum: Fraction
    certified_lower_bound: int | None = None
    blocking_vertex: str | None = None
    blocking_edge: tuple[str, str] | None = None
    blocking_independent_set: tuple[str, ...] | None = None
    blocking_independent_set_weight: Fraction | None = None


def _source_wire_bytes(
    graph: SimpleUndirectedGraph,
    claimed_chromatic_number: int,
    coloring: tuple[int, ...],
    weights: tuple[CanonicalRational, ...],
) -> int:
    return len(
        encode_strict_json(
            {
                "graph": graph.model_dump(mode="json"),
                "claimed_chromatic_number": claimed_chromatic_number,
                "coloring": list(coloring),
                "weights": [weight.model_dump(mode="json") for weight in weights],
            }
        )
    )


def _intermediate_digit_bound(weights: tuple[CanonicalRational, ...]) -> int:
    """Bound the common denominator, scaled weights, and every subset sum."""
    if not weights:
        return 1
    unique_denominators = {weight.den for weight in weights}
    denominator_product_digits = sum(len(value) for value in unique_denominators)
    scaled_digits = max(
        len(weight.num.lstrip("-")) + denominator_product_digits - len(weight.den)
        for weight in weights
    )
    # A sum of n signed scaled integers gains at most digits(n), with one
    # further digit reserved for a sign/carry in the conservative bound.
    return max(1, scaled_digits + len(str(max(1, len(weights)))) + 1)


def _estimated_digit_work(
    graph: SimpleUndirectedGraph,
    intermediate_digits: int,
) -> int:
    order = len(graph.vertices)
    subset_states = 1 << order
    arithmetic_setup = 6 * order * intermediate_digits * intermediate_digits
    subset_replay = subset_states * (1 + 2 * intermediate_digits)
    graph_setup = order + 3 * len(graph.edges)
    return (
        CHROMATIC_CERTIFICATE_REPLAY_PASSES
        * (arithmetic_setup + subset_replay + graph_setup)
        + _RESULT_RATIONAL_VALIDATION_DIGIT_WORK
    )


def _require_bounded_sources(
    graph: SimpleUndirectedGraph,
    claimed_chromatic_number: int,
    coloring: tuple[int, ...],
    weights: tuple[CanonicalRational, ...],
) -> None:
    order = len(graph.vertices)
    if order > MAX_CHROMATIC_CERTIFICATE_VERTICES:
        raise PydanticCustomError(
            "graph.chromatic_number_certificate_checking_supports_at_most",
            "chromatic-number certificate checking supports at most "
            f"{MAX_CHROMATIC_CERTIFICATE_VERTICES} vertices",
        )
    if len(graph.edges) > MAX_CHROMATIC_CERTIFICATE_EDGES:
        raise PydanticCustomError(
            "graph.chromatic_number_certificate_checking_supports_at_most",
            "chromatic-number certificate checking supports at most "
            f"{MAX_CHROMATIC_CERTIFICATE_EDGES} edges",
        )
    subset_states = 1 << order
    if subset_states > MAX_CHROMATIC_CERTIFICATE_SUBSET_STATES:
        raise PydanticCustomError(
            "graph.chromatic_number_certificate_independent_set_replay_exceeds",
            "chromatic-number certificate independent-set replay exceeds the "
            f"{MAX_CHROMATIC_CERTIFICATE_SUBSET_STATES}-subset bound",
        )
    if len(coloring) != order:
        raise PydanticCustomError(
            "graph.coloring_must_assign_one_color_per_graph_vertex",
            "coloring must assign one color per graph vertex",
        )
    if len(weights) != order:
        raise PydanticCustomError(
            "graph.weights_must_assign_one_exact_rational_per_graph",
            "weights must assign one exact rational per graph vertex",
        )
    for weight in weights:
        require_bounded_rational(
            weight,
            max_digits=MAX_CHROMATIC_CERTIFICATE_RATIONAL_DIGITS,
            label="chromatic-number certificate weight",
        )

    source_bytes = _source_wire_bytes(
        graph,
        claimed_chromatic_number,
        coloring,
        weights,
    )
    limits = CanonicalLimits()
    if source_bytes > limits.max_input_bytes:
        raise PydanticCustomError(
            "graph.chromatic_number_certificate_source_exceeds_canonical_input",
            "chromatic-number certificate source exceeds the canonical input limit",
        )

    intermediate_digits = _intermediate_digit_bound(weights)
    digit_work = _estimated_digit_work(graph, intermediate_digits)
    if digit_work > MAX_CHROMATIC_CERTIFICATE_DIGIT_WORK:
        raise PydanticCustomError(
            "graph.chromatic_number_certificate_exact_replay_work_exceeds",
            "chromatic-number certificate exact replay work exceeds the "
            f"{MAX_CHROMATIC_CERTIFICATE_DIGIT_WORK} decimal-digit-operation bound",
        )

    label_wire_bytes = sum(
        len(encode_strict_json(vertex)) + 1 for vertex in graph.vertices
    )
    estimated_result_bytes = (
        source_bytes
        + label_wire_bytes
        + 4 * intermediate_digits
        + _RESULT_ENVELOPE_RESERVE_BYTES
    )
    if estimated_result_bytes > limits.max_output_bytes:
        raise PydanticCustomError(
            "graph.chromatic_number_certificate_retained_result_would_exceed",
            "chromatic-number certificate retained result would exceed the "
            f"{limits.max_output_bytes}-byte canonical output limit",
        )


def _scaled_weights(
    weights: tuple[CanonicalRational, ...],
) -> tuple[int, tuple[int, ...]]:
    ratios = tuple(weight.as_integer_ratio() for weight in weights)
    common_denominator = 1
    for _, denominator in ratios:
        common_denominator = lcm(common_denominator, denominator)
    return common_denominator, tuple(
        numerator * (common_denominator // denominator)
        for numerator, denominator in ratios
    )


def _rejected(
    reason: CertificateReason,
    weight_sum: Fraction,
    *,
    certified_lower_bound: int | None = None,
    blocking_vertex: str | None = None,
    blocking_edge: tuple[str, str] | None = None,
    blocking_independent_set: tuple[str, ...] | None = None,
    blocking_independent_set_weight: Fraction | None = None,
) -> _CertificateEvaluation:
    return _CertificateEvaluation(
        verdict="REJECTED",
        reason=reason,
        weight_sum=weight_sum,
        certified_lower_bound=certified_lower_bound,
        blocking_vertex=blocking_vertex,
        blocking_edge=blocking_edge,
        blocking_independent_set=blocking_independent_set,
        blocking_independent_set_weight=blocking_independent_set_weight,
    )


def _evaluate_chromatic_number_certificate(
    graph: SimpleUndirectedGraph,
    claimed_chromatic_number: int,
    coloring: tuple[int, ...],
    weights: tuple[CanonicalRational, ...],
) -> _CertificateEvaluation:
    """Replay the proper-coloring upper bound and fractional-clique lower bound."""
    common_denominator, scaled_weights = _scaled_weights(weights)
    weight_sum = Fraction(sum(scaled_weights), common_denominator)
    order = len(graph.vertices)

    if (order == 0 and claimed_chromatic_number != 0) or (
        order > 0 and not 1 <= claimed_chromatic_number <= order
    ):
        return _rejected("CLAIM_OUT_OF_RANGE", weight_sum)

    for index, color in enumerate(coloring):
        if color >= claimed_chromatic_number:
            return _rejected(
                "COLOR_OUT_OF_PALETTE",
                weight_sum,
                blocking_vertex=graph.vertices[index],
            )

    vertex_index = {vertex: index for index, vertex in enumerate(graph.vertices)}
    for left, right in sorted(graph.edges):
        if coloring[vertex_index[left]] == coloring[vertex_index[right]]:
            return _rejected(
                "MONOCHROMATIC_EDGE",
                weight_sum,
                blocking_edge=(left, right),
            )

    for index, scaled_weight in enumerate(scaled_weights):
        if scaled_weight < 0:
            return _rejected(
                "NEGATIVE_WEIGHT",
                weight_sum,
                blocking_vertex=graph.vertices[index],
            )

    adjacency = [0] * order
    for left, right in graph.edges:
        left_index = vertex_index[left]
        right_index = vertex_index[right]
        adjacency[left_index] |= 1 << right_index
        adjacency[right_index] |= 1 << left_index

    subset_states = 1 << order
    independent = bytearray(subset_states)
    independent[0] = 1
    for mask in range(1, subset_states):
        bit = mask & -mask
        vertex = bit.bit_length() - 1
        remainder = mask ^ bit
        independent[mask] = independent[remainder] and not (
            adjacency[vertex] & remainder
        )

    subset_weight = 0
    previous_mask = 0
    for step in range(1, subset_states):
        mask = step ^ (step >> 1)
        changed_bit = mask ^ previous_mask
        vertex = changed_bit.bit_length() - 1
        if mask & changed_bit:
            subset_weight += scaled_weights[vertex]
        else:
            subset_weight -= scaled_weights[vertex]
        if independent[mask] and subset_weight > common_denominator:
            blocking_set = tuple(
                label
                for index, label in enumerate(graph.vertices)
                if mask & (1 << index)
            )
            return _rejected(
                "INDEPENDENT_SET_OVERWEIGHT",
                weight_sum,
                blocking_independent_set=blocking_set,
                blocking_independent_set_weight=Fraction(
                    subset_weight, common_denominator
                ),
            )
        previous_mask = mask

    certified_lower_bound = ceil(weight_sum)
    if certified_lower_bound < claimed_chromatic_number:
        return _rejected(
            "LOWER_BOUND_BELOW_CLAIM",
            weight_sum,
            certified_lower_bound=certified_lower_bound,
        )
    return _CertificateEvaluation(
        verdict="ACCEPTED",
        reason="ACCEPTED",
        weight_sum=weight_sum,
        certified_lower_bound=certified_lower_bound,
    )


class ChromaticNumberCertificateCheckRequest(StrictModel):
    """Check an exact upper/lower certificate for ``chi(graph) = k``.

    The coloring and weights are aligned to ``graph.vertices``.  Feasible
    nonnegative weights put total weight at most one on every independent set;
    therefore every proper coloring has at least ``ceil(sum(weights))`` color
    classes.  The full independent-set family is replayed exhaustively.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Check a claimed exact chromatic number from a proper coloring "
                "upper witness and nonnegative rational vertex weights whose "
                "sum on every independent set is at most one. The coloring and "
                "weights are aligned to graph.vertices order. Admission bounds "
                "the full 2^n independent-set replay, exact rational growth, "
                "retained-source bytes, and producer plus result-validation work."
            )
        }
    )

    graph: ChromaticCertificateGraph
    claimed_chromatic_number: StrictInt = Field(
        ge=0,
        le=MAX_CHROMATIC_CERTIFICATE_VERTICES,
        description=(
            "Claimed value k. The empty graph may claim 0; a nonempty graph "
            "must claim a value in 1..order or receives a typed rejection."
        ),
    )
    coloring: tuple[CertificateColor, ...] = Field(
        max_length=MAX_CHROMATIC_CERTIFICATE_VERTICES,
        description=(
            "Candidate vertex colors aligned to graph.vertices order; each "
            "entry is structurally bounded to 0..20, while membership in the "
            "claimed palette 0..k-1 is checked as mathematical evidence."
        ),
    )
    weights: tuple[CertificateWeight, ...] = Field(
        max_length=MAX_CHROMATIC_CERTIFICATE_VERTICES,
        description=(
            "Exact rational weights aligned to graph.vertices order. A valid "
            "lower certificate is nonnegative and gives every independent set "
            "weight at most one. Each component has at most 64 digits."
        ),
    )

    @model_validator(mode="after")
    def require_bounded_sources(self) -> Self:
        _require_bounded_sources(
            self.graph,
            self.claimed_chromatic_number,
            self.coloring,
            self.weights,
        )
        return self


class ChromaticNumberCertificateCheckResult(StrictModel):
    """Source-bound acceptance or concrete rejection of one optimum certificate."""

    graph: ChromaticCertificateGraph
    claimed_chromatic_number: StrictInt = Field(
        ge=0,
        le=MAX_CHROMATIC_CERTIFICATE_VERTICES,
        description="Retained claimed value k.",
    )
    coloring: tuple[CertificateColor, ...] = Field(
        max_length=MAX_CHROMATIC_CERTIFICATE_VERTICES,
        description="Retained coloring in graph.vertices order.",
    )
    weights: tuple[CertificateWeight, ...] = Field(
        max_length=MAX_CHROMATIC_CERTIFICATE_VERTICES,
        description=(
            "Retained exact rational weights in graph.vertices order; each "
            "component has at most 64 digits."
        ),
    )
    verdict: Literal["ACCEPTED", "REJECTED"] = Field(
        description=(
            "ACCEPTED exactly when the supplied upper and lower witnesses prove "
            "the claim. REJECTED means this evidence does not prove the claim; "
            "it does not by itself refute the claimed chromatic number."
        )
    )
    reason: CertificateReason = Field(
        description="Deterministic first failed certificate condition."
    )
    weight_sum: CertificateDerivedRational = Field(
        description=(
            "Exact sum of the retained vertex weights, with at most "
            f"{MAX_CHROMATIC_CERTIFICATE_DERIVED_RATIONAL_DIGITS} digits per "
            "component."
        )
    )
    certified_lower_bound: StrictInt | None = Field(
        default=None,
        ge=0,
        le=MAX_CHROMATIC_CERTIFICATE_VERTICES,
        description=(
            "ceil(weight_sum), present after all independent-set constraints "
            "have been replayed."
        ),
    )
    blocking_vertex: str | None = Field(
        default=None,
        description="First vertex with an out-of-palette color or negative weight.",
    )
    blocking_edge: tuple[str, str] | None = Field(
        default=None,
        description="First canonical edge whose endpoints have the same color.",
    )
    blocking_independent_set: tuple[str, ...] | None = Field(
        default=None,
        max_length=MAX_CHROMATIC_CERTIFICATE_VERTICES,
        description=(
            "First overweight independent set in binary-reflected Gray-code "
            "order, listed in graph.vertices axis order."
        ),
    )
    blocking_independent_set_weight: CertificateDerivedRational | None = Field(
        default=None,
        description=(
            "Exact weight of the returned overweight independent set, with at "
            "most "
            f"{MAX_CHROMATIC_CERTIFICATE_DERIVED_RATIONAL_DIGITS} digits per "
            "component."
        ),
    )

    @classmethod
    def _from_kernel(
        cls,
        *,
        graph: SimpleUndirectedGraph,
        claimed_chromatic_number: int,
        coloring: tuple[int, ...],
        weights: tuple[CanonicalRational, ...],
        evaluation: _CertificateEvaluation,
    ) -> Self:
        """Construct a result whose exact certificate relation was established."""

        return cls.model_construct(
            graph=graph,
            claimed_chromatic_number=claimed_chromatic_number,
            coloring=coloring,
            weights=weights,
            verdict=evaluation.verdict,
            reason=evaluation.reason,
            weight_sum=CanonicalRational.from_fraction(evaluation.weight_sum),
            certified_lower_bound=evaluation.certified_lower_bound,
            blocking_vertex=evaluation.blocking_vertex,
            blocking_edge=evaluation.blocking_edge,
            blocking_independent_set=evaluation.blocking_independent_set,
            blocking_independent_set_weight=(
                None
                if evaluation.blocking_independent_set_weight is None
                else CanonicalRational.from_fraction(
                    evaluation.blocking_independent_set_weight
                )
            ),
        )

    @model_validator(mode="after")
    def require_structural_result_bounds(self) -> Self:
        _require_bounded_sources(
            self.graph,
            self.claimed_chromatic_number,
            self.coloring,
            self.weights,
        )
        return self


__all__ = [
    "MAX_CHROMATIC_CERTIFICATE_DERIVED_RATIONAL_DIGITS",
    "MAX_CHROMATIC_CERTIFICATE_EDGES",
    "MAX_CHROMATIC_CERTIFICATE_RATIONAL_DIGITS",
    "MAX_CHROMATIC_CERTIFICATE_VERTICES",
    "ChromaticNumberCertificateCheckRequest",
    "ChromaticNumberCertificateCheckResult",
]
