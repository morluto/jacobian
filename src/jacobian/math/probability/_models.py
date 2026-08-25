"""Bounded exact finite-probability contracts."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    CanonicalInteger,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian._models import StrictModel
from jacobian.canonical import (
    canonicalize_json,
    encode_strict_json,
    format_canonical_integer,
)
from jacobian.math.graphs.directed._models import DirectedGraph
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.probability._gaussian_moments import gaussian_univariate_moment


def _validation_error(message: str) -> PydanticCustomError:
    return PydanticCustomError("probability.model_invariant", message)


MAX_FINITE_DISTRIBUTION_ATOMS = 256
MAX_FINITE_CONVOLUTION_PAIRS = 4096
MAX_INPUT_RATIONAL_DIGITS = 128
MAX_RESULT_RATIONAL_DIGITS = 512
MAX_GAUSSIAN_VARIABLES = 16
MAX_GAUSSIAN_POLYNOMIAL_TERMS = 16
MAX_GAUSSIAN_TERM_DEGREE = 8
MAX_GAUSSIAN_MOMENT_ORDER = 16
MAX_GAUSSIAN_EXPANSION_PATHS = 65536
MAX_GAUSSIAN_RESULT_RATIONAL_DIGITS = 4096
MAX_GRAPH_RELIABILITY_VERTICES = 16
MAX_GRAPH_RELIABILITY_EDGES = 12
MAX_GRAPH_RELIABILITY_STATES = 1 << MAX_GRAPH_RELIABILITY_EDGES
MAX_GRAPH_RELIABILITY_LEDGER_BYTES = 9 * 1024 * 1024
# Reference vertex count used only to derive the work budget below; admission
# is bounded by that derived budget, not by this count.
MAX_DIRECTED_BOND_RELIABILITY_VERTICES = 16
MAX_DIRECTED_BOND_RELIABILITY_ARCS = 12
MAX_DIRECTED_BOND_RELIABILITY_STATES = 1 << MAX_DIRECTED_BOND_RELIABILITY_ARCS
MAX_DIRECTED_BOND_RELIABILITY_LEDGER_BYTES = 9 * 1024 * 1024
# One state mass has at most one numerator and denominator factor per arc.
# Summing at most 2**arcs masses can add at most ``arcs`` decimal digits.
MAX_DIRECTED_BOND_RELIABILITY_RATIONAL_DIGITS = (
    MAX_INPUT_RATIONAL_DIGITS * MAX_DIRECTED_BOND_RELIABILITY_ARCS
    + MAX_DIRECTED_BOND_RELIABILITY_ARCS
)
# The producer enumerates every arc subset and result validation replays it.
# Per state and pass it selects open arcs, evaluates every probability, adds
# open arcs to the directed graph, and traverses them: at most four arc visits.
# It also adds every graph vertex, visits vertices during descendants, and
# materializes both the reachable and unreachable partitions: four vertex
# visits.  The producer constructs one complete state record and validation
# compares one; each records or compares at most every open arc and its three
# scalar fields.  The final result also compares a fixed set of aggregate
# fields, charged below.
MAX_DIRECTED_BOND_RELIABILITY_LOGICAL_WORK = (
    2
    * MAX_DIRECTED_BOND_RELIABILITY_STATES
    * (
        5 * MAX_DIRECTED_BOND_RELIABILITY_ARCS
        + 4 * MAX_DIRECTED_BOND_RELIABILITY_VERTICES
        + 3
    )
    + 8
)


def _require_bounded_fraction(
    value: Fraction,
    *,
    max_digits: int,
    label: str,
) -> None:
    if (
        len(format_canonical_integer(abs(value.numerator))) > max_digits
        or len(format_canonical_integer(value.denominator)) > max_digits
    ):
        raise _validation_error(f"{label} exceeds the {max_digits}-digit bound")


def _require_strictly_increasing(
    values: tuple[CanonicalRational, ...],
    *,
    label: str,
) -> tuple[Fraction, ...]:
    fractions = tuple(value.as_fraction() for value in values)
    if any(left >= right for left, right in pairwise(fractions)):
        raise _validation_error(f"{label} must be strictly increasing")
    return fractions


class ExactComplexRational(StrictModel):
    """One exact element of Q(i), encoded without floating-point values."""

    real: CanonicalRational
    imaginary: CanonicalRational

    def as_fractions(self) -> tuple[Fraction, Fraction]:
        return self.real.as_fraction(), self.imaginary.as_fraction()

    @model_validator(mode="after")
    def require_bounded_components(self) -> Self:
        for label, value in (
            ("complex real component", self.real),
            ("complex imaginary component", self.imaginary),
        ):
            require_bounded_rational(
                value,
                max_digits=MAX_GAUSSIAN_RESULT_RATIONAL_DIGITS,
                label=label,
            )
        return self


class GaussianPolynomialTerm(StrictModel):
    coefficient: ExactComplexRational
    exponents: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_GAUSSIAN_VARIABLES,
    )

    @model_validator(mode="after")
    def require_bounded_nonzero_term(self) -> Self:
        if any(
            type(exponent) is not int or exponent < 0 for exponent in self.exponents
        ):
            raise _validation_error(
                "Gaussian polynomial exponents must be nonnegative integers"
            )
        if sum(self.exponents) > MAX_GAUSSIAN_TERM_DEGREE:
            raise _validation_error(
                "Gaussian polynomial term exceeds the "
                f"{MAX_GAUSSIAN_TERM_DEGREE}-degree bound"
            )
        if self.coefficient.as_fractions() == (Fraction(), Fraction()):
            raise _validation_error(
                "Gaussian polynomial terms must have nonzero coefficients"
            )
        return self


class GaussianPolynomial(StrictModel):
    """A canonical sparse polynomial in independent standard real Gaussians."""

    variable_count: StrictInt = Field(ge=1, le=MAX_GAUSSIAN_VARIABLES)
    terms: tuple[GaussianPolynomialTerm, ...] = Field(
        min_length=1,
        max_length=MAX_GAUSSIAN_POLYNOMIAL_TERMS,
        description=(
            "Nonzero sparse terms ordered lexicographically by their complete "
            "exponent vectors, for example [0, 1] before [1, 0]."
        ),
    )

    @model_validator(mode="after")
    def require_canonical_sparse_polynomial(self) -> Self:
        exponents = tuple(term.exponents for term in self.terms)
        if any(len(item) != self.variable_count for item in exponents):
            raise _validation_error(
                "every Gaussian polynomial exponent vector must match variable_count"
            )
        for left, right in pairwise(exponents):
            if left >= right:
                raise _validation_error(
                    "Gaussian polynomial terms must use strictly increasing "
                    "lexicographic exponent-vector order; first offending adjacent "
                    f"pair is {list(left)} then {list(right)}"
                )
        return self


class GaussianPolynomialMomentRequest(StrictModel):
    polynomial: GaussianPolynomial
    order: StrictInt = Field(ge=0, le=MAX_GAUSSIAN_MOMENT_ORDER)

    @model_validator(mode="after")
    def require_bounded_complete_expansion(self) -> Self:
        expansion_paths = len(self.polynomial.terms) ** self.order
        if expansion_paths > MAX_GAUSSIAN_EXPANSION_PATHS:
            raise _validation_error(
                "Gaussian polynomial power exceeds the "
                f"{MAX_GAUSSIAN_EXPANSION_PATHS}-path expansion bound"
            )
        components = tuple(
            component
            for term in self.polynomial.terms
            for component in (term.coefficient.real, term.coefficient.imaginary)
        )
        distinct_denominator_digits = sum(
            len(denominator)
            for denominator in {component.den for component in components}
        )
        maximum_numerator_digits = max(
            len(component.num.lstrip("-")) for component in components
        )
        result_digit_bound = (
            self.order * (distinct_denominator_digits + maximum_numerator_digits)
            + len(str(max(1, expansion_paths)))
            + 64
        )
        if result_digit_bound > MAX_GAUSSIAN_RESULT_RATIONAL_DIGITS:
            raise _validation_error(
                "Gaussian polynomial coefficient denominators can exceed the "
                f"{MAX_GAUSSIAN_RESULT_RATIONAL_DIGITS}-digit result bound"
            )
        return self


class GaussianMomentContraction(StrictModel):
    exponents: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_GAUSSIAN_VARIABLES,
    )
    expanded_coefficient: ExactComplexRational
    variable_moment_factors: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GAUSSIAN_VARIABLES,
    )
    gaussian_moment_factor: CanonicalInteger
    contribution: ExactComplexRational

    @model_validator(mode="after")
    def bind_gaussian_contraction(self) -> Self:
        if len(self.exponents) != len(self.variable_moment_factors):
            raise _validation_error("Gaussian contraction dimensions disagree")
        expected_factors = tuple(
            gaussian_univariate_moment(exponent) for exponent in self.exponents
        )
        actual_factors = tuple(int(value) for value in self.variable_moment_factors)
        if actual_factors != expected_factors:
            raise _validation_error("Gaussian variable moment factors are invalid")
        expected_factor = 1
        for factor in expected_factors:
            expected_factor *= factor
        if int(self.gaussian_moment_factor) != expected_factor:
            raise _validation_error(
                "Gaussian moment factor does not match its variables"
            )
        coefficient = self.expanded_coefficient.as_fractions()
        contribution = self.contribution.as_fractions()
        if contribution != (
            coefficient[0] * expected_factor,
            coefficient[1] * expected_factor,
        ):
            raise _validation_error("Gaussian contraction contribution is invalid")
        return self


class GaussianPolynomialMomentResult(StrictModel):
    order: StrictInt = Field(ge=0, le=MAX_GAUSSIAN_MOMENT_ORDER)
    moment: ExactComplexRational
    expansion_path_count: StrictInt = Field(ge=1, le=MAX_GAUSSIAN_EXPANSION_PATHS)
    expanded_monomial_count: StrictInt = Field(ge=1, le=MAX_GAUSSIAN_EXPANSION_PATHS)
    contractions: tuple[GaussianMomentContraction, ...] = Field(
        min_length=1,
        max_length=MAX_GAUSSIAN_EXPANSION_PATHS,
    )
    gaussian_model: Literal["INDEPENDENT_STANDARD_REAL"] = "INDEPENDENT_STANDARD_REAL"
    completeness: Literal["COMPLETE_BOUNDED_EXPANSION"] = "COMPLETE_BOUNDED_EXPANSION"
    exactness: Literal["EXACT_COMPLEX_RATIONAL"] = "EXACT_COMPLEX_RATIONAL"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"

    @model_validator(mode="after")
    def bind_complete_contraction_ledger(self) -> Self:
        if self.expanded_monomial_count != len(self.contractions):
            raise _validation_error("expanded monomial count does not match the ledger")
        exponents = tuple(item.exponents for item in self.contractions)
        if any(left >= right for left, right in pairwise(exponents)):
            raise _validation_error(
                "Gaussian contractions must use strictly increasing exponent order"
            )
        if any(len(item) != len(exponents[0]) for item in exponents):
            raise _validation_error("Gaussian contraction dimensions disagree")
        total_real = Fraction()
        total_imaginary = Fraction()
        for item in self.contractions:
            real, imaginary = item.contribution.as_fractions()
            total_real += real
            total_imaginary += imaginary
        if self.moment.as_fractions() != (total_real, total_imaginary):
            raise _validation_error(
                "Gaussian polynomial moment does not match its ledger"
            )
        return self


class GraphReliabilityEdgeProbability(StrictModel):
    edge: tuple[str, str]
    open_probability: CanonicalRational

    @model_validator(mode="after")
    def require_canonical_bounded_probability(self) -> Self:
        if len(self.edge) != 2 or self.edge[0] >= self.edge[1]:
            raise _validation_error(
                "reliability edge must contain two ordered vertices"
            )
        require_bounded_rational(
            self.open_probability,
            max_digits=MAX_INPUT_RATIONAL_DIGITS,
            label="graph reliability edge probability",
        )
        if not 0 <= self.open_probability.as_fraction() <= 1:
            raise _validation_error(
                "graph reliability probabilities must lie in [0, 1]"
            )
        return self


class GraphConnectionProbabilityRequest(StrictModel):
    graph: SimpleUndirectedGraph
    edge_probabilities: tuple[GraphReliabilityEdgeProbability, ...] = Field(
        max_length=MAX_GRAPH_RELIABILITY_EDGES
    )
    terminals: tuple[str, str]
    event: Literal["TERMINALS_CONNECTED"] = "TERMINALS_CONNECTED"

    @model_validator(mode="after")
    def require_bounded_fully_weighted_graph(self) -> Self:
        if len(self.graph.vertices) > MAX_GRAPH_RELIABILITY_VERTICES:
            raise _validation_error(
                "graph reliability exceeds the "
                f"{MAX_GRAPH_RELIABILITY_VERTICES}-vertex bound"
            )
        if len(self.graph.edges) > MAX_GRAPH_RELIABILITY_EDGES:
            raise _validation_error(
                "graph reliability exceeds the "
                f"{MAX_GRAPH_RELIABILITY_EDGES}-edge bound"
            )
        if tuple(item.edge for item in self.edge_probabilities) != self.graph.edges:
            raise _validation_error(
                "edge probabilities must cover graph edges in canonical graph order"
            )
        if (
            len(self.terminals) != 2
            or self.terminals[0] == self.terminals[1]
            or any(terminal not in self.graph.vertices for terminal in self.terminals)
        ):
            raise _validation_error(
                "terminals must be two distinct declared graph vertices"
            )
        edge_count = len(self.graph.edges)
        state_count = 1 << edge_count
        repeated_edge_bytes = (
            (1 << (edge_count - 1))
            * sum(len(canonicalize_json(list(edge))) + 1 for edge in self.graph.edges)
            if edge_count
            else 0
        )
        probability_numerator_digits = sum(
            max(
                len(
                    format_canonical_integer(
                        item.open_probability.as_fraction().numerator
                    )
                ),
                len(
                    format_canonical_integer(
                        (1 - item.open_probability.as_fraction()).numerator
                    )
                ),
            )
            for item in self.edge_probabilities
        )
        probability_denominator_digits = sum(
            len(
                format_canonical_integer(
                    item.open_probability.as_fraction().denominator
                )
            )
            for item in self.edge_probabilities
        )
        maximum_state = {
            "state_index": state_count - 1,
            "open_edges": [],
            "terminals_connected": False,
            "state_probability": {
                "num": "9" * max(1, probability_numerator_digits),
                "den": "9" * max(1, probability_denominator_digits),
            },
        }
        estimated_ledger_bytes = (
            repeated_edge_bytes
            + state_count * len(canonicalize_json(maximum_state))
            + 16 * 1024
        )
        if estimated_ledger_bytes > MAX_GRAPH_RELIABILITY_LEDGER_BYTES:
            raise _validation_error(
                "graph reliability request can exceed the complete ledger "
                f"budget of {MAX_GRAPH_RELIABILITY_LEDGER_BYTES} bytes"
            )
        return self


class GraphReliabilityState(StrictModel):
    state_index: StrictInt = Field(ge=0, lt=MAX_GRAPH_RELIABILITY_STATES)
    open_edges: tuple[tuple[str, str], ...] = Field(
        max_length=MAX_GRAPH_RELIABILITY_EDGES
    )
    terminals_connected: bool
    state_probability: CanonicalRational

    @model_validator(mode="after")
    def require_bounded_probability(self) -> Self:
        if not 0 <= self.state_probability.as_fraction() <= 1:
            raise _validation_error(
                "graph reliability state probability must lie in [0, 1]"
            )
        return self


class GraphConnectionProbabilityResult(StrictModel):
    terminals: tuple[str, str]
    connection_probability: CanonicalRational
    edge_count: StrictInt = Field(ge=0, le=MAX_GRAPH_RELIABILITY_EDGES)
    visited_states: StrictInt = Field(ge=1, le=MAX_GRAPH_RELIABILITY_STATES)
    states: tuple[GraphReliabilityState, ...] = Field(
        min_length=1,
        max_length=MAX_GRAPH_RELIABILITY_STATES,
    )
    event: Literal["TERMINALS_CONNECTED"] = "TERMINALS_CONNECTED"
    edge_independence: Literal["INDEPENDENT_BERNOULLI"] = "INDEPENDENT_BERNOULLI"
    enumeration: Literal["COMPLETE_EDGE_SUBSETS"] = "COMPLETE_EDGE_SUBSETS"
    completeness: Literal["COMPLETE"] = "COMPLETE"
    truncated: Literal[False] = False
    termination_reason: Literal["EXHAUSTED"] = "EXHAUSTED"
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"

    @model_validator(mode="after")
    def bind_complete_state_mass(self) -> Self:
        if self.visited_states != 1 << self.edge_count:
            raise _validation_error("visited state count is not the full edge powerset")
        if len(self.states) != self.visited_states:
            raise _validation_error("state ledger length does not match visited states")
        if tuple(item.state_index for item in self.states) != tuple(
            range(self.visited_states)
        ):
            raise _validation_error(
                "state ledger indices must be complete and canonical"
            )
        total = sum(
            (item.state_probability.as_fraction() for item in self.states),
            start=Fraction(),
        )
        connected = sum(
            (
                item.state_probability.as_fraction()
                for item in self.states
                if item.terminals_connected
            ),
            start=Fraction(),
        )
        if total != 1:
            raise _validation_error(
                "graph reliability state probabilities must sum to one"
            )
        if self.connection_probability.as_fraction() != connected:
            raise _validation_error(
                "connection probability does not match connected states"
            )
        return self


class DirectedBondReliabilityArcProbability(StrictModel):
    """The independent open probability attached to one directed arc."""

    arc: tuple[StrictInt, StrictInt]
    open_probability: CanonicalRational

    @model_validator(mode="after")
    def require_bounded_directed_arc_probability(self) -> Self:
        if self.arc[0] == self.arc[1]:
            raise _validation_error("directed reliability arcs must not be self-loops")
        require_bounded_rational(
            self.open_probability,
            max_digits=MAX_INPUT_RATIONAL_DIGITS,
            label="directed bond reliability arc probability",
        )
        if not 0 <= self.open_probability.as_fraction() <= 1:
            raise _validation_error(
                "directed bond reliability probabilities must lie in [0, 1]"
            )
        return self


class DirectedBondConnectionProbabilitySource(StrictModel):
    """One finite directed bond-percolation source in canonical arc order.

    Sources have at most 12 arcs; the derived producer-and-replay work budget
    bounds the vertex count, so sparse sources may declare more vertices.  The
    probability map must contain every graph arc exactly once and is empty for
    an edgeless source, and source/target are distinct declared vertices.
    Input arc rows are normalized to lexicographic arc order before state
    indices and result records are assigned.
    """

    graph: DirectedGraph = Field(
        description=(
            "A directed graph with at most 12 arcs for this "
            "complete-enumeration operation."
        )
    )
    arc_probabilities: tuple[DirectedBondReliabilityArcProbability, ...] = Field(
        max_length=MAX_DIRECTED_BOND_RELIABILITY_ARCS,
        description=(
            "One independent open probability for every graph arc exactly once. "
            "Input rows are accepted in any order and normalized to lexicographic "
            "arc order."
        ),
    )
    source: StrictInt = Field(
        description="A declared source vertex, distinct from target."
    )
    target: StrictInt = Field(
        description="A declared target vertex, distinct from source."
    )

    @model_validator(mode="after")
    def require_bounded_fully_weighted_directed_graph(self) -> Self:
        if len(self.graph.edges) > MAX_DIRECTED_BOND_RELIABILITY_ARCS:
            raise _validation_error(
                "directed bond reliability exceeds the "
                f"{MAX_DIRECTED_BOND_RELIABILITY_ARCS}-arc bound"
            )
        if self.source == self.target or any(
            vertex < 0 or vertex >= self.graph.vertex_count
            for vertex in (self.source, self.target)
        ):
            raise _validation_error(
                "source and target must be distinct declared graph vertices"
            )

        probabilities_by_arc = {
            item.arc: item.open_probability for item in self.arc_probabilities
        }
        if len(probabilities_by_arc) != len(self.arc_probabilities) or frozenset(
            probabilities_by_arc
        ) != frozenset(self.graph.edges):
            raise _validation_error(
                "arc probabilities must contain every directed graph arc exactly once"
            )

        canonical_arcs = tuple(sorted(self.graph.edges))
        object.__setattr__(
            self,
            "graph",
            DirectedGraph(
                vertex_count=self.graph.vertex_count,
                edges=canonical_arcs,
            ),
        )
        object.__setattr__(
            self,
            "arc_probabilities",
            tuple(
                DirectedBondReliabilityArcProbability(
                    arc=arc,
                    open_probability=probabilities_by_arc[arc],
                )
                for arc in canonical_arcs
            ),
        )

        arc_count = len(canonical_arcs)
        state_count = 1 << arc_count
        logical_work = (
            2 * state_count * (5 * arc_count + 4 * self.graph.vertex_count + 3) + 8
        )
        if logical_work > MAX_DIRECTED_BOND_RELIABILITY_LOGICAL_WORK:
            raise _validation_error(
                "directed bond reliability exceeds the complete producer and "
                "replay work budget"
            )
        repeated_arc_bytes = (state_count // 2) * sum(
            len(encode_strict_json(list(arc))) + 1 for arc in canonical_arcs
        )
        # Every state mass multiplies one open or one closed factor per arc,
        # so a state's numerator and denominator digit counts are bounded by
        # the corresponding sums over its selected factors, and each factor's
        # digits occur in exactly half of the powerset states.
        per_arc_numerator_bytes = sum(
            len(format_canonical_integer(item.open_probability.as_fraction().numerator))
            + len(
                format_canonical_integer(
                    (1 - item.open_probability.as_fraction()).numerator
                )
            )
            for item in self.arc_probabilities
        )
        per_arc_denominator_bytes = sum(
            len(
                format_canonical_integer(
                    item.open_probability.as_fraction().denominator
                )
            )
            + len(
                format_canonical_integer(
                    (1 - item.open_probability.as_fraction()).denominator
                )
            )
            for item in self.arc_probabilities
        )
        maximum_state_template = {
            "state_index": 0,
            "open_arcs": [],
            "source_reaches_target": False,
            "state_probability": {"num": "", "den": ""},
        }
        fixed_state_bytes = len(encode_strict_json(maximum_state_template))
        source_bytes = len(
            encode_strict_json(
                {
                    "graph": {
                        "vertex_count": self.graph.vertex_count,
                        "edges": [list(arc) for arc in canonical_arcs],
                    },
                    "arc_probabilities": [
                        {
                            "arc": list(item.arc),
                            "open_probability": item.open_probability.model_dump(),
                        }
                        for item in self.arc_probabilities
                    ],
                    "source": self.source,
                    "target": self.target,
                }
            )
        )
        estimated_ledger_bytes = (
            source_bytes
            + repeated_arc_bytes
            # Each record adds its index digits and at least one numerator
            # and denominator character beyond the empty template, plus one
            # ledger separator byte.
            + state_count * (fixed_state_bytes + len(str(state_count - 1)) + 2 + 1)
            + (state_count // 2) * (per_arc_numerator_bytes + per_arc_denominator_bytes)
            + 16 * 1024
        )
        if estimated_ledger_bytes > MAX_DIRECTED_BOND_RELIABILITY_LEDGER_BYTES:
            raise _validation_error(
                "directed bond reliability request can exceed the complete ledger "
                f"budget of {MAX_DIRECTED_BOND_RELIABILITY_LEDGER_BYTES} bytes"
            )
        return self


class DirectedBondConnectionProbabilityRequest(StrictModel):
    """Compute directed source-to-target bond connection probability.

    The request admits at most 12 arcs, requires one probability for every
    arc exactly once (empty for an edgeless graph), and requires distinct
    declared source and target vertices.  The derived work budget bounds the
    vertex count, so sparse graphs may declare more vertices.  It normalizes
    arc rows lexicographically, so state indices and the source-bound result
    do not depend on input order.
    """

    graph: DirectedGraph = Field(
        description=(
            "A directed graph with at most 12 arcs for this "
            "complete-enumeration operation."
        )
    )
    arc_probabilities: tuple[DirectedBondReliabilityArcProbability, ...] = Field(
        max_length=MAX_DIRECTED_BOND_RELIABILITY_ARCS,
        description=(
            "One independent open probability for every graph arc exactly once. "
            "Input rows are accepted in any order and normalized to lexicographic "
            "arc order."
        ),
    )
    source: StrictInt = Field(
        description="A declared source vertex, distinct from target."
    )
    target: StrictInt = Field(
        description="A declared target vertex, distinct from source."
    )
    event: Literal["DIRECTED_PATH_EXISTS"] = "DIRECTED_PATH_EXISTS"

    @model_validator(mode="after")
    def require_canonical_source(self) -> Self:
        canonical_source = DirectedBondConnectionProbabilitySource(
            graph=self.graph,
            arc_probabilities=self.arc_probabilities,
            source=self.source,
            target=self.target,
        )
        object.__setattr__(self, "graph", canonical_source.graph)
        object.__setattr__(
            self, "arc_probabilities", canonical_source.arc_probabilities
        )
        return self


class DirectedBondReliabilityState(StrictModel):
    """One exact directed arc-subset state from a bond-percolation source."""

    state_index: StrictInt = Field(
        ge=0,
        lt=MAX_DIRECTED_BOND_RELIABILITY_STATES,
    )
    open_arcs: tuple[tuple[StrictInt, StrictInt], ...] = Field(
        max_length=MAX_DIRECTED_BOND_RELIABILITY_ARCS
    )
    source_reaches_target: bool
    state_probability: CanonicalRational

    @model_validator(mode="after")
    def require_bounded_probability(self) -> Self:
        require_bounded_rational(
            self.state_probability,
            max_digits=MAX_DIRECTED_BOND_RELIABILITY_RATIONAL_DIGITS,
            label="directed bond reliability state probability",
        )
        if not 0 <= self.state_probability.as_fraction() <= 1:
            raise _validation_error(
                "directed bond reliability state probability must lie in [0, 1]"
            )
        return self


class DirectedBondConnectionProbabilityResult(StrictModel):
    """An exact, complete, source-bound directed bond reliability result."""

    source: DirectedBondConnectionProbabilitySource
    connection_probability: CanonicalRational
    arc_count: StrictInt = Field(ge=0, le=MAX_DIRECTED_BOND_RELIABILITY_ARCS)
    visited_states: StrictInt = Field(
        ge=1,
        le=MAX_DIRECTED_BOND_RELIABILITY_STATES,
    )
    states: tuple[DirectedBondReliabilityState, ...] = Field(
        min_length=1,
        max_length=MAX_DIRECTED_BOND_RELIABILITY_STATES,
    )
    event: Literal["DIRECTED_PATH_EXISTS"] = "DIRECTED_PATH_EXISTS"
    arc_independence: Literal["INDEPENDENT_BERNOULLI"] = "INDEPENDENT_BERNOULLI"
    enumeration: Literal["COMPLETE_ARC_SUBSETS"] = "COMPLETE_ARC_SUBSETS"
    completeness: Literal["COMPLETE"] = "COMPLETE"
    truncated: Literal[False] = False
    termination_reason: Literal["EXHAUSTED"] = "EXHAUSTED"
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"

    @model_validator(mode="after")
    def bind_to_directed_bond_source(self) -> Self:
        require_bounded_rational(
            self.connection_probability,
            max_digits=MAX_DIRECTED_BOND_RELIABILITY_RATIONAL_DIGITS,
            label="directed bond connection probability",
        )
        from jacobian.math.probability._operations import (
            _directed_bond_connection_probability_data,
        )

        connection_probability, expected_states = (
            _directed_bond_connection_probability_data(self.source)
        )
        if self.arc_count != len(self.source.graph.edges):
            raise _validation_error("arc_count must match the source graph")
        if self.visited_states != 1 << self.arc_count:
            raise _validation_error("visited state count is not the full arc powerset")
        if len(self.states) != self.visited_states:
            raise _validation_error("state ledger length does not match visited states")
        if tuple(item.state_index for item in self.states) != tuple(
            range(self.visited_states)
        ):
            raise _validation_error(
                "state ledger indices must be complete and canonical"
            )
        if len(expected_states) != len(self.states):
            raise _validation_error("state ledger length does not match source replay")
        for state, expected in zip(self.states, expected_states, strict=True):
            open_arcs, reaches_target, state_probability = expected
            if state.open_arcs != open_arcs:
                raise _validation_error("state open arcs do not match source subset")
            if state.source_reaches_target != reaches_target:
                raise _validation_error(
                    "state reachability does not match source subset"
                )
            if state.state_probability.as_fraction() != state_probability:
                raise _validation_error(
                    "state probability does not match source subset"
                )
        if self.connection_probability.as_fraction() != connection_probability:
            raise _validation_error(
                "connection probability does not match source replay"
            )
        return self


class FiniteDistributionAtom(StrictModel):
    value: CanonicalRational
    probability: CanonicalRational

    @model_validator(mode="after")
    def require_bounded_nonnegative_probability(self) -> Self:
        require_bounded_rational(
            self.value,
            max_digits=MAX_RESULT_RATIONAL_DIGITS,
            label="finite-distribution atom",
        )
        require_bounded_rational(
            self.probability,
            max_digits=MAX_RESULT_RATIONAL_DIGITS,
            label="finite-distribution probability",
        )
        if self.probability.as_fraction() < 0:
            raise _validation_error(
                "finite-distribution probabilities must be nonnegative"
            )
        return self


class FiniteRationalDistribution(StrictModel):
    atoms: tuple[FiniteDistributionAtom, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_DISTRIBUTION_ATOMS,
    )

    @model_validator(mode="after")
    def require_canonical_probability_distribution(self) -> Self:
        _require_strictly_increasing(
            tuple(atom.value for atom in self.atoms),
            label="finite-distribution support values",
        )
        if (
            sum(
                (atom.probability.as_fraction() for atom in self.atoms),
                start=Fraction(),
            )
            != 1
        ):
            raise _validation_error(
                "finite-distribution probabilities must sum exactly to 1"
            )
        return self


def require_input_distribution(
    atoms: tuple[FiniteDistributionAtom, ...],
    *,
    require_canonical: bool,
) -> tuple[Fraction, ...]:
    values = tuple(atom.value.as_fraction() for atom in atoms)
    if len(values) != len(set(values)):
        raise _validation_error("finite-distribution support values must be unique")
    if require_canonical and any(left >= right for left, right in pairwise(values)):
        raise _validation_error(
            "finite-distribution support values must be strictly increasing"
        )
    for atom in atoms:
        require_bounded_rational(
            atom.value,
            max_digits=MAX_INPUT_RATIONAL_DIGITS,
            label="finite-distribution input atom",
        )
        require_bounded_rational(
            atom.probability,
            max_digits=MAX_INPUT_RATIONAL_DIGITS,
            label="finite-distribution input probability",
        )
    if (
        sum(
            (atom.probability.as_fraction() for atom in atoms),
            start=Fraction(),
        )
        != 1
    ):
        raise _validation_error(
            "finite-distribution probabilities must sum exactly to 1"
        )
    return values


class FiniteRawMomentRequest(StrictModel):
    atoms: tuple[FiniteDistributionAtom, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_DISTRIBUTION_ATOMS,
    )
    order: StrictInt = Field(ge=0, le=128)

    @model_validator(mode="after")
    def require_probability_distribution(self) -> Self:
        require_input_distribution(self.atoms, require_canonical=False)
        return self


class FiniteRawMomentContribution(StrictModel):
    value: CanonicalRational
    probability: CanonicalRational
    powered_value: CanonicalRational
    contribution: CanonicalRational


class FiniteRawMomentResult(StrictModel):
    order: StrictInt = Field(ge=0, le=128)
    moment: CanonicalRational
    contributions: tuple[FiniteRawMomentContribution, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_DISTRIBUTION_ATOMS,
    )

    @model_validator(mode="after")
    def bind_exact_contributions(self) -> Self:
        total = Fraction()
        for item in self.contributions:
            expected_power = item.value.as_fraction() ** self.order
            if item.powered_value.as_fraction() != expected_power:
                raise _validation_error("moment powered value does not match its atom")
            expected_contribution = item.probability.as_fraction() * expected_power
            if item.contribution.as_fraction() != expected_contribution:
                raise _validation_error("moment contribution does not match its atom")
            total += expected_contribution
        if self.moment.as_fraction() != total:
            raise _validation_error(
                "moment does not equal the sum of atom contributions"
            )
        return self


class FiniteEventRequest(StrictModel):
    distribution: FiniteRationalDistribution
    event_values: tuple[CanonicalRational, ...] = Field(
        max_length=MAX_FINITE_DISTRIBUTION_ATOMS
    )

    @model_validator(mode="after")
    def require_explicit_support_subset(self) -> Self:
        support = set(
            require_input_distribution(
                self.distribution.atoms,
                require_canonical=True,
            )
        )
        event = _require_strictly_increasing(
            self.event_values,
            label="finite event values",
        )
        for value in self.event_values:
            require_bounded_rational(
                value,
                max_digits=MAX_INPUT_RATIONAL_DIGITS,
                label="finite event value",
            )
        if not set(event).issubset(support):
            raise _validation_error(
                "finite event values must belong to the distribution"
            )
        event_mass = sum(
            (
                atom.probability.as_fraction()
                for atom in self.distribution.atoms
                if atom.value.as_fraction() in set(event)
            ),
            start=Fraction(),
        )
        _require_bounded_fraction(
            event_mass,
            max_digits=MAX_RESULT_RATIONAL_DIGITS,
            label="finite event probability",
        )
        return self


class FiniteConditionRequest(FiniteEventRequest):
    """A finite event known to have positive exact probability."""

    @model_validator(mode="after")
    def require_positive_event_mass(self) -> Self:
        selected = {value.as_fraction() for value in self.event_values}
        mass = sum(
            (
                atom.probability.as_fraction()
                for atom in self.distribution.atoms
                if atom.value.as_fraction() in selected
            ),
            start=Fraction(),
        )
        if mass <= 0:
            raise _validation_error(
                "conditioning requires a positive-mass finite event"
            )
        return self


class FiniteEventProbabilityResult(StrictModel):
    event_probability: CanonicalRational
    selected_atoms: tuple[FiniteDistributionAtom, ...] = Field(
        max_length=MAX_FINITE_DISTRIBUTION_ATOMS
    )

    @model_validator(mode="after")
    def bind_selected_atom_contributions(self) -> Self:
        _require_strictly_increasing(
            tuple(atom.value for atom in self.selected_atoms),
            label="selected finite-event atoms",
        )
        total = sum(
            (atom.probability.as_fraction() for atom in self.selected_atoms),
            start=Fraction(),
        )
        if self.event_probability.as_fraction() != total:
            raise _validation_error(
                "event probability does not equal selected atom mass"
            )
        return self


class FiniteConditionalContribution(StrictModel):
    value: CanonicalRational
    source_probability: CanonicalRational
    conditioned_probability: CanonicalRational

    @model_validator(mode="after")
    def require_bounded_nonnegative_masses(self) -> Self:
        for label, value in (
            ("conditional value", self.value),
            ("conditional source probability", self.source_probability),
            ("conditioned probability", self.conditioned_probability),
        ):
            require_bounded_rational(
                value,
                max_digits=MAX_RESULT_RATIONAL_DIGITS,
                label=label,
            )
        if (
            self.source_probability.as_fraction() < 0
            or self.conditioned_probability.as_fraction() < 0
        ):
            raise _validation_error(
                "conditional contribution masses must be nonnegative"
            )
        return self


class FiniteConditionResult(StrictModel):
    event_probability: CanonicalRational
    distribution: FiniteRationalDistribution
    contributions: tuple[FiniteConditionalContribution, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_DISTRIBUTION_ATOMS,
    )

    @model_validator(mode="after")
    def bind_normalized_contributions(self) -> Self:
        event_probability = self.event_probability.as_fraction()
        if event_probability <= 0:
            raise _validation_error(
                "conditional distribution requires positive event mass"
            )
        values = tuple(item.value for item in self.contributions)
        _require_strictly_increasing(
            values,
            label="conditional contribution values",
        )
        expected_atoms: list[tuple[Fraction, Fraction]] = []
        source_total = Fraction()
        for item in self.contributions:
            source = item.source_probability.as_fraction()
            conditioned = item.conditioned_probability.as_fraction()
            if source < 0 or conditioned != source / event_probability:
                raise _validation_error(
                    "conditioned probability does not match source mass"
                )
            source_total += source
            expected_atoms.append((item.value.as_fraction(), conditioned))
        if source_total != event_probability:
            raise _validation_error("conditional contributions do not equal event mass")
        actual_atoms = [
            (atom.value.as_fraction(), atom.probability.as_fraction())
            for atom in self.distribution.atoms
        ]
        if actual_atoms != expected_atoms:
            raise _validation_error(
                "conditional distribution does not match contributions"
            )
        return self


class FinitePushforwardMapEntry(StrictModel):
    source: CanonicalRational
    target: CanonicalRational


class FinitePushforwardRequest(StrictModel):
    distribution: FiniteRationalDistribution
    mapping: tuple[FinitePushforwardMapEntry, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_DISTRIBUTION_ATOMS,
    )

    @model_validator(mode="after")
    def require_total_canonical_lookup(self) -> Self:
        source_values = require_input_distribution(
            self.distribution.atoms,
            require_canonical=True,
        )
        mapping_sources = tuple(item.source.as_fraction() for item in self.mapping)
        if mapping_sources != source_values:
            raise _validation_error(
                "pushforward mapping must cover each source atom in canonical order"
            )
        aggregated: dict[Fraction, Fraction] = {}
        for atom, item in zip(self.distribution.atoms, self.mapping, strict=True):
            require_bounded_rational(
                item.source,
                max_digits=MAX_INPUT_RATIONAL_DIGITS,
                label="pushforward source",
            )
            require_bounded_rational(
                item.target,
                max_digits=MAX_INPUT_RATIONAL_DIGITS,
                label="pushforward target",
            )
            target = item.target.as_fraction()
            aggregated[target] = (
                aggregated.get(target, Fraction()) + atom.probability.as_fraction()
            )
        for target, probability in aggregated.items():
            _require_bounded_fraction(
                target,
                max_digits=MAX_RESULT_RATIONAL_DIGITS,
                label="pushforward target",
            )
            _require_bounded_fraction(
                probability,
                max_digits=MAX_RESULT_RATIONAL_DIGITS,
                label="pushforward probability",
            )
        return self


class FinitePushforwardContribution(StrictModel):
    source: CanonicalRational
    target: CanonicalRational
    probability: CanonicalRational

    @model_validator(mode="after")
    def require_bounded_nonnegative_mass(self) -> Self:
        for label, value in (
            ("pushforward source", self.source),
            ("pushforward target", self.target),
            ("pushforward probability", self.probability),
        ):
            require_bounded_rational(
                value,
                max_digits=MAX_RESULT_RATIONAL_DIGITS,
                label=label,
            )
        if self.probability.as_fraction() < 0:
            raise _validation_error("pushforward contribution mass must be nonnegative")
        return self


class FinitePushforwardResult(StrictModel):
    distribution: FiniteRationalDistribution
    contributions: tuple[FinitePushforwardContribution, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_DISTRIBUTION_ATOMS,
    )

    @model_validator(mode="after")
    def bind_aggregated_pushforward(self) -> Self:
        _require_strictly_increasing(
            tuple(item.source for item in self.contributions),
            label="pushforward contribution sources",
        )
        aggregated: dict[Fraction, Fraction] = {}
        for item in self.contributions:
            target = item.target.as_fraction()
            probability = item.probability.as_fraction()
            aggregated[target] = aggregated.get(target, Fraction()) + probability
        expected = sorted(aggregated.items())
        actual = [
            (atom.value.as_fraction(), atom.probability.as_fraction())
            for atom in self.distribution.atoms
        ]
        if actual != expected:
            raise _validation_error(
                "pushforward distribution does not match contributions"
            )
        return self


class FiniteConvolutionRequest(StrictModel):
    left: FiniteRationalDistribution
    right: FiniteRationalDistribution

    @model_validator(mode="after")
    def require_bounded_pair_product(self) -> Self:
        require_input_distribution(self.left.atoms, require_canonical=True)
        require_input_distribution(self.right.atoms, require_canonical=True)
        pair_count = len(self.left.atoms) * len(self.right.atoms)
        if pair_count > MAX_FINITE_CONVOLUTION_PAIRS:
            raise _validation_error(
                "finite convolution exceeds the "
                f"{MAX_FINITE_CONVOLUTION_PAIRS}-pair bound"
            )
        aggregated: dict[Fraction, Fraction] = {}
        for left in self.left.atoms:
            for right in self.right.atoms:
                value = left.value.as_fraction() + right.value.as_fraction()
                probability = (
                    left.probability.as_fraction() * right.probability.as_fraction()
                )
                aggregated[value] = aggregated.get(value, Fraction()) + probability
        if len(aggregated) > MAX_FINITE_DISTRIBUTION_ATOMS:
            raise _validation_error(
                "finite convolution exceeds the "
                f"{MAX_FINITE_DISTRIBUTION_ATOMS}-atom output bound"
            )
        for value, probability in aggregated.items():
            _require_bounded_fraction(
                value,
                max_digits=MAX_RESULT_RATIONAL_DIGITS,
                label="convolution atom",
            )
            _require_bounded_fraction(
                probability,
                max_digits=MAX_RESULT_RATIONAL_DIGITS,
                label="convolution probability",
            )
        return self


class FiniteConvolutionContribution(StrictModel):
    left_value: CanonicalRational
    right_value: CanonicalRational
    sum_value: CanonicalRational
    probability: CanonicalRational

    @model_validator(mode="after")
    def require_bounded_nonnegative_mass(self) -> Self:
        for label, value in (
            ("convolution left value", self.left_value),
            ("convolution right value", self.right_value),
            ("convolution sum value", self.sum_value),
            ("convolution probability", self.probability),
        ):
            require_bounded_rational(
                value,
                max_digits=MAX_RESULT_RATIONAL_DIGITS,
                label=label,
            )
        if self.probability.as_fraction() < 0:
            raise _validation_error("convolution contribution mass must be nonnegative")
        return self


class FiniteConvolutionResult(StrictModel):
    distribution: FiniteRationalDistribution
    contributions: tuple[FiniteConvolutionContribution, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_CONVOLUTION_PAIRS,
    )
    independence: Literal["PRODUCT_MEASURE"] = "PRODUCT_MEASURE"

    @model_validator(mode="after")
    def bind_aggregated_pairs(self) -> Self:
        aggregated: dict[Fraction, Fraction] = {}
        previous: tuple[Fraction, Fraction] | None = None
        for item in self.contributions:
            left = item.left_value.as_fraction()
            right = item.right_value.as_fraction()
            pair = (left, right)
            if previous is not None and pair <= previous:
                raise _validation_error(
                    "convolution contributions must use canonical pair order"
                )
            previous = pair
            value = item.sum_value.as_fraction()
            if value != left + right:
                raise _validation_error("convolution sum value does not match its pair")
            probability = item.probability.as_fraction()
            aggregated[value] = aggregated.get(value, Fraction()) + probability
        expected = sorted(aggregated.items())
        actual = [
            (atom.value.as_fraction(), atom.probability.as_fraction())
            for atom in self.distribution.atoms
        ]
        if actual != expected:
            raise _validation_error(
                "convolution distribution does not match pair contributions"
            )
        return self


__all__ = [
    "MAX_DIRECTED_BOND_RELIABILITY_ARCS",
    "MAX_DIRECTED_BOND_RELIABILITY_STATES",
    "MAX_DIRECTED_BOND_RELIABILITY_VERTICES",
    "MAX_FINITE_CONVOLUTION_PAIRS",
    "MAX_FINITE_DISTRIBUTION_ATOMS",
    "MAX_GAUSSIAN_EXPANSION_PATHS",
    "MAX_GAUSSIAN_MOMENT_ORDER",
    "MAX_GAUSSIAN_POLYNOMIAL_TERMS",
    "MAX_GAUSSIAN_TERM_DEGREE",
    "MAX_GAUSSIAN_VARIABLES",
    "MAX_GRAPH_RELIABILITY_EDGES",
    "MAX_GRAPH_RELIABILITY_STATES",
    "MAX_GRAPH_RELIABILITY_VERTICES",
    "DirectedBondConnectionProbabilityRequest",
    "DirectedBondConnectionProbabilityResult",
    "DirectedBondConnectionProbabilitySource",
    "DirectedBondReliabilityArcProbability",
    "DirectedBondReliabilityState",
    "ExactComplexRational",
    "FiniteConditionResult",
    "FiniteConditionalContribution",
    "FiniteConvolutionContribution",
    "FiniteConvolutionRequest",
    "FiniteConvolutionResult",
    "FiniteDistributionAtom",
    "FiniteEventProbabilityResult",
    "FiniteEventRequest",
    "FinitePushforwardContribution",
    "FinitePushforwardMapEntry",
    "FinitePushforwardRequest",
    "FinitePushforwardResult",
    "FiniteRationalDistribution",
    "FiniteRawMomentContribution",
    "FiniteRawMomentRequest",
    "FiniteRawMomentResult",
    "GaussianMomentContraction",
    "GaussianPolynomial",
    "GaussianPolynomialMomentRequest",
    "GaussianPolynomialMomentResult",
    "GaussianPolynomialTerm",
    "GraphConnectionProbabilityRequest",
    "GraphConnectionProbabilityResult",
    "GraphReliabilityEdgeProbability",
    "GraphReliabilityState",
    "require_input_distribution",
]
