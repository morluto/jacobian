"""Bounded exact finite-probability contracts."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian.canonical import canonicalize_json, format_canonical_integer
from jacobian.contracts.exact import (
    CanonicalInteger,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian.contracts.graph_isomorphism import SimpleUndirectedGraph
from jacobian.contracts.results import ContractModel

MAX_FINITE_DISTRIBUTION_ATOMS = 256
MAX_FINITE_CONVOLUTION_PAIRS = 4096
MAX_INPUT_RATIONAL_DIGITS = 128
MAX_RESULT_RATIONAL_DIGITS = 512
MAX_GAUSSIAN_VARIABLES = 8
MAX_GAUSSIAN_POLYNOMIAL_TERMS = 16
MAX_GAUSSIAN_TERM_DEGREE = 8
MAX_GAUSSIAN_MOMENT_ORDER = 16
MAX_GAUSSIAN_EXPANSION_PATHS = 65536
MAX_GAUSSIAN_RESULT_RATIONAL_DIGITS = 4096
MAX_GRAPH_RELIABILITY_VERTICES = 16
MAX_GRAPH_RELIABILITY_EDGES = 12
MAX_GRAPH_RELIABILITY_STATES = 1 << MAX_GRAPH_RELIABILITY_EDGES
MAX_GRAPH_RELIABILITY_LEDGER_BYTES = 9 * 1024 * 1024


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
        raise ValueError(f"{label} exceeds the {max_digits}-digit bound")


def _require_strictly_increasing(
    values: tuple[CanonicalRational, ...],
    *,
    label: str,
) -> tuple[Fraction, ...]:
    fractions = tuple(value.as_fraction() for value in values)
    if any(left >= right for left, right in pairwise(fractions)):
        raise ValueError(f"{label} must be strictly increasing")
    return fractions


def _gaussian_univariate_moment(exponent: int) -> int:
    if exponent % 2:
        return 0
    result = 1
    for factor in range(1, exponent, 2):
        result *= factor
    return result


class ExactComplexRational(ContractModel):
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


class GaussianPolynomialTerm(ContractModel):
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
            raise ValueError(
                "Gaussian polynomial exponents must be nonnegative integers"
            )
        if sum(self.exponents) > MAX_GAUSSIAN_TERM_DEGREE:
            raise ValueError(
                "Gaussian polynomial term exceeds the "
                f"{MAX_GAUSSIAN_TERM_DEGREE}-degree bound"
            )
        if self.coefficient.as_fractions() == (Fraction(), Fraction()):
            raise ValueError("Gaussian polynomial terms must have nonzero coefficients")
        for component in (self.coefficient.real, self.coefficient.imaginary):
            require_bounded_rational(
                component,
                max_digits=MAX_INPUT_RATIONAL_DIGITS,
                label="Gaussian polynomial input coefficient",
            )
        return self


class GaussianPolynomial(ContractModel):
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
            raise ValueError(
                "every Gaussian polynomial exponent vector must match variable_count"
            )
        for left, right in pairwise(exponents):
            if left >= right:
                raise ValueError(
                    "Gaussian polynomial terms must use strictly increasing "
                    "lexicographic exponent-vector order; first offending adjacent "
                    f"pair is {list(left)} then {list(right)}"
                )
        return self


class GaussianPolynomialMomentRequest(ContractModel):
    polynomial: GaussianPolynomial
    order: StrictInt = Field(ge=0, le=MAX_GAUSSIAN_MOMENT_ORDER)

    @model_validator(mode="after")
    def require_bounded_complete_expansion(self) -> Self:
        expansion_paths = len(self.polynomial.terms) ** self.order
        if expansion_paths > MAX_GAUSSIAN_EXPANSION_PATHS:
            raise ValueError(
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
            raise ValueError(
                "Gaussian polynomial coefficient denominators can exceed the "
                f"{MAX_GAUSSIAN_RESULT_RATIONAL_DIGITS}-digit result bound"
            )
        return self


class GaussianMomentContraction(ContractModel):
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
            raise ValueError("Gaussian contraction dimensions disagree")
        expected_factors = tuple(
            _gaussian_univariate_moment(exponent) for exponent in self.exponents
        )
        actual_factors = tuple(int(value) for value in self.variable_moment_factors)
        if actual_factors != expected_factors:
            raise ValueError("Gaussian variable moment factors are invalid")
        expected_factor = 1
        for factor in expected_factors:
            expected_factor *= factor
        if int(self.gaussian_moment_factor) != expected_factor:
            raise ValueError("Gaussian moment factor does not match its variables")
        coefficient = self.expanded_coefficient.as_fractions()
        contribution = self.contribution.as_fractions()
        if contribution != (
            coefficient[0] * expected_factor,
            coefficient[1] * expected_factor,
        ):
            raise ValueError("Gaussian contraction contribution is invalid")
        return self


class GaussianPolynomialMomentResult(ContractModel):
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
    backend: Literal["python-flint"] = "python-flint"
    backend_version: Literal["0.9.0"] = "0.9.0"
    verification: Literal["UNVERIFIED"] = "UNVERIFIED"

    @model_validator(mode="after")
    def bind_complete_contraction_ledger(self) -> Self:
        if self.expanded_monomial_count != len(self.contractions):
            raise ValueError("expanded monomial count does not match the ledger")
        exponents = tuple(item.exponents for item in self.contractions)
        if any(left >= right for left, right in pairwise(exponents)):
            raise ValueError(
                "Gaussian contractions must use strictly increasing exponent order"
            )
        if any(len(item) != len(exponents[0]) for item in exponents):
            raise ValueError("Gaussian contraction dimensions disagree")
        total_real = Fraction()
        total_imaginary = Fraction()
        for item in self.contractions:
            real, imaginary = item.contribution.as_fractions()
            total_real += real
            total_imaginary += imaginary
        if self.moment.as_fractions() != (total_real, total_imaginary):
            raise ValueError("Gaussian polynomial moment does not match its ledger")
        return self


class GraphReliabilityEdgeProbability(ContractModel):
    edge: tuple[str, str]
    open_probability: CanonicalRational

    @model_validator(mode="after")
    def require_canonical_bounded_probability(self) -> Self:
        if len(self.edge) != 2 or self.edge[0] >= self.edge[1]:
            raise ValueError("reliability edge must contain two ordered vertices")
        require_bounded_rational(
            self.open_probability,
            max_digits=MAX_INPUT_RATIONAL_DIGITS,
            label="graph reliability edge probability",
        )
        if not 0 <= self.open_probability.as_fraction() <= 1:
            raise ValueError("graph reliability probabilities must lie in [0, 1]")
        return self


class GraphConnectionProbabilityRequest(ContractModel):
    graph: SimpleUndirectedGraph
    edge_probabilities: tuple[GraphReliabilityEdgeProbability, ...] = Field(
        max_length=MAX_GRAPH_RELIABILITY_EDGES
    )
    terminals: tuple[str, str]
    event: Literal["TERMINALS_CONNECTED"] = "TERMINALS_CONNECTED"

    @model_validator(mode="after")
    def require_bounded_fully_weighted_graph(self) -> Self:
        if len(self.graph.vertices) > MAX_GRAPH_RELIABILITY_VERTICES:
            raise ValueError(
                "graph reliability exceeds the "
                f"{MAX_GRAPH_RELIABILITY_VERTICES}-vertex bound"
            )
        if len(self.graph.edges) > MAX_GRAPH_RELIABILITY_EDGES:
            raise ValueError(
                "graph reliability exceeds the "
                f"{MAX_GRAPH_RELIABILITY_EDGES}-edge bound"
            )
        if tuple(item.edge for item in self.edge_probabilities) != self.graph.edges:
            raise ValueError(
                "edge probabilities must cover graph edges in canonical graph order"
            )
        if (
            len(self.terminals) != 2
            or self.terminals[0] == self.terminals[1]
            or any(terminal not in self.graph.vertices for terminal in self.terminals)
        ):
            raise ValueError("terminals must be two distinct declared graph vertices")
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
            raise ValueError(
                "graph reliability request can exceed the complete ledger "
                f"budget of {MAX_GRAPH_RELIABILITY_LEDGER_BYTES} bytes"
            )
        return self


class GraphReliabilityState(ContractModel):
    state_index: StrictInt = Field(ge=0, lt=MAX_GRAPH_RELIABILITY_STATES)
    open_edges: tuple[tuple[str, str], ...] = Field(
        max_length=MAX_GRAPH_RELIABILITY_EDGES
    )
    terminals_connected: bool
    state_probability: CanonicalRational


class GraphConnectionProbabilityResult(ContractModel):
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
    backend: Literal["python-flint"] = "python-flint"
    backend_version: Literal["0.9.0"] = "0.9.0"
    verification: Literal["UNVERIFIED"] = "UNVERIFIED"

    @model_validator(mode="after")
    def bind_complete_state_mass(self) -> Self:
        if self.visited_states != 1 << self.edge_count:
            raise ValueError("visited state count is not the full edge powerset")
        if len(self.states) != self.visited_states:
            raise ValueError("state ledger length does not match visited states")
        if tuple(item.state_index for item in self.states) != tuple(
            range(self.visited_states)
        ):
            raise ValueError("state ledger indices must be complete and canonical")
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
            raise ValueError("graph reliability state probabilities must sum to one")
        if self.connection_probability.as_fraction() != connected:
            raise ValueError("connection probability does not match connected states")
        return self


class FiniteDistributionAtom(ContractModel):
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
            raise ValueError("finite-distribution probabilities must be nonnegative")
        return self


class FiniteRationalDistribution(ContractModel):
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
            raise ValueError("finite-distribution probabilities must sum exactly to 1")
        return self


def require_input_distribution(
    atoms: tuple[FiniteDistributionAtom, ...],
    *,
    require_canonical: bool,
) -> tuple[Fraction, ...]:
    values = tuple(atom.value.as_fraction() for atom in atoms)
    if len(values) != len(set(values)):
        raise ValueError("finite-distribution support values must be unique")
    if require_canonical and any(left >= right for left, right in pairwise(values)):
        raise ValueError(
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
        raise ValueError("finite-distribution probabilities must sum exactly to 1")
    return values


class FiniteEventRequest(ContractModel):
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
            raise ValueError("finite event values must belong to the distribution")
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


class FiniteEventProbabilityResult(ContractModel):
    event_probability: CanonicalRational
    selected_atoms: tuple[FiniteDistributionAtom, ...] = Field(
        max_length=MAX_FINITE_DISTRIBUTION_ATOMS
    )
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"
    backend: Literal["python-flint"] = "python-flint"
    backend_version: Literal["0.9.0"] = "0.9.0"
    verification: Literal["UNVERIFIED"] = "UNVERIFIED"

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
            raise ValueError("event probability does not equal selected atom mass")
        return self


class FiniteConditionalContribution(ContractModel):
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
            raise ValueError("conditional contribution masses must be nonnegative")
        return self


class FiniteConditionResult(ContractModel):
    event_probability: CanonicalRational
    distribution: FiniteRationalDistribution
    contributions: tuple[FiniteConditionalContribution, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_DISTRIBUTION_ATOMS,
    )
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"
    backend: Literal["python-flint"] = "python-flint"
    backend_version: Literal["0.9.0"] = "0.9.0"
    verification: Literal["UNVERIFIED"] = "UNVERIFIED"

    @model_validator(mode="after")
    def bind_normalized_contributions(self) -> Self:
        event_probability = self.event_probability.as_fraction()
        if event_probability <= 0:
            raise ValueError("conditional distribution requires positive event mass")
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
                raise ValueError("conditioned probability does not match source mass")
            source_total += source
            expected_atoms.append((item.value.as_fraction(), conditioned))
        if source_total != event_probability:
            raise ValueError("conditional contributions do not equal event mass")
        actual_atoms = [
            (atom.value.as_fraction(), atom.probability.as_fraction())
            for atom in self.distribution.atoms
        ]
        if actual_atoms != expected_atoms:
            raise ValueError("conditional distribution does not match contributions")
        return self


class FinitePushforwardMapEntry(ContractModel):
    source: CanonicalRational
    target: CanonicalRational


class FinitePushforwardRequest(ContractModel):
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
            raise ValueError(
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


class FinitePushforwardContribution(ContractModel):
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
            raise ValueError("pushforward contribution mass must be nonnegative")
        return self


class FinitePushforwardResult(ContractModel):
    distribution: FiniteRationalDistribution
    contributions: tuple[FinitePushforwardContribution, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_DISTRIBUTION_ATOMS,
    )
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"
    backend: Literal["python-flint"] = "python-flint"
    backend_version: Literal["0.9.0"] = "0.9.0"
    verification: Literal["UNVERIFIED"] = "UNVERIFIED"

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
            raise ValueError("pushforward distribution does not match contributions")
        return self


class FiniteConvolutionRequest(ContractModel):
    left: FiniteRationalDistribution
    right: FiniteRationalDistribution

    @model_validator(mode="after")
    def require_bounded_pair_product(self) -> Self:
        require_input_distribution(self.left.atoms, require_canonical=True)
        require_input_distribution(self.right.atoms, require_canonical=True)
        pair_count = len(self.left.atoms) * len(self.right.atoms)
        if pair_count > MAX_FINITE_CONVOLUTION_PAIRS:
            raise ValueError(
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
            raise ValueError(
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


class FiniteConvolutionContribution(ContractModel):
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
            raise ValueError("convolution contribution mass must be nonnegative")
        return self


class FiniteConvolutionResult(ContractModel):
    distribution: FiniteRationalDistribution
    contributions: tuple[FiniteConvolutionContribution, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_CONVOLUTION_PAIRS,
    )
    independence: Literal["PRODUCT_MEASURE"] = "PRODUCT_MEASURE"
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"
    backend: Literal["python-flint"] = "python-flint"
    backend_version: Literal["0.9.0"] = "0.9.0"
    verification: Literal["UNVERIFIED"] = "UNVERIFIED"

    @model_validator(mode="after")
    def bind_aggregated_pairs(self) -> Self:
        aggregated: dict[Fraction, Fraction] = {}
        previous: tuple[Fraction, Fraction] | None = None
        for item in self.contributions:
            left = item.left_value.as_fraction()
            right = item.right_value.as_fraction()
            pair = (left, right)
            if previous is not None and pair <= previous:
                raise ValueError(
                    "convolution contributions must use canonical pair order"
                )
            previous = pair
            value = item.sum_value.as_fraction()
            if value != left + right:
                raise ValueError("convolution sum value does not match its pair")
            probability = item.probability.as_fraction()
            aggregated[value] = aggregated.get(value, Fraction()) + probability
        expected = sorted(aggregated.items())
        actual = [
            (atom.value.as_fraction(), atom.probability.as_fraction())
            for atom in self.distribution.atoms
        ]
        if actual != expected:
            raise ValueError(
                "convolution distribution does not match pair contributions"
            )
        return self


__all__ = [
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
