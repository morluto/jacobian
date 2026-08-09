"""Exact finite probability operations backed by Python-FLINT."""

from __future__ import annotations

from fractions import Fraction
from math import lcm
from typing import Any

from jacobian.canonical import format_canonical_integer
from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.probability import (
    ExactComplexRational,
    FiniteConditionalContribution,
    FiniteConditionResult,
    FiniteConvolutionContribution,
    FiniteConvolutionRequest,
    FiniteConvolutionResult,
    FiniteDistributionAtom,
    FiniteEventProbabilityResult,
    FiniteEventRequest,
    FiniteJointLikelihoodRatio,
    FiniteJointTableMutualInformationRequest,
    FiniteJointTableMutualInformationResult,
    FinitePushforwardContribution,
    FinitePushforwardRequest,
    FinitePushforwardResult,
    FiniteRationalDistribution,
    GaussianMomentContraction,
    GaussianPolynomialMomentRequest,
    GaussianPolynomialMomentResult,
    GraphConnectionProbabilityRequest,
    GraphConnectionProbabilityResult,
    GraphReliabilityState,
    MutualInformationLogProductCertificate,
)
from jacobian.contracts.validated_analysis import (
    FiniteRawMomentContribution,
    FiniteRawMomentRequest,
    FiniteRawMomentResult,
)
from jacobian.domains._examples import example
from jacobian.operations import (
    ComputedNotApplicable,
    ComputedOperation,
    ComputedOutcome,
    ComputedSuccess,
)


def _wire(value: Any) -> CanonicalRational:
    return CanonicalRational(
        num=format_canonical_integer(int(value.p)),
        den=format_canonical_integer(int(value.q)),
    )


def _fmpq(value: CanonicalRational) -> Any:
    from flint import fmpq

    fraction = value.as_fraction()
    return fmpq(fraction.numerator, fraction.denominator)


def _complex_wire(value: tuple[Any, Any]) -> ExactComplexRational:
    return ExactComplexRational(real=_wire(value[0]), imaginary=_wire(value[1]))


def _complex_multiply(
    left: tuple[Any, Any],
    right: tuple[Any, Any],
) -> tuple[Any, Any]:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _gaussian_univariate_moment(exponent: int) -> int:
    if exponent % 2:
        return 0
    result = 1
    for factor in range(1, exponent, 2):
        result *= factor
    return result


def _distribution(values: dict[Fraction, Any]) -> FiniteRationalDistribution:
    return FiniteRationalDistribution(
        atoms=tuple(
            FiniteDistributionAtom(
                value=CanonicalRational(
                    num=format_canonical_integer(value.numerator),
                    den=format_canonical_integer(value.denominator),
                ),
                probability=_wire(probability),
            )
            for value, probability in sorted(values.items())
        )
    )


def _raw_moment(
    request: FiniteRawMomentRequest,
) -> ComputedOutcome[FiniteRawMomentResult]:
    from flint import fmpq

    contributions: list[FiniteRawMomentContribution] = []
    total = fmpq(0)
    for atom in request.atoms:
        value = _fmpq(atom.value)
        probability = _fmpq(atom.probability)
        powered = value**request.order
        contribution = probability * powered
        total += contribution
        contributions.append(
            FiniteRawMomentContribution(
                value=atom.value,
                probability=atom.probability,
                powered_value=_wire(powered),
                contribution=_wire(contribution),
            )
        )
    return ComputedSuccess(
        FiniteRawMomentResult(
            order=request.order,
            moment=_wire(total),
            contributions=tuple(contributions),
        )
    )


def _power_exponent(value: int, base: int) -> int | None:
    exponent = 0
    while value > 1 and value % base == 0:
        value //= base
        exponent += 1
    return exponent if value == 1 else None


def _mutual_information(
    request: FiniteJointTableMutualInformationRequest,
) -> ComputedOutcome[FiniteJointTableMutualInformationResult]:
    from flint import fmpq

    table = [[_fmpq(value) for value in row] for row in request.probabilities]
    row_marginals = [sum(row, fmpq(0)) for row in table]
    column_marginals = [
        sum((table[row][column] for row in range(len(table))), fmpq(0))
        for column in range(len(request.column_labels))
    ]
    support: list[FiniteJointLikelihoodRatio] = []
    denominators: list[int] = []
    ratios: list[tuple[Any, Any]] = []
    for row_index, row in enumerate(table):
        for column_index, probability in enumerate(row):
            if probability == 0:
                continue
            product_marginal = row_marginals[row_index] * column_marginals[column_index]
            if product_marginal == 0:
                raise ValueError("positive joint mass has zero marginal support")
            ratio = probability / product_marginal
            denominators.append(int(probability.denom()))
            ratios.append((probability, ratio))
            support.append(
                FiniteJointLikelihoodRatio(
                    row_index=row_index,
                    column_index=column_index,
                    probability=_wire(probability),
                    row_marginal=_wire(row_marginals[row_index]),
                    column_marginal=_wire(column_marginals[column_index]),
                    likelihood_ratio=_wire(ratio),
                )
            )
    scale = lcm(*denominators)
    product = fmpq(1)
    for probability, ratio in ratios:
        exponent = scale * int(probability.numer()) // int(probability.denom())
        product *= ratio**exponent
    if product < 1:
        raise ValueError("mutual-information log product contradicts Gibbs inequality")
    numerator_exponent = _power_exponent(int(product.numer()), request.log_base)
    denominator_exponent = _power_exponent(int(product.denom()), request.log_base)
    exact_value = None
    if numerator_exponent is not None and denominator_exponent is not None:
        exact_value = _wire(fmpq(numerator_exponent - denominator_exponent, scale))
    return ComputedSuccess(
        FiniteJointTableMutualInformationResult(
            row_marginals=tuple(_wire(value) for value in row_marginals),
            column_marginals=tuple(_wire(value) for value in column_marginals),
            positive_support=tuple(support),
            log_base=request.log_base,
            log_product_certificate=MutualInformationLogProductCertificate(
                scale=format_canonical_integer(scale),
                product=_wire(product),
            ),
            exact_value=exact_value,
            sign="ZERO" if product == 1 else "POSITIVE",
        )
    )


def _event_probability(
    request: FiniteEventRequest,
) -> ComputedOutcome[FiniteEventProbabilityResult]:
    from flint import fmpq

    selected_values = {value.as_fraction() for value in request.event_values}
    selected = tuple(
        atom
        for atom in request.distribution.atoms
        if atom.value.as_fraction() in selected_values
    )
    total = fmpq(0)
    for atom in selected:
        total += _fmpq(atom.probability)
    return ComputedSuccess(
        FiniteEventProbabilityResult(
            event_probability=_wire(total),
            selected_atoms=selected,
        )
    )


def _condition(
    request: FiniteEventRequest,
) -> ComputedOutcome[FiniteConditionResult]:
    from flint import fmpq

    selected_values = {value.as_fraction() for value in request.event_values}
    selected = tuple(
        atom
        for atom in request.distribution.atoms
        if atom.value.as_fraction() in selected_values
    )
    event_probability = fmpq(0)
    for atom in selected:
        event_probability += _fmpq(atom.probability)
    if event_probability == 0:
        return ComputedNotApplicable(
            CapabilityDiagnostic(
                code="FINITE_CONDITIONING_ZERO_MASS",
                stage="finite_probability_conditioning",
                message="The selected finite event has exact probability zero.",
                hint="Condition only on an explicitly selected positive-mass event.",
            )
        )
    contributions = tuple(
        FiniteConditionalContribution(
            value=atom.value,
            source_probability=atom.probability,
            conditioned_probability=_wire(_fmpq(atom.probability) / event_probability),
        )
        for atom in selected
    )
    return ComputedSuccess(
        FiniteConditionResult(
            event_probability=_wire(event_probability),
            distribution=FiniteRationalDistribution(
                atoms=tuple(
                    FiniteDistributionAtom(
                        value=item.value,
                        probability=item.conditioned_probability,
                    )
                    for item in contributions
                )
            ),
            contributions=contributions,
        )
    )


def _pushforward(
    request: FinitePushforwardRequest,
) -> ComputedOutcome[FinitePushforwardResult]:
    from flint import fmpq

    aggregated: dict[Fraction, Any] = {}
    contributions: list[FinitePushforwardContribution] = []
    for atom, mapping in zip(
        request.distribution.atoms,
        request.mapping,
        strict=True,
    ):
        target = mapping.target.as_fraction()
        probability = _fmpq(atom.probability)
        aggregated[target] = aggregated.get(target, fmpq(0)) + probability
        contributions.append(
            FinitePushforwardContribution(
                source=atom.value,
                target=mapping.target,
                probability=atom.probability,
            )
        )
    return ComputedSuccess(
        FinitePushforwardResult(
            distribution=_distribution(aggregated),
            contributions=tuple(contributions),
        )
    )


def _convolution(
    request: FiniteConvolutionRequest,
) -> ComputedOutcome[FiniteConvolutionResult]:
    from flint import fmpq

    aggregated: dict[Fraction, Any] = {}
    contributions: list[FiniteConvolutionContribution] = []
    for left in request.left.atoms:
        for right in request.right.atoms:
            sum_value = left.value.as_fraction() + right.value.as_fraction()
            probability = _fmpq(left.probability) * _fmpq(right.probability)
            aggregated[sum_value] = aggregated.get(sum_value, fmpq(0)) + probability
            contributions.append(
                FiniteConvolutionContribution(
                    left_value=left.value,
                    right_value=right.value,
                    sum_value=CanonicalRational(
                        num=format_canonical_integer(sum_value.numerator),
                        den=format_canonical_integer(sum_value.denominator),
                    ),
                    probability=_wire(probability),
                )
            )
    return ComputedSuccess(
        FiniteConvolutionResult(
            distribution=_distribution(aggregated),
            contributions=tuple(contributions),
        )
    )


def _gaussian_polynomial_moment(
    request: GaussianPolynomialMomentRequest,
) -> ComputedOutcome[GaussianPolynomialMomentResult]:
    from flint import fmpq, fmpq_mpoly_ctx

    zero = fmpq(0)
    one = fmpq(1)
    dimension = request.polynomial.variable_count
    base = tuple(
        (
            term.exponents,
            (
                _fmpq(term.coefficient.real),
                _fmpq(term.coefficient.imaginary),
            ),
        )
        for term in request.polynomial.terms
    )

    # Power the complex-coefficient polynomial via FLINT fmpq_mpoly binary
    # exponentiation.  A complex coefficient (a + b i) is represented as a
    # pair of real fmpq_mpoly (real_part, imag_part); complex multiplication
    # is (r1*r2 - i1*i2, r1*i2 + i1*r2).  This replaces the previous
    # path-by-path dictionary expansion with native FLINT arithmetic.
    names = tuple(f"x{index}" for index in range(dimension))
    ctx = fmpq_mpoly_ctx.get(names, "lex")
    real_base = ctx.from_dict(
        {
            term.exponents: _fmpq(term.coefficient.real)
            for term in request.polynomial.terms
        }
    )
    imag_base = ctx.from_dict(
        {
            term.exponents: _fmpq(term.coefficient.imaginary)
            for term in request.polynomial.terms
        }
    )

    def _complex_poly_multiply(
        left: Any,
        right: tuple[Any, Any],
    ) -> tuple[Any, Any]:
        left_real, left_imag = left
        right_real, right_imag = right
        return (
            left_real * right_real - left_imag * right_imag,
            left_real * right_imag + left_imag * right_real,
        )

    constant_one = ctx.from_dict({(0,) * dimension: one})
    constant_zero = ctx.from_dict({})
    powered: tuple[Any, Any] = (constant_one, constant_zero)
    if request.order > 0:
        powered = (real_base, imag_base)
        remaining = request.order - 1
        while remaining:
            if remaining & 1:
                powered = _complex_poly_multiply(powered, (real_base, imag_base))
            remaining >>= 1
            if remaining:
                real_base, imag_base = _complex_poly_multiply(
                    (real_base, imag_base), (real_base, imag_base)
                )

    real_poly, imag_poly = powered
    real_terms = {
        tuple(int(value) for value in exps): coeff for exps, coeff in real_poly.terms()
    }
    imag_terms = {
        tuple(int(value) for value in exps): coeff for exps, coeff in imag_poly.terms()
    }

    expanded: dict[tuple[int, ...], tuple[Any, Any]] = {}
    for exponents in sorted(set(real_terms) | set(imag_terms)):
        real_coeff = real_terms.get(exponents, zero)
        imag_coeff = imag_terms.get(exponents, zero)
        if real_coeff != zero or imag_coeff != zero:
            expanded[exponents] = (real_coeff, imag_coeff)

    contractions: list[GaussianMomentContraction] = []
    total = (zero, zero)
    for exponents, coefficient in sorted(expanded.items()):
        variable_factors = tuple(
            _gaussian_univariate_moment(exponent) for exponent in exponents
        )
        gaussian_factor = 1
        for factor in variable_factors:
            gaussian_factor *= factor
        contribution = (
            coefficient[0] * gaussian_factor,
            coefficient[1] * gaussian_factor,
        )
        total = (total[0] + contribution[0], total[1] + contribution[1])
        contractions.append(
            GaussianMomentContraction(
                exponents=exponents,
                expanded_coefficient=_complex_wire(coefficient),
                variable_moment_factors=tuple(str(value) for value in variable_factors),
                gaussian_moment_factor=str(gaussian_factor),
                contribution=_complex_wire(contribution),
            )
        )

    return ComputedSuccess(
        GaussianPolynomialMomentResult(
            order=request.order,
            moment=_complex_wire(total),
            expansion_path_count=len(base) ** request.order,
            expanded_monomial_count=len(contractions),
            contractions=tuple(contractions),
        )
    )


def _terminals_connected(
    vertices: tuple[str, ...],
    open_edges: tuple[tuple[str, str], ...],
    terminals: tuple[str, str],
) -> bool:
    adjacency: dict[str, set[str]] = {vertex: set() for vertex in vertices}
    for left, right in open_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen = {terminals[0]}
    pending = [terminals[0]]
    while pending:
        vertex = pending.pop()
        for neighbor in adjacency[vertex] - seen:
            if neighbor == terminals[1]:
                return True
            seen.add(neighbor)
            pending.append(neighbor)
    return terminals[1] in seen


def _graph_connection_probability(
    request: GraphConnectionProbabilityRequest,
) -> ComputedOutcome[GraphConnectionProbabilityResult]:
    from flint import fmpq

    probabilities = tuple(
        _fmpq(item.open_probability) for item in request.edge_probabilities
    )
    states: list[GraphReliabilityState] = []
    connection_probability = fmpq(0)
    for state_index in range(1 << len(request.graph.edges)):
        open_edges = tuple(
            edge
            for index, edge in enumerate(request.graph.edges)
            if state_index & (1 << index)
        )
        state_probability = fmpq(1)
        for index, probability in enumerate(probabilities):
            state_probability *= (
                probability if state_index & (1 << index) else 1 - probability
            )
        connected = _terminals_connected(
            request.graph.vertices,
            open_edges,
            request.terminals,
        )
        if connected:
            connection_probability += state_probability
        states.append(
            GraphReliabilityState(
                state_index=state_index,
                open_edges=open_edges,
                terminals_connected=connected,
                state_probability=_wire(state_probability),
            )
        )
    return ComputedSuccess(
        GraphConnectionProbabilityResult(
            terminals=request.terminals,
            connection_probability=_wire(connection_probability),
            edge_count=len(request.graph.edges),
            visited_states=len(states),
            states=tuple(states),
        )
    )


_FAIR_BIT = {
    "atoms": [
        {
            "value": {"num": "0", "den": "1"},
            "probability": {"num": "1", "den": "2"},
        },
        {
            "value": {"num": "1", "den": "1"},
            "probability": {"num": "1", "den": "2"},
        },
    ]
}


FINITE_PROBABILITY_CAPABILITIES = (
    ComputedOperation(
        capability_id="probability.joint.mutual_information.compute",
        title="Exact finite-table mutual information certificate",
        description=(
            "Compute ordered marginals and every positive-support likelihood "
            "ratio for one bounded normalized rational joint table, returning "
            "an exact scaled logarithmic product certificate without floating point."
        ),
        request_model=FiniteJointTableMutualInformationRequest,
        result_model=FiniteJointTableMutualInformationResult,
        implementation=_mutual_information,
        relation_id="probability.joint.mutual_information.relation",
        tags=(
            "probability",
            "information-theory",
            "mutual-information",
            "finite",
            "exact",
            "certificate",
        ),
        invocation_examples=(
            example(
                "perfectly_correlated_fair_bits",
                "Compute exact base-two mutual information for two identical fair bits.",
                {
                    "row_labels": ["0", "1"],
                    "column_labels": ["0", "1"],
                    "probabilities": [
                        [
                            {"num": "1", "den": "2"},
                            {"num": "0", "den": "1"},
                        ],
                        [
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "2"},
                        ],
                    ],
                    "log_base": 2,
                },
            ),
        ),
    ),
    ComputedOperation(
        capability_id="probability.finite_distribution.raw_moment.compute",
        title="Exact finite-distribution raw moment",
        description=(
            "Compute one bounded raw moment of a normalized finite exact "
            "rational distribution, preserving every atom contribution."
        ),
        request_model=FiniteRawMomentRequest,
        result_model=FiniteRawMomentResult,
        implementation=_raw_moment,
        relation_id="probability.finite_distribution.raw_moment.relation",
        tags=("probability", "moment", "finite", "exact", "python-flint"),
        invocation_examples=(
            example(
                "fair_bit_second_moment",
                "Compute the second raw moment of a fair distribution on 0 and 1.",
                {
                    "atoms": _FAIR_BIT["atoms"],
                    "order": 2,
                },
            ),
        ),
    ),
    ComputedOperation(
        capability_id="probability.finite_distribution.event_probability.compute",
        title="Exact finite-event probability",
        description=(
            "Sum the exact mass of one explicit subset of a canonical finite "
            "rational distribution and preserve every selected atom."
        ),
        request_model=FiniteEventRequest,
        result_model=FiniteEventProbabilityResult,
        implementation=_event_probability,
        relation_id="probability.finite_distribution.event_probability.relation",
        tags=("probability", "event", "finite", "exact", "python-flint"),
        invocation_examples=(
            example(
                "fair_bit_is_one",
                "Compute the exact probability that a fair bit equals one.",
                {
                    "distribution": _FAIR_BIT,
                    "event_values": [{"num": "1", "den": "1"}],
                },
            ),
        ),
    ),
    ComputedOperation(
        capability_id="probability.finite_distribution.condition.compute",
        title="Condition an exact finite distribution",
        description=(
            "Normalize one explicit positive-mass event of a canonical finite "
            "rational distribution, preserving each source contribution."
        ),
        request_model=FiniteEventRequest,
        result_model=FiniteConditionResult,
        implementation=_condition,
        relation_id="probability.finite_distribution.condition.relation",
        tags=("probability", "conditioning", "finite", "exact", "python-flint"),
        invocation_examples=(
            example(
                "fair_bit_given_one",
                "Condition a fair bit on the positive-mass event that it equals one.",
                {
                    "distribution": _FAIR_BIT,
                    "event_values": [{"num": "1", "den": "1"}],
                },
            ),
        ),
    ),
    ComputedOperation(
        capability_id="probability.finite_distribution.pushforward.compute",
        title="Push forward an exact finite distribution",
        description=(
            "Apply one explicit total rational lookup map and exactly aggregate "
            "all source masses with the same target."
        ),
        request_model=FinitePushforwardRequest,
        result_model=FinitePushforwardResult,
        implementation=_pushforward,
        relation_id="probability.finite_distribution.pushforward.relation",
        tags=("probability", "pushforward", "finite", "exact", "python-flint"),
        invocation_examples=(
            example(
                "collapse_fair_bit",
                "Map both atoms of a fair bit to one exact target.",
                {
                    "distribution": _FAIR_BIT,
                    "mapping": [
                        {
                            "source": {"num": "0", "den": "1"},
                            "target": {"num": "0", "den": "1"},
                        },
                        {
                            "source": {"num": "1", "den": "1"},
                            "target": {"num": "0", "den": "1"},
                        },
                    ],
                },
            ),
        ),
    ),
    ComputedOperation(
        capability_id="probability.finite_distribution.convolution.compute",
        title="Convolve two exact finite distributions",
        description=(
            "Compute the bounded product-measure distribution of the sum of "
            "two independent finite rational random variables."
        ),
        request_model=FiniteConvolutionRequest,
        result_model=FiniteConvolutionResult,
        implementation=_convolution,
        relation_id="probability.finite_distribution.convolution.relation",
        tags=(
            "probability",
            "convolution",
            "independence",
            "finite",
            "exact",
            "python-flint",
        ),
        invocation_examples=(
            example(
                "two_fair_bits",
                "Compute the exact distribution of the sum of two fair bits.",
                {"left": _FAIR_BIT, "right": _FAIR_BIT},
            ),
        ),
    ),
    ComputedOperation(
        capability_id="probability.gaussian_polynomial.moment.compute",
        title="Exact bounded Gaussian polynomial moment",
        description=(
            "Compute one fixed-order exact moment of a bounded sparse complex-"
            "rational polynomial in independent standard real Gaussian variables, "
            "preserving the complete coefficient-contraction ledger. This does not "
            "establish an identity for every order."
        ),
        request_model=GaussianPolynomialMomentRequest,
        result_model=GaussianPolynomialMomentResult,
        implementation=_gaussian_polynomial_moment,
        relation_id="probability.gaussian_polynomial.moment.relation",
        tags=(
            "probability",
            "Gaussian",
            "polynomial",
            "moment",
            "Wick",
            "Isserlis",
            "exact",
            "bounded",
            "python-flint",
        ),
        invocation_examples=(
            example(
                "sum_of_two_gaussians_second_moment",
                "Compute E[(X_1 + X_2)^2] for independent standard real Gaussians.",
                {
                    "polynomial": {
                        "variable_count": 2,
                        "terms": [
                            {
                                "coefficient": {
                                    "real": {"num": "1", "den": "1"},
                                    "imaginary": {"num": "0", "den": "1"},
                                },
                                "exponents": [0, 1],
                            },
                            {
                                "coefficient": {
                                    "real": {"num": "1", "den": "1"},
                                    "imaginary": {"num": "0", "den": "1"},
                                },
                                "exponents": [1, 0],
                            },
                        ],
                    },
                    "order": 2,
                },
            ),
        ),
    ),
    ComputedOperation(
        capability_id="probability.graph_reliability.connection_probability.compute",
        title="Exact small-graph terminal connection probability",
        description=(
            "Compute the exact probability that two explicit terminals are "
            "connected in one bounded undirected graph with independent rational "
            "edge-open probabilities, preserving the complete edge-subset ledger."
        ),
        request_model=GraphConnectionProbabilityRequest,
        result_model=GraphConnectionProbabilityResult,
        implementation=_graph_connection_probability,
        relation_id="probability.graph_reliability.connection_probability.relation",
        tags=(
            "probability",
            "graph",
            "reliability",
            "percolation",
            "connection",
            "terminals",
            "exact",
            "bounded",
            "python-flint",
        ),
        invocation_examples=(
            example(
                "triangle_terminal_reliability",
                "Compute the exact terminal connection probability in a fair-edge triangle.",
                {
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["a", "c"], ["b", "c"]],
                    },
                    "edge_probabilities": [
                        {
                            "edge": ["a", "b"],
                            "open_probability": {"num": "1", "den": "2"},
                        },
                        {
                            "edge": ["a", "c"],
                            "open_probability": {"num": "1", "den": "2"},
                        },
                        {
                            "edge": ["b", "c"],
                            "open_probability": {"num": "1", "den": "2"},
                        },
                    ],
                    "terminals": ["a", "c"],
                },
            ),
        ),
    ),
)

__all__ = ["FINITE_PROBABILITY_CAPABILITIES"]
