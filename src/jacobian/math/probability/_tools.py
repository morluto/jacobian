"""Exact finite-probability operation declarations."""

from __future__ import annotations

from typing import Any

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.probability import operations as native
from jacobian.math.probability._directed_bond_reliability import (
    DIRECTED_BOND_CONNECTION_PROBABILITY_OPERATION,
)
from jacobian.math.probability._distribution import (
    FiniteConditionRequest,
    FiniteConditionResult,
    FiniteConvolutionRequest,
    FiniteConvolutionResult,
    FiniteEventProbabilityResult,
    FiniteEventRequest,
    FinitePushforwardRequest,
    FinitePushforwardResult,
    FiniteRawMomentRequest,
    FiniteRawMomentResult,
)
from jacobian.math.probability._gaussian import (
    GaussianPolynomialMomentResult,
)
from jacobian.math.probability._gaussian_inputs import (
    CanonicalGaussianPolynomialMomentRequest,
)
from jacobian.math.probability._graph_connection_probability import (
    GRAPH_CONNECTION_PROBABILITY_OPERATION,
)
from jacobian.math.probability._local_lemma import ASYMMETRIC_LOCAL_LEMMA_OPERATION


def _event_probability(
    request: FiniteEventRequest,
) -> FiniteEventProbabilityResult:
    return native.event_probability(request.distribution, request.event_values)


def _raw_moment(request: FiniteRawMomentRequest) -> FiniteRawMomentResult:
    return native.raw_moment(request.atoms, request.order)


def _condition(request: FiniteConditionRequest) -> FiniteConditionResult:
    return native.condition(request.distribution, request.event_values)


def _pushforward(request: FinitePushforwardRequest) -> FinitePushforwardResult:
    return native.pushforward(request.distribution, request.mapping)


def _convolution(request: FiniteConvolutionRequest) -> FiniteConvolutionResult:
    return native.convolution(request.left, request.right)


def _gaussian_polynomial_moment(
    request: CanonicalGaussianPolynomialMomentRequest,
) -> GaussianPolynomialMomentResult:
    return native.gaussian_polynomial_moment(request.polynomial, request.order)


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
    ],
}

_FAIR_DIE_3 = {
    "atoms": [
        {
            "value": {"num": "0", "den": "1"},
            "probability": {"num": "1", "den": "3"},
        },
        {
            "value": {"num": "1", "den": "1"},
            "probability": {"num": "1", "den": "3"},
        },
        {
            "value": {"num": "2", "den": "1"},
            "probability": {"num": "1", "den": "3"},
        },
    ],
}

FINITE_PROBABILITY_OPERATIONS = (
    MathTool(
        operation_id="probability.finite_distribution.event_probability.compute",
        title="Exact finite-event probability",
        description=(
            "Compute the exact probability of a finite event selected from a "
            "canonical finite rational distribution, retaining the selected atoms."
        ),
        request_type=FiniteEventRequest,
        result_type=FiniteEventProbabilityResult,
        run=_event_probability,
        tags=("probability", "event", "finite", "exact", "python-flint"),
        examples=(
            example(
                "fair_bit_event",
                "Compute the probability that a fair bit equals one.",
                {
                    "distribution": _FAIR_BIT,
                    "event_values": [{"num": "1", "den": "1"}],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="probability.finite_distribution.raw_moment.compute",
        title="Exact finite-distribution raw moment",
        description=(
            "Compute one bounded raw moment of a normalized finite exact "
            "rational distribution, preserving every atom contribution. "
            "Order one is the distribution's exact expectation or expected value."
        ),
        request_type=FiniteRawMomentRequest,
        result_type=FiniteRawMomentResult,
        run=_raw_moment,
        tags=(
            "probability",
            "moment",
            "expectation",
            "expected-value",
            "discrete-random-variable",
            "finite",
            "exact",
            "python-flint",
        ),
        examples=(
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
    MathTool(
        operation_id="probability.finite_distribution.condition.compute",
        title="Condition an exact finite distribution",
        description=(
            "Normalize one explicit positive-mass event of a canonical finite "
            "rational distribution, preserving each source contribution."
        ),
        request_type=FiniteConditionRequest,
        result_type=FiniteConditionResult,
        run=_condition,
        tags=("probability", "conditioning", "finite", "exact", "python-flint"),
        examples=(
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
    MathTool(
        operation_id="probability.finite_distribution.pushforward.compute",
        title="Push forward an exact finite distribution",
        description=(
            "Apply one explicit total rational lookup map and exactly aggregate "
            "all source masses with the same target."
        ),
        request_type=FinitePushforwardRequest,
        result_type=FinitePushforwardResult,
        run=_pushforward,
        tags=("probability", "pushforward", "finite", "exact", "python-flint"),
        examples=(
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
            example(
                "fair_die_pair_merge",
                "Push forward a fair die by merging atoms; mapping sources must cover distribution atoms in canonical order.",
                {
                    "distribution": _FAIR_DIE_3,
                    "mapping": [
                        {
                            "source": {"num": "0", "den": "1"},
                            "target": {"num": "0", "den": "1"},
                        },
                        {
                            "source": {"num": "1", "den": "1"},
                            "target": {"num": "1", "den": "2"},
                        },
                        {
                            "source": {"num": "2", "den": "1"},
                            "target": {"num": "1", "den": "2"},
                        },
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="probability.finite_distribution.convolution.compute",
        title="Convolve two exact finite distributions",
        description=(
            "Compute the bounded product-measure distribution of the sum of "
            "two independent finite rational random variables."
        ),
        request_type=FiniteConvolutionRequest,
        result_type=FiniteConvolutionResult,
        run=_convolution,
        tags=(
            "probability",
            "convolution",
            "independence",
            "finite",
            "exact",
            "python-flint",
        ),
        examples=(
            example(
                "two_fair_bits",
                "Compute the exact distribution of the sum of two fair bits.",
                {"left": _FAIR_BIT, "right": _FAIR_BIT},
            ),
            example(
                "die_plus_bit",
                "Convolve a fair die with a fair bit; pair product and aggregated atoms have bounded limits.",
                {"left": _FAIR_DIE_3, "right": _FAIR_BIT},
            ),
        ),
    ),
    MathTool(
        operation_id="probability.gaussian_polynomial.moment.compute",
        title="Exact bounded Gaussian polynomial moment",
        description=(
            "Compute one fixed-order exact moment of a bounded sparse complex-"
            "rational polynomial in independent standard real Gaussian variables, "
            "preserving the complete coefficient-contraction ledger. This does not "
            "establish an identity for every order."
        ),
        request_type=CanonicalGaussianPolynomialMomentRequest,
        result_type=GaussianPolynomialMomentResult,
        run=_gaussian_polynomial_moment,
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
        examples=(
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
    GRAPH_CONNECTION_PROBABILITY_OPERATION,
    DIRECTED_BOND_CONNECTION_PROBABILITY_OPERATION,
)


def finite_probability_operations() -> MathTools:
    from dataclasses import replace

    from jacobian.math.probability._all_terminal_reliability import (
        ALL_TERMINAL_RELIABILITY_OPERATION,
    )
    from jacobian.math.probability._gaussian_inputs import (
        CanonicalGaussianPolynomialMomentRequest,
    )
    from jacobian.math.probability._mutual_information import (
        MUTUAL_INFORMATION_OPERATION,
    )

    def _with_canonical_gaussian_input(operation: Any) -> Any:
        if operation.operation_id != "probability.gaussian_polynomial.moment.compute":
            return operation
        return replace(
            operation,
            request_type=CanonicalGaussianPolynomialMomentRequest,
        )

    return (
        MUTUAL_INFORMATION_OPERATION,
        *(
            _with_canonical_gaussian_input(operation)
            for operation in FINITE_PROBABILITY_OPERATIONS
        ),
        ALL_TERMINAL_RELIABILITY_OPERATION,
    )


TOOLS: MathTools = (*finite_probability_operations(), ASYMMETRIC_LOCAL_LEMMA_OPERATION)


__all__ = ["TOOLS", "finite_probability_operations"]
