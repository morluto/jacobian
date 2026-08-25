"""Exact finite probability operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.probability._directed_bond_reliability import (
    DIRECTED_BOND_CONNECTION_PROBABILITY_OPERATION,
)
from jacobian.math.probability._distribution import (
    FiniteConditionalContribution,
    FiniteConditionRequest,
    FiniteConditionResult,
    FiniteConvolutionContribution,
    FiniteConvolutionRequest,
    FiniteConvolutionResult,
    FiniteDistributionAtom,
    FiniteEventProbabilityResult,
    FiniteEventRequest,
    FinitePushforwardContribution,
    FinitePushforwardRequest,
    FinitePushforwardResult,
    FiniteRationalDistribution,
    FiniteRawMomentContribution,
    FiniteRawMomentRequest,
    FiniteRawMomentResult,
)
from jacobian.math.probability._gaussian import (
    ExactComplexRational,
    GaussianMomentContraction,
    GaussianPolynomialMomentRequest,
    GaussianPolynomialMomentResult,
)
from jacobian.math.probability._gaussian_inputs import (
    CanonicalGaussianPolynomialMomentRequest,
)
from jacobian.math.probability._gaussian_moments import gaussian_univariate_moment
from jacobian.math.probability._graph_connection_probability import (
    GRAPH_CONNECTION_PROBABILITY_OPERATION,
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
) -> FiniteRawMomentResult:
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
    return FiniteRawMomentResult(
        order=request.order,
        moment=_wire(total),
        contributions=tuple(contributions),
    )


def _event_probability(
    request: FiniteEventRequest,
) -> FiniteEventProbabilityResult:
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
    return FiniteEventProbabilityResult(
        event_probability=_wire(total),
        selected_atoms=selected,
    )


def _condition(
    request: FiniteConditionRequest,
) -> FiniteConditionResult:
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
    contributions = tuple(
        FiniteConditionalContribution(
            value=atom.value,
            source_probability=atom.probability,
            conditioned_probability=_wire(_fmpq(atom.probability) / event_probability),
        )
        for atom in selected
    )
    return FiniteConditionResult(
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


def _pushforward(
    request: FinitePushforwardRequest,
) -> FinitePushforwardResult:
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
    return FinitePushforwardResult(
        distribution=_distribution(aggregated),
        contributions=tuple(contributions),
    )


def _convolution(
    request: FiniteConvolutionRequest,
) -> FiniteConvolutionResult:
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
    return FiniteConvolutionResult(
        distribution=_distribution(aggregated),
        contributions=tuple(contributions),
    )


def _gaussian_polynomial_moment(
    request: GaussianPolynomialMomentRequest,
) -> GaussianPolynomialMomentResult:
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
            gaussian_univariate_moment(exponent) for exponent in exponents
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

    return GaussianPolynomialMomentResult(
        order=request.order,
        moment=_complex_wire(total),
        expansion_path_count=len(base) ** request.order,
        expanded_monomial_count=len(contractions),
        contractions=tuple(contractions),
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
        operation_id="probability.finite_distribution.event_probability.compute",
        title="Exact finite-event probability",
        description=(
            "Sum the exact mass of one explicit subset of a canonical finite "
            "rational distribution and preserve every selected atom."
        ),
        request_type=FiniteEventRequest,
        result_type=FiniteEventProbabilityResult,
        run=_event_probability,
        tags=("probability", "event", "finite", "exact", "python-flint"),
        examples=(
            example(
                "fair_bit_is_one",
                "Compute the exact probability that a fair bit equals one.",
                {
                    "distribution": _FAIR_BIT,
                    "event_values": [{"num": "1", "den": "1"}],
                },
            ),
            example(
                "fair_die_event_subset",
                "Compute a fair-die event probability; event values must be increasing, bounded, and supported by the distribution.",
                {
                    "distribution": _FAIR_DIE_3,
                    "event_values": [
                        {"num": "0", "den": "1"},
                        {"num": "2", "den": "1"},
                    ],
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

__all__ = ["FINITE_PROBABILITY_OPERATIONS", "finite_probability_operations"]


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
