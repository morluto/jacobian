"""Exact native operations for finite rational probability."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.probability._distribution import (
    MAX_FINITE_CONVOLUTION_PAIRS,
    MAX_FINITE_DISTRIBUTION_ATOMS,
    FiniteConditionalContribution,
    FiniteConditionResult,
    FiniteConvolutionContribution,
    FiniteConvolutionResult,
    FiniteDistributionAtom,
    FiniteEventProbabilityResult,
    FinitePushforwardContribution,
    FinitePushforwardMapEntry,
    FinitePushforwardResult,
    FiniteRationalDistribution,
    FiniteRawMomentContribution,
    FiniteRawMomentResult,
    require_input_distribution,
)
from jacobian.math.probability._gaussian import (
    GAUSSIAN_RESULT_DIGIT_SAFETY_MARGIN,
    MAX_GAUSSIAN_EXPANSION_PATHS,
    MAX_GAUSSIAN_MOMENT_ORDER,
    MAX_GAUSSIAN_RESULT_RATIONAL_DIGITS,
    ExactComplexRational,
    GaussianMomentContraction,
    GaussianPolynomial,
    GaussianPolynomialMomentResult,
)
from jacobian.math.probability._gaussian_moments import gaussian_univariate_moment
from jacobian.math.probability._models import (
    MAX_INPUT_RATIONAL_DIGITS,
    MAX_RESULT_RATIONAL_DIGITS,
    _require_bounded_fraction,
    _require_strictly_increasing,
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


def _admit_distribution(
    atoms: tuple[FiniteDistributionAtom, ...], *, require_canonical: bool
) -> tuple[Fraction, ...]:
    try:
        return require_input_distribution(atoms, require_canonical=require_canonical)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("atoms",),
            code="probability.distribution.admission",
            message=str(exc),
        ) from exc


def _admit_event(
    distribution: FiniteRationalDistribution,
    event_values: tuple[CanonicalRational, ...],
    *,
    require_positive: bool,
) -> None:
    support = set(_admit_distribution(distribution.atoms, require_canonical=True))
    try:
        event = _require_strictly_increasing(event_values, label="finite event values")
        for value in event_values:
            require_bounded_rational(
                value,
                max_digits=MAX_INPUT_RATIONAL_DIGITS,
                label="finite event value",
            )
        if not set(event).issubset(support):
            raise ValueError("finite event values must belong to the distribution")
        selected = set(event)
        event_mass = sum(
            (
                atom.probability.as_fraction()
                for atom in distribution.atoms
                if atom.value.as_fraction() in selected
            ),
            start=Fraction(),
        )
        _require_bounded_fraction(
            event_mass,
            max_digits=MAX_RESULT_RATIONAL_DIGITS,
            label="finite event probability",
        )
        if require_positive and event_mass <= 0:
            raise ValueError("conditioning requires a positive-mass finite event")
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("event_values",),
            code="probability.event.admission",
            message=str(exc),
        ) from exc


def _admit_pushforward(
    distribution: FiniteRationalDistribution,
    mapping: tuple[FinitePushforwardMapEntry, ...],
) -> None:
    source_values = _admit_distribution(distribution.atoms, require_canonical=True)
    try:
        mapping_sources = tuple(item.source.as_fraction() for item in mapping)
        if mapping_sources != source_values:
            raise ValueError(
                "pushforward mapping must cover each source atom in canonical order"
            )
        aggregated: dict[Fraction, Fraction] = {}
        for atom, item in zip(distribution.atoms, mapping, strict=True):
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
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("mapping",),
            code="probability.pushforward.admission",
            message=str(exc),
        ) from exc


def _admit_convolution(
    left: FiniteRationalDistribution,
    right: FiniteRationalDistribution,
) -> None:
    _admit_distribution(left.atoms, require_canonical=True)
    _admit_distribution(right.atoms, require_canonical=True)
    if len(left.atoms) * len(right.atoms) > MAX_FINITE_CONVOLUTION_PAIRS:
        raise OperationDomainValidationError(
            location=("left", "right"),
            code="probability.convolution.pair_bound",
            message=(
                f"finite convolution exceeds the {MAX_FINITE_CONVOLUTION_PAIRS}-pair bound"
            ),
        )
    try:
        aggregated: dict[Fraction, Fraction] = {}
        for left_atom in left.atoms:
            for right_atom in right.atoms:
                value = left_atom.value.as_fraction() + right_atom.value.as_fraction()
                probability = (
                    left_atom.probability.as_fraction()
                    * right_atom.probability.as_fraction()
                )
                aggregated[value] = aggregated.get(value, Fraction()) + probability
        if len(aggregated) > MAX_FINITE_DISTRIBUTION_ATOMS:
            raise ValueError(
                f"finite convolution exceeds the {MAX_FINITE_DISTRIBUTION_ATOMS}-atom output bound"
            )
        for value, probability in aggregated.items():
            _require_bounded_fraction(
                value, max_digits=MAX_RESULT_RATIONAL_DIGITS, label="convolution atom"
            )
            _require_bounded_fraction(
                probability,
                max_digits=MAX_RESULT_RATIONAL_DIGITS,
                label="convolution probability",
            )
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("left", "right"),
            code="probability.convolution.output_bound",
            message=str(exc),
        ) from exc


def _admit_gaussian_polynomial_moment(
    polynomial: GaussianPolynomial, order: int
) -> int:
    """Admit the complete expansion and exact-result envelope."""
    if type(order) is not int or not 0 <= order <= MAX_GAUSSIAN_MOMENT_ORDER:
        raise ValueError(
            f"Gaussian moment order must be between 0 and {MAX_GAUSSIAN_MOMENT_ORDER}"
        )
    expansion_paths: int = len(polynomial.terms) ** order
    if expansion_paths > MAX_GAUSSIAN_EXPANSION_PATHS:
        raise OperationDomainValidationError(
            location=("polynomial", "order"),
            code="probability.gaussian.expansion_path_bound",
            message=(
                "Gaussian polynomial power exceeds the "
                f"{MAX_GAUSSIAN_EXPANSION_PATHS}-path expansion bound"
            ),
        )
    components = tuple(
        component
        for term in polynomial.terms
        for component in (term.coefficient.real, term.coefficient.imaginary)
    )
    distinct_denominator_digits = sum(
        len(denominator) for denominator in {component.den for component in components}
    )
    maximum_numerator_digits = max(
        len(component.num.lstrip("-")) for component in components
    )
    result_digit_bound = (
        order * (distinct_denominator_digits + maximum_numerator_digits)
        + len(str(max(1, expansion_paths)))
        + GAUSSIAN_RESULT_DIGIT_SAFETY_MARGIN
    )
    if result_digit_bound > MAX_GAUSSIAN_RESULT_RATIONAL_DIGITS:
        raise OperationDomainValidationError(
            location=("polynomial", "order"),
            code="probability.gaussian.result_digit_bound",
            message=(
                "Gaussian polynomial coefficient denominators can exceed the "
                f"{MAX_GAUSSIAN_RESULT_RATIONAL_DIGITS}-digit result bound"
            ),
        )
    return expansion_paths


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


def raw_moment(
    atoms: tuple[FiniteDistributionAtom, ...], order: int
) -> FiniteRawMomentResult:
    from flint import fmpq

    if type(order) is not int or not 0 <= order <= 128:
        raise ValueError("raw moment order must be between 0 and 128")
    _admit_distribution(atoms, require_canonical=False)
    contributions: list[FiniteRawMomentContribution] = []
    total = fmpq(0)
    for atom in atoms:
        value = _fmpq(atom.value)
        probability = _fmpq(atom.probability)
        powered = value**order
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
    return FiniteRawMomentResult._from_kernel(
        order=order,
        moment=_wire(total),
        contributions=tuple(contributions),
    )


def event_probability(
    distribution: FiniteRationalDistribution,
    event_values: tuple[CanonicalRational, ...],
) -> FiniteEventProbabilityResult:
    from flint import fmpq

    _admit_event(distribution, event_values, require_positive=False)
    selected_values = {value.as_fraction() for value in event_values}
    selected = tuple(
        atom
        for atom in distribution.atoms
        if atom.value.as_fraction() in selected_values
    )
    total = fmpq(0)
    for atom in selected:
        total += _fmpq(atom.probability)
    return FiniteEventProbabilityResult._from_kernel(
        event_probability=_wire(total),
        selected_atoms=selected,
    )


def condition(
    distribution: FiniteRationalDistribution,
    event_values: tuple[CanonicalRational, ...],
) -> FiniteConditionResult:
    from flint import fmpq

    _admit_event(distribution, event_values, require_positive=True)
    selected_values = {value.as_fraction() for value in event_values}
    selected = tuple(
        atom
        for atom in distribution.atoms
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
    return FiniteConditionResult._from_kernel(
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


def pushforward(
    distribution: FiniteRationalDistribution,
    mapping: tuple[FinitePushforwardMapEntry, ...],
) -> FinitePushforwardResult:
    from flint import fmpq

    _admit_pushforward(distribution, mapping)
    aggregated: dict[Fraction, Any] = {}
    contributions: list[FinitePushforwardContribution] = []
    for atom, mapping_entry in zip(
        distribution.atoms,
        mapping,
        strict=True,
    ):
        target = mapping_entry.target.as_fraction()
        probability = _fmpq(atom.probability)
        aggregated[target] = aggregated.get(target, fmpq(0)) + probability
        contributions.append(
            FinitePushforwardContribution(
                source=atom.value,
                target=mapping_entry.target,
                probability=atom.probability,
            )
        )
    return FinitePushforwardResult._from_kernel(
        distribution=_distribution(aggregated),
        contributions=tuple(contributions),
    )


def convolution(
    left: FiniteRationalDistribution,
    right: FiniteRationalDistribution,
) -> FiniteConvolutionResult:
    from flint import fmpq

    _admit_convolution(left, right)
    aggregated: dict[Fraction, Any] = {}
    contributions: list[FiniteConvolutionContribution] = []
    for left_atom in left.atoms:
        for right_atom in right.atoms:
            sum_value = left_atom.value.as_fraction() + right_atom.value.as_fraction()
            probability = _fmpq(left_atom.probability) * _fmpq(right_atom.probability)
            aggregated[sum_value] = aggregated.get(sum_value, fmpq(0)) + probability
            contributions.append(
                FiniteConvolutionContribution(
                    left_value=left_atom.value,
                    right_value=right_atom.value,
                    sum_value=CanonicalRational(
                        num=format_canonical_integer(sum_value.numerator),
                        den=format_canonical_integer(sum_value.denominator),
                    ),
                    probability=_wire(probability),
                )
            )
    return FiniteConvolutionResult._from_kernel(
        distribution=_distribution(aggregated),
        contributions=tuple(contributions),
    )


def gaussian_polynomial_moment(
    polynomial: GaussianPolynomial, order: int
) -> GaussianPolynomialMomentResult:
    from flint import fmpq, fmpq_mpoly_ctx

    expansion_paths = _admit_gaussian_polynomial_moment(polynomial, order)
    zero = fmpq(0)
    one = fmpq(1)
    dimension = polynomial.variable_count

    # Power the complex-coefficient polynomial via FLINT fmpq_mpoly binary
    # exponentiation.  A complex coefficient (a + b i) is represented as a
    # pair of real fmpq_mpoly (real_part, imag_part); complex multiplication
    # is (r1*r2 - i1*i2, r1*i2 + i1*r2).  This replaces the previous
    # path-by-path dictionary expansion with native FLINT arithmetic.
    names = tuple(f"x{index}" for index in range(dimension))
    ctx = fmpq_mpoly_ctx.get(names, "lex")
    real_base = ctx.from_dict(
        {term.exponents: _fmpq(term.coefficient.real) for term in polynomial.terms}
    )
    imag_base = ctx.from_dict(
        {term.exponents: _fmpq(term.coefficient.imaginary) for term in polynomial.terms}
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
    if order > 0:
        powered = (real_base, imag_base)
        remaining = order - 1
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

    return GaussianPolynomialMomentResult._from_kernel(
        order=order,
        moment=_complex_wire(total),
        expansion_path_count=expansion_paths,
        expanded_monomial_count=len(contractions),
        contractions=tuple(contractions),
    )


__all__ = [
    "condition",
    "convolution",
    "event_probability",
    "gaussian_polynomial_moment",
    "pushforward",
    "raw_moment",
]
