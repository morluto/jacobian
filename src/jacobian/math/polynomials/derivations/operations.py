"""Native exact polynomial-derivation operations."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.derivations._models import (
    MAX_DERIVATION_CERTIFICATE_CHAIN,
    MAX_DERIVATION_COEFFICIENT_DIGITS,
    MAX_DERIVATION_CONTRIBUTION_CELLS,
    MAX_DERIVATION_EXPONENT,
    MAX_DERIVATION_IMAGE_TERMS,
    MAX_DERIVATION_ITERATE_COUNT,
    MAX_DERIVATION_ITERATE_TERMS,
    MAX_DERIVATION_SOURCE_TERMS,
    DerivationApplyResult,
    DerivationIteratesResult,
    LocallyNilpotentCertificate,
    PolynomialDerivation,
    PolynomialGaAction,
    _require_derivation_polynomial,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

_TermMap = dict[tuple[int, ...], Fraction]


def _as_derivation(
    value: PolynomialDerivation | Mapping[str, Any],
) -> PolynomialDerivation:
    try:
        payload = (
            value.model_dump() if isinstance(value, PolynomialDerivation) else value
        )
        # Typed/model_construct values are caller-authored at this boundary;
        # reparse nested image polynomials before any owner arithmetic.
        return PolynomialDerivation.model_validate(payload)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("derivation",),
            code="polynomial_derivation.shape",
            message="the derivation is not a valid typed value",
        ) from exc


def _as_polynomial(value: RationalPolynomial | Mapping[str, Any]) -> RationalPolynomial:
    try:
        payload = value.model_dump() if isinstance(value, RationalPolynomial) else value
        return RationalPolynomial.model_validate(payload)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial_derivation.polynomial_shape",
            message="the polynomial is not a valid typed value",
        ) from exc


def _run_admission(admission: Any, *, location: tuple[str | int, ...]) -> None:
    try:
        admission()
    except OperationDomainValidationError:
        raise
    except OperationResourceAdmissionError:
        raise
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=location, code=exc.type, message=exc.message()
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=location,
            code="polynomial_derivation.admission",
            message=str(exc),
        ) from exc


def _admit_derivation(derivation: PolynomialDerivation) -> None:
    """Admit a complete derivation independently of Pydantic construction."""
    if not isinstance(derivation, PolynomialDerivation):
        raise OperationDomainValidationError(
            location=("derivation",),
            code="polynomial_derivation.shape",
            message="the derivation is not a valid typed value",
        )
    variables = derivation.variables
    if (
        not isinstance(variables, tuple)
        or not variables
        or any(not isinstance(variable, str) for variable in variables)
        or len(set(variables)) != len(variables)
        or len(derivation.images) != len(variables)
    ):
        raise OperationDomainValidationError(
            location=("derivation",),
            code="polynomial_derivation.image_count",
            message="a derivation needs exactly one image per distinct generator",
        )
    for index, image in enumerate(derivation.images):
        if not isinstance(image, RationalPolynomial) or image.variables != variables:
            raise OperationDomainValidationError(
                location=("derivation", "images", index),
                code="polynomial_derivation.ordered_ring",
                message="derivation images must use the declared ordered ring",
            )
        _run_admission(
            lambda image=image, index=index: _require_derivation_polynomial(
                image,
                label=f"generator image {index}",
                maximum_terms=MAX_DERIVATION_IMAGE_TERMS,
            ),
            location=("derivation", "images", index),
        )


def _admit_source(
    derivation: PolynomialDerivation, polynomial: RationalPolynomial
) -> RationalPolynomial:
    """Admit and canonicalize one source against an admitted derivation."""
    polynomial = _as_polynomial(polynomial)
    if not isinstance(polynomial, RationalPolynomial):
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial_derivation.polynomial_shape",
            message="the polynomial is not a valid typed value",
        )
    if polynomial.variables != derivation.variables:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial_derivation.ordered_ring",
            message="the polynomial must use the derivation's ordered ring",
        )
    if len(polynomial.polynomial.terms) > MAX_DERIVATION_SOURCE_TERMS:
        raise OperationResourceAdmissionError(
            location=("polynomial", "polynomial", "terms"),
            code="polynomial_derivation.source_term_budget",
            message="source polynomial exceeds the admitted term envelope",
        )
    _run_admission(
        lambda: _require_derivation_polynomial(
            polynomial,
            label="source polynomial",
            maximum_terms=MAX_DERIVATION_SOURCE_TERMS,
        ),
        location=("polynomial",),
    )
    predicted_cells = 0
    for axis, image in enumerate(derivation.images):
        surviving = sum(
            1 for term in polynomial.polynomial.terms if term.exponents[axis] > 0
        )
        predicted_cells += len(image.polynomial.terms) * surviving
    if predicted_cells > MAX_DERIVATION_CONTRIBUTION_CELLS:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial_derivation.contribution_budget",
            message="derivation application exceeds the contribution-cell budget",
        )
    _admit_result_growth(derivation, polynomial)
    return polynomial


def _component_digits(value: int) -> int:
    return len(str(abs(value)))


def _admit_result_growth(
    derivation: PolynomialDerivation, polynomial: RationalPolynomial
) -> None:
    """Soundly bound one exact derivative before multiplying coefficients."""

    grouped_bounds: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for axis, image in enumerate(derivation.images):
        for source_term in polynomial.polynomial.terms:
            multiplier = source_term.exponents[axis]
            if multiplier == 0:
                continue
            lowered = list(source_term.exponents)
            lowered[axis] -= 1
            for image_term in image.polynomial.terms:
                exponents = tuple(
                    left + right
                    for left, right in zip(lowered, image_term.exponents, strict=True)
                )
                if (
                    any(value > MAX_DERIVATION_EXPONENT for value in exponents)
                    or sum(exponents) > MAX_DERIVATION_EXPONENT
                ):
                    raise OperationResourceAdmissionError(
                        location=("polynomial",),
                        code="polynomial_derivation.result_exponent_budget",
                        message="derivation result exceeds the admitted exponent envelope",
                    )
                source_coefficient = source_term.coefficient
                image_coefficient = image_term.coefficient
                numerator_digits = (
                    _component_digits(source_coefficient.num)
                    + _component_digits(image_coefficient.num)
                    + _component_digits(multiplier)
                )
                denominator_digits = _component_digits(
                    source_coefficient.den
                ) + _component_digits(image_coefficient.den)
                grouped_bounds.setdefault(exponents, []).append(
                    (numerator_digits, denominator_digits)
                )

    if len(grouped_bounds) > MAX_DERIVATION_ITERATE_TERMS:
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial_derivation.result_term_budget",
            message="derivation result exceeds the admitted term envelope",
        )
    for bounds in grouped_bounds.values():
        denominator_digits = sum(bound[1] for bound in bounds)
        numerator_digits = (
            max(
                numerator + denominator_digits - denominator
                for numerator, denominator in bounds
            )
            + len(str(len(bounds)))
            + 1
        )
        if (
            denominator_digits > MAX_DERIVATION_COEFFICIENT_DIGITS
            or numerator_digits > MAX_DERIVATION_COEFFICIENT_DIGITS
        ):
            raise OperationResourceAdmissionError(
                location=("polynomial",),
                code="polynomial_derivation.result_coefficient_budget",
                message=(
                    "derivation result coefficients exceed the admitted exact envelope"
                ),
            )


def _admit_derivation_apply(
    derivation: PolynomialDerivation, polynomial: RationalPolynomial
) -> None:
    """Enforce the shared ring and the pre-expansion work/output envelope."""
    _admit_derivation(derivation)
    _admit_source(derivation, polynomial)


def _term_map(polynomial: RationalPolynomial) -> _TermMap:
    return {
        tuple(term.exponents): term.coefficient.as_fraction()
        for term in polynomial.polynomial.terms
    }


def _partial(terms: _TermMap, axis: int) -> _TermMap:
    derived: _TermMap = {}
    for exponents, coefficient in terms.items():
        if exponents[axis] > 0:
            lowered = list(exponents)
            lowered[axis] -= 1
            derived[tuple(lowered)] = coefficient * exponents[axis]
    return derived


def _multiply(left: _TermMap, right: _TermMap) -> _TermMap:
    product: _TermMap = {}
    for left_exponents, left_coefficient in left.items():
        for right_exponents, right_coefficient in right.items():
            exponents = tuple(
                left_exponent + right_exponent
                for left_exponent, right_exponent in zip(
                    left_exponents, right_exponents, strict=True
                )
            )
            product[exponents] = (
                product.get(exponents, Fraction(0))
                + left_coefficient * right_coefficient
            )
    return {exponents: value for exponents, value in product.items() if value != 0}


def _add(into: _TermMap, extra: _TermMap) -> None:
    for exponents, value in extra.items():
        combined = into.get(exponents, Fraction(0)) + value
        if combined == 0:
            into.pop(exponents, None)
        else:
            into[exponents] = combined


def _encode(variables: tuple[str, ...], terms: _TermMap) -> RationalPolynomial:
    ordered = sorted(terms.items(), key=lambda item: item[0], reverse=True)
    return RationalPolynomial(
        domain="QQ",
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(coefficient),
                    exponents=exponents,
                )
                for exponents, coefficient in ordered
            )
        ),
    )


def _apply_admitted(
    derivation: PolynomialDerivation, polynomial: RationalPolynomial
) -> DerivationApplyResult:
    """Apply a derivation after its shared admission has already run."""
    source = _term_map(polynomial)
    total: _TermMap = {}
    contributions: list[RationalPolynomial] = []
    for axis, image in enumerate(derivation.images):
        contribution = _multiply(_term_map(image), _partial(source, axis))
        contributions.append(_encode(derivation.variables, contribution))
        _add(total, contribution)
    return DerivationApplyResult._from_kernel(
        derivation,
        polynomial,
        result=_encode(derivation.variables, total),
        contributions=tuple(contributions),
    )


def _generator(variables: tuple[str, ...], index: int) -> RationalPolynomial:
    return _encode(
        variables,
        {
            tuple(
                1 if axis == index else 0 for axis in range(len(variables))
            ): Fraction(1)
        },
    )


def _as_certificate(
    certificate: LocallyNilpotentCertificate,
) -> LocallyNilpotentCertificate:
    try:
        if not isinstance(certificate, LocallyNilpotentCertificate):
            raise TypeError("certificate must be a typed value")
        return LocallyNilpotentCertificate.model_validate(certificate.model_dump())
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("certificate",),
            code="polynomial_derivation.certificate_shape",
            message="the certificate is not a valid typed value",
        ) from exc


def _admit_certificate(
    certificate: LocallyNilpotentCertificate,
) -> LocallyNilpotentCertificate:
    """Recheck the authored relation before exponentiating it."""
    certificate = _as_certificate(certificate)
    if not isinstance(certificate, LocallyNilpotentCertificate):
        raise OperationDomainValidationError(
            location=("certificate",),
            code="polynomial_derivation.certificate_shape",
            message="the certificate is not a valid typed value",
        )
    derivation = _as_derivation(certificate.derivation)
    if not isinstance(derivation, PolynomialDerivation):
        raise OperationDomainValidationError(
            location=("certificate", "derivation"),
            code="polynomial_derivation.certificate_shape",
            message="certificate derivation must be a typed derivation",
        )
    _admit_derivation(derivation)
    chains = certificate.generator_iterates
    if len(chains) != len(derivation.variables):
        raise OperationDomainValidationError(
            location=("certificate", "generator_iterates"),
            code="polynomial_derivation.certificate_shape",
            message="one complete iterate chain is required per generator",
        )
    canonical_chains: list[tuple[RationalPolynomial, ...]] = []
    for index, chain in enumerate(chains):
        if (
            not isinstance(chain, tuple)
            or not 1 <= len(chain) <= MAX_DERIVATION_CERTIFICATE_CHAIN
        ):
            raise OperationDomainValidationError(
                location=("certificate", "generator_iterates", index),
                code="polynomial_derivation.certificate_shape",
                message="generator chains must be nonempty and bounded",
            )
        try:
            canonical_chain = tuple(_as_polynomial(value) for value in chain)
        except OperationDomainValidationError as exc:
            raise OperationDomainValidationError(
                location=("certificate", "generator_iterates", index),
                code="polynomial_derivation.certificate_ring",
                message="certificate chains must use the derivation ring",
            ) from exc
        if any(value.variables != derivation.variables for value in canonical_chain):
            raise OperationDomainValidationError(
                location=("certificate", "generator_iterates", index),
                code="polynomial_derivation.certificate_ring",
                message="certificate chains must use the derivation ring",
            )
        expected_source = _generator(derivation.variables, index)
        if canonical_chain[0] != expected_source:
            raise OperationDomainValidationError(
                location=("certificate", "generator_iterates", index, 0),
                code="polynomial_derivation.certificate_source",
                message="each chain must begin with its generator",
            )
        for step, value in enumerate(canonical_chain):
            value = _admit_source(derivation, value)
            if step:
                expected = _apply_admitted(derivation, canonical_chain[step - 1]).result
                if expected != value:
                    raise OperationDomainValidationError(
                        location=("certificate", "generator_iterates", index, step),
                        code="polynomial_derivation.certificate_mismatch",
                        message="generator iterate chain does not match the derivation",
                    )
        if canonical_chain[-1].polynomial.terms:
            raise OperationDomainValidationError(
                location=("certificate", "generator_iterates", index),
                code="polynomial_derivation.certificate_not_zero",
                message="each generator chain must end at zero",
            )
        canonical_chains.append(canonical_chain)
    return LocallyNilpotentCertificate.model_construct(
        derivation=derivation,
        generator_iterates=tuple(canonical_chains),
    )


def derivation_iterates(
    derivation: PolynomialDerivation | Mapping[str, Any],
    polynomial: RationalPolynomial | Mapping[str, Any],
    bound: int,
) -> DerivationIteratesResult:
    """Return the exact prefix D^0(f), ..., D^N(f), stopping at first zero."""
    derivation_value = _as_derivation(derivation)
    polynomial_value = _as_polynomial(polynomial)
    if (
        not isinstance(bound, int)
        or isinstance(bound, bool)
        or not 0 <= bound <= MAX_DERIVATION_ITERATE_COUNT
    ):
        raise OperationResourceAdmissionError(
            location=("bound",),
            code="polynomial_derivation.iterate_bound",
            message="iterate bound is outside the admitted envelope",
        )
    _admit_derivation_apply(derivation_value, polynomial_value)
    current = polynomial_value
    values = [current]
    if not current.polynomial.terms:
        return DerivationIteratesResult.model_construct(
            derivation=derivation_value,
            source=polynomial_value,
            iterates=tuple(values),
            status="FIRST_ZERO_ON_F",
            first_zero_index=0,
        )
    for index in range(1, bound + 1):
        # Every generated source is re-admitted before the next application;
        # otherwise term and coefficient growth could escape the public
        # envelope after the first iterate.
        if index > 1:
            current = _admit_source(derivation_value, current)
        result = _apply_admitted(derivation_value, current).result
        if len(result.polynomial.terms) > MAX_DERIVATION_ITERATE_TERMS:
            raise OperationResourceAdmissionError(
                location=("bound",),
                code="polynomial_derivation.iterate_terms",
                message="iterate result exceeds the admitted term envelope",
            )
        values.append(result)
        current = result
        if not result.polynomial.terms:
            return DerivationIteratesResult.model_construct(
                derivation=derivation_value,
                source=polynomial_value,
                iterates=tuple(values),
                status="FIRST_ZERO_ON_F",
                first_zero_index=index,
            )
    return DerivationIteratesResult.model_construct(
        derivation=derivation_value,
        source=polynomial_value,
        iterates=tuple(values),
        status="NONZERO_THROUGH_BOUND",
        first_zero_index=None,
    )


def construct_locally_nilpotent_certificate(
    derivation: PolynomialDerivation | Mapping[str, Any],
    chains: tuple[tuple[RationalPolynomial, ...], ...],
) -> LocallyNilpotentCertificate:
    """Check complete generator chains and return the typed certificate."""
    derivation_value = _as_derivation(derivation)
    _admit_derivation(derivation_value)
    if len(chains) != len(derivation_value.variables):
        raise OperationDomainValidationError(
            location=("chains",),
            code="polynomial_derivation.certificate_shape",
            message="one chain is required per generator",
        )
    canonical: list[tuple[RationalPolynomial, ...]] = []
    for index, chain in enumerate(chains):
        if not 1 <= len(chain) <= MAX_DERIVATION_CERTIFICATE_CHAIN:
            raise OperationResourceAdmissionError(
                location=("chains", index),
                code="polynomial_derivation.certificate_bound",
                message="generator chain exceeds the admitted bound",
            )
        generator = _generator(derivation_value.variables, index)
        # The supplied chain is caller-authored; re-admit each polynomial and
        # replay every transition.
        canonical_chain = tuple(_as_polynomial(value) for value in chain)
        if canonical_chain[0] != generator:
            raise OperationDomainValidationError(
                location=("chains", index, 0),
                code="polynomial_derivation.certificate_source",
                message="each chain must begin with its generator",
            )
        for step, value in enumerate(canonical_chain[1:], 1):
            previous = _admit_source(derivation_value, canonical_chain[step - 1])
            expected = _apply_admitted(derivation_value, previous).result
            if expected != value:
                raise OperationDomainValidationError(
                    location=("chains", index, step),
                    code="polynomial_derivation.certificate_mismatch",
                    message="generator iterate chain does not match the derivation",
                )
        if canonical_chain[-1].polynomial.terms:
            raise OperationDomainValidationError(
                location=("chains", index),
                code="polynomial_derivation.certificate_not_zero",
                message="each generator chain must end at zero",
            )
        canonical.append(canonical_chain)
    return LocallyNilpotentCertificate.model_construct(
        derivation=derivation_value, generator_iterates=tuple(canonical)
    )


def ga_action_from_certificate(
    certificate: LocallyNilpotentCertificate,
) -> PolynomialGaAction:
    """Exponentiate a checked locally nilpotent derivation on generators."""
    certificate = _admit_certificate(certificate)
    derivation = certificate.derivation
    variables = derivation.variables
    parameter = "t"
    extended = (*variables, parameter)
    images: list[RationalPolynomial] = []
    for chain in certificate.generator_iterates:
        terms: dict[tuple[int, ...], Fraction] = {}
        factorial = 1
        for index, value in enumerate(chain):
            factorial = factorial * index if index else 1
            for term in value.polynomial.terms:
                exponents = (*term.exponents, index)
                coefficient = Fraction(
                    term.coefficient.num, term.coefficient.den * factorial
                )
                terms[exponents] = terms.get(exponents, Fraction(0)) + coefficient
        images.append(
            _encode(extended, {key: value for key, value in terms.items() if value})
        )
    return PolynomialGaAction.model_construct(
        source_variables=variables, parameter=parameter, generator_images=tuple(images)
    )


def apply_derivation(
    derivation: PolynomialDerivation | Mapping[str, Any],
    polynomial: RationalPolynomial | Mapping[str, Any],
) -> DerivationApplyResult:
    """Compute D(f) = sum_i D(x_i) * partial_i(f) with a per-variable ledger."""
    derivation_value = _as_derivation(derivation)
    polynomial_value = _as_polynomial(polynomial)
    _admit_derivation_apply(derivation_value, polynomial_value)
    return _apply_admitted(derivation_value, polynomial_value)


__all__ = [
    "apply_derivation",
    "construct_locally_nilpotent_certificate",
    "derivation_iterates",
    "ga_action_from_certificate",
]
