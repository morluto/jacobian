"""Regressions for the derivation native boundaries and growth envelopes.

Each repaired case is paired with a negative control showing the enclosing
bound still refuses the genuinely over-limit request.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.derivations._models import PolynomialGaAction
from jacobian.math.polynomials.derivations._stable_kernels import (
    ga_stable_subrepresentation,
)
from jacobian.math.polynomials.derivations._weight_models import (
    PolynomialWeightAction,
    PolynomialWeightSubrepresentationRequest,
)
from jacobian.math.polynomials.derivations._weight_operations import (
    gm_generated_subrepresentation,
)
from jacobian.math.polynomials.derivations.orbits._kernels import ga_polynomial_orbit
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _rational(numerator: int, denominator: int = 1) -> CanonicalRational:
    return CanonicalRational(num=numerator, den=denominator)


def _poly(
    variables: tuple[str, ...],
    terms: Sequence[tuple[Sequence[int], int]],
) -> RationalPolynomial:
    built = sorted(
        (
            RationalPolynomialTerm(
                exponents=tuple(exponents), coefficient=_rational(coefficient)
            )
            for exponents, coefficient in terms
        ),
        key=lambda term: term.exponents,
        reverse=True,
    )
    return RationalPolynomial(
        variables=variables, polynomial=SparseRationalPolynomial(terms=tuple(built))
    )


def _translation_action(coefficient: int = 1) -> PolynomialGaAction:
    """The translation x -> x + coefficient*t on the one-variable source ring."""
    return PolynomialGaAction(
        source_variables=("x",),
        parameter="t",
        generator_images=(_poly(("x", "t"), [((0, 1), coefficient), ((1, 0), 1)]),),
    )


def _weight_action(variables: tuple[str, ...]) -> PolynomialWeightAction:
    return PolynomialWeightAction(variables=variables, weights=(1,) * len(variables))


# --- the exported native callable takes domain values -----------------------


def test_gm_subrepresentation_accepts_domain_values() -> None:
    action = _weight_action(("x", "y"))
    generators = (_poly(("x", "y"), [((1, 0), 1), ((0, 1), 1)]),)
    result = gm_generated_subrepresentation(action, generators, "t")
    assert result.basis
    assert result.matrix


def test_gm_subrepresentation_rejects_a_generator_on_another_ring() -> None:
    action = _weight_action(("x", "y"))
    generators = (_poly(("x", "z"), [((1, 0), 1)]),)
    with pytest.raises(OperationDomainValidationError) as error:
        gm_generated_subrepresentation(action, generators, "t")
    assert error.value.errors()[0]["type"] == "gm_subrepresentation.ordered_ring"


def test_gm_subrepresentation_rejects_a_parameter_inside_the_ring() -> None:
    action = _weight_action(("x", "t"))
    generators = (_poly(("x", "t"), [((1, 0), 1)]),)
    with pytest.raises(OperationDomainValidationError):
        gm_generated_subrepresentation(action, generators, "t")


# Negative control: the generator-count envelope is unchanged.


def test_gm_subrepresentation_still_enforces_the_generator_count() -> None:
    from jacobian.math.polynomials.derivations._weight_models import (
        MAX_GM_SUBREP_GENERATORS,
    )

    action = _weight_action(("x",))
    generators = tuple(
        _poly(("x",), [((0,), 1)]) for _ in range(MAX_GM_SUBREP_GENERATORS + 1)
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        gm_generated_subrepresentation(action, generators, "t")
    assert error.value.errors()[0]["type"] == "gm_subrepresentation.generator_count"


# --- the subrepresentation ring admits the shared eight-variable maximum ---


def test_gm_subrepresentation_accepts_eight_source_variables() -> None:
    variables = tuple("abcdefgh")
    action = _weight_action(variables)
    generators = (_poly(variables, [((0,) * 8, 1)]),)
    result = gm_generated_subrepresentation(action, generators, "t")
    assert len(result.basis) == 1


def test_subrepresentation_request_schema_publishes_the_ring_prerequisites() -> None:
    """The coupled constraints must be schema-visible, not validator-only."""
    fields = PolynomialWeightSubrepresentationRequest.model_fields
    assert "same variable tuple as the action" in str(fields["generators"].description)
    assert "must not appear" in str(fields["parameter"].description)
    assert "ordered variable axis" in str(fields["action"].description)


# Negative control: the Laurent coaction still reserves its eighth axis.


def test_diagonal_weight_action_still_rejects_eight_source_variables() -> None:
    from jacobian.math.polynomials.derivations._weight_operations import (
        diagonal_weight_action,
    )

    variables = tuple("abcdefgh")
    action = _weight_action(variables)
    source = _poly(variables, [((0,) * 8, 1)])
    with pytest.raises(OperationDomainValidationError):
        diagonal_weight_action(action, source, "t")


# --- published envelope no longer claims a serialized byte limit ------------


def test_published_subrepresentation_description_omits_the_byte_limit() -> None:
    from jacobian.math.polynomials.derivations._tools import TOOLS

    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "algebraic_group.gm.finite_subrepresentation.compute"
    )
    assert "10 MiB" not in tool.description
    assert "retained cells" in tool.description


# --- shared denominators bound the basis expansion, not the product count ---


def test_cubic_translation_basis_is_admitted() -> None:
    action = _translation_action()
    basis = tuple(_poly(("x",), [((degree,), 1)]) for degree in range(4))
    result = ga_stable_subrepresentation(action, basis)
    assert len(result.action_matrix) == 4
    # Every basis element is a nonzero row of the exact action matrix.
    assert all(len(row) == 4 for row in result.action_matrix)
    assert all(
        any(entry.polynomial.terms for entry in row) for row in result.action_matrix
    )


def test_quartic_translation_basis_is_admitted() -> None:
    action = _translation_action()
    basis = tuple(_poly(("x",), [((degree,), 1)]) for degree in range(5))
    result = ga_stable_subrepresentation(action, basis)
    assert len(result.action_matrix) == 5


# Negative control: a genuinely wide source coefficient is still refused.


def test_stable_subrepresentation_still_enforces_source_coefficient_digits() -> None:
    action = _translation_action()
    basis = (
        _poly(
            ("x",),
            [((0,), 10**200)],
        ),
    )
    with pytest.raises(OperationResourceAdmissionError):
        ga_stable_subrepresentation(action, basis)


# --- orbit preflight counts merged supports ---------------------------------


def test_degree_thirteen_translation_orbit_is_admitted() -> None:
    action = _translation_action()
    source = _poly(("x",), [((13,), 1)])
    image = ga_polynomial_orbit(action, source)
    # (x+t)^13 has exactly 14 distinct monomials.
    assert len(image.polynomial.terms) == 14


def test_translation_orbit_matches_binomial_coefficients() -> None:
    action = _translation_action()
    source = _poly(("x",), [((3,), 1)])
    image = ga_polynomial_orbit(action, source)
    coefficients = {
        term.exponents: term.coefficient.as_fraction()
        for term in image.polynomial.terms
    }
    assert coefficients[(3, 0)] == 1
    assert coefficients[(2, 1)] == 3
    assert coefficients[(1, 2)] == 3
    assert coefficients[(0, 3)] == 1


def test_higher_degree_translation_orbit_is_admitted() -> None:
    action = _translation_action()
    source = _poly(("x",), [((32,), 1)])
    image = ga_polynomial_orbit(action, source)
    assert len(image.polynomial.terms) == 33


# Negative controls for the orbit bounds.


def test_orbit_still_enforces_the_expansion_budget() -> None:
    """A wide generator image per variable still exceeds the support bound."""
    from jacobian.math.polynomials.derivations.orbits._models import (
        MAX_GA_ORBIT_EXPANSIONS,
    )

    # A generator image with many distinct terms, raised to a high exponent,
    # has a genuine multiset support beyond the admitted envelope.
    many = tuple(((index, 0), 1) for index in range(1, 9))
    action = PolynomialGaAction(
        source_variables=("x",),
        parameter="t",
        generator_images=(_poly(("x", "t"), many),),
    )
    with pytest.raises(OperationResourceAdmissionError):
        ga_polynomial_orbit(action, _poly(("x",), [((64,), 1)]))
    assert MAX_GA_ORBIT_EXPANSIONS > 0


def test_orbit_still_enforces_the_output_exponent_bound() -> None:
    """A source exponent whose image would exceed the exact bound is refused."""
    from jacobian.math.polynomials.values import MAX_POLYNOMIAL_EXPONENT

    # The image has a parameter-axis exponent of one per copy, so a source
    # monomial above the shared exponent ceiling still overflows it.
    half = MAX_POLYNOMIAL_EXPONENT // 2 + 2
    image = _poly(("x", "t"), [((0, 1), 1), ((1, 0), 1)])
    action = PolynomialGaAction(
        source_variables=("x",), parameter="t", generator_images=(image,)
    )
    with pytest.raises(OperationResourceAdmissionError):
        ga_polynomial_orbit(action, _poly(("x",), [((MAX_POLYNOMIAL_EXPONENT,), 1)]))
    assert half > 0


# --- malformed native input keeps the operation's error classification ------


def test_stable_subrepresentation_translates_an_empty_mapping() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        ga_stable_subrepresentation({})
    assert (
        error.value.errors()[0]["type"]
        == "polynomial_ga_subrepresentation.request_shape"
    )


def test_stable_subrepresentation_translates_a_malformed_mapping() -> None:
    with pytest.raises(OperationDomainValidationError):
        ga_stable_subrepresentation({"action": {}, "basis": "not-a-basis"})


def test_stable_subrepresentation_translates_a_forged_request() -> None:
    from jacobian.math.polynomials.derivations._stable_models import (
        PolynomialGaStableSubrepresentationRequest,
    )

    forged = PolynomialGaStableSubrepresentationRequest.model_construct(
        action=_translation_action(), basis=()
    )
    with pytest.raises(OperationDomainValidationError) as error:
        ga_stable_subrepresentation(forged)
    assert (
        error.value.errors()[0]["type"]
        == "polynomial_ga_subrepresentation.request_shape"
    )


# Negative control: a well-formed action plus a wrong-ring basis is still a
# distinct, correctly classified rejection.
def test_stable_subrepresentation_still_rejects_a_wrong_ring_basis() -> None:
    action = _translation_action()
    with pytest.raises(OperationDomainValidationError):
        ga_stable_subrepresentation(action, (_poly(("y",), [((0,), 1)]),))
