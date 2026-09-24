"""Galois theory operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.galois._models import (
    AutomorphismApplyRequest,
    AutomorphismApplyResult,
    AutomorphismComposeRequest,
    AutomorphismElementApplyRequest,
    AutomorphismInverseRequest,
    AutomorphismRequest,
    AutomorphismResult,
    FrobeniusCycleRequest,
    FrobeniusCycleResult,
    GaloisFactorRequest,
    GaloisFactorResult,
    GaloisGroupRequest,
    GaloisGroupResult,
    PolynomialDiscriminantRequest,
    PolynomialDiscriminantResult,
    QQFieldAutomorphism,
    SolvableRequest,
    SolvableResult,
    SplittingFieldRequest,
    SplittingFieldResult,
)
from jacobian.math.number_theory.galois.operations import (
    apply_automorphism,
    apply_automorphism_to_element,
    automorphisms,
    compose_automorphisms,
    frobenius_cycle,
    galois_factor,
    galois_group,
    inverse_automorphism,
    polynomial_discriminant,
    solvable,
    splitting_field,
)
from jacobian.math.number_theory.number_fields.values import SimpleNumberFieldElement


def _galois_factor(request: GaloisFactorRequest) -> GaloisFactorResult:
    return galois_factor(request.field_order, request.coefficients)


def _frobenius_cycle(request: FrobeniusCycleRequest) -> FrobeniusCycleResult:
    return frobenius_cycle(
        request.field_order,
        request.polynomial_degree,
        request.factorization_degrees,
    )


def _galois_group(request: GaloisGroupRequest) -> GaloisGroupResult:
    return galois_group(request.coefficients)


def _solvable(request: SolvableRequest) -> SolvableResult:
    return solvable(request.coefficients)


def _splitting(request: SplittingFieldRequest) -> SplittingFieldResult:
    return splitting_field(request)


def _automorphisms(request: AutomorphismRequest) -> AutomorphismResult:
    return automorphisms(request.field)


def _compose(request: AutomorphismComposeRequest) -> QQFieldAutomorphism:
    return compose_automorphisms(request.first, request.second)


def _inverse(request: AutomorphismInverseRequest) -> QQFieldAutomorphism:
    return inverse_automorphism(request.automorphism)


def _apply(request: AutomorphismApplyRequest) -> AutomorphismApplyResult:
    return AutomorphismApplyResult(
        root=apply_automorphism(request.automorphism, request.root)
    )


def _apply_element(
    request: AutomorphismElementApplyRequest,
) -> SimpleNumberFieldElement:
    return apply_automorphism_to_element(request.automorphism, request.element)


def _discriminant(
    request: PolynomialDiscriminantRequest,
) -> PolynomialDiscriminantResult:
    return polynomial_discriminant(request.polynomial)


_SPLIT_X2 = {
    "polynomial": {
        "variables": ["x"],
        "polynomial": {
            "terms": [
                {"coefficient": {"num": "1", "den": "1"}, "exponents": [2]},
                {"coefficient": {"num": "-2", "den": "1"}, "exponents": [0]},
            ]
        },
    }
}


_X2_EXTENSION = {"domain": "QQ", "coefficients_descending": ["1", "0", "-2"]}
_QQ_EXTENSION = {"domain": "QQ", "coefficients_descending": ["1", "0"]}
_X2_ALPHA = {
    "presentation": _X2_EXTENSION,
    "coefficients_ascending": [
        {"num": "0", "den": "1"},
        {"num": "1", "den": "1"},
    ],
}
_X2_NEG_ALPHA = {
    "presentation": _X2_EXTENSION,
    "coefficients_ascending": [
        {"num": "0", "den": "1"},
        {"num": "-1", "den": "1"},
    ],
}
_QQ_ONE = {
    "presentation": _X2_EXTENSION,
    "coefficients_ascending": [
        {"num": "1", "den": "1"},
        {"num": "0", "den": "1"},
    ],
}
_FIELD_X2 = {
    "source": _SPLIT_X2["polynomial"],
    "extension": _X2_EXTENSION,
    "root_values": [_X2_ALPHA, _X2_NEG_ALPHA],
    "root_multiplicities": [1, 1],
}
_AUT_X2_ID = {
    "field": _FIELD_X2,
    "root_permutation": [0, 1],
    "basis_images": [_QQ_ONE, _X2_ALPHA],
}
_AUT_X2_CONJUGATION = {
    "field": _FIELD_X2,
    "root_permutation": [1, 0],
    "basis_images": [_QQ_ONE, _X2_NEG_ALPHA],
}
_ROOT_X2_ALPHA = {
    "field": _FIELD_X2,
    "index": 0,
    "multiplicity": 1,
    "value": _X2_ALPHA,
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="polynomial.galois.discriminant_profile.compute",
        title="Compute a polynomial discriminant profile",
        description="Compute the exact integer discriminant and whether it is a square in QQ for a univariate integer polynomial of degree at most six with coefficients bounded by 10^12. Reducible and nonmonic inputs are supported; no root approximations or factorization are used.",
        request_type=PolynomialDiscriminantRequest,
        result_type=PolynomialDiscriminantResult,
        run=_discriminant,
        tags=("galois-theory", "discriminant", "exact"),
        discovery_terms=(
            "polynomial discriminant",
            "Galois discriminant square test",
            "discriminant parity profile",
        ),
        examples=(
            OperationExample(
                name="reducible_x2_minus1",
                description="The discriminant of x^2-1 is 4, a square in QQ.",
                input={
                    "polynomial": {
                        "variables": ["x"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [2],
                                },
                                {
                                    "coefficient": {"num": "-1", "den": "1"},
                                    "exponents": [0],
                                },
                            ]
                        },
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="number_field.polynomial.splitting_field.compute",
        title="Construct an exact degree-at-most-two QQ splitting field",
        description="Construct the complete exact splitting field for any univariate integer polynomial of degree one or two. The extension is a canonical simple-number-field presentation; every root is an exact reduced power-basis element, with multiplicity and linear-factor reconstruction. Cubics and higher degrees are rejected before expansion.",
        request_type=SplittingFieldRequest,
        result_type=SplittingFieldResult,
        run=_splitting,
        tags=("galois-theory", "splitting-field", "exact"),
        examples=(
            OperationExample(
                name="split_x2_minus2",
                description="Represent both roots of x^2-2 exactly in QQ(alpha), alpha^2=2.",
                input=_SPLIT_X2,
            ),
        ),
    ),
    MathTool(
        operation_id="number_field.extension.automorphisms.compute",
        title="Compute all exact automorphisms of a small splitting field",
        description="Enumerate the complete QQ-automorphism group of a degree-at-most-two splitting field. Each map contains exact power-basis images and the induced permutation of the complete exact root family.",
        request_type=AutomorphismRequest,
        result_type=AutomorphismResult,
        run=_automorphisms,
        tags=("galois-theory", "automorphism", "exact"),
        examples=(
            OperationExample(
                name="automorphisms_x2_minus2",
                description="The two automorphisms of QQ(sqrt(2)) fix 1 and send alpha to alpha or -alpha.",
                input={"field": _FIELD_X2},
            ),
        ),
    ),
    MathTool(
        operation_id="number_field.automorphism.inverse.compute",
        title="Invert an exact field automorphism",
        description="Return the exact inverse of a QQ-field automorphism on the same degree-at-most-two splitting field, retaining its basis images and root action.",
        request_type=AutomorphismInverseRequest,
        result_type=QQFieldAutomorphism,
        run=_inverse,
        tags=("galois-theory", "automorphism", "exact"),
        examples=(
            OperationExample(
                name="inverse_quadratic_conjugation",
                description="Quadratic conjugation is its own exact inverse on QQ(sqrt(2)).",
                input={"automorphism": _AUT_X2_CONJUGATION},
            ),
        ),
    ),
    MathTool(
        operation_id="number_field.automorphism.compose.compute",
        title="Compose exact field automorphisms",
        description="Compose two exact QQ-field maps in one degree-at-most-two splitting field, retaining the exact basis images and root action.",
        request_type=AutomorphismComposeRequest,
        result_type=QQFieldAutomorphism,
        run=_compose,
        tags=("galois-theory", "automorphism", "exact"),
        examples=(
            OperationExample(
                name="identity_composition",
                description="Compose the identity map of QQ(sqrt(2)) with itself.",
                input={"first": _AUT_X2_ID, "second": _AUT_X2_ID},
            ),
        ),
    ),
    MathTool(
        operation_id="number_field.automorphism.apply.compute",
        title="Apply an exact field automorphism to a root",
        description="Apply a QQ-field map to an exact root value and return the resulting exact root in the same splitting-field parent.",
        request_type=AutomorphismApplyRequest,
        result_type=AutomorphismApplyResult,
        run=_apply,
        tags=("galois-theory", "automorphism", "exact"),
        examples=(
            OperationExample(
                name="conjugate_sqrt2",
                description="Apply conjugation to the exact root alpha in QQ(alpha), alpha^2=2.",
                input={"automorphism": _AUT_X2_CONJUGATION, "root": _ROOT_X2_ALPHA},
            ),
        ),
    ),
    MathTool(
        operation_id="number_field.automorphism.apply_element.compute",
        title="Apply an exact field automorphism to a field element",
        description=(
            "Apply a supplied exact QQ-automorphism to any reduced power-basis "
            "element of its degree-at-most-two source field. The parent is "
            "preserved and a conservative coordinate-growth bound is checked "
            "before exact arithmetic."
        ),
        request_type=AutomorphismElementApplyRequest,
        result_type=SimpleNumberFieldElement,
        run=_apply_element,
        tags=("galois-theory", "automorphism", "field-element", "exact"),
        discovery_terms=(
            "apply field automorphism to an element",
            "Galois action on a number field element",
            "exact automorphism image",
        ),
        examples=(
            OperationExample(
                name="conjugate_three_plus_two_sqrt2",
                description="Conjugate 3+2*sqrt(2) to 3-2*sqrt(2) in QQ(sqrt(2)).",
                input={
                    "automorphism": _AUT_X2_CONJUGATION,
                    "element": {
                        "presentation": _X2_EXTENSION,
                        "coefficients_ascending": [
                            {"num": "3", "den": "1"},
                            {"num": "2", "den": "1"},
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.galois.factor_mod_p.compute",
        title="Factor a polynomial over GF(p)",
        description="Factor a polynomial over a prime finite field GF(p), retaining "
        "the source polynomial, unit, monic irreducible factors and multiplicities. "
        "Determines finite-field polynomial irreducibility from the supplied "
        "polynomial coefficients, including a degree-108 polynomial, using "
        "Frobenius linear algebra over GF(p). Admits degree d<=128, "
        "prime p<=251 and 32*(p+1)*d^3<=1,000,000,000 scalar updates; "
        "deterministic Berlekamp factorization has a shared 60-second safety deadline.",
        request_type=GaloisFactorRequest,
        result_type=GaloisFactorResult,
        run=_galois_factor,
        tags=("galois-theory", "factorization", "exact"),
        discovery_terms=(
            "finite field irreducibility",
            "degree 108 polynomial using Frobenius",
            "irreducibility from coefficients",
            "factor a polynomial over a finite field with Frobenius splitting",
            "decide finite-field polynomial irreducibility from coefficients",
        ),
        examples=(
            OperationExample(
                name="factor_x2_plus_1_over_f5",
                description="Factor x^2 + 1 over F_5.",
                input={"field_order": 5, "coefficients": [1, 0, 1]},
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.galois.frobenius_cycle.compute",
        title="Compute the Frobenius cycle type",
        description="Compute the Frobenius cycle type from supplied distinct "
        "factor degrees over GF(p), with total degree at most 128 and prime at "
        "most 251. Consumes a degree partition, not polynomial coefficients; "
        "it does not check that a source polynomial has the supplied factors.",
        request_type=FrobeniusCycleRequest,
        result_type=FrobeniusCycleResult,
        run=_frobenius_cycle,
        tags=("galois-theory", "frobenius", "exact"),
        discovery_terms=(
            "Frobenius cycle type from supplied factor degrees",
            "factorization degrees partition",
        ),
        examples=(
            OperationExample(
                name="irreducible_quadratic",
                description="Frobenius cycle of an irreducible quadratic.",
                input={
                    "field_order": 3,
                    "polynomial_degree": 2,
                    "factorization_degrees": [2],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.galois_group.compute",
        title="Compute the Galois group of a polynomial over Q",
        description="Compute the Galois group of a polynomial with rational coefficients "
        "using SymPy's galois_group function. The request must be irreducible "
        "and degree at most six; the result includes explicit generators.",
        request_type=GaloisGroupRequest,
        result_type=GaloisGroupResult,
        run=_galois_group,
        tags=("galois-theory", "galois-group", "exact"),
        examples=(
            OperationExample(
                name="galois_group_of_x2_minus_2",
                description="Galois group of x^2 - 2 over Q.",
                input={
                    "polynomial": {
                        "variables": ["x"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [2],
                                },
                                {
                                    "coefficient": {"num": "-2", "den": "1"},
                                    "exponents": [0],
                                },
                            ]
                        },
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial.solvable_by_radicals.decide",
        title="Decide if a polynomial is solvable by radicals",
        description="Check whether a polynomial is solvable by radicals based on its "
        "Galois group, within SymPy's irreducible degree-at-most-six domain.",
        request_type=SolvableRequest,
        result_type=SolvableResult,
        run=_solvable,
        tags=("galois-theory", "solvable", "exact"),
        examples=(
            OperationExample(
                name="x3_solvable",
                description="Check x^3 - 2 is solvable by radicals.",
                input={
                    "polynomial": {
                        "variables": ["x"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [3],
                                },
                                {
                                    "coefficient": {"num": "-2", "den": "1"},
                                    "exponents": [0],
                                },
                            ]
                        },
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
