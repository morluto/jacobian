"""Polynomial-derivation operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.derivations._models import (
    DerivationApplyRequest,
    DerivationApplyResult,
    DerivationCertificateRequest,
    DerivationFromVectorFieldRequest,
    DerivationIteratesRequest,
    DerivationIteratesResult,
    GaActionRequest,
    LocallyNilpotentCertificate,
    PolynomialDerivation,
    PolynomialGaAction,
)
from jacobian.math.polynomials.derivations._stable_models import (
    PolynomialGaFixedSubspace,
    PolynomialGaFixedSubspaceRequest,
    PolynomialGaStableSubrepresentation,
    PolynomialGaStableSubrepresentationRequest,
)
from jacobian.math.polynomials.derivations._stable_operations import (
    ga_fixed_subspace,
    ga_stable_subrepresentation,
)
from jacobian.math.polynomials.derivations._weight_models import (
    PolynomialWeightActionRequest,
    PolynomialWeightActionResult,
    PolynomialWeightInvariantRequest,
    PolynomialWeightInvariantResult,
)
from jacobian.math.polynomials.derivations._weight_operations import (
    diagonal_weight_action,
    gm_invariants_through_degree,
)
from jacobian.math.polynomials.derivations.operations import (
    apply_derivation,
    construct_locally_nilpotent_certificate,
    derivation_from_vector_field,
    derivation_iterates,
    ga_action_from_derivation,
)


def _run_apply_derivation(request: DerivationApplyRequest) -> DerivationApplyResult:
    return apply_derivation(request.derivation, request.polynomial)


def _rational(value: int) -> dict[str, str]:
    return {"num": str(value), "den": "1"}


def _poly(variables: list[str], terms: list[tuple[int, list[int]]]) -> dict[str, Any]:
    return {
        "domain": "QQ",
        "variables": variables,
        "polynomial": {
            "terms": [
                {"coefficient": _rational(coefficient), "exponents": exponents}
                for coefficient, exponents in terms
            ]
        },
    }


_TRIANGULAR_IMAGES = [
    _poly(["x", "y"], [(1, [0, 1])]),
    _poly(["x", "y"], []),
]

_TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="polynomial_derivation.apply.compute",
        title="Apply an exact polynomial derivation to a polynomial",
        description=(
            "Compute D(f) = sum_i D(x_i) * partial_i(f) for a checked "
            "polynomial derivation on QQ[x_1, ..., x_n] with n between 1 and "
            "8 and a source polynomial in the same ordered ring, returning "
            "the canonical result with one contribution polynomial per "
            "generator. Source images and polynomials are bounded in terms, "
            "degree, and coefficient digits, and predicted contribution "
            "cells are preflighted before expansion."
        ),
        request_type=DerivationApplyRequest,
        result_type=DerivationApplyResult,
        run=_run_apply_derivation,
        tags=("polynomial", "derivation", "exact", "ga-action"),
        discovery_terms=(
            "polynomial derivation action",
            "Leibniz rule application",
            "locally nilpotent derivation iterate",
            "additive group infinitesimal action",
        ),
        examples=(
            OperationExample(
                name="triangular_derivation_on_square",
                description=(
                    "Apply D(x)=y, D(y)=0 to x^2 to get 2*x*y with ledger "
                    "(2*x*y, 0); the derivation images and the source must "
                    "share one ordered QQ ring."
                ),
                input={
                    "derivation": {
                        "variables": ["x", "y"],
                        "images": _TRIANGULAR_IMAGES,
                    },
                    "polynomial": _poly(["x", "y"], [(1, [2, 0])]),
                },
            ),
        ),
    ),
)

_ITERATE_EXAMPLE = {
    "derivation": {"variables": ["x", "y"], "images": _TRIANGULAR_IMAGES},
    "polynomial": _poly(["x", "y"], [(1, [2, 0])]),
    "bound": 3,
}
_CERTIFICATE_EXAMPLE = {
    "derivation": {"variables": ["x", "y"], "images": _TRIANGULAR_IMAGES},
    "chains": [
        [
            _poly(["x", "y"], [(1, [1, 0])]),
            _poly(["x", "y"], [(1, [0, 1])]),
            _poly(["x", "y"], []),
        ],
        [_poly(["x", "y"], [(1, [0, 1])]), _poly(["x", "y"], [])],
    ],
}

_TOOLS += (
    MathTool(
        operation_id="polynomial_derivation.from_vector_field.compute",
        title="Convert a polynomial vector field into its algebra derivation",
        description=(
            "Bind one exact polynomial component f_i to each generator x_i, "
            "representing D=sum_i f_i partial_i on the same ordered QQ ring. "
            "The conversion retains the complete variable axis and admits all "
            "component term, degree, exponent, and coefficient bounds before "
            "publishing the derivation."
        ),
        request_type=DerivationFromVectorFieldRequest,
        result_type=PolynomialDerivation,
        run=lambda request: derivation_from_vector_field(request),
        tags=("polynomial", "vector-field", "derivation", "exact"),
        discovery_terms=(
            "polynomial vector field as derivation",
            "derivation from vector field",
            "infinitesimal polynomial action",
        ),
        examples=(
            OperationExample(
                name="triangular_vector_field",
                description=(
                    "The vector field y partial_x has generator images D(x)=y "
                    "and D(y)=0 in the ordered ring QQ[x,y]."
                ),
                input={"components": _TRIANGULAR_IMAGES},
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial_derivation.iterates.compute",
        title="Compute a bounded exact derivation iterate profile",
        description="Compute D^0(f) through D^N(f), returning the first zero when reached or a nonzero-through-bound profile; a finite nonzero prefix is not a global nonnilpotence claim.",
        request_type=DerivationIteratesRequest,
        result_type=DerivationIteratesResult,
        run=lambda request: derivation_iterates(
            request.derivation, request.polynomial, request.bound
        ),
        tags=("polynomial", "derivation", "iterate", "exact"),
        examples=(
            OperationExample(
                name="triangular_iterates",
                description="Compute the bounded iterates of D(x)=y,D(y)=0 on x^2; the bound is finite and the source ring is shared.",
                input=_ITERATE_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial_derivation.local_nilpotence_certificate.construct",
        title="Construct a checked locally nilpotent derivation certificate",
        description="Check complete exact generator iterate chains ending at zero and return the locally nilpotent derivation certificate; mutated chains are rejected.",
        request_type=DerivationCertificateRequest,
        result_type=LocallyNilpotentCertificate,
        run=lambda request: construct_locally_nilpotent_certificate(
            request.derivation, request.chains
        ),
        tags=("polynomial", "derivation", "locally-nilpotent", "certificate"),
        examples=(
            OperationExample(
                name="triangular_certificate",
                description="Construct the generator-chain certificate for D(x)=y,D(y)=0; every chain must begin at its generator and end at zero.",
                input=_CERTIFICATE_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="algebraic_group.ga.action_from_derivation.compute",
        title="Exponentiate a checked locally nilpotent derivation",
        description=(
            "Check exact generator chains for local nilpotence, then return the "
            "finite exponential coaction exp(tD) on generators. Admission "
            "bounds source variables and total exact output terms and bytes."
        ),
        request_type=GaActionRequest,
        result_type=PolynomialGaAction,
        run=lambda request: ga_action_from_derivation(
            request.derivation, request.chains
        ),
        tags=("algebraic-group", "ga", "derivation", "exact"),
        examples=(
            OperationExample(
                name="triangular_ga_action",
                description="Exponentiate D(x)=y,D(y)=0 to x maps to x+t*y and y maps to y; the input generator chains must be complete.",
                input=_CERTIFICATE_EXAMPLE,
            ),
        ),
    ),
)

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="algebraic_group.ga.fixed_subspace.compute",
        title="Compute fixed polynomials in a finite Ga-subrepresentation",
        description=(
            "Given a serialized finite-dimensional Ga-stable polynomial span, "
            "recheck its action matrix and return a deterministic exact basis "
            "for the fixed subspace, with coordinates in the supplied basis. "
            "Over QQ this is the kernel of the coefficient of t in the action "
            "matrix. The supplied representation dimension is at most 32; "
            "polynomial matrix reconstruction uses the stable-subrepresentation "
            "operation's bounded admission."
        ),
        request_type=PolynomialGaFixedSubspaceRequest,
        result_type=PolynomialGaFixedSubspace,
        run=lambda request: ga_fixed_subspace(request.subrepresentation),
        tags=("algebraic-group", "ga", "fixed-subspace", "invariants", "exact"),
        discovery_terms=(
            "Ga fixed polynomials",
            "additive group invariants in a finite subrepresentation",
            "kernel of infinitesimal representation",
        ),
        examples=(
            OperationExample(
                name="translation_fixes_constants",
                description=(
                    "For the translation action x -> x+t on the span (1,x), "
                    "the fixed subspace is the constants, represented by "
                    "coordinate vector (1,0)."
                ),
                input={
                    "subrepresentation": {
                        "action": {
                            "source_variables": ["x"],
                            "parameter": "t",
                            "generator_images": [
                                _poly(["x", "t"], [(1, [1, 0]), (1, [0, 1])])
                            ],
                        },
                        "basis": [
                            _poly(["x"], [(1, [0])]),
                            _poly(["x"], [(1, [1])]),
                        ],
                        "action_matrix": [
                            [
                                _poly(["t"], [(1, [0])]),
                                _poly(["t"], [(1, [1])]),
                            ],
                            [
                                _poly(["t"], []),
                                _poly(["t"], [(1, [0])]),
                            ],
                        ],
                    }
                },
            ),
        ),
    ),
    *_TOOLS,
    MathTool(
        operation_id="algebraic_group.ga.stable_subrepresentation.compute",
        title="Check a supplied finite polynomial Ga-stable subspace",
        description=(
            "Given a checked additive-group action and an ordered linearly "
            "independent finite list of QQ polynomials, compute the exact "
            "QQ[t] action matrix on their span if every action image lies in "
            "that span. This operation checks only the explicitly supplied "
            "span; it does not search for all finite-dimensional "
            "subrepresentations. Expansion work, degree, dimension, terms, "
            "and coefficient growth are admitted before polynomial expansion."
        ),
        request_type=PolynomialGaStableSubrepresentationRequest,
        result_type=PolynomialGaStableSubrepresentation,
        run=lambda request: ga_stable_subrepresentation(request),
        tags=("algebraic-group", "ga", "polynomial", "representation", "exact"),
        discovery_terms=(
            "finite-dimensional additive group subrepresentation",
            "Ga-stable polynomial span",
            "polynomial action matrix on a supplied basis",
        ),
        examples=(
            OperationExample(
                name="translation_on_linear_polynomials",
                description=(
                    "For x maps to x+t, the supplied ordered basis (1,x) is "
                    "stable, with action matrix columns (1,0) and (t,1)."
                ),
                input={
                    "action": {
                        "source_variables": ["x"],
                        "parameter": "t",
                        "generator_images": [
                            {
                                "domain": "QQ",
                                "variables": ["x", "t"],
                                "polynomial": {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [1, 0],
                                        },
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [0, 1],
                                        },
                                    ]
                                },
                            }
                        ],
                    },
                    "basis": [
                        {
                            "domain": "QQ",
                            "variables": ["x"],
                            "polynomial": {
                                "terms": [
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "exponents": [0],
                                    }
                                ]
                            },
                        },
                        {
                            "domain": "QQ",
                            "variables": ["x"],
                            "polynomial": {
                                "terms": [
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "exponents": [1],
                                    }
                                ]
                            },
                        },
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="algebraic_group.gm.invariants_through_degree.compute",
        title="Compute the bounded exact G_m invariant polynomial slice",
        description=(
            "Enumerate the complete monomial basis of QQ[x_1,...,x_n] in "
            "total degree at most d, retain precisely those exponent vectors "
            "whose dot product with the declared integer variable weights is "
            "zero, and return the basis and Hilbert-function prefix. This is "
            "the exact invariant subspace of the finite degree slice, not a "
            "generating set for the global invariant ring. The full candidate "
            "monomial count is admitted before enumeration and is bounded by "
            "4096; weights retain the diagonal G_m coaction semantics, including "
            "negative weights."
        ),
        request_type=PolynomialWeightInvariantRequest,
        result_type=PolynomialWeightInvariantResult,
        run=gm_invariants_through_degree,
        tags=("algebraic-group", "gm", "invariants", "graded", "exact"),
        discovery_terms=(
            "G_m invariant polynomials through degree",
            "weight-zero monomial basis",
            "multiplicative group Hilbert prefix",
        ),
        examples=(
            OperationExample(
                name="opposite_weights_invariants_through_degree_four",
                description="For weights (1,-1), the invariant monomials through degree four are 1, x*y, and x^2*y^2.",
                input={
                    "action": {"variables": ["x", "y"], "weights": [1, -1]},
                    "degree": 4,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="algebraic_group.gm.diagonal_weight_action.compute",
        title="Apply a bounded diagonal integer-weight G_m action",
        description=(
            "For an exact QQ polynomial ring with 1 to 8 ordered variables and "
            "one integer weight in [-64,64] per variable, return the Laurent "
            "coaction image, exact integer-weight decomposition, and weight-zero "
            "invariant projection of a same-parent polynomial. Source degree is "
            "at most 64 and source terms at most 256; these bounds are checked "
            "before constructing Laurent monomials. The diagonal formula "
            "lambda.x_i=lambda^w_i*x_i satisfies counit and coassociativity "
            "termwise, including negative weights."
        ),
        request_type=PolynomialWeightActionRequest,
        result_type=PolynomialWeightActionResult,
        run=diagonal_weight_action,
        tags=("algebraic-group", "gm", "polynomial", "weights", "exact"),
        discovery_terms=(
            "multiplicative group polynomial action",
            "diagonal integer weights",
            "polynomial weight decomposition",
            "weight zero invariant polynomial slice",
        ),
        examples=(
            OperationExample(
                name="positive_zero_and_negative_weights",
                description=(
                    "For weights (1,-1,0), x*y*z is fixed, while x^2 and y^2 "
                    "have weights 2 and -2. The Laurent parameter is separate."
                ),
                input={
                    "action": {"variables": ["x", "y", "z"], "weights": [1, -1, 0]},
                    "polynomial": _poly(
                        ["x", "y", "z"],
                        [(1, [2, 0, 0]), (1, [1, 1, 1]), (1, [0, 2, 0])],
                    ),
                    "parameter": "t",
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
