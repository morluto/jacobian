"""Polynomial-derivation operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.derivations._models import (
    DerivationApplyRequest,
    DerivationApplyResult,
    DerivationCertificateRequest,
    DerivationIteratesRequest,
    DerivationIteratesResult,
    GaActionRequest,
    LocallyNilpotentCertificate,
    PolynomialGaAction,
)
from jacobian.math.polynomials.derivations._weight_models import (
    PolynomialWeightActionRequest,
    PolynomialWeightActionResult,
    PolynomialWeightInvariantRequest,
    PolynomialWeightInvariantResult,
    PolynomialWeightSubrepresentationRequest,
    PolynomialWeightSubrepresentationResult,
)
from jacobian.math.polynomials.derivations._weight_operations import (
    diagonal_weight_action,
    gm_generated_subrepresentation,
    gm_invariants_through_degree,
)
from jacobian.math.polynomials.derivations.operations import (
    apply_derivation,
    construct_locally_nilpotent_certificate,
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
            "bounds source variables and total exact output terms and "
            "expansion cells."
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
    *_TOOLS,
    MathTool(
        operation_id="algebraic_group.gm.finite_subrepresentation.compute",
        title="Compute the smallest G_m-stable polynomial span generated by polynomials",
        description=(
            "For one bounded diagonal integer-weight action, return the smallest "
            "stable QQ-subspace containing the supplied polynomial generators. "
            "The basis is deterministic and homogeneous by weight; exact source "
            "coordinates and the diagonal Laurent representation matrix retain "
            "the source ring and parameter axes. Generator count is at most 16, "
            "combined support at most 256 terms, output dimension at most 256, "
            "and projection row-reduction work is admitted before expansion. "
            "The complete serialized result, including echoed inputs, is "
            "bounded by the canonical 10 MiB output limit."
        ),
        request_type=PolynomialWeightSubrepresentationRequest,
        result_type=PolynomialWeightSubrepresentationResult,
        run=gm_generated_subrepresentation,
        tags=("algebraic-group", "gm", "subrepresentation", "weights", "exact"),
        discovery_terms=(
            "G_m generated finite subrepresentation",
            "stable polynomial span under diagonal torus action",
            "weight projections of polynomial generators",
            "multiplicative group representation matrix",
        ),
        examples=(
            OperationExample(
                name="span_of_a_mixed_weight_generator",
                description=(
                    "For weights (1,-1), the generated stable span of x+y has "
                    "basis (y,x), weights (-1,1), and source coordinates (1,1)."
                ),
                input={
                    "action": {"variables": ["x", "y"], "weights": [1, -1]},
                    "generators": [_poly(["x", "y"], [(1, [1, 0]), (1, [0, 1])])],
                    "parameter": "t",
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
        run=lambda request: gm_invariants_through_degree(
            request.action, request.degree
        ),
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
            "For an exact QQ polynomial ring with 1 to 7 ordered variables "
            "(the eighth Laurent carrier axis is reserved for the parameter) "
            "and one integer weight in [-64,64] per variable, return the "
            "Laurent coaction image, exact integer-weight decomposition, and "
            "weight-zero invariant projection of a same-parent polynomial. "
            "Source degree is at most 64 and source terms at most 256; these "
            "bounds are checked before constructing Laurent monomials. The "
            "diagonal formula "
            "lambda.x_i=lambda^w_i*x_i satisfies counit and coassociativity "
            "termwise, including negative weights."
        ),
        request_type=PolynomialWeightActionRequest,
        result_type=PolynomialWeightActionResult,
        run=lambda request: diagonal_weight_action(
            request.action, request.polynomial, request.parameter
        ),
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
