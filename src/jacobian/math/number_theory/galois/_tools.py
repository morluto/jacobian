"""Galois theory operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.galois._models import (
    AutomorphismApplyRequest,
    AutomorphismApplyResult,
    AutomorphismComposeRequest,
    AutomorphismRequest,
    AutomorphismResult,
    FrobeniusCycleRequest,
    FrobeniusCycleResult,
    GaloisFactorRequest,
    GaloisFactorResult,
    GaloisGroupRequest,
    GaloisGroupResult,
    QQFieldAutomorphism,
    SolvableRequest,
    SolvableResult,
    SplittingFieldRequest,
    SplittingFieldResult,
)
from jacobian.math.number_theory.galois.operations import (
    apply_automorphism,
    automorphisms,
    compose_automorphisms,
    frobenius_cycle,
    galois_factor,
    galois_group,
    solvable,
    splitting_field,
)


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
    return splitting_field(request.coefficients)


def _automorphisms(request: AutomorphismRequest) -> AutomorphismResult:
    return automorphisms(request.field)


def _compose(request: AutomorphismComposeRequest) -> QQFieldAutomorphism:
    return compose_automorphisms(request.first, request.second)


def _apply(request: AutomorphismApplyRequest) -> AutomorphismApplyResult:
    return AutomorphismApplyResult(
        root=apply_automorphism(request.automorphism, request.root)
    )


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


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="number_field.polynomial.splitting_field.compute",
        title="Construct a bounded exact QQ splitting field",
        description="Construct an exact QQ splitting-field carrier with a complete simple-root axis, basis axis, and source-polynomial reconstruction; the source must be irreducible over QQ and degree at most six.",
        request_type=SplittingFieldRequest,
        result_type=SplittingFieldResult,
        run=_splitting,
        tags=("galois-theory", "splitting-field", "exact"),
        examples=(
            OperationExample(
                name="split_x2_minus2",
                description="Construct the splitting field of x^2-2; the polynomial must have bounded exact QQ coefficients.",
                input=_SPLIT_X2,
            ),
        ),
    ),
    MathTool(
        operation_id="number_field.extension.automorphisms.compute",
        title="Compute exact field automorphism generators",
        description="Compute exact QQ-automorphism generators with their faithful root permutations for a bounded splitting field; the field must come from an irreducible degree-at-most-six polynomial.",
        request_type=AutomorphismRequest,
        result_type=AutomorphismResult,
        run=_automorphisms,
        tags=("galois-theory", "automorphism", "exact"),
        examples=(
            OperationExample(
                name="automorphisms_x2_minus2",
                description="Compute automorphism generators of the x^2-2 splitting field; the field parent must retain its exact source polynomial and root axis.",
                input={
                    "field": {
                        "source": _SPLIT_X2["polynomial"],
                        "basis_labels": ["b_0", "b_1"],
                        "root_labels": ["root_0", "root_1"],
                        "degree": 2,
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="number_field.automorphism.compose.compute",
        title="Compose exact field automorphisms",
        description="Compose two root-permuting automorphisms in one exact splitting-field parent; both maps must retain the same field and root axis.",
        request_type=AutomorphismComposeRequest,
        result_type=QQFieldAutomorphism,
        run=_compose,
        tags=("galois-theory", "automorphism", "exact"),
        examples=(
            OperationExample(
                name="identity_composition",
                description="Compose two identity root permutations; both automorphisms must share one exact splitting-field parent.",
                input={
                    "first": {
                        "field": {
                            "source": _SPLIT_X2["polynomial"],
                            "basis_labels": ["b_0", "b_1"],
                            "root_labels": ["root_0", "root_1"],
                            "degree": 2,
                        },
                        "root_permutation": [0, 1],
                    },
                    "second": {
                        "field": {
                            "source": _SPLIT_X2["polynomial"],
                            "basis_labels": ["b_0", "b_1"],
                            "root_labels": ["root_0", "root_1"],
                            "degree": 2,
                        },
                        "root_permutation": [0, 1],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="number_field.automorphism.apply.compute",
        title="Apply an exact field automorphism to a root",
        description="Apply a root-permuting exact automorphism while retaining the splitting-field parent; root and automorphism must share the same field.",
        request_type=AutomorphismApplyRequest,
        result_type=AutomorphismApplyResult,
        run=_apply,
        tags=("galois-theory", "automorphism", "exact"),
        examples=(
            OperationExample(
                name="identity_root",
                description="Apply the identity automorphism to a root; both objects must retain the same exact splitting-field parent.",
                input={
                    "automorphism": {
                        "field": {
                            "source": _SPLIT_X2["polynomial"],
                            "basis_labels": ["b_0", "b_1"],
                            "root_labels": ["root_0", "root_1"],
                            "degree": 2,
                        },
                        "root_permutation": [0, 1],
                    },
                    "root": {
                        "field": {
                            "source": _SPLIT_X2["polynomial"],
                            "basis_labels": ["b_0", "b_1"],
                            "root_labels": ["root_0", "root_1"],
                            "degree": 2,
                        },
                        "index": 0,
                        "multiplicity": 1,
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
