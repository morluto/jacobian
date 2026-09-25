"""Finite-dimensional Lie-algebra operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.lie_algebras._models import (
    FiniteDimensionalLieAlgebra,
    LieAdjointRepresentationResult,
    LieAdjointRequest,
    LieAdjointResult,
    LieAlgebraRequest,
    LieBracketRequest,
    LieBracketResult,
    LieCenterResult,
    LieCentralizerRequest,
    LieCentralizerResult,
    LieDerivedSeriesResult,
    LieDirectSumRequest,
    LieGeneratedIdealRequest,
    LieGeneratedSubalgebraRequest,
    LieIdeal,
    LieIdealCheckResult,
    LieIdealRequest,
    LieKillingRadicalResult,
    LieKillingResult,
    LieLowerCentralSeriesResult,
    LieQuotientRequest,
    LieQuotientResult,
    LieSubalgebra,
    LieSubalgebraCheckResult,
    LieSubalgebraRequest,
    LieUpperCentralSeriesResult,
)
from jacobian.math.lie_algebras.operations import (
    check_ideal,
    check_subalgebra,
    lie_adjoint,
    lie_adjoint_representation,
    lie_bracket,
    lie_center,
    lie_derived_series,
    lie_derived_subalgebra,
    lie_direct_sum,
    lie_generated_ideal,
    lie_generated_subalgebra,
    lie_killing_form,
    lie_killing_form_radical,
    lie_lower_central_series,
    lie_quotient,
    lie_subalgebra_centralizer,
    lie_upper_central_series,
)


def _run_lie_bracket(request: LieBracketRequest) -> LieBracketResult:
    return lie_bracket(request.algebra, request.left, request.right)


def _run_lie_killing_form(request: LieAlgebraRequest) -> LieKillingResult:
    return lie_killing_form(request.algebra)


def _run_lie_killing_form_radical(
    request: LieAlgebraRequest,
) -> LieKillingRadicalResult:
    return lie_killing_form_radical(request.algebra)


def _run_lie_adjoint_representation(
    request: LieAlgebraRequest,
) -> LieAdjointRepresentationResult:
    return lie_adjoint_representation(request.algebra)


def _run_lie_adjoint(request: LieAdjointRequest) -> LieAdjointResult:
    return lie_adjoint(request.algebra, request.element)


def _run_lie_center(request: LieAlgebraRequest) -> LieCenterResult:
    return lie_center(request.algebra)


def _run_lie_centralizer(request: LieCentralizerRequest) -> LieCentralizerResult:
    return lie_subalgebra_centralizer(request.algebra, request.elements)


def _run_lie_derived_series(request: LieAlgebraRequest) -> LieDerivedSeriesResult:
    return lie_derived_series(request.algebra)


def _run_lie_derived_subalgebra(request: LieAlgebraRequest) -> LieIdeal:
    return lie_derived_subalgebra(request.algebra)


def _run_lie_lower_central_series(
    request: LieAlgebraRequest,
) -> LieLowerCentralSeriesResult:
    return lie_lower_central_series(request.algebra)


def _run_lie_upper_central_series(
    request: LieAlgebraRequest,
) -> LieUpperCentralSeriesResult:
    return lie_upper_central_series(request.algebra)


def _run_lie_ideal_check(request: LieIdealRequest) -> LieIdealCheckResult:
    return check_ideal(request.algebra, request.candidate)


def _run_lie_subalgebra_check(
    request: LieSubalgebraRequest,
) -> LieSubalgebraCheckResult:
    return check_subalgebra(request.algebra, request.candidate)


def _run_lie_generated_subalgebra(
    request: LieGeneratedSubalgebraRequest,
) -> LieSubalgebra:
    return lie_generated_subalgebra(request.algebra, request.generators)


def _run_lie_generated_ideal(request: LieGeneratedIdealRequest) -> LieIdeal:
    return lie_generated_ideal(request.algebra, request.generators)


def _run_lie_quotient(request: LieQuotientRequest) -> LieQuotientResult:
    return lie_quotient(request.algebra, request.ideal, request.quotient_basis)


def _run_lie_direct_sum(
    request: LieDirectSumRequest,
) -> FiniteDimensionalLieAlgebra:
    return lie_direct_sum(request.left, request.right, request.basis)


def _rational(value: int) -> dict[str, str]:
    return {"num": str(value), "den": "1"}


_SL2_ALGEBRA = {
    "basis": ["e", "f", "h"],
    "structure_constants": [
        {"i": 0, "j": 1, "k": 2, "coefficient": _rational(1)},
        {"i": 0, "j": 2, "k": 0, "coefficient": _rational(-2)},
        {"i": 1, "j": 2, "k": 1, "coefficient": _rational(2)},
    ],
}


def _element(coords: list[int]) -> dict[str, Any]:
    return {
        "basis": ["e", "f", "h"],
        "coordinates": [_rational(value) for value in coords],
    }


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="lie_algebra.adjoint.compute",
        title="Compute the exact adjoint action of a Lie-algebra element",
        description=(
            "Return the exact rational matrix ad_x(y) = [x,y] for one element "
            "x in a finite-dimensional Lie algebra over QQ, acting on column "
            "coordinate vectors in the retained ordered basis. The element must "
            "use the algebra's basis axis. Dimension is at most 8; admission "
            "establishes every Jacobi identity and bounds coefficient growth, "
            "arithmetic work, and matrix output before expansion."
        ),
        request_type=LieAdjointRequest,
        result_type=LieAdjointResult,
        run=_run_lie_adjoint,
        tags=("lie-algebra", "adjoint", "exact", "rational"),
        discovery_terms=(
            "adjoint action of a Lie algebra element",
            "ad_x matrix",
            "inner derivation matrix",
            "matrix of x bracket y",
        ),
        examples=(
            OperationExample(
                name="heisenberg_adjoint_x",
                description=(
                    "Compute ad_x in the Heisenberg algebra [x,y] = z, "
                    "retaining the exact ordered basis."
                ),
                input={
                    "algebra": {
                        "basis": ["x", "y", "z"],
                        "structure_constants": [
                            {
                                "i": 0,
                                "j": 1,
                                "k": 2,
                                "coefficient": _rational(1),
                            }
                        ],
                    },
                    "element": {
                        "basis": ["x", "y", "z"],
                        "coordinates": [_rational(1), _rational(0), _rational(0)],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="lie_algebra.bracket.compute",
        title="Compute the exact bracket of two Lie-algebra elements",
        description=(
            "Compute [x, y] for two exact rational coordinate vectors in one "
            "finite-dimensional Lie algebra over QQ given by ordered basis "
            "labels and sparse structure constants with dimension at most 8, "
            "returning exact bracket coordinates with one ledger row per "
            "nonzero basis pair. Antisymmetry is canonical in the stored "
            "table and every basis-triple Jacobi identity is established by "
            "operation admission before expansion."
        ),
        request_type=LieBracketRequest,
        result_type=LieBracketResult,
        run=_run_lie_bracket,
        tags=("lie-algebra", "bracket", "exact", "rational"),
        discovery_terms=(
            "Lie bracket of basis elements",
            "structure constant expansion",
            "sl2 commutator",
            "Heisenberg bracket",
        ),
        examples=(
            OperationExample(
                name="sl2_bracket_e_f",
                description=(
                    "Compute [e, f] = h in sl2(QQ); both elements must use "
                    "the algebra's ordered basis and the constants must "
                    "satisfy antisymmetry and Jacobi."
                ),
                input={
                    "algebra": _SL2_ALGEBRA,
                    "left": _element([1, 0, 0]),
                    "right": _element([0, 1, 0]),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="lie_algebra.adjoint_representation.compute",
        title="Compute the exact adjoint representation matrices",
        description=(
            "Return one exact rational matrix ad(b_i) per ordered basis element "
            "of a finite-dimensional Lie algebra over QQ, acting on column "
            "coordinate vectors. The source algebra and basis order are retained. "
            "Dimension is at most 8; admission verifies every Jacobi identity, "
            "establishing [ad(x), ad(y)] = ad([x,y])."
        ),
        request_type=LieAlgebraRequest,
        result_type=LieAdjointRepresentationResult,
        run=_run_lie_adjoint_representation,
        tags=("lie-algebra", "adjoint", "representation", "exact", "rational"),
        discovery_terms=(
            "Lie algebra adjoint representation",
            "adjoint matrices",
            "ad_x matrix",
            "adjoint action of basis vectors",
        ),
        examples=(
            OperationExample(
                name="sl2_adjoint_representation",
                description=(
                    "Return ad(e), ad(f), and ad(h) in the declared sl2 basis "
                    "order after Jacobi admission."
                ),
                input={"algebra": _SL2_ALGEBRA},
            ),
        ),
    ),
    MathTool(
        operation_id="lie_algebra.killing_form.compute",
        title="Compute the exact Killing form of a Lie algebra",
        description=(
            "Compute the exact Killing-form Gram matrix tr(ad_x ad_y) of a "
            "finite-dimensional Lie algebra over QQ in its ordered basis, "
            "retaining the source algebra. Every basis-triple Jacobi "
            "identity is established by operation admission before the "
            "adjoint traces run, so the matrix is the Killing form of a Lie "
            "algebra. Consuming operations read semisimplicity and the "
            "radical off this matrix."
        ),
        request_type=LieAlgebraRequest,
        result_type=LieKillingResult,
        run=_run_lie_killing_form,
        tags=("lie-algebra", "killing-form", "exact", "rational"),
        discovery_terms=(
            "Killing form matrix",
            "Cartan criterion semisimplicity",
            "adjoint trace form",
            "Lie algebra invariant bilinear form",
        ),
        examples=(
            OperationExample(
                name="sl2_killing_form",
                description=(
                    "Compute the sl2(QQ) Killing matrix with κ(e,f) = 4 and "
                    "κ(h,h) = 8; the constants must satisfy antisymmetry "
                    "and Jacobi."
                ),
                input={"algebra": _SL2_ALGEBRA},
            ),
        ),
    ),
    MathTool(
        operation_id="lie_algebra.killing_form.radical.compute",
        title="Compute the radical of a Lie algebra's Killing form",
        description=(
            "Compute the nullspace of the exact Killing-form matrix over QQ, "
            "returning its canonical RREF rows on the source algebra basis "
            "together with the source algebra and complete KillingResult. "
            "This is the radical of the bilinear form and is not asserted to "
            "be the solvable radical of the Lie algebra."
        ),
        request_type=LieAlgebraRequest,
        result_type=LieKillingRadicalResult,
        run=_run_lie_killing_form_radical,
        tags=("lie-algebra", "killing-form", "radical", "exact", "rational"),
        discovery_terms=(
            "Killing form radical",
            "nullspace of Killing form",
            "radical of the Killing bilinear form",
            "Cartan semisimplicity criterion",
            "is a Lie algebra semisimple",
            "Killing form nondegenerate",
            "Lie algebra semisimplicity decision",
        ),
        examples=(
            OperationExample(
                name="heisenberg_killing_form_radical",
                description=(
                    "The three-dimensional Heisenberg algebra has zero Killing "
                    "form, so its bilinear-form radical is the full source "
                    "space; this result does not label it the solvable radical."
                ),
                input={
                    "algebra": {
                        "basis": ["x", "y", "z"],
                        "structure_constants": [
                            {
                                "i": 0,
                                "j": 1,
                                "k": 2,
                                "coefficient": {"num": "1", "den": "1"},
                            }
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="lie_algebra.center.compute",
        title="Compute the exact center of a Lie algebra",
        description=(
            "Compute the exact center of a finite-dimensional Lie algebra "
            "over QQ as a canonical RREF subspace of its ordered basis, "
            "retaining the source algebra. Centrality against every basis "
            "element is one exact rational linear system solved through "
            "maintained nullspace and RREF kernels after operation "
            "admission establishes every Jacobi identity. Consuming "
            "series, quotient, and nilpotency operations accept the "
            "subspace unchanged."
        ),
        request_type=LieAlgebraRequest,
        result_type=LieCenterResult,
        run=_run_lie_center,
        tags=("lie-algebra", "center", "exact", "rational"),
        discovery_terms=(
            "Lie algebra center",
            "central elements",
            "abelian center subspace",
            "bracket kernel subspace",
        ),
        examples=(
            OperationExample(
                name="heisenberg_center",
                description=(
                    "Compute the one-dimensional span of z as the "
                    "Heisenberg center; the constants must satisfy "
                    "antisymmetry and Jacobi."
                ),
                input={
                    "algebra": {
                        "basis": ["x", "y", "z"],
                        "structure_constants": [
                            {
                                "i": 0,
                                "j": 1,
                                "k": 2,
                                "coefficient": {"num": "1", "den": "1"},
                            }
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="lie_algebra.subalgebra.centralizer.compute",
        title="Compute the exact centralizer of Lie algebra elements",
        description=(
            "Return the complete common centralizer of up to dim(g) exact "
            "rational vectors in a finite-dimensional Lie algebra over QQ as "
            "a canonical RREF LieSubalgebra bound to the source algebra. The "
            "empty family has centralizer g. Admission establishes Jacobi, "
            "bounds coordinate digits and the stacked linear system, and "
            "the defining equations are [x,s]=0 for every supplied vector s."
        ),
        request_type=LieCentralizerRequest,
        result_type=LieCentralizerResult,
        run=_run_lie_centralizer,
        tags=("lie-algebra", "centralizer", "subalgebra", "exact", "rational"),
        discovery_terms=(
            "Lie algebra centralizer",
            "centralizer of a Lie subalgebra",
            "vectors commuting with a Lie algebra element",
            "common kernel of adjoint maps",
        ),
        examples=(
            OperationExample(
                name="heisenberg_centralizer_of_x",
                description=(
                    "In the three-dimensional Heisenberg algebra, the "
                    "centralizer of x is span(x,z)."
                ),
                input={
                    "algebra": {
                        "basis": ["x", "y", "z"],
                        "structure_constants": [
                            {"i": 0, "j": 1, "k": 2, "coefficient": _rational(1)}
                        ],
                    },
                    "elements": [
                        {
                            "basis": ["x", "y", "z"],
                            "coordinates": [_rational(1), _rational(0), _rational(0)],
                        }
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="lie_algebra.derived_subalgebra.compute",
        title="Compute the derived subalgebra",
        description=(
            "Return [g,g] = span{[b_i,b_j]} as a canonical exact LieIdeal "
            "in the source algebra's ordered basis. Admission establishes "
            "antisymmetry and every Jacobi identity before bracket expansion; "
            "the result retains the ambient algebra and composes with ideal "
            "operations such as quotient construction."
        ),
        request_type=LieAlgebraRequest,
        result_type=LieIdeal,
        run=_run_lie_derived_subalgebra,
        tags=("lie-algebra", "derived-subalgebra", "commutator", "exact", "rational"),
        discovery_terms=(
            "Lie derived subalgebra",
            "commutator ideal",
            "span of Lie brackets",
            "[g,g]",
        ),
        examples=(
            OperationExample(
                name="heisenberg_derived_subalgebra",
                description="The Heisenberg derived ideal is the central span of z.",
                input={
                    "algebra": {
                        "basis": ["x", "y", "z"],
                        "structure_constants": [
                            {"i": 0, "j": 1, "k": 2, "coefficient": _rational(1)}
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="lie_algebra.derived_series.compute",
        title="Compute the derived series with its solvability decision",
        description=(
            "Compute the derived series D^0 = g, D^{k+1} = [D^k, D^k] of a "
            "finite-dimensional Lie algebra over QQ as canonical RREF "
            "subspaces, stopping at zero or the first fixed term, with the "
            "solvability decision. Every Jacobi identity is established by "
            "operation admission first, so each bracket family spans an "
            "ideal and the series descends."
        ),
        request_type=LieAlgebraRequest,
        result_type=LieDerivedSeriesResult,
        run=_run_lie_derived_series,
        tags=("lie-algebra", "derived-series", "solvable", "exact", "rational"),
        discovery_terms=(
            "Lie algebra derived series",
            "solvability decision",
            "commutator subspace chain",
        ),
        examples=(
            OperationExample(
                name="heisenberg_derived_series",
                description=(
                    "Compute the Heisenberg derived dimensions 3, 1, 0 with "
                    "a solvable decision; the constants must satisfy "
                    "antisymmetry and Jacobi."
                ),
                input={
                    "algebra": {
                        "basis": ["x", "y", "z"],
                        "structure_constants": [
                            {
                                "i": 0,
                                "j": 1,
                                "k": 2,
                                "coefficient": {"num": "1", "den": "1"},
                            }
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="lie_algebra.lower_central_series.compute",
        title="Compute the lower central series with its nilpotency decision",
        description=(
            "Compute the lower central series L^0 = g, L^{k+1} = [g, L^k] "
            "of a finite-dimensional Lie algebra over QQ as canonical RREF "
            "subspaces, stopping at zero or the first fixed term, with the "
            "nilpotency decision. Every Jacobi identity is established by "
            "operation admission first, so each bracket family spans an "
            "ideal and the series descends."
        ),
        request_type=LieAlgebraRequest,
        result_type=LieLowerCentralSeriesResult,
        run=_run_lie_lower_central_series,
        tags=("lie-algebra", "central-series", "nilpotent", "exact", "rational"),
        discovery_terms=(
            "Lie algebra lower central series",
            "nilpotency decision",
            "nilpotency class",
        ),
        examples=(
            OperationExample(
                name="affine_lower_central_series",
                description=(
                    "Compute the ax+b lower central dimensions 2, 1 with a "
                    "non-nilpotent decision; the constants must satisfy "
                    "antisymmetry and Jacobi."
                ),
                input={
                    "algebra": {
                        "basis": ["a", "b"],
                        "structure_constants": [
                            {
                                "i": 0,
                                "j": 1,
                                "k": 1,
                                "coefficient": {"num": "1", "den": "1"},
                            }
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="lie_algebra.upper_central_series.compute",
        title="Compute the upper central series with its nilpotency decision",
        description=(
            "Compute Z_0 = 0 and Z_(i+1)/Z_i = center(L/Z_i) over QQ as "
            "canonical RREF subspaces on the original ordered basis. The chain "
            "ends at L exactly for nilpotent algebras; otherwise it stops at "
            "the first stable proper term. Dimension is bounded by 8."
        ),
        request_type=LieAlgebraRequest,
        result_type=LieUpperCentralSeriesResult,
        run=_run_lie_upper_central_series,
        tags=("lie-algebra", "central-series", "nilpotent", "exact", "rational"),
        discovery_terms=(
            "Lie algebra upper central series",
            "nilpotency decision",
            "ascending central series",
        ),
        examples=(
            OperationExample(
                name="heisenberg_upper_central_series",
                description=(
                    "Compute the Heisenberg upper central dimensions 0, 1, 3; "
                    "the constants must satisfy antisymmetry and Jacobi."
                ),
                input={
                    "algebra": {
                        "basis": ["x", "y", "z"],
                        "structure_constants": [
                            {"i": 0, "j": 1, "k": 2, "coefficient": _rational(1)}
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="lie_algebra.ideal.check",
        title="Decide whether a subspace is a Lie-algebra ideal",
        description=(
            "Decide whether an RREF subspace of a finite-dimensional Lie "
            "algebra over QQ absorbs every algebra bracket, retaining the "
            "first escaping bracket in increasing basis and generator order "
            "on failure. Every Jacobi identity is established by operation "
            "admission first. Consuming quotient operations accept an IDEAL "
            "verdict unchanged."
        ),
        request_type=LieIdealRequest,
        result_type=LieIdealCheckResult,
        run=_run_lie_ideal_check,
        tags=("lie-algebra", "ideal", "exact", "rational"),
        discovery_terms=(
            "Lie algebra ideal check",
            "ideal membership subspace",
            "invariant subspace bracket",
        ),
        examples=(
            OperationExample(
                name="heisenberg_center_is_ideal",
                description=(
                    "Decide the Heisenberg z-axis is an ideal; the subspace "
                    "must use the algebra basis and the constants must "
                    "satisfy antisymmetry and Jacobi."
                ),
                input={
                    "algebra": {
                        "basis": ["x", "y", "z"],
                        "structure_constants": [
                            {
                                "i": 0,
                                "j": 1,
                                "k": 2,
                                "coefficient": {"num": "1", "den": "1"},
                            }
                        ],
                    },
                    "candidate": {
                        "basis": ["x", "y", "z"],
                        "generators": {
                            "domain": "QQ",
                            "row_count": 1,
                            "column_count": 3,
                            "entries": [
                                [
                                    {"num": "0", "den": "1"},
                                    {"num": "0", "den": "1"},
                                    {"num": "1", "den": "1"},
                                ]
                            ],
                        },
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="lie_algebra.subalgebra.check",
        title="Decide whether a subspace is a Lie subalgebra",
        description=(
            "Decide whether an RREF subspace is closed under its own Lie "
            "bracket. Bilinearity reduces the check to generator pairs; a "
            "failed decision retains the first exact escaping bracket."
        ),
        request_type=LieSubalgebraRequest,
        result_type=LieSubalgebraCheckResult,
        run=_run_lie_subalgebra_check,
        tags=("lie-algebra", "subalgebra", "closure", "exact", "rational"),
        discovery_terms=(
            "Lie subalgebra check",
            "bracket-closed subspace",
            "subalgebra membership",
        ),
        examples=(
            OperationExample(
                name="sl2_positive_borel_is_subalgebra",
                description="The span of h and e is closed under brackets in sl2.",
                input={
                    "algebra": _SL2_ALGEBRA,
                    "candidate": {
                        "basis": ["e", "f", "h"],
                        "generators": {
                            "domain": "QQ",
                            "row_count": 2,
                            "column_count": 3,
                            "entries": [
                                [_rational(1), _rational(0), _rational(0)],
                                [_rational(0), _rational(0), _rational(1)],
                            ],
                        },
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="lie_algebra.subalgebra.generated.compute",
        title="Generate the Lie subalgebra spanned by exact vectors",
        description=(
            "Return the smallest Lie subalgebra containing up to dim(g) exact "
            "rational vectors in a finite-dimensional Lie algebra over QQ. The "
            "canonical RREF rows retain the exact ambient algebra and ordered "
            "basis; the empty generator family generates zero. Admission "
            "establishes Jacobi and bounds closure work and coefficient growth."
        ),
        request_type=LieGeneratedSubalgebraRequest,
        result_type=LieSubalgebra,
        run=_run_lie_generated_subalgebra,
        tags=("lie-algebra", "generated-subalgebra", "exact", "rational"),
        discovery_terms=(
            "Lie subalgebra generated by vectors",
            "Lie closure of generators",
            "smallest Lie subalgebra containing elements",
        ),
        examples=(
            OperationExample(
                name="sl2_generators_e_f_generate_sl2",
                description="The generators e and f generate h by their bracket, hence all of sl2.",
                input={
                    "algebra": _SL2_ALGEBRA,
                    "generators": [_element([1, 0, 0]), _element([0, 1, 0])],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="lie_algebra.ideal.generated.compute",
        title="Generate the Lie ideal spanned by exact vectors",
        description=(
            "Return the smallest ideal of a finite-dimensional Lie algebra "
            "over QQ containing up to dim(g) exact rational vectors. The "
            "canonical RREF rows retain the exact ambient algebra and ordered "
            "basis; the empty family generates the zero ideal. Admission "
            "establishes Jacobi and bounds closure work, coefficient growth, "
            "and intermediate output size."
        ),
        request_type=LieGeneratedIdealRequest,
        result_type=LieIdeal,
        run=_run_lie_generated_ideal,
        tags=("lie-algebra", "generated-ideal", "exact", "rational"),
        discovery_terms=(
            "Lie ideal generated by vectors",
            "smallest ideal containing elements",
            "ideal closure under ambient brackets",
        ),
        examples=(
            OperationExample(
                name="heisenberg_ideal_generated_by_x",
                description=(
                    "In the Heisenberg algebra, the ideal generated by x is "
                    "span(x, z), since bracketing x with y adds z."
                ),
                input={
                    "algebra": {
                        "basis": ["x", "y", "z"],
                        "structure_constants": [
                            {
                                "i": 0,
                                "j": 1,
                                "k": 2,
                                "coefficient": _rational(1),
                            }
                        ],
                    },
                    "generators": [
                        {
                            "basis": ["x", "y", "z"],
                            "coordinates": [_rational(1), _rational(0), _rational(0)],
                        }
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="lie_algebra.quotient.compute",
        title="Form the quotient of a Lie algebra by an ideal",
        description=(
            "Form the quotient Lie algebra on cosets of an ideal subspace, "
            "with caller-supplied ordered labels attached in increasing "
            "free-column order of the ideal RREF and bracket constants "
            "reduced against the ideal. Ideal absorption is verified by "
            "operation admission before cosets form, so the quotient "
            "bracket is well defined."
        ),
        request_type=LieQuotientRequest,
        result_type=LieQuotientResult,
        run=_run_lie_quotient,
        tags=("lie-algebra", "quotient", "exact", "rational"),
        discovery_terms=(
            "Lie algebra quotient",
            "quotient by ideal",
            "coset bracket",
        ),
        examples=(
            OperationExample(
                name="heisenberg_mod_center",
                description=(
                    "Form the abelian 2-dimensional quotient of Heisenberg "
                    "by its center; the labels must number dimension minus "
                    "ideal dimension and the constants must satisfy Jacobi."
                ),
                input={
                    "algebra": {
                        "basis": ["x", "y", "z"],
                        "structure_constants": [
                            {
                                "i": 0,
                                "j": 1,
                                "k": 2,
                                "coefficient": {"num": "1", "den": "1"},
                            }
                        ],
                    },
                    "ideal": {
                        "basis": ["x", "y", "z"],
                        "generators": {
                            "domain": "QQ",
                            "row_count": 1,
                            "column_count": 3,
                            "entries": [
                                [
                                    {"num": "0", "den": "1"},
                                    {"num": "0", "den": "1"},
                                    {"num": "1", "den": "1"},
                                ]
                            ],
                        },
                    },
                    "quotient_basis": ["u", "v"],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="lie_algebra.direct_sum.compute",
        title="Construct the exact direct sum of two Lie algebras",
        description=(
            "Construct the finite-dimensional direct sum L ⊕ M over QQ on a "
            "caller-declared ordered basis. Each summand has dimension at most 8, "
            "the combined dimension is at most 8, both source brackets must satisfy "
            "Jacobi, source structure constants are embedded in their respective "
            "blocks, and every mixed bracket is zero."
        ),
        request_type=LieDirectSumRequest,
        result_type=FiniteDimensionalLieAlgebra,
        run=_run_lie_direct_sum,
        tags=("lie-algebra", "direct-sum", "exact", "rational"),
        discovery_terms=(
            "direct sum of Lie algebras",
            "block diagonal Lie bracket",
            "product Lie algebra",
            "zero mixed bracket",
        ),
        examples=(
            OperationExample(
                name="sl2_plus_abelian_line",
                description=(
                    "Form sl2(QQ) direct sum a one-dimensional abelian algebra; "
                    "the output basis lists the left block before the right block."
                ),
                input={
                    "left": _SL2_ALGEBRA,
                    "right": {"basis": ["t"], "structure_constants": []},
                    "basis": ["e", "f", "h", "t"],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
