"""Algebraic-torus operation declarations."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.algebraic_tori._models import HomogeneousMonomialSolutionRequest
from jacobian.math.algebraic_tori.operations import (
    homogeneous_monomial_solution_subgroup,
)
from jacobian.math.algebraic_tori.values import AlgebraicTorusSolutionSubgroup


def _solve(
    request: HomogeneousMonomialSolutionRequest,
) -> AlgebraicTorusSolutionSubgroup:
    return homogeneous_monomial_solution_subgroup(request.system)


TOOLS: MathTools = (
    MathTool(
        operation_id="algebraic_torus.monomial_system.solution_subgroup.compute",
        title="Solve a homogeneous monomial system on an algebraic torus",
        description=(
            "Return the complete exact solution subgroup of x^A = 1 as compact "
            "Smith torsion characters times a free complex torus, with source-bound "
            "Laurent-monomial coordinate maps."
        ),
        request_type=HomogeneousMonomialSolutionRequest,
        result_type=AlgebraicTorusSolutionSubgroup,
        run=_solve,
        tags=(
            "algebraic-torus",
            "monomial-system",
            "smith-normal-form",
            "character-lattice",
            "exact",
        ),
        examples=(
            example(
                "two_component_curve",
                "Solve x^2 y^6 = 1 on the two-dimensional complex torus.",
                {
                    "system": {
                        "exponent_matrix": {
                            "row_count": 1,
                            "column_count": 2,
                            "entries": [["2", "6"]],
                        },
                        "equation_axis": ["relation"],
                        "coordinate_axis": ["x", "y"],
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
