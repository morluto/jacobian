"""Public declarations for bounded prime-field GL and SL operations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.groups.finite_matrix._models import (
    GeneralLinearNaturalActionRequest,
    PrimeFieldGeneralLinearGroup,
    PrimeFieldGeneralLinearNaturalAction,
    PrimeFieldLinearGroupRequest,
    PrimeFieldSpecialLinearGroup,
    PrimeFieldSpecialLinearNaturalAction,
    SpecialLinearNaturalActionRequest,
)
from jacobian.math.groups.finite_matrix.operations import (
    construct_general_linear_group,
    construct_special_linear_group,
    general_linear_nonzero_vector_action,
    special_linear_nonzero_vector_action,
)


def _construct_gl(
    request: PrimeFieldLinearGroupRequest,
) -> PrimeFieldGeneralLinearGroup:
    return construct_general_linear_group(request.prime, request.dimension)


def _construct_sl(
    request: PrimeFieldLinearGroupRequest,
) -> PrimeFieldSpecialLinearGroup:
    return construct_special_linear_group(request.prime, request.dimension)


def _gl_action(
    request: GeneralLinearNaturalActionRequest,
) -> PrimeFieldGeneralLinearNaturalAction:
    return general_linear_nonzero_vector_action(request.group)


def _sl_action(
    request: SpecialLinearNaturalActionRequest,
) -> PrimeFieldSpecialLinearNaturalAction:
    return special_linear_nonzero_vector_action(request.group)


_GL_2_2 = {
    "prime": 2,
    "dimension": 2,
    "order": "6",
    "generators": [
        {"prime": 2, "entries": [[1, 1], [0, 1]], "columns": 2},
        {"prime": 2, "entries": [[1, 0], [1, 1]], "columns": 2},
    ],
}

_SL_2_3 = {
    "prime": 3,
    "dimension": 2,
    "order": "24",
    "generators": [
        {"prime": 3, "entries": [[1, 1], [0, 1]], "columns": 2},
        {"prime": 3, "entries": [[1, 0], [1, 1]], "columns": 2},
    ],
    "ambient_general_linear_order": "48",
    "determinant_index": "2",
}


TOOLS: MathTools = (
    MathTool(
        operation_id="finite_matrix_group.general_linear.construct",
        title="Construct a full prime-field general linear group",
        description=(
            "Construct the complete standard matrix group GL(n,p), returning its "
            "exact ordered-basis count and a canonical complete generator family: "
            "adjacent elementary transvections for the determinant kernel plus a "
            "primitive determinant generator. This is the named full group, not an "
            "arbitrary matrix-generated subgroup."
        ),
        request_type=PrimeFieldLinearGroupRequest,
        result_type=PrimeFieldGeneralLinearGroup,
        run=_construct_gl,
        tags=("finite-matrix-group", "general-linear-group", "prime-field", "exact"),
        discovery_terms=(
            "GL n q",
            "general linear group over finite field",
            "invertible prime field matrices",
        ),
        examples=(
            OperationExample(
                name="general_linear_two_over_f2",
                description=(
                    "Construct the full six-element group GL(2,2) with complete "
                    "elementary generators; the characteristic must be prime and "
                    "the dimension must fit the bounded generator envelope."
                ),
                input={"prime": 2, "dimension": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="finite_matrix_group.special_linear.construct",
        title="Construct a full prime-field special linear group",
        description=(
            "Construct the determinant kernel SL(n,p), returning its exact order, "
            "its index in GL(n,p), and canonical adjacent elementary generators "
            "that generate the complete named group."
        ),
        request_type=PrimeFieldLinearGroupRequest,
        result_type=PrimeFieldSpecialLinearGroup,
        run=_construct_sl,
        tags=("finite-matrix-group", "special-linear-group", "prime-field", "exact"),
        discovery_terms=(
            "SL n q",
            "special linear group over finite field",
            "determinant one prime field matrices",
        ),
        examples=(
            OperationExample(
                name="special_linear_two_over_f3",
                description=(
                    "Construct the full 24-element group SL(2,3) with complete "
                    "elementary generators; the characteristic must be prime."
                ),
                input={"prime": 3, "dimension": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="finite_matrix_group.general_linear.nonzero_vector_action.compute",
        title="Compute the natural GL action on nonzero vectors",
        description=(
            "Re-admit a canonical full GL(n,p) value and materialize its faithful "
            "natural permutation action on every nonzero vector of GF(p)^n. The "
            "complete vector axis must contain at most 50 points."
        ),
        request_type=GeneralLinearNaturalActionRequest,
        result_type=PrimeFieldGeneralLinearNaturalAction,
        run=_gl_action,
        tags=("finite-matrix-group", "group-action", "nonzero-vectors", "exact"),
        discovery_terms=(
            "general linear natural vector action",
            "GL action on nonzero vectors",
        ),
        examples=(
            OperationExample(
                name="gl_2_2_nonzero_vectors",
                description=(
                    "Materialize the faithful GL(2,2) action on its three nonzero "
                    "vectors; the authored group must equal the canonical full "
                    "named group."
                ),
                input={"group": _GL_2_2},
            ),
        ),
    ),
    MathTool(
        operation_id="finite_matrix_group.special_linear.nonzero_vector_action.compute",
        title="Compute the natural SL action on nonzero vectors",
        description=(
            "Re-admit a canonical full SL(n,p) value and materialize its natural "
            "permutation action on every nonzero vector of GF(p)^n. The complete "
            "vector axis must contain at most 50 points."
        ),
        request_type=SpecialLinearNaturalActionRequest,
        result_type=PrimeFieldSpecialLinearNaturalAction,
        run=_sl_action,
        tags=("finite-matrix-group", "group-action", "nonzero-vectors", "exact"),
        discovery_terms=(
            "special linear natural vector action",
            "SL action on nonzero vectors",
        ),
        examples=(
            OperationExample(
                name="sl_2_3_nonzero_vectors",
                description=(
                    "Materialize the SL(2,3) action on its eight nonzero vectors; "
                    "the authored group must equal the canonical determinant "
                    "kernel with its exact index."
                ),
                input={"group": _SL_2_3},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
