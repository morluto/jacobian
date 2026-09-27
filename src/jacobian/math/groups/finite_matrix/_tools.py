"""Public declarations for bounded GL and SL operations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.groups.finite_matrix._models import (
    ExtensionFieldGeneralLinearGroup,
    ExtensionFieldGeneralLinearProjectiveAction,
    ExtensionFieldGeneralLinearProjectiveActionRequest,
    ExtensionFieldLinearGroupRequest,
    ExtensionFieldSpecialLinearGroup,
    ExtensionFieldSpecialLinearProjectiveAction,
    ExtensionFieldSpecialLinearProjectiveActionRequest,
    GeneralLinearNaturalActionRequest,
    PrimeFieldGeneralLinearGroup,
    PrimeFieldGeneralLinearNaturalAction,
    PrimeFieldLinearGroupRequest,
    PrimeFieldSpecialLinearGroup,
    PrimeFieldSpecialLinearNaturalAction,
    SpecialLinearNaturalActionRequest,
)
from jacobian.math.groups.finite_matrix.extension_operations import (
    construct_extension_general_linear_group,
    construct_extension_special_linear_group,
    extension_general_linear_projective_action,
    extension_special_linear_projective_action,
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


def _construct_extension_gl(
    request: ExtensionFieldLinearGroupRequest,
) -> ExtensionFieldGeneralLinearGroup:
    return construct_extension_general_linear_group(
        request.presentation, request.vector_axis
    )


def _construct_extension_sl(
    request: ExtensionFieldLinearGroupRequest,
) -> ExtensionFieldSpecialLinearGroup:
    return construct_extension_special_linear_group(
        request.presentation, request.vector_axis
    )


def _extension_gl_action(
    request: ExtensionFieldGeneralLinearProjectiveActionRequest,
) -> ExtensionFieldGeneralLinearProjectiveAction:
    return extension_general_linear_projective_action(request.group)


def _extension_sl_action(
    request: ExtensionFieldSpecialLinearProjectiveActionRequest,
) -> ExtensionFieldSpecialLinearProjectiveAction:
    return extension_special_linear_projective_action(request.group)


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


_GF4: dict[str, object] = {
    "characteristic": "2",
    "modulus_coefficients": ["1", "1", "1"],
    "generator": "a",
}
_GF4_AXIS: dict[str, object] = {"name": "v", "labels": ["x", "y"]}
_GF4_ONE: dict[str, object] = {"presentation": _GF4, "coordinates": ["1", "0"]}
_GF4_ALPHA: dict[str, object] = {"presentation": _GF4, "coordinates": ["0", "1"]}
_GF4_ZERO: dict[str, object] = {"presentation": _GF4, "coordinates": ["0", "0"]}


def _gf4_matrix(entries: list[list[dict[str, object]]]) -> dict[str, object]:
    return {
        "presentation": _GF4,
        "row_axis": _GF4_AXIS,
        "column_axis": _GF4_AXIS,
        "entries": entries,
    }


_GF4_E12_ONE: dict[str, object] = _gf4_matrix(
    [[_GF4_ONE, _GF4_ONE], [_GF4_ZERO, _GF4_ONE]]
)
_GF4_E21_ONE: dict[str, object] = _gf4_matrix(
    [[_GF4_ONE, _GF4_ZERO], [_GF4_ONE, _GF4_ONE]]
)
_GF4_E12_ALPHA: dict[str, object] = _gf4_matrix(
    [[_GF4_ONE, _GF4_ALPHA], [_GF4_ZERO, _GF4_ONE]]
)
_GF4_E21_ALPHA: dict[str, object] = _gf4_matrix(
    [[_GF4_ONE, _GF4_ZERO], [_GF4_ALPHA, _GF4_ONE]]
)
_GF4_DIAGONAL_ALPHA: dict[str, object] = _gf4_matrix(
    [[_GF4_ALPHA, _GF4_ZERO], [_GF4_ZERO, _GF4_ONE]]
)
_GL_2_4 = {
    "presentation": _GF4,
    "vector_axis": _GF4_AXIS,
    "order": "180",
    "generators": [
        _GF4_E12_ONE,
        _GF4_E21_ONE,
        _GF4_E12_ALPHA,
        _GF4_E21_ALPHA,
        _GF4_DIAGONAL_ALPHA,
    ],
}
_SL_2_4 = {
    "presentation": _GF4,
    "vector_axis": _GF4_AXIS,
    "order": "60",
    "generators": [
        _GF4_E12_ONE,
        _GF4_E21_ONE,
        _GF4_E12_ALPHA,
        _GF4_E21_ALPHA,
    ],
    "ambient_general_linear_order": "180",
    "determinant_index": "3",
}

EXTENSION_TOOLS: MathTools = (
    MathTool(
        operation_id="finite_matrix_group.extension.general_linear.construct",
        title="Construct a full general linear group over a presented extension field",
        description=(
            "Construct the complete GL(V) over one exact finite-field presentation "
            "of degree at least two. The result retains the presentation, ordered "
            "coordinate axis, exact group order, and a canonical complete generator "
            "family; it is not an arbitrary generated subgroup."
        ),
        request_type=ExtensionFieldLinearGroupRequest,
        result_type=ExtensionFieldGeneralLinearGroup,
        run=_construct_extension_gl,
        tags=(
            "finite-matrix-group",
            "general-linear-group",
            "extension-field",
            "exact",
        ),
        discovery_terms=(
            "GL n q extension field",
            "general linear group over finite extension field",
        ),
        examples=(
            OperationExample(
                name="general_linear_two_over_gf_four",
                description="Construct GL(2,4) on a labelled power-basis coordinate axis.",
                input={"presentation": _GF4, "vector_axis": _GF4_AXIS},
            ),
        ),
    ),
    MathTool(
        operation_id="finite_matrix_group.extension.special_linear.construct",
        title="Construct a full special linear group over a presented extension field",
        description=(
            "Construct the determinant-one subgroup SL(V) over one exact finite-field "
            "presentation of degree at least two, retaining its exact order, index in "
            "GL(V), and canonical elementary-matrix generators."
        ),
        request_type=ExtensionFieldLinearGroupRequest,
        result_type=ExtensionFieldSpecialLinearGroup,
        run=_construct_extension_sl,
        tags=(
            "finite-matrix-group",
            "special-linear-group",
            "extension-field",
            "exact",
        ),
        discovery_terms=(
            "SL n q extension field",
            "special linear group over finite extension field",
        ),
        examples=(
            OperationExample(
                name="special_linear_two_over_gf_four",
                description="Construct SL(2,4) and its exact index in GL(2,4).",
                input={"presentation": _GF4, "vector_axis": _GF4_AXIS},
            ),
        ),
    ),
    MathTool(
        operation_id="finite_matrix_group.extension.general_linear.projective_action.compute",
        title="Compute the natural projective action of an extension-field GL group",
        description=(
            "Re-admit the canonical full GL(V) value and compute its action on all "
            "projective points of the retained coordinate space. This is an action "
            "of GL; scalar matrices may act trivially, so the result does not claim "
            "to be a faithful PGL quotient. The complete point axis is capped at 50."
        ),
        request_type=ExtensionFieldGeneralLinearProjectiveActionRequest,
        result_type=ExtensionFieldGeneralLinearProjectiveAction,
        run=_extension_gl_action,
        tags=("finite-matrix-group", "projective-action", "extension-field", "exact"),
        discovery_terms=(
            "GL projective point action",
            "general linear action on projective space",
        ),
        examples=(
            OperationExample(
                name="gl_2_4_projective_point_action",
                description="Compute the GL(2,4) action on the five points of PG(1,4).",
                input={"group": _GL_2_4},
            ),
        ),
    ),
    MathTool(
        operation_id="finite_matrix_group.extension.special_linear.projective_action.compute",
        title="Compute the natural projective action of an extension-field SL group",
        description=(
            "Re-admit the canonical full SL(V) value and compute its action on all "
            "projective points of the retained coordinate space. The complete point "
            "axis is capped at 50."
        ),
        request_type=ExtensionFieldSpecialLinearProjectiveActionRequest,
        result_type=ExtensionFieldSpecialLinearProjectiveAction,
        run=_extension_sl_action,
        tags=("finite-matrix-group", "projective-action", "extension-field", "exact"),
        discovery_terms=(
            "SL projective point action",
            "special linear action on projective space",
        ),
        examples=(
            OperationExample(
                name="sl_2_4_projective_point_action",
                description="Compute the SL(2,4) action on the five points of PG(1,4).",
                input={"group": _SL_2_4},
            ),
        ),
    ),
)

TOOLS = (*TOOLS, *EXTENSION_TOOLS)

__all__ = ["TOOLS"]
