"""Public exact scalar-extension and field-valued q-prefix operations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.number_theory.modular_forms._models import (
    ModularFormCoordinatesFieldExtensionRequest,
    ModularFormCoordinatesQExpansionRequest,
)
from jacobian.math.number_theory.modular_forms.field_coordinates import (
    modular_form_coordinates_extend_field,
    modular_form_field_coordinates_q_expansion,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormFieldQExpansion,
)


def _extend(
    request: ModularFormCoordinatesFieldExtensionRequest,
) -> ModularFormCoordinates:
    return modular_form_coordinates_extend_field(
        request.form, request.coefficient_field
    )


def _q_expansion(
    request: ModularFormCoordinatesQExpansionRequest,
) -> ModularFormFieldQExpansion:
    return modular_form_field_coordinates_q_expansion(request.form, request.precision)


_FIELD = {"domain": "QQ_CYCLOTOMIC", "order": 6, "generator": "CLASS_OF_X"}
_FIELD_ZERO = {
    "field": _FIELD,
    "coefficients_ascending": [{"num": "0", "den": "1"}, {"num": "0", "den": "1"}],
}
_FIELD_ONE = {
    "field": _FIELD,
    "coefficients_ascending": [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
}
_SPACE = {
    "group": "GAMMA0",
    "level": 2,
    "weight": 4,
    "kind": "M",
    "character": "TRIVIAL",
    "coefficient_domain": _FIELD,
}

TOOLS: MathTools = (
    MathTool(
        operation_id="modular_form.coordinates.extend_field.compute",
        title="Extend modular-form coordinates to an explicit coefficient field",
        description=(
            "Embed supported rational coordinates into the identical basis over "
            "Q(zeta_6), retaining level, weight, kind, and basis identity. The "
            "result is a field-valued modular-form coordinate value."
        ),
        request_type=ModularFormCoordinatesFieldExtensionRequest,
        result_type=ModularFormCoordinates,
        run=_extend,
        tags=("modular-forms", "coefficient-fields", "coordinates", "exact"),
        examples=(
            OperationExample(
                name="extend_level_two_weight_four_coordinates",
                description="Extend (2, 3) in M4(Gamma0(2)) to Q(zeta_6).",
                input={
                    "form": {
                        "space": {
                            "level": 2,
                            "weight": 4,
                            "kind": "M",
                            "character": "TRIVIAL",
                            "coefficient_domain": "QQ",
                        },
                        "basis_id": "gamma0-two-weight-2-4-monomials-v1",
                        "coordinates": [
                            {"num": "2", "den": "1"},
                            {"num": "3", "den": "1"},
                        ],
                    },
                    "coefficient_field": _FIELD,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="modular_form.field_coordinates.q_expansion.compute",
        title="Expand field-valued modular-form coordinates",
        description=(
            "Return a finite exact q-prefix of modular-form coordinates over "
            "Q(zeta_6), using their retained space and coefficient parent."
        ),
        request_type=ModularFormCoordinatesQExpansionRequest,
        result_type=ModularFormFieldQExpansion,
        run=_q_expansion,
        tags=("modular-forms", "coefficient-fields", "q-expansion", "exact"),
        examples=(
            OperationExample(
                name="expand_level_two_weight_four_field_coordinates",
                description="Expand the basis vector (1, 0) through q^2.",
                input={
                    "form": {
                        "space": _SPACE,
                        "basis_id": "gamma0-two-weight-2-4-monomials-v1",
                        "coordinates": [_FIELD_ONE, _FIELD_ZERO],
                    },
                    "precision": 3,
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
