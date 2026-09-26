"""Public declaration for exact global character-form equality."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.number_theory.modular_forms.global_equality.models import (
    ModularFormGlobalEqualityRequest,
    ModularFormGlobalEqualityResult,
)
from jacobian.math.number_theory.modular_forms.global_equality.operations import (
    modular_form_coordinates_global_equal,
)


def _compute(
    request: ModularFormGlobalEqualityRequest,
) -> ModularFormGlobalEqualityResult:
    return ModularFormGlobalEqualityResult(
        equal=modular_form_coordinates_global_equal(
            request.left,
            request.left_embedding,
            request.right,
            request.right_embedding,
        )
    )


def _example_form_payload(character_coordinate: int) -> dict[str, object]:
    field = {"domain": "QQ_CYCLOTOMIC", "order": 6, "generator": "CLASS_OF_X"}
    return {
        "space": {
            "group": "GAMMA0",
            "level": 13,
            "weight": 2,
            "kind": "M",
            "character": {
                "group": {
                    "modulus": 13,
                    "unit_residues": list(range(1, 13)),
                    "character_count": 12,
                    "invariant_factors": [12],
                    "generators": [2],
                    "generator_orders": [12],
                    "unit_coordinates": [
                        [0],
                        [1],
                        [4],
                        [2],
                        [9],
                        [5],
                        [11],
                        [3],
                        [8],
                        [10],
                        [7],
                        [6],
                    ],
                    "exponent": 12,
                },
                "coordinates": [character_coordinate],
            },
            "coefficient_domain": field,
        },
        "basis_id": "gamma0-cyclotomic-character-sturm-rref-v1",
        "coordinates": [
            {
                "field": field,
                "coefficients_ascending": [
                    {"num": 1, "den": 1},
                    {"num": 0, "den": 1},
                ],
            },
            {
                "field": field,
                "coefficients_ascending": [
                    {"num": 0, "den": 1},
                    {"num": 0, "den": 1},
                ],
            },
            {
                "field": field,
                "coefficients_ascending": [
                    {"num": 0, "den": 1},
                    {"num": 0, "den": 1},
                ],
            },
        ],
    }


def _example_embedding_payload() -> dict[str, object]:
    field = {"domain": "QQ_CYCLOTOMIC", "order": 12, "generator": "CLASS_OF_X"}
    return {
        "source_order": 6,
        "target_field": field,
        "generator_image": {
            "field": field,
            "coefficients_ascending": [
                {"num": 0, "den": 1},
                {"num": 0, "den": 1},
                {"num": 1, "den": 1},
                {"num": 0, "den": 1},
            ],
        },
    }


def _example_conjugate_embedding_payload() -> dict[str, object]:
    payload = _example_embedding_payload()
    payload["generator_image"] = {
        "field": payload["target_field"],
        "coefficients_ascending": [
            {"num": 1, "den": 1},
            {"num": 0, "den": 1},
            {"num": -1, "den": 1},
            {"num": 0, "den": 1},
        ],
    }
    return payload


TOOLS: MathTools = (
    MathTool(
        operation_id="modular_form.equal.global_character.check",
        title="Check global equality across character spaces",
        description=(
            "Compare exact same-weight forms from distinct source embeddings whose "
            "characters agree at levels 13, 26, or 39. Each source has an "
            "explicit cyclotomic field embedding into the same target field. "
            "Equality is decided through the Sturm bound for Gamma1 of the "
            "least common multiple of the source levels."
        ),
        request_type=ModularFormGlobalEqualityRequest,
        result_type=ModularFormGlobalEqualityResult,
        run=_compute,
        tags=("modular-forms", "characters", "equality", "exact"),
        examples=(
            OperationExample(
                name="equal-nonzero-forms-with-conjugate-character-embeddings",
                description=(
                    "Compare the same nonzero form from Galois-conjugate character spaces "
                    "after explicit embeddings identify their common character."
                ),
                input={
                    "left": _example_form_payload(2),
                    "left_embedding": _example_embedding_payload(),
                    "right": _example_form_payload(10),
                    "right_embedding": _example_conjugate_embedding_payload(),
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
