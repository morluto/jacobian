"""Public-dispatch coverage for formal-context operation contracts."""

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import OperationRequestValidationError, invoke_operation


@pytest.mark.parametrize(
    ("operation_id", "context"),
    (
        (
            "formal_context.attributes.derivation.compute",
            {
                "objects": ["o0", "o1"],
                "attributes": ["a0"],
                "incidence": [[0, 0], [1, 0]],
            },
        ),
        (
            "formal_context.concept.from_attributes.compute",
            {
                "objects": ["o0", "o1"],
                "attributes": ["a0"],
                "incidence": [[0, 0], [1, 0]],
            },
        ),
        (
            "formal_context.objects.derivation.compute",
            {
                "objects": ["o0"],
                "attributes": ["a0", "a1"],
                "incidence": [[0, 0], [0, 1]],
            },
        ),
    ),
)
def test_public_operations_reject_indices_outside_their_axis(
    operation_id: str, context: dict[str, object]
) -> None:
    with pytest.raises(OperationRequestValidationError):
        invoke_operation(
            operation_id,
            {"context": context, "subset": [1]},
            Catalog.open(),
        )


def test_public_duquenne_guigues_basis_retains_complete_closure_matrix() -> None:
    public_result = invoke_operation(
        "formal_context.duquenne_guigues_basis.compute",
        {
            "context": {
                "objects": ["g0", "g1"],
                "attributes": ["always", "sometimes"],
                "incidence": [[0, 0], [1, 0]],
            }
        },
        Catalog.open(),
    )

    output = public_result.output
    assert output["source_attribute_indices"] == [0, 1]
    assert output["closure_matrix"] == [
        {"candidate_state": 0, "subset": [], "closure": [0]},
        {"candidate_state": 1, "subset": [0], "closure": [0]},
        {"candidate_state": 2, "subset": [1], "closure": [0, 1]},
        {"candidate_state": 3, "subset": [0, 1], "closure": [0, 1]},
    ]
    assert output["pseudo_intents"] == [
        {
            "candidate_state": 0,
            "premise": [],
            "closure": [0],
            "basis_implication_index": 0,
        }
    ]
