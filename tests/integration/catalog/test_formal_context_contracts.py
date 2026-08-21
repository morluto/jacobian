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
