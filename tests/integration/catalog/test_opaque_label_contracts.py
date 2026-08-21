"""Public opaque-label contracts shared by finite mathematical domains."""

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import OperationRequestValidationError, invoke_operation


def _payloads(label: str) -> tuple[tuple[str, dict[str, object]], ...]:
    return (
        (
            "semigroup.element.power.compute",
            {
                "semigroup": {
                    "elements": [label],
                    "multiplication": [[label]],
                },
                "element": label,
                "exponent": 1,
            },
        ),
        (
            "topology.finite.interior.compute",
            {
                "space": {"points": [label], "preorder": [[0]]},
                "subset": [0],
            },
        ),
        (
            "finite_category.profile.compute",
            {
                "objects": [label],
                "morphisms": [
                    {
                        "morphism_id": label,
                        "source": label,
                        "target": label,
                    }
                ],
                "identities": [[label, label]],
                "composition": [[label, label, label]],
            },
        ),
        (
            "probability.bh_step_up.compute",
            {
                "hypotheses": [
                    {
                        "hypothesis_id": label,
                        "p_value": {"num": "1", "den": "2"},
                    }
                ],
                "level": {"num": "1", "den": "2"},
            },
        ),
    )


@pytest.mark.parametrize("label", ["", " ", " x", "x ", "x\n", "x\x00y", "x\u0085y"])
def test_public_opaque_labels_reject_empty_whitespace_and_controls(label: str) -> None:
    catalog = Catalog.open()
    for operation_id, payload in _payloads(label):
        with pytest.raises(OperationRequestValidationError):
            invoke_operation(operation_id, payload, catalog)


@pytest.mark.parametrize("label", ["0", "a:b", "X_1", "point.with-punctuation"])
def test_public_opaque_labels_remain_mathematical_labels(label: str) -> None:
    catalog = Catalog.open()
    for operation_id, payload in _payloads(label):
        invoke_operation(operation_id, payload, catalog)


def test_kolmogorov_quotient_retains_long_labels_as_a_class() -> None:
    left = "a" * 64
    right = "b" * 64
    result = invoke_operation(
        "topology.finite.kolmogorov_quotient.compute",
        {
            "space": {
                "points": [left, right],
                "preorder": [[0, 1], [0, 1]],
            }
        },
        Catalog.open(),
    )

    assert result.output["quotient_points"] == [[left, right]]
