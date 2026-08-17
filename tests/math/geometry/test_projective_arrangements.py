"""Contract coverage for inline projective line-arrangement operations."""

from jacobian.math.geometry.projective._arrangements import (
    compute_projective_line_flats,
)
from jacobian.math.geometry.projective._models import ProjectiveLineArrangementRequest


def test_projective_arrangement_result_has_no_removed_verification_route() -> None:
    request = ProjectiveLineArrangementRequest.model_validate(
        {
            "lines": [
                {
                    "label": "x",
                    "coefficients": [
                        {"num": "1", "den": "1"},
                        {"num": "0", "den": "1"},
                        {"num": "0", "den": "1"},
                    ],
                },
                {
                    "label": "y",
                    "coefficients": [
                        {"num": "0", "den": "1"},
                        {"num": "1", "den": "1"},
                        {"num": "0", "den": "1"},
                    ],
                },
            ]
        }
    )

    result = compute_projective_line_flats(request).model_dump()

    assert "verification_operation_id" not in result
    assert "verification_input_field" not in result
