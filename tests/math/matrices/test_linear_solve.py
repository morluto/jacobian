from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import MathTool
from jacobian.math.matrices._operation_models import RationalLinearSolveResult
from jacobian.math.matrices._tools import TOOLS


def _operation() -> MathTool:
    return next(
        tool
        for tool in TOOLS
        if tool.operation_id == "matrix.rational_linear_system.solve"
    )


def _request(rhs: tuple[str, str]) -> dict[str, object]:
    def rational(value: str) -> dict[str, str]:
        return {"num": value, "den": "1"}

    return {
        "matrix": {
            "entries": [
                [rational("1"), rational("1")],
                [rational("1"), rational("1")],
            ]
        },
        "rhs": [rational(value) for value in rhs],
    }


@pytest.mark.parametrize("rhs", [("0", "1"), ("1", "1")])
def test_unique_linear_solve_rejects_singular_systems(
    rhs: tuple[str, str],
) -> None:
    with pytest.raises(ValidationError, match="singular"):
        _operation().request_type.model_validate(_request(rhs))


def test_unique_linear_solve_returns_exact_solution() -> None:
    operation = _operation()
    request = operation.request_type.model_validate(
        {
            "matrix": {
                "entries": [
                    [
                        {"num": "1", "den": "1"},
                        {"num": "1", "den": "1"},
                    ],
                    [
                        {"num": "1", "den": "1"},
                        {"num": "-1", "den": "1"},
                    ],
                ]
            },
            "rhs": [
                {"num": "3", "den": "1"},
                {"num": "1", "den": "1"},
            ],
        }
    )

    result = operation.run(request)

    assert isinstance(result, RationalLinearSolveResult)
    assert tuple(value.as_fraction() for value in result.solution) == (2, 1)
