"""Explicit public-entrypoint inventory for the bounded rational request family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def ratio(num: str = "1", den: str = "2") -> dict[str, str]:
    return {"num": num, "den": den}


def dense(value: dict[str, Any]) -> dict[str, Any]:
    return {"domain": "QQ", "row_count": 1, "column_count": 1, "entries": [[value]]}


def sparse(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "domain": "QQ",
        "row_count": 1,
        "column_count": 1,
        "entries": [{"row": 0, "column": 0, "value": value}],
    }


@dataclass(frozen=True)
class RationalRequestCase:
    operation_id: str
    argument: str
    sparse: bool = False

    @property
    def name(self) -> str:
        return f"{self.operation_id}:{self.argument}:{'sparse' if self.sparse else 'dense'}"

    def payload(self, value: dict[str, Any]) -> dict[str, Any]:
        if self.operation_id.startswith("linear."):
            return {
                "system": {
                    "domain": "QQ",
                    "relation": "AX_EQUALS_B",
                    "variables": ["x"],
                    "coefficients": sparse(
                        value if self.argument == "coefficients" else ratio()
                    ),
                    "rhs": [value if self.argument == "rhs" else ratio()],
                }
            }
        if self.argument in {"left", "right"}:
            return {
                "left": dense(value if self.argument == "left" else ratio()),
                "right": dense(value if self.argument == "right" else ratio()),
            }
        result: dict[str, Any] = {"matrix": (sparse if self.sparse else dense)(value)}
        if self.operation_id == "matrix.rational_linear_system.solve":
            result["rhs"] = [value if self.argument == "rhs" else ratio()]
            if self.argument == "rhs":
                result["matrix"] = dense(ratio())
        if self.operation_id == "matrix.partial_trace.compute":
            result.update(traced_dimension=1, kept_dimension=1)
        return result


RATIONAL_REQUEST_CASES = (
    RationalRequestCase("matrix.determinant.compute", "matrix"),
    RationalRequestCase("matrix.permanent.compute", "matrix"),
    RationalRequestCase("matrix.characteristic_polynomial.compute", "matrix"),
    RationalRequestCase("matrix.multiply.compute", "left"),
    RationalRequestCase("matrix.multiply.compute", "right"),
    RationalRequestCase("matrix.kronecker_product.compute", "left"),
    RationalRequestCase("matrix.kronecker_product.compute", "right"),
    RationalRequestCase("matrix.normal_form.rref.compute", "matrix"),
    RationalRequestCase("matrix.rank.compute", "matrix"),
    RationalRequestCase("matrix.rank.compute", "matrix", sparse=True),
    RationalRequestCase("matrix.nullspace.compute", "matrix"),
    RationalRequestCase("matrix.nullspace.compute", "matrix", sparse=True),
    RationalRequestCase("matrix.rational_linear_system.solve", "matrix"),
    RationalRequestCase("matrix.rational_linear_system.solve", "rhs"),
    RationalRequestCase("matrix.partial_trace.compute", "matrix"),
    RationalRequestCase(
        "linear.rational_solution.compute", "coefficients", sparse=True
    ),
    RationalRequestCase("linear.rational_solution.compute", "rhs"),
    RationalRequestCase(
        "linear.rational_inconsistency.compute", "coefficients", sparse=True
    ),
    RationalRequestCase("linear.rational_inconsistency.compute", "rhs"),
)
