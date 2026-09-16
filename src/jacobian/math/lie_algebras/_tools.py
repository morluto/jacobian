"""Finite-dimensional Lie-algebra operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.lie_algebras._models import LieBracketRequest, LieBracketResult
from jacobian.math.lie_algebras.operations import lie_bracket


def _run_lie_bracket(request: LieBracketRequest) -> LieBracketResult:
    return lie_bracket(request.algebra, request.left, request.right)


def _rational(value: int) -> dict[str, str]:
    return {"num": str(value), "den": "1"}


_SL2_ALGEBRA = {
    "basis": ["e", "f", "h"],
    "structure_constants": [
        {"i": 0, "j": 1, "k": 2, "coefficient": _rational(1)},
        {"i": 0, "j": 2, "k": 0, "coefficient": _rational(-2)},
        {"i": 1, "j": 2, "k": 1, "coefficient": _rational(2)},
    ],
}


def _element(coords: list[int]) -> dict[str, Any]:
    return {
        "basis": ["e", "f", "h"],
        "coordinates": [_rational(value) for value in coords],
    }


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="lie_algebra.bracket.compute",
        title="Compute the exact bracket of two Lie-algebra elements",
        description=(
            "Compute [x, y] for two exact rational coordinate vectors in one "
            "finite-dimensional Lie algebra over QQ given by ordered basis "
            "labels and sparse structure constants with dimension at most 8, "
            "returning exact bracket coordinates with one ledger row per "
            "nonzero basis pair. Antisymmetry is canonical in the stored "
            "table and every basis-triple Jacobi identity is established by "
            "operation admission before expansion."
        ),
        request_type=LieBracketRequest,
        result_type=LieBracketResult,
        run=_run_lie_bracket,
        tags=("lie-algebra", "bracket", "exact", "rational"),
        discovery_terms=(
            "Lie bracket of basis elements",
            "structure constant expansion",
            "sl2 commutator",
            "Heisenberg bracket",
        ),
        examples=(
            OperationExample(
                name="sl2_bracket_e_f",
                description=(
                    "Compute [e, f] = h in sl2(QQ); both elements must use "
                    "the algebra's ordered basis and the constants must "
                    "satisfy antisymmetry and Jacobi."
                ),
                input={
                    "algebra": _SL2_ALGEBRA,
                    "left": _element([1, 0, 0]),
                    "right": _element([0, 1, 0]),
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
