from fractions import Fraction

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.lie_algebras._models import (
    FiniteDimensionalLieAlgebra,
    LieSubalgebraResult,
    LieSubspace,
)
from jacobian.math.lie_algebras.operations import lie_subalgebra

SL2 = FiniteDimensionalLieAlgebra.model_validate(
    {
        "basis": ["e", "f", "h"],
        "structure_constants": [
            {"i": 0, "j": 1, "k": 2, "coefficient": {"num": 1, "den": 1}},
            {"i": 0, "j": 2, "k": 0, "coefficient": {"num": -2, "den": 1}},
            {"i": 1, "j": 2, "k": 1, "coefficient": {"num": 2, "den": 1}},
        ],
    }
)


def _subspace(rows: list[list[int]]) -> LieSubspace:
    from jacobian.math.matrices.values import rational_matrix_from_fractions

    return LieSubspace.model_validate(
        {
            "basis": SL2.basis,
            "generators": rational_matrix_from_fractions(
                tuple(tuple(Fraction(value) for value in row) for row in rows)
            ).model_dump(),
        }
    )


def test_borel_subalgebra_has_exact_induced_structure_constants_and_inclusion() -> None:
    result = lie_subalgebra(SL2, _subspace([[1, 0, 0], [0, 0, 1]]), ["e", "h"])

    assert result.induced.basis == ("e", "h")
    assert [
        (constant.i, constant.j, constant.k, constant.coefficient.as_fraction())
        for constant in result.induced.structure_constants
    ] == [(0, 1, 0, Fraction(-2))]
    assert result.inclusion.entries == result.subspace.generators.entries
    assert result.subspace.algebra == SL2
    assert (
        LieSubalgebraResult.model_validate_json(result.model_dump_json(), strict=True)
        == result
    )

    # Independent matrix oracle with e=[[0,1],[0,0]], h=diag(1,-1):
    # e*h-h*e = [[0,-2],[0,0]] = -2e.
    e_h = ((0, -1), (0, 0))
    h_e = ((0, 1), (0, 0))
    assert tuple(
        tuple(a - b for a, b in zip(row_a, row_b, strict=True))
        for row_a, row_b in zip(e_h, h_e, strict=True)
    ) == ((0, -2), (0, 0))


def test_subspace_that_is_not_closed_is_rejected() -> None:
    with pytest.raises(OperationDomainValidationError, match="bracket escapes"):
        lie_subalgebra(SL2, _subspace([[1, 0, 0], [0, 1, 0]]), ["e", "f"])


def test_induced_bracket_transports_under_an_invertible_rational_basis_change() -> None:
    # New ambient basis u=e+h, v=f, w=h. Exact transport gives
    # [u,v]=w-2v, [u,w]=-2u+2w, [v,w]=2v.
    changed = FiniteDimensionalLieAlgebra.model_validate(
        {
            "basis": ["u", "v", "w"],
            "structure_constants": [
                {"i": 0, "j": 1, "k": 1, "coefficient": {"num": -2, "den": 1}},
                {"i": 0, "j": 1, "k": 2, "coefficient": {"num": 1, "den": 1}},
                {"i": 0, "j": 2, "k": 0, "coefficient": {"num": -2, "den": 1}},
                {"i": 0, "j": 2, "k": 2, "coefficient": {"num": 2, "den": 1}},
                {"i": 1, "j": 2, "k": 1, "coefficient": {"num": 2, "den": 1}},
            ],
        }
    )
    candidate = LieSubspace.model_validate(
        {
            "basis": changed.basis,
            "generators": {
                "domain": "QQ",
                "row_count": 2,
                "column_count": 3,
                "entries": [
                    [{"num": 1, "den": 1}, {"num": 0, "den": 1}, {"num": 0, "den": 1}],
                    [{"num": 0, "den": 1}, {"num": 0, "den": 1}, {"num": 1, "den": 1}],
                ],
            },
        }
    )

    result = lie_subalgebra(changed, candidate, ["u", "w"])

    assert [
        (constant.i, constant.j, constant.k, constant.coefficient.as_fraction())
        for constant in result.induced.structure_constants
    ] == [(0, 1, 0, Fraction(-2)), (0, 1, 1, Fraction(2))]
    # This is the coordinate transform of [e,h]=-2e under e=u-w and h=w.


def test_wrong_number_of_induced_basis_labels_is_rejected() -> None:
    with pytest.raises(OperationDomainValidationError, match="one unique basis label"):
        lie_subalgebra(SL2, _subspace([[1, 0, 0], [0, 0, 1]]), ["e"])


def test_induced_coefficient_bound_precedes_canonical_result_construction() -> None:
    scale = 10**32
    large_product = scale**2
    algebra = FiniteDimensionalLieAlgebra.model_validate(
        {
            "basis": ["x", "y", "z"],
            "structure_constants": [
                {"i": 0, "j": 1, "k": 2, "coefficient": {"num": scale, "den": 1}},
                {
                    "i": 1,
                    "j": 2,
                    "k": 0,
                    "coefficient": {"num": -scale, "den": 1},
                },
                {
                    "i": 1,
                    "j": 2,
                    "k": 2,
                    "coefficient": {"num": -(large_product - 1), "den": 1},
                },
            ],
        }
    )
    candidate = LieSubspace.model_validate(
        {
            "basis": algebra.basis,
            "generators": {
                "domain": "QQ",
                "row_count": 2,
                "column_count": 3,
                "entries": [
                    [
                        {"num": 1, "den": 1},
                        {"num": 0, "den": 1},
                        {"num": scale, "den": 1},
                    ],
                    [
                        {"num": 0, "den": 1},
                        {"num": 1, "den": 1},
                        {"num": 0, "den": 1},
                    ],
                ],
            },
        }
    )

    # The source brackets satisfy [x,y]=scale*z and
    # [z,y]=scale*x+(scale^2-1)*z. Thus the candidate basis
    # r=x+scale*z, y is closed with [r,y]=scale^2*r. Its induced
    # coefficient has 65 digits although every source coefficient has <=64.
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        lie_subalgebra(algebra, candidate, ["r", "y"])
    assert (
        exc_info.value.errors()[0]["type"]
        == "lie_algebra.subalgebra_result_height_bound"
    )


def test_catalog_declares_subalgebra_structure_constant_transform() -> None:
    from jacobian.math.lie_algebras._tools import TOOLS

    tool = next(
        tool for tool in TOOLS if tool.operation_id == "lie_algebra.subalgebra.compute"
    )
    assert tool.request_type.__name__ == "LieSubalgebraConstructionRequest"
