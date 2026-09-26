from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.lie_algebras._models import (
    FiniteDimensionalLieAlgebra,
    LieAlgebraElement,
    LieSubalgebra,
)
from jacobian.math.lie_algebras.operations import (
    check_subalgebra,
    lie_generated_subalgebra,
)

HEISENBERG = FiniteDimensionalLieAlgebra.model_validate(
    {
        "basis": ["x", "y", "z"],
        "structure_constants": [
            {"i": 0, "j": 1, "k": 2, "coefficient": {"num": 1, "den": 1}}
        ],
    }
)


def _v(*coordinates: int) -> LieAlgebraElement:
    return LieAlgebraElement.model_validate(
        {
            "basis": HEISENBERG.basis,
            "coordinates": [{"num": value, "den": 1} for value in coordinates],
        }
    )


def _rows(value: LieSubalgebra) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(entry.as_fraction() for entry in row) for row in value.generators.entries
    )


def test_heisenberg_generators_close_to_full_algebra_and_compose() -> None:
    generated = lie_generated_subalgebra(HEISENBERG, [_v(1, 0, 0), _v(0, 1, 0)])

    assert generated.algebra == HEISENBERG
    assert generated.basis == HEISENBERG.basis
    assert _rows(generated) == (
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    assert check_subalgebra(HEISENBERG, generated).is_subalgebra
    assert LieSubalgebra.model_validate(generated.model_dump()) == generated


def test_empty_and_single_generator_cases() -> None:
    assert _rows(lie_generated_subalgebra(HEISENBERG, [])) == ()
    assert _rows(lie_generated_subalgebra(HEISENBERG, [_v(2, 0, 3)])) == (
        (Fraction(1), Fraction(0), Fraction(3, 2)),
    )


def test_request_rejects_vectors_on_a_different_axis() -> None:
    alien = LieAlgebraElement.model_validate(
        {"basis": ["u", "v", "w"], "coordinates": [{"num": 1, "den": 1}] * 3}
    )
    from jacobian.math.lie_algebras._models import LieGeneratedSubalgebraRequest

    with pytest.raises(ValidationError, match="generators must use"):
        LieGeneratedSubalgebraRequest(algebra=HEISENBERG, generators=[alien])


def test_subalgebra_model_rejects_forged_ambient_axis() -> None:
    with pytest.raises(ValidationError, match="source algebra's ordered basis"):
        LieSubalgebra.model_validate(
            {
                "algebra": HEISENBERG.model_dump(),
                "basis": ["y", "x", "z"],
                "generators": {
                    "domain": "QQ",
                    "row_count": 0,
                    "column_count": 3,
                    "entries": [],
                },
            }
        )


def test_large_input_height_is_rejected_by_iterated_growth_preflight() -> None:
    large = 10**63
    algebra = FiniteDimensionalLieAlgebra.model_validate(
        {
            "basis": ["x", "y", "z"],
            "structure_constants": [
                {
                    "i": 0,
                    "j": 1,
                    "k": 2,
                    "coefficient": {"num": large, "den": 1},
                }
            ],
        }
    )
    generators = [
        LieAlgebraElement.model_validate(
            {
                "basis": ["x", "y", "z"],
                "coordinates": [
                    {"num": large, "den": 1},
                    {"num": 1, "den": 1},
                    {"num": 0, "den": 1},
                ],
            }
        ),
        LieAlgebraElement.model_validate(
            {
                "basis": ["x", "y", "z"],
                "coordinates": [
                    {"num": 1, "den": 1},
                    {"num": large, "den": 1},
                    {"num": 0, "den": 1},
                ],
            }
        ),
    ]
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        lie_generated_subalgebra(algebra, generators)
    assert (
        exc_info.value.errors()[0]["type"]
        == "lie_algebra.generated_subalgebra_height_bound"
    )


def test_moderate_height_full_rank_generators_are_admitted() -> None:
    coefficient = 10**7
    algebra = FiniteDimensionalLieAlgebra.model_validate(
        {
            "basis": ["a", "b"],
            "structure_constants": [
                {
                    "i": 0,
                    "j": 1,
                    "k": 1,
                    "coefficient": {"num": coefficient, "den": 1},
                }
            ],
        }
    )
    result = lie_generated_subalgebra(
        algebra,
        [
            LieAlgebraElement.model_validate(
                {
                    "basis": ["a", "b"],
                    "coordinates": [{"num": 1, "den": 1}, {"num": 0, "den": 1}],
                }
            ),
            LieAlgebraElement.model_validate(
                {
                    "basis": ["a", "b"],
                    "coordinates": [{"num": 0, "den": 1}, {"num": 1, "den": 1}],
                }
            ),
        ],
    )
    assert result.generators.row_count == 2


def test_full_rank_basis_generators_admit_without_iterated_blowup() -> None:
    algebra = FiniteDimensionalLieAlgebra.model_validate(
        {
            "basis": ["a", "b", "c", "d"],
            "structure_constants": [
                {"i": 0, "j": 1, "k": 1, "coefficient": {"num": 1, "den": 1}}
            ],
        }
    )
    generators = [
        LieAlgebraElement.model_validate(
            {
                "basis": ["a", "b", "c", "d"],
                "coordinates": [
                    {"num": 1 if position == index else 0, "den": 1}
                    for position in range(4)
                ],
            }
        )
        for index in range(4)
    ]

    generated = lie_generated_subalgebra(algebra, generators)

    assert _rows(generated) == (
        (Fraction(1), Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
    )
    assert check_subalgebra(algebra, generated).is_subalgebra


def test_operation_rejects_generators_on_a_different_axis() -> None:
    alien = LieAlgebraElement.model_validate(
        {"basis": ["u", "v", "w"], "coordinates": [{"num": 1, "den": 1}] * 3}
    )
    with pytest.raises(
        OperationDomainValidationError, match="generators must use"
    ) as exc_info:
        lie_generated_subalgebra(HEISENBERG, [alien])
    assert (
        exc_info.value.errors()[0]["type"]
        == "lie_algebra.generated_subalgebra_generator_basis"
    )


def test_catalog_declares_constructive_subalgebra_operation() -> None:
    from jacobian.math.lie_algebras._tools import TOOLS

    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "lie_algebra.subalgebra.generated.compute"
    )
    assert operation.result_type is LieSubalgebra
