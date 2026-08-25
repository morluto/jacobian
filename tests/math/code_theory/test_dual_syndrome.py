"""Regression tests for the retained dual-code and syndrome operations."""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.code_linear._models import (
    DualCodeRequest,
    DualCodeResult,
    ParityCheckMatrix,
    SyndromeRequest,
    SyndromeResult,
)
from jacobian.math.code_linear._operations import (
    compute_dual_code,
    compute_syndrome,
)
from jacobian.math.code_linear.values import PrimeFieldLinearEncoder
from jacobian.math.code_theory._tools import TOOLS


def _encoder(
    generator_matrix: tuple[tuple[int, ...], ...], field_order: int = 2
) -> PrimeFieldLinearEncoder:
    return PrimeFieldLinearEncoder(
        field_order=field_order,
        message_axis=tuple(f"m{index}" for index in range(len(generator_matrix))),
        coordinate_axis=tuple(f"x{index}" for index in range(len(generator_matrix[0]))),
        generator_matrix=generator_matrix,
    )


def test_dual_and_syndrome_operation_ids_remain_published() -> None:
    tools_by_id = {tool.operation_id: tool for tool in TOOLS}

    assert {"code.dual_code.compute", "code.syndrome.compute"} <= set(tools_by_id)
    assert tools_by_id["code.dual_code.compute"].request_type is DualCodeRequest
    assert tools_by_id["code.dual_code.compute"].result_type is DualCodeResult
    assert tools_by_id["code.syndrome.compute"].request_type is SyndromeRequest
    assert tools_by_id["code.syndrome.compute"].result_type is SyndromeResult


def test_dual_hamming_7_4() -> None:
    result = compute_dual_code(
        DualCodeRequest(
            encoder=_encoder(
                (
                    (1, 0, 0, 0, 1, 1, 0),
                    (0, 1, 0, 0, 1, 0, 1),
                    (0, 0, 1, 0, 0, 1, 1),
                    (0, 0, 0, 1, 1, 1, 1),
                )
            )
        )
    )
    assert result.dimension == 4
    assert result.length == 7
    assert result.dual_dimension == 3
    assert len(result.parity_check.rows) == 3


def test_dual_code_is_orthogonal_over_the_prime_field() -> None:
    request = DualCodeRequest(encoder=_encoder(((2, 1),), field_order=3))
    result = compute_dual_code(request)
    assert result.dimension == result.dual_dimension == 1
    assert all(
        sum(
            generator * dual
            for generator, dual in zip(
                request.encoder.generator_matrix[0], row, strict=True
            )
        )
        % request.encoder.field_order
        == 0
        for row in result.parity_check.rows
    )


def test_syndrome_is_computed_modulo_the_field_order() -> None:
    result = compute_syndrome(
        SyndromeRequest(
            parity_check=ParityCheckMatrix(
                field_order=3,
                coordinate_axis=("x0", "x1"),
                rows=((1, 1), (0, 1)),
            ),
            coordinate_axis=("x0", "x1"),
            word=(2, 2),
        )
    )
    assert result.syndrome == (1, 2)


def test_full_space_dual_composes_into_syndrome_without_axis_reconstruction() -> None:
    dual = compute_dual_code(DualCodeRequest(encoder=_encoder(((1, 0), (0, 1)))))

    assert dual.parity_check.rows == ()
    syndrome = compute_syndrome(
        SyndromeRequest(
            parity_check=dual.parity_check,
            coordinate_axis=dual.parity_check.coordinate_axis,
            word=(1, 1),
        )
    )

    assert syndrome.syndrome == ()
    assert syndrome.is_member is True


def test_serialized_dual_parity_check_dispatches_into_syndrome() -> None:
    catalog = Catalog.open()
    dual = invoke_operation(
        "code.dual_code.compute",
        {
            "encoder": {
                "field_order": 2,
                "message_axis": ["m0", "m1"],
                "coordinate_axis": ["left", "right"],
                "generator_matrix": [[1, 0], [0, 1]],
            }
        },
        catalog,
    )
    parity_check = dual.output["parity_check"]
    syndrome = invoke_operation(
        "code.syndrome.compute",
        {
            "parity_check": parity_check,
            "coordinate_axis": parity_check["coordinate_axis"],
            "word": [1, 1],
        },
        catalog,
    )

    assert syndrome.output["syndrome"] == []
    assert syndrome.output["is_member"] is True
