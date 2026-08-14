from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError
from pytest import MonkeyPatch, fixture, raises
from tests.support.rationals import rational_payload as _q
from tests.support.services import (
    DomainTestServices,
    open_domain_services,
)

from jacobian.contracts.matrices import (
    MAX_MATRIX_SCALAR_DIGITS,
    IntegerMatrix,
    RationalMatrix,
)
from jacobian.contracts.matrix_operations import (
    MAX_INPUT_SCALAR_DIGITS,
    IntegerMatrixRequest,
    LatticeReductionRequest,
    LatticeReductionResult,
    MatrixProductResult,
    MatrixTraceResult,
    NullspaceResult,
    RationalMatrixProductRequest,
    RrefResult,
    SquareIntegerMatrixRequest,
)
from jacobian.contracts.operations import (
    OperationDiscoveryRequest,
    OperationRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.matrix_lattice.domain_declarations import matrix_operations
from jacobian.domains.matrix_lattice.lattice import reduce_lattice_basis
from jacobian.domains.matrix_lattice.lattice_declarations import lattice_operations
from jacobian.domains.matrix_lattice.operation_declarations import matrix_operation
from jacobian.domains.matrix_lattice.operations import (
    compute_smith_normal_form,
)
from jacobian.operations import OperationAbortError
from jacobian.process_policy import ProcessResult, ProcessTermination


@fixture
def matrix_domain_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(
        tmp_path, matrix_operations(), lattice_operations()
    ) as services:
        yield services


def _qq(rows: list[list[int]]) -> dict[str, object]:
    return {
        "matrix_schema_version": "1",
        "domain": "QQ",
        "entries": [[_q(value) for value in row] for row in rows],
    }


def _zz(rows: list[list[int]]) -> dict[str, object]:
    return {
        "matrix_schema_version": "1",
        "domain": "ZZ",
        "entries": [[str(value) for value in row] for row in rows],
    }


def _result_payload(services: DomainTestServices, result: object) -> dict[str, object]:
    result_uri = result.output["result_uri"]  # type: ignore[attr-defined]
    return services.core.store.get(result_uri).payload


def test_exact_matrix_domain_results_and_lineage(
    matrix_domain_services: DomainTestServices,
) -> None:
    runtime = matrix_domain_services
    cases = (
        (
            "matrix.inverse.compute",
            {
                "matrix": {
                    **_zz([[1, 2], [3, 4]]),
                }
            },
            {
                "inverse": {
                    "matrix_schema_version": "1",
                    "domain": "QQ",
                    "entries": [
                        [_q(-2), _q(1)],
                        [_q(3, 2), _q(-1, 2)],
                    ],
                },
                "convention": "TWO_SIDED_INVERSE_OVER_QQ",
            },
        ),
        (
            "matrix.trace.compute",
            {
                "matrix": {
                    "domain": "ZZ",
                    "entries": [["1", "2"], ["3", "4"]],
                }
            },
            {
                "trace": "5",
                "convention": "SUM_OF_DIAGONAL_ENTRIES",
            },
        ),
        (
            "matrix.multiply.compute",
            {
                "left": _qq([[1, 2, 0], [0, 1, 1]]),
                "right": _qq([[1, 0], [0, 1], [1, 1]]),
            },
            {
                "product": _qq([[1, 2], [1, 2]]),
                "left_rows": 2,
                "inner_dimension": 3,
                "right_columns": 2,
                "convention": "STANDARD_ROW_BY_COLUMN_PRODUCT_OVER_QQ",
            },
        ),
        (
            "matrix.normal_form.rref.compute",
            {"matrix": _qq([[1, 2, 3], [2, 4, 7]])},
            {
                "reduced_matrix": {
                    "matrix_schema_version": "1",
                    "domain": "QQ",
                    "entries": [
                        [_q(1), _q(2), _q(0)],
                        [_q(0), _q(0), _q(1)],
                    ],
                },
                "rank": 2,
                "pivot_columns": [0, 2],
                "free_columns": [1],
                "convention": "UNIQUE_RREF_OVER_QQ",
            },
        ),
        (
            "matrix.nullspace.compute",
            {"matrix": _qq([[1, 2, 3], [2, 4, 6]])},
            {
                "ambient_dimension": 3,
                "rank": 1,
                "nullity": 2,
                "basis_vectors": [
                    [_q(-2), _q(1), _q(0)],
                    [_q(-3), _q(0), _q(1)],
                ],
                "free_columns": [1, 2],
                "convention": "RREF_FUNDAMENTAL_BASIS",
            },
        ),
        (
            "matrix.characteristic_polynomial.compute",
            {"matrix": _qq([[1, 2], [3, 4]])},
            {
                "variable": "lambda",
                "degree": 2,
                "coefficients_descending": [_q(1), _q(-5), _q(-2)],
                "monic": True,
                "convention": "DET_LAMBDA_I_MINUS_A",
            },
        ),
        (
            "matrix.normal_form.smith.compute",
            {
                "matrix": {
                    **_zz([[2, 4, 4], [6, 6, 12]]),
                }
            },
            {
                "normal_form": _zz([[2, 0, 0], [0, 6, 0]]),
                "rank": 2,
                "invariant_factors": ["2", "6"],
                "transformation_available": False,
                "convention": "POSITIVE_DIVISIBILITY_DIAGONAL",
            },
        ),
        (
            "matrix.rational_linear_system.solve",
            {
                "matrix": {
                    "domain": "QQ",
                    "entries": [
                        [
                            {"num": "1", "den": "1"},
                            {"num": "2", "den": "1"},
                        ],
                        [
                            {"num": "3", "den": "1"},
                            {"num": "4", "den": "1"},
                        ],
                    ],
                },
                "rhs": [
                    {"num": "5", "den": "1"},
                    {"num": "11", "den": "1"},
                ],
            },
            {
                "solution": [
                    {"num": "1", "den": "1"},
                    {"num": "2", "den": "1"},
                ],
                "convention": "UNIQUE_SOLUTION_OVER_QQ",
            },
        ),
        (
            "matrix.adjugate.compute",
            {
                "matrix": {
                    "domain": "ZZ",
                    "entries": [["1", "2"], ["3", "4"]],
                }
            },
            {
                "adjugate": {
                    "matrix_schema_version": "1",
                    "domain": "ZZ",
                    "entries": [["4", "-2"], ["-3", "1"]],
                },
                "convention": "CLASSICAL_ADJUGATE",
            },
        ),
    )

    for operation_id, payload, expected in cases:
        result = runtime.core.operations.invoke(
            OperationRequest(operation_id=operation_id, input=payload)
        )
        assert result.execution.status is ExecutionStatus.COMPLETED
        assert result.output["result"] == expected
        assert result.artifact_uris == ()


def test_rational_relation_intent_reuses_the_exact_nullspace_operation(
    matrix_domain_services: DomainTestServices,
) -> None:
    discovered = matrix_domain_services.core.operations.search(
        OperationDiscoveryRequest(
            query=(
                "all exact rational dependencies among named vectors and a "
                "normalized relation basis"
            ),
            limit=5,
        )
    )

    assert discovered.matches[0].operation_id == "matrix.nullspace.compute"
    assert discovered.matches[0].relevance_score > 0
    descriptor = next(
        descriptor
        for descriptor in matrix_domain_services.core.operations.snapshot().operations
        if descriptor.operation_id == "matrix.nullspace.compute"
    )
    assert descriptor.version == "2"
    assert descriptor.examples[0].name == ("rational_relation_among_columns")

    result = matrix_domain_services.core.operations.invoke(
        OperationRequest(
            operation_id=descriptor.operation_id,
            input=descriptor.examples[0].input,
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {
        "ambient_dimension": 3,
        "rank": 2,
        "nullity": 1,
        "basis_vectors": [[_q(-1), _q(-1), _q(1)]],
        "free_columns": [2],
        "convention": "RREF_FUNDAMENTAL_BASIS",
    }


def test_matrix_multiplication_intent_is_discoverable(
    matrix_domain_services: DomainTestServices,
) -> None:
    discovered = matrix_domain_services.core.operations.search(
        OperationDiscoveryRequest(
            query=(
                "multiply an exact matrix by itself and inspect whether its square "
                "is zero"
            ),
            limit=5,
        )
    )

    assert discovered.matches[0].operation_id == "matrix.multiply.compute"
    assert discovered.matches[0].relevance_score > 0
    descriptor = next(
        descriptor
        for descriptor in matrix_domain_services.core.operations.snapshot().operations
        if descriptor.operation_id == "matrix.multiply.compute"
    )
    assert descriptor.examples[0].name == "multiply_rectangular_matrices"

    result = matrix_domain_services.core.operations.invoke(
        OperationRequest(
            operation_id=descriptor.operation_id,
            input=descriptor.examples[0].input,
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {
        "product": _qq([[1, 2], [1, 2]]),
        "left_rows": 2,
        "inner_dimension": 3,
        "right_columns": 2,
        "convention": "STANDARD_ROW_BY_COLUMN_PRODUCT_OVER_QQ",
    }


def test_matrix_product_contracts_reject_incompatible_or_mismatched_shapes() -> None:
    with raises(ValidationError, match="left column count"):
        RationalMatrixProductRequest.model_validate(
            {
                "left": _qq([[1, 2]]),
                "right": _qq([[1, 2]]),
            }
        )

    with raises(ValidationError, match="product row count"):
        MatrixProductResult.model_validate(
            {
                "product": _qq([[1, 2]]),
                "left_rows": 2,
                "inner_dimension": 1,
                "right_columns": 2,
            }
        )


def test_nullspace_result_enforces_rank_nullity() -> None:
    with raises(ValidationError, match="rank plus nullity"):
        NullspaceResult.model_validate(
            {
                "ambient_dimension": 3,
                "rank": 2,
                "nullity": 2,
                "basis_vectors": [
                    [_q(-2), _q(1), _q(0)],
                    [_q(-3), _q(0), _q(1)],
                ],
                "free_columns": [1, 2],
            }
        )


def test_invalid_matrix_request_fails_before_operation_artifacts(
    matrix_domain_services: DomainTestServices,
) -> None:
    runtime = matrix_domain_services
    result = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="matrix.characteristic_polynomial.compute",
            input={"matrix": _qq([[1, 2, 3], [4, 5, 6]])},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_EXACT_MATRIX_REQUEST"
    assert result.artifact_uris == ()

    incompatible_product = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="matrix.multiply.compute",
            input={
                "left": _qq([[1, 2]]),
                "right": _qq([[1, 2]]),
            },
        )
    )

    assert incompatible_product.execution.status is ExecutionStatus.ERROR
    assert incompatible_product.diagnostics[0].code == "INVALID_EXACT_MATRIX_REQUEST"
    assert incompatible_product.artifact_uris == ()


def test_singular_matrix_inverse_is_not_applicable(
    matrix_domain_services: DomainTestServices,
) -> None:
    runtime = matrix_domain_services
    result = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="matrix.inverse.compute",
            input={
                "matrix": {
                    "domain": "ZZ",
                    "entries": [["1", "2"], ["2", "4"]],
                }
            },
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "MATRIX_OPERATION_NOT_APPLICABLE"
    assert result.artifact_uris == ()


def test_inverse_accepts_exact_growth_from_maximum_size_input(
    matrix_domain_services: DomainTestServices,
) -> None:
    runtime = matrix_domain_services
    diagonal = "9" * 256
    determinant = str(int(diagonal) ** 2 - 1)
    result = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="matrix.inverse.compute",
            input={
                "matrix": {
                    "domain": "ZZ",
                    "entries": [[diagonal, "1"], ["1", diagonal]],
                }
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"]["inverse"]["entries"] == [
        [
            {"num": diagonal, "den": determinant},
            {"num": "-1", "den": determinant},
        ],
        [
            {"num": "-1", "den": determinant},
            {"num": diagonal, "den": determinant},
        ],
    ]


def test_matrix_output_contract_failure_is_operational_error() -> None:
    request = SquareIntegerMatrixRequest(matrix=IntegerMatrix(entries=(("1",),)))
    operation = matrix_operation(
        "matrix.test.compute",
        "Test matrix output boundary",
        "Exercise result validation in the matrix operation declaration.",
        SquareIntegerMatrixRequest,
        MatrixTraceResult,
        lambda _request: MatrixTraceResult(trace="9" * (MAX_MATRIX_SCALAR_DIGITS + 1)),
    )

    with raises(OperationAbortError) as exc_info:
        operation.execute(request)

    assert exc_info.value.status is ExecutionStatus.ERROR
    assert exc_info.value.diagnostic.code == "MATRIX_OUTPUT_LIMIT_EXCEEDED"


def test_smith_normal_form_preserves_rectangular_shape_and_zero_tail() -> None:
    cases = (
        ((("2", "4", "6"),), (("2", "0", "0"),), ("2",)),
        ((("2",), ("4",), ("6",)), (("2",), ("0",), ("0",)), ("2",)),
        (
            (("0", "0", "0"), ("0", "0", "0")),
            (("0", "0", "0"), ("0", "0", "0")),
            (),
        ),
        (
            (("0", "0", "0"), ("0", "2", "0")),
            (("2", "0", "0"), ("0", "0", "0")),
            ("2",),
        ),
    )

    for source, expected, factors in cases:
        result = compute_smith_normal_form(
            IntegerMatrixRequest(matrix=IntegerMatrix(entries=source))
        )
        assert result.normal_form.entries == expected
        assert result.invariant_factors == factors
        assert result.rank == len(factors)


def test_lll_worker_allows_result_growth_beyond_input_digit_limit() -> None:
    scalar = "9" * 256
    outcome = reduce_lattice_basis(
        LatticeReductionRequest(
            basis=IntegerMatrix(
                entries=(
                    ("1", scalar, "0"),
                    ("0", "1", scalar),
                    ("0", "0", "1"),
                )
            )
        )
    )
    assert isinstance(outcome, LatticeReductionResult)
    largest_output = max(
        len(value.lstrip("-"))
        for matrix in (outcome.reduced_basis, outcome.transformation)
        for row in matrix.entries
        for value in row
    )

    assert largest_output == 512
    assert largest_output <= MAX_MATRIX_SCALAR_DIGITS


def test_lattice_lll_returns_exact_left_transformation(
    matrix_domain_services: DomainTestServices,
) -> None:
    runtime = matrix_domain_services
    source = [[4, 1], [1, 3]]
    result = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="lattice.basis.reduce",
            input={
                "basis": {
                    "domain": "ZZ",
                    "entries": [[str(value) for value in row] for row in source],
                },
                "resource_budget": {"wall_seconds": 10},
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    computed = _result_payload(runtime, result)
    reduced = [
        [int(value) for value in row] for row in computed["reduced_basis"]["entries"]
    ]
    transformation = [
        [int(value) for value in row] for row in computed["transformation"]["entries"]
    ]
    assert reduced == [
        [
            sum(
                transformation[row][inner] * source[inner][column] for inner in range(2)
            )
            for column in range(2)
        ]
        for row in range(2)
    ]
    assert computed["delta"] == "0.99"
    assert computed["eta"] == "0.51"
    assert len(result.artifact_uris) == 2


def test_lattice_lll_supports_advertised_one_row_basis(
    matrix_domain_services: DomainTestServices,
) -> None:
    runtime = matrix_domain_services
    for entries, expected_rank in (([["1"]], 1), ([["3", "-4"]], 1)):
        result = runtime.core.operations.invoke(
            OperationRequest(
                operation_id="lattice.basis.reduce",
                input={"basis": {"domain": "ZZ", "entries": entries}},
            )
        )

        assert result.execution.status is ExecutionStatus.COMPLETED
        computed = _result_payload(runtime, result)
        assert computed["reduced_basis"]["entries"] == entries
        assert computed["transformation"]["entries"] == [["1"]]
        assert computed["rank"] == expected_rank
        assert len(result.artifact_uris) == 2


def test_lattice_lll_timeout_retains_no_operation_artifacts(
    matrix_domain_services: DomainTestServices,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = matrix_domain_services
    from jacobian.domains.matrix_lattice import lattice

    monkeypatch.setattr(
        lattice,
        "execute_process",
        lambda *args, **kwargs: ProcessResult(
            termination=ProcessTermination.TIMED_OUT,
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
        ),
    )
    result = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="lattice.basis.reduce",
            input={
                "basis": {"domain": "ZZ", "entries": [["1"]]},
                "resource_budget": {"wall_seconds": 1},
            },
        )
    )

    assert result.execution.status is ExecutionStatus.TIMEOUT
    assert result.diagnostics[0].code == "FLINT_LLL_TIMEOUT"
    assert result.artifact_uris == ()


def test_rref_result_feeds_product_request_without_artifact_uri(
    matrix_domain_services: DomainTestServices,
) -> None:
    """RREF reduced_matrix is a RationalMatrix that composes directly into product."""
    runtime = matrix_domain_services
    rref_result = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="matrix.normal_form.rref.compute",
            input={"matrix": _qq([[1, 2], [2, 4]])},
        )
    )
    assert rref_result.execution.status is ExecutionStatus.COMPLETED
    assert rref_result.artifact_uris == ()

    rref = RrefResult.model_validate(rref_result.output["result"])
    assert isinstance(rref.reduced_matrix, RationalMatrix)

    # 1. Insert the in-process Pydantic object directly as both operands.
    direct_request = RationalMatrixProductRequest(
        left=rref.reduced_matrix,
        right=rref.reduced_matrix,
    )
    assert direct_request.left is rref.reduced_matrix
    assert direct_request.right is rref.reduced_matrix

    # 2. Round-trip through serialized payload and reconstruct.
    serialized = rref.model_dump(mode="json")
    reconstructed = RationalMatrix.model_validate(serialized["reduced_matrix"])
    round_trip_request = RationalMatrixProductRequest(
        left=reconstructed,
        right=reconstructed,
    )
    assert round_trip_request.left.entries == rref.reduced_matrix.entries
    assert round_trip_request.right.entries == rref.reduced_matrix.entries

    # 3. The composition produces a correct product via the operation.
    product_result = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="matrix.multiply.compute",
            input={
                "left": serialized["reduced_matrix"],
                "right": serialized["reduced_matrix"],
            },
        )
    )
    assert product_result.execution.status is ExecutionStatus.COMPLETED
    assert product_result.artifact_uris == ()
    assert product_result.output["result"]["product"] == _qq([[1, 2], [0, 0]])


def test_oversized_authoritative_matrix_rejected_by_downstream_operation_budget() -> (
    None
):
    """A structurally valid RationalMatrix beyond 256 digits is rejected by request budget."""
    oversized = "9" * (MAX_INPUT_SCALAR_DIGITS + 1)
    assert len(oversized.lstrip("-")) <= MAX_MATRIX_SCALAR_DIGITS

    matrix = RationalMatrix.model_validate(
        {
            "domain": "QQ",
            "entries": [
                [_q_from_int(oversized), _q(0)],
                [_q(0), _q(1)],
            ],
        }
    )
    # The authoritative type accepts it — it is structurally valid.
    assert len(matrix.entries[0][0].num.lstrip("-")) == MAX_INPUT_SCALAR_DIGITS + 1

    # The downstream operation rejects it via its input budget validator.
    with raises(ValidationError, match="256 decimal digit"):
        RationalMatrixProductRequest(left=matrix, right=matrix)


def _q_from_int(value: str) -> dict[str, str]:
    return {"num": value, "den": "1"}
