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

from jacobian.bounded_process import BoundedProcessResult
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityDiscoveryRequest,
    CapabilityRequest,
)
from jacobian.contracts.matrix_operations import (
    MAX_OUTPUT_SCALAR_DIGITS,
    IntegerMatrix,
    IntegerMatrixRequest,
    LatticeReductionRequest,
    MatrixProductResult,
    MatrixTraceResult,
    NullspaceResult,
    RationalMatrixProductRequest,
    SquareIntegerMatrixRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.matrix_lattice.bundle import build_matrix_bundle
from jacobian.domains.matrix_lattice.capabilities import matrix_operation
from jacobian.domains.matrix_lattice.lattice import reduce_lattice_basis
from jacobian.domains.matrix_lattice.lattice_bundle import build_lattice_bundle
from jacobian.domains.matrix_lattice.operations import compute_smith_normal_form
from jacobian.operations import ComputedSuccess, OperationExecutionFailure


@fixture
def matrix_domain_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(
        tmp_path, build_matrix_bundle(), build_lattice_bundle()
    ) as services:
        yield services


def _qq(rows: list[list[int]]) -> dict[str, object]:
    return {
        "domain": "QQ",
        "entries": [[_q(value) for value in row] for row in rows],
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
                    "domain": "ZZ",
                    "entries": [["1", "2"], ["3", "4"]],
                }
            },
            {
                "inverse": {
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
                    "domain": "ZZ",
                    "entries": [["2", "4", "4"], ["6", "6", "12"]],
                }
            },
            {
                "normal_form": {
                    "domain": "ZZ",
                    "entries": [["2", "0", "0"], ["0", "6", "0"]],
                },
                "rank": 2,
                "invariant_factors": ["2", "6"],
                "transformation_available": False,
                "convention": "POSITIVE_DIVISIBILITY_DIAGONAL",
            },
        ),
    )

    for capability_id, payload, expected in cases:
        result = runtime.core.capabilities.invoke(
            CapabilityRequest(capability_id=capability_id, input=payload)
        )
        assert result.execution.status is ExecutionStatus.COMPLETED
        assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
        assert result.output["result"] == expected
        assert result.artifact_uris == ()
        assert result.episode_uri is None


def test_rational_relation_intent_reuses_the_exact_nullspace_operation(
    matrix_domain_services: DomainTestServices,
) -> None:
    discovered = matrix_domain_services.core.capabilities.discover(
        CapabilityDiscoveryRequest(
            query=(
                "all exact rational dependencies among named vectors and a "
                "normalized relation basis"
            ),
            limit=5,
        )
    )

    assert discovered.matches[0].capability_id == "matrix.nullspace.compute"
    assert discovered.matches[0].lexical_fit == "STRONG_CANDIDATE"
    descriptor = next(
        descriptor
        for descriptor in matrix_domain_services.core.capabilities.catalog().capabilities
        if descriptor.capability_id == "matrix.nullspace.compute"
    )
    assert descriptor.version == "2"
    assert descriptor.invocation_examples[0].name == ("rational_relation_among_columns")

    result = matrix_domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=descriptor.capability_id,
            input=descriptor.invocation_examples[0].input,
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
    discovered = matrix_domain_services.core.capabilities.discover(
        CapabilityDiscoveryRequest(
            query=(
                "multiply an exact matrix by itself and inspect whether its square "
                "is zero"
            ),
            limit=5,
        )
    )

    assert discovered.matches[0].capability_id == "matrix.multiply.compute"
    assert discovered.matches[0].lexical_fit == "STRONG_CANDIDATE"
    descriptor = next(
        descriptor
        for descriptor in matrix_domain_services.core.capabilities.catalog().capabilities
        if descriptor.capability_id == "matrix.multiply.compute"
    )
    assert descriptor.invocation_examples[0].name == "multiply_rectangular_matrices"

    result = matrix_domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=descriptor.capability_id,
            input=descriptor.invocation_examples[0].input,
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
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.characteristic_polynomial.compute",
            input={"matrix": _qq([[1, 2, 3], [4, 5, 6]])},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_EXACT_MATRIX_REQUEST"
    assert result.artifact_uris == ()

    incompatible_product = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.multiply.compute",
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
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.inverse.compute",
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
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.inverse.compute",
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
        lambda _request: MatrixTraceResult(trace="9" * (MAX_OUTPUT_SCALAR_DIGITS + 1)),
        "matrix.relation.test-of",
    )

    outcome = operation.implementation(request)

    assert isinstance(outcome, OperationExecutionFailure)
    assert outcome.status is ExecutionStatus.ERROR
    assert outcome.diagnostic.code == "MATRIX_OUTPUT_LIMIT_EXCEEDED"


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
    assert isinstance(outcome, ComputedSuccess)
    largest_output = max(
        len(value.lstrip("-"))
        for matrix in (outcome.value.reduced_basis, outcome.value.transformation)
        for row in matrix.entries
        for value in row
    )

    assert largest_output == 512
    assert largest_output <= MAX_OUTPUT_SCALAR_DIGITS


def test_lattice_lll_returns_exact_left_transformation(
    matrix_domain_services: DomainTestServices,
) -> None:
    runtime = matrix_domain_services
    source = [[4, 1], [1, 3]]
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="lattice.basis.reduce",
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
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
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


def test_lattice_lll_timeout_retains_no_operation_artifacts(
    matrix_domain_services: DomainTestServices,
    monkeypatch: MonkeyPatch,
) -> None:
    runtime = matrix_domain_services
    from jacobian.domains.matrix_lattice import lattice

    monkeypatch.setattr(
        lattice,
        "run_bounded_process",
        lambda *args, **kwargs: BoundedProcessResult(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=True,
        ),
    )
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="lattice.basis.reduce",
            input={
                "basis": {"domain": "ZZ", "entries": [["1"]]},
                "resource_budget": {"wall_seconds": 1},
            },
        )
    )

    assert result.execution.status is ExecutionStatus.TIMEOUT
    assert result.diagnostics[0].code == "FLINT_LLL_TIMEOUT"
    assert result.artifact_uris == ()
