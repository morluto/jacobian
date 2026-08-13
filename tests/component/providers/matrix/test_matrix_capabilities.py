from __future__ import annotations

from fractions import Fraction
from itertools import permutations
from random import Random
from typing import Any

import pytest
import sympy

from jacobian.contracts.capabilities import (
    CapabilityProviderAvailability,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus

_LARGE_CANONICAL_INTEGER = "1" + ("0" * 4_999) + "1"


def _rational(value: int | Fraction) -> dict[str, str]:
    exact = Fraction(value)
    return {"num": str(exact.numerator), "den": str(exact.denominator)}


def _matrix(rows: list[list[int | Fraction]]) -> dict[str, Any]:
    return {
        "matrix_schema_version": "1",
        "domain": "QQ",
        "entries": [[_rational(value) for value in row] for row in rows],
    }


def _reference_determinant(rows: list[list[Fraction]]) -> Fraction:
    total = Fraction(0)
    for permutation in permutations(range(len(rows))):
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(len(rows))
            for right in range(left + 1, len(rows))
        )
        term = Fraction(-1 if inversions % 2 else 1)
        for row, column in enumerate(permutation):
            term *= rows[row][column]
        total += term
    return total


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        pytest.param([[Fraction(-3, 7)]], Fraction(-3, 7), id="one-by-one"),
        pytest.param([[0, 2], [3, 4]], Fraction(-6), id="row-swap-sign"),
        pytest.param([[1, 2], [2, 4]], Fraction(0), id="singular"),
        pytest.param(
            [[Fraction(1, 2), 1], [3, Fraction(5, 2)]],
            Fraction(-7, 4),
            id="rational",
        ),
    ],
)
def test_matrix_determinant_compute_is_exact_and_unverified(
    matrix_services: Any,
    rows: list[list[int | Fraction]],
    expected: Fraction,
) -> None:
    runtime = matrix_services

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.determinant.compute",
            input={"matrix": _matrix(rows)},
        )
    )

    assert result.output["result"]["determinant"] == _rational(expected)
    assert result.output["backend_version"] == sympy.__version__
    assert result.artifact_uris == ()


def test_matrix_determinant_verify_independently_recomputes_exact_value(
    matrix_checker_services: Any,
) -> None:
    runtime = matrix_checker_services
    matrix = _matrix(
        [
            [1, 0, 1],
            [2, -1, 3],
            [4, 3, 2],
        ]
    )
    computed = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.determinant.compute",
            input={"matrix": matrix},
        )
    )
    verified = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.determinant.verify",
            input={
                "input": {"matrix": matrix},
                "candidate": computed.output["result"],
            },
        )
    )

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["conclusion"] == "TRUE"
    assert verified.output["verification_record_uri"].startswith("artifact://sha256/")
    assert verified.verification_record_uri is not None


def test_matrix_determinant_verify_rejects_wrong_bound_value(
    matrix_checker_services: Any,
) -> None:
    runtime = matrix_checker_services
    matrix = _matrix([[1, 2], [3, 4]])
    computed = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.determinant.compute",
            input={"matrix": matrix},
        )
    )
    rejected = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.determinant.verify",
            input={
                "input": {"matrix": matrix},
                "candidate": {
                    **computed.output["result"],
                    "determinant": _rational(2),
                },
            },
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None


def test_matrix_determinant_verify_timeout_is_not_a_conclusion(
    matrix_checker_services: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = matrix_checker_services
    matrix = _matrix([[1]])
    computed = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.determinant.compute",
            input={"matrix": matrix},
        )
    )
    monkeypatch.setattr(
        runtime.verification._checker_executor,
        "execute",
        lambda **_kwargs: (_ for _ in ()).throw(
            TimeoutError("checker execution timed out")
        ),
    )

    timed_out = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.determinant.verify",
            input={
                "input": {"matrix": matrix},
                "candidate": computed.output["result"],
            },
        )
    )

    assert timed_out.execution.status is ExecutionStatus.TIMEOUT
    assert timed_out.output == {}
    assert timed_out.verification_record_uri is None


def test_matrix_rank_compute_returns_rectangular_pivot_evidence(
    matrix_services: Any,
) -> None:
    runtime = matrix_services

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.rank.compute",
            input={
                "matrix": _matrix(
                    [
                        [1, 2, 3, 4],
                        [2, 4, 6, 8],
                        [0, 1, 1, 0],
                    ]
                )
            },
        )
    )

    assert result.output["result"]["rank"] == 2
    assert result.output["result"]["pivot_columns"] == [0, 1]
    assert result.output["backend_version"] == sympy.__version__
    assert result.artifact_uris == ()


def test_matrix_rank_verify_independently_recomputes_inline_candidate(
    matrix_checker_services: Any,
) -> None:
    matrix = _matrix([[1, 2, 3], [2, 4, 6], [0, 1, 1]])
    computed = matrix_checker_services.core.capabilities.invoke(
        CapabilityRequest(capability_id="matrix.rank.compute", input={"matrix": matrix})
    )

    verified = matrix_checker_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.rank.verify",
            input={
                "input": {"matrix": matrix},
                "candidate": computed.output["result"],
            },
        )
    )

    assert verified.output["status"] == "VERIFIED"
    assert verified.output["conclusion"] == "TRUE"
    assert verified.verification_record_uri is not None


def test_matrix_rank_rejects_authoritative_values_above_its_operation_budget(
    matrix_services: Any,
) -> None:
    result = matrix_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.rank.compute",
            input={
                "matrix": {
                    "matrix_schema_version": "1",
                    "domain": "QQ",
                    "entries": [[{"num": _LARGE_CANONICAL_INTEGER, "den": "1"}]],
                }
            },
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_EXACT_MATRIX_REQUEST"


def test_matrix_determinant_rejects_rectangular_input(
    matrix_services: Any,
) -> None:
    runtime = matrix_services

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.determinant.compute",
            input={"matrix": _matrix([[1, 2, 3], [4, 5, 6]])},
        )
    )

    assert result.execution.status.value == "ERROR"
    assert result.diagnostics[0].code == "INVALID_EXACT_MATRIX_REQUEST"


def test_matrix_determinant_matches_independent_bounded_oracle(
    matrix_services: Any,
) -> None:
    runtime = matrix_services
    random = Random(20260726)

    for size in range(1, 5):
        for _ in range(6):
            rows = [
                [
                    Fraction(random.randint(-5, 5), random.randint(1, 5))
                    for _ in range(size)
                ]
                for _ in range(size)
            ]
            result = runtime.core.capabilities.invoke(
                CapabilityRequest(
                    capability_id="matrix.determinant.compute",
                    input={"matrix": _matrix(rows)},
                )
            )

            assert result.output["result"]["determinant"] == _rational(
                _reference_determinant(rows)
            )


def test_matrix_capabilities_report_sympy_provider_identity(
    matrix_services: Any,
) -> None:
    runtime = matrix_services
    descriptors = {
        descriptor.capability_id: descriptor
        for descriptor in runtime.core.capabilities.catalog().capabilities
    }

    for capability_id in ("matrix.determinant.compute", "matrix.rank.compute"):
        descriptor = descriptors[capability_id]
        assert descriptor.provider == "jacobian.sympy"
        assert descriptor.provider_runtime.provider == "jacobian.sympy"
        assert (
            descriptor.provider_runtime.availability
            is CapabilityProviderAvailability.AVAILABLE
        )
        assert descriptor.provider_runtime.version == sympy.__version__
