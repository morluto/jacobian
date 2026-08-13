"""Jacobian-syzygy behavior and independent verification."""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.polynomial.bundle import build_polynomial_bundle
from tests.support.exact_domain import open_exact_domain_services
from tests.support.services import DomainTestServices


@pytest.fixture
def polynomial_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_exact_domain_services(
        tmp_path / "state",
        build_polynomial_bundle(),
    ) as services:
        yield services


def _q(value: int) -> dict[str, str]:
    return {"num": str(value), "den": "1"}


def _polynomial(
    terms: list[tuple[int, tuple[int, int, int]]],
) -> dict[str, object]:
    return {
        "variables": ["x", "y", "z"],
        "polynomial": {
            "terms": [
                {
                    "coefficient": _q(coefficient),
                    "exponents": list(exponents),
                }
                for coefficient, exponents in terms
            ]
        },
    }


def _result_payload(
    runtime: DomainTestServices,
    computed: Any,
) -> dict[str, Any]:
    if "result_uri" in computed.output:
        return runtime.core.store.get(computed.output["result_uri"]).payload
    return computed.output["result"]


def test_graded_jacobian_syzygy_finds_and_verifies_the_first_kernel(
    polynomial_services: DomainTestServices,
) -> None:
    descriptor = next(
        item
        for item in polynomial_services.core.capabilities.catalog().capabilities
        if item.capability_id == "polynomial.jacobian_syzygy.minimum_degree.compute"
    )
    sparse_example = next(
        item
        for item in descriptor.invocation_examples
        if item.name == "sparse-homogeneous-polynomial"
    )
    example_result = polynomial_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=descriptor.capability_id,
            input=sparse_example.input,
        )
    )
    assert example_result.execution.status is ExecutionStatus.COMPLETED
    assert "unique exponent tuples in descending lexicographic order" in (
        descriptor.description
    )

    computed = polynomial_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=("polynomial.jacobian_syzygy.minimum_degree.compute"),
            input={
                "polynomial": _polynomial([(1, (1, 1, 1))]),
                "max_degree": 3,
            },
        )
    )

    assert computed.execution.status is ExecutionStatus.COMPLETED
    result = _result_payload(polynomial_services, computed)
    assert result["status"] == "FOUND"
    assert result["first_syzygy_degree"] == 1
    assert [(item["rank"], item["nullity"]) for item in result["degree_maps"]] == [
        (3, 0),
        (7, 2),
    ]
    assert result["degree_maps"][0]["rank_minor"]["determinant"] != _q(0)
    assert result["coefficient_map_detail"] == "CERTIFICATES"
    assert all(not item["sparse_entries"] for item in result["degree_maps"])

    verified = polynomial_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=("polynomial.jacobian_syzygy.minimum_degree.verify"),
            input={
                "input": {
                    "polynomial": _polynomial([(1, (1, 1, 1))]),
                    "max_degree": 3,
                },
                "candidate": computed.output["result"],
            },
        )
    )
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"

    verifier = next(
        descriptor
        for descriptor in polynomial_services.core.capabilities.catalog().capabilities
        if descriptor.capability_id
        == "polynomial.jacobian_syzygy.minimum_degree.verify"
    )
    assert "complete, unmodified producer output.result object" in (
        verifier.description
    )


def test_graded_jacobian_syzygy_handles_a_zero_partial_derivative(
    polynomial_services: DomainTestServices,
) -> None:
    input_payload = {
        "polynomial": _polynomial([(1, (2, 0, 1))]),
        "max_degree": 0,
    }
    computed = polynomial_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.jacobian_syzygy.minimum_degree.compute",
            input=input_payload,
        )
    )

    assert computed.execution.status is ExecutionStatus.COMPLETED
    result = computed.output["result"]
    assert result["first_syzygy_degree"] == 0
    assert [(item["rank"], item["nullity"]) for item in result["degree_maps"]] == [
        (2, 1)
    ]
    assert result["partial_derivatives"][1]["polynomial"]["terms"] == []
    assert [item["num"] for item in result["kernel_witness"]["coefficient_vector"]] == [
        "0",
        "1",
        "0",
    ]

    verified = polynomial_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.jacobian_syzygy.minimum_degree.verify",
            input={"input": input_payload, "candidate": result},
        )
    )
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"


@pytest.mark.parametrize(
    "forgery",
    ("map_digest", "rank_minor", "kernel_vector", "partial_derivative"),
)
def test_syzygy_checker_rejects_schema_valid_forged_evidence(
    polynomial_services: DomainTestServices,
    forgery: str,
) -> None:
    computed = polynomial_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=("polynomial.jacobian_syzygy.minimum_degree.compute"),
            input={
                "polynomial": _polynomial([(1, (1, 1, 1))]),
                "max_degree": 1,
            },
        )
    )
    input_payload = {
        "polynomial": _polynomial([(1, (1, 1, 1))]),
        "max_degree": 1,
    }
    forged = deepcopy(computed.output["result"])
    if forgery == "map_digest":
        forged["degree_maps"][0]["matrix_digest"] = f"sha256:{'0' * 64}"
    elif forgery == "rank_minor":
        determinant = forged["degree_maps"][0]["rank_minor"]["determinant"]
        determinant["num"] = str(-int(determinant["num"]))
    elif forgery == "kernel_vector":
        forged["kernel_witness"]["coefficient_vector"][0] = _q(2)
    else:
        forged["partial_derivatives"][0]["polynomial"]["terms"][0]["coefficient"] = _q(
            2
        )
    checked = polynomial_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=("polynomial.jacobian_syzygy.minimum_degree.verify"),
            input={"input": input_payload, "candidate": forged},
        )
    )
    assert checked.execution.status is ExecutionStatus.COMPLETED
    assert checked.output["status"] == "REJECTED"
    assert checked.output["conclusion"] == "UNKNOWN"


def test_sparse_map_detail_is_explicitly_opt_in(
    polynomial_services: DomainTestServices,
) -> None:
    computed = polynomial_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=("polynomial.jacobian_syzygy.coefficients.materialize"),
            input={
                "polynomial": _polynomial([(1, (1, 1, 1))]),
                "max_degree": 0,
                "coefficient_map_detail": "SPARSE_ENTRIES",
            },
        )
    )
    entries = _result_payload(polynomial_services, computed)["degree_maps"][0][
        "sparse_entries"
    ]
    assert entries
    assert entries == sorted(entries, key=lambda item: (item["row"], item["column"]))


@pytest.mark.parametrize(
    ("factors", "expected_degree"),
    (
        (
            (
                (0, 1, -1),
                (0, 1, 2),
                (0, 2, 1),
                (1, -2, -1),
                (1, -1, -2),
                (1, -1, 1),
                (1, 1, -1),
                (1, 1, 2),
                (2, -1, -2),
            ),
            4,
        ),
        (
            (
                (1, 0, 0),
                (1, -1, 0),
                (1, 1, 0),
                (1, -1, -1),
                (0, 1, 1),
                (0, 0, 1),
                (1, 0, -1),
                (1, 1, 2),
                (1, -2, -1),
            ),
            5,
        ),
    ),
    ids=("mdr-4", "mdr-5"),
)
def test_nine_line_challenge_mdr_values_are_end_to_end_verified(
    polynomial_services: DomainTestServices,
    factors: tuple[tuple[int, int, int], ...],
    expected_degree: int,
) -> None:
    computed = polynomial_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=("polynomial.jacobian_syzygy.minimum_degree.compute"),
            input={
                "linear_factors": [
                    {
                        "label": str(index),
                        "coefficients": [_q(value) for value in coefficients],
                    }
                    for index, coefficients in enumerate(factors, start=1)
                ],
                "linear_factor_variables": ["x", "y", "z"],
                "max_degree": expected_degree,
            },
        )
    )
    assert computed.execution.status is ExecutionStatus.COMPLETED
    assert (
        _result_payload(polynomial_services, computed)["first_syzygy_degree"]
        == expected_degree
    )
    verified = polynomial_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=("polynomial.jacobian_syzygy.minimum_degree.verify"),
            input={
                "input": {
                    "linear_factors": [
                        {
                            "label": str(index),
                            "coefficients": [_q(value) for value in coefficients],
                        }
                        for index, coefficients in enumerate(factors, start=1)
                    ],
                    "linear_factor_variables": ["x", "y", "z"],
                    "max_degree": expected_degree,
                },
                "candidate": computed.output["result"],
            },
        )
    )
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
