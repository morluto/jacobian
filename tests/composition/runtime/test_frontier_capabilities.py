"""Reproductions and attack tests for accepted frontier capabilities."""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.capabilities import CapabilityMode, CapabilityRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.graph_optimization.bundle import (
    build_graph_optimization_bundle,
)
from jacobian.domains.polynomial.bundle import build_polynomial_bundle
from jacobian.domains.projective_geometry.bundle import (
    build_projective_geometry_bundle,
)
from jacobian.exact_domain_checkers import install_exact_domain_verification
from jacobian.portfolio.domain_installation import DomainBundleInstaller
from jacobian.portfolio.model import PortfolioPlan
from jacobian.runtime.config import CheckerAuthorityMode


@pytest.fixture
def frontier_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    bundles = (
        build_projective_geometry_bundle(),
        build_graph_optimization_bundle(),
        build_polynomial_bundle(),
    )
    with open_domain_services(
        tmp_path / "state",
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    ) as services:
        installed = DomainBundleInstaller(services.installation).install(
            PortfolioPlan(domain_bundles=bundles)
        )
        verifier_adapters, _ = install_exact_domain_verification(
            services.core.store,
            services.core.schemas,
            services.core.artifacts,
            services.application.verification,
            services.core.checkers,
            bundles={
                bundle.domain_id: (bundle, installed.installed[bundle.domain_id])
                for bundle in bundles
            },
            authorize=services.installation.authorizes_bundled_checkers,
        )
        for adapter in verifier_adapters:
            services.installation.register_capability(adapter)
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


def test_projective_arrangement_materializes_the_nine_line_flat_lattice(
    frontier_services: DomainTestServices,
) -> None:
    coefficients = (
        (0, 1, -1),
        (0, 1, 2),
        (0, 2, 1),
        (1, -2, -1),
        (1, -1, -2),
        (1, -1, 1),
        (1, 1, -1),
        (1, 1, 2),
        (2, -1, -2),
    )
    result = frontier_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=("geometry.projective_line_arrangement.flats.materialize"),
            input={
                "lines": [
                    {
                        "label": str(index),
                        "coefficients": [_q(value) for value in line],
                    }
                    for index, line in enumerate(coefficients, start=1)
                ]
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    payload = _result_payload(frontier_services, result)
    assert payload["non_double_flats"] == [
        ["1", "2", "3"],
        ["1", "4", "5"],
        ["1", "6", "7"],
        ["2", "4", "6"],
        ["2", "5", "8", "9"],
        ["3", "5", "7"],
        ["3", "6", "8"],
        ["4", "7", "9"],
    ]
    assert payload["multiplicity_histogram"] == [
        {"multiplicity": 2, "flat_count": 9},
        {"multiplicity": 3, "flat_count": 7},
        {"multiplicity": 4, "flat_count": 1},
    ]
    assert payload["pair_count_total"] == 36
    verified = frontier_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=("geometry.projective_line_arrangement.flats.verify"),
            mode=CapabilityMode.VERIFY,
            input={"result_uri": result.output["result_uri"]},
        )
    )
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"


def test_projective_arrangement_rejects_projectively_duplicate_lines(
    frontier_services: DomainTestServices,
) -> None:
    result = frontier_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=("geometry.projective_line_arrangement.flats.materialize"),
            input={
                "lines": [
                    {"label": "L1", "coefficients": [_q(1), _q(2), _q(3)]},
                    {"label": "L2", "coefficients": [_q(2), _q(4), _q(6)]},
                ]
            },
        )
    )
    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_PROJECTIVE_ARRANGEMENT_REQUEST"
    assert result.artifact_uris == ()


def test_arrangement_checker_rejects_schema_valid_forged_normalization(
    frontier_services: DomainTestServices,
) -> None:
    computed = frontier_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=("geometry.projective_line_arrangement.flats.materialize"),
            input={
                "lines": [
                    {"label": "L1", "coefficients": [_q(1), _q(0), _q(0)]},
                    {"label": "L2", "coefficients": [_q(0), _q(1), _q(0)]},
                ]
            },
        )
    )
    stored = frontier_services.core.store.get(computed.output["result_uri"])
    forged = deepcopy(stored.payload)
    forged["normalized_lines"][0]["coefficients"]["coordinates"] = [
        "1",
        "1",
        "0",
    ]
    forged_uri = frontier_services.core.artifacts.put(
        schema_uri=stored.manifest.schema_uri,
        semantics_uri=stored.manifest.semantics_uri,
        payload=forged,
        parents=(computed.output["input_uri"],),
        summary="schema-valid forged projective normalization",
    ).artifact_uri
    checked = frontier_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=("geometry.projective_line_arrangement.flats.verify"),
            mode=CapabilityMode.VERIFY,
            input={"result_uri": forged_uri},
        )
    )
    assert checked.execution.status is ExecutionStatus.COMPLETED
    assert checked.output["status"] == "REJECTED"


@pytest.mark.parametrize(
    ("graph", "decision"),
    [
        (
            {
                "vertices": ["a", "b", "c", "d"],
                "edges": [["a", "b"], ["b", "c"], ["c", "d"]],
            },
            "EXISTS",
        ),
        (
            {
                "vertices": ["c", "a", "b", "d"],
                "edges": [["c", "a"], ["c", "b"], ["c", "d"]],
            },
            "DOES_NOT_EXIST",
        ),
    ],
)
def test_hamiltonian_path_decision_has_independent_replay(
    frontier_services: DomainTestServices,
    graph: dict[str, object],
    decision: str,
) -> None:
    computed = frontier_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.hamiltonian_path.decide",
            input={"graph": graph},
        )
    )
    assert computed.execution.status is ExecutionStatus.COMPLETED
    assert computed.output["result"]["decision"] == decision
    assert computed.artifact_uris == ()

    verified = frontier_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.hamiltonian_path.verify",
            mode=CapabilityMode.VERIFY,
            input={"input": {"graph": graph}, "candidate": computed.output["result"]},
        )
    )
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"


def test_graded_jacobian_syzygy_finds_and_verifies_the_first_kernel(
    frontier_services: DomainTestServices,
) -> None:
    computed = frontier_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=("polynomial.jacobian_syzygy.minimum_degree.compute"),
            input={
                "polynomial": _polynomial([(1, (1, 1, 1))]),
                "max_degree": 3,
            },
        )
    )

    assert computed.execution.status is ExecutionStatus.COMPLETED
    result = _result_payload(frontier_services, computed)
    assert result["status"] == "FOUND"
    assert result["first_syzygy_degree"] == 1
    assert [(item["rank"], item["nullity"]) for item in result["degree_maps"]] == [
        (3, 0),
        (7, 2),
    ]
    assert result["degree_maps"][0]["rank_minor"]["determinant"] != _q(0)
    assert result["coefficient_map_detail"] == "CERTIFICATES"
    assert all(not item["sparse_entries"] for item in result["degree_maps"])

    verified = frontier_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=("polynomial.jacobian_syzygy.minimum_degree.verify"),
            mode=CapabilityMode.VERIFY,
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


def test_graded_jacobian_syzygy_handles_a_zero_partial_derivative(
    frontier_services: DomainTestServices,
) -> None:
    input_payload = {
        "polynomial": _polynomial([(1, (2, 0, 1))]),
        "max_degree": 0,
    }
    computed = frontier_services.core.capabilities.invoke(
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

    verified = frontier_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.jacobian_syzygy.minimum_degree.verify",
            mode=CapabilityMode.VERIFY,
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
    frontier_services: DomainTestServices,
    forgery: str,
) -> None:
    computed = frontier_services.core.capabilities.invoke(
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
    checked = frontier_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=("polynomial.jacobian_syzygy.minimum_degree.verify"),
            mode=CapabilityMode.VERIFY,
            input={"input": input_payload, "candidate": forged},
        )
    )
    assert checked.execution.status is ExecutionStatus.COMPLETED
    assert checked.output["status"] == "REJECTED"
    assert checked.output["conclusion"] == "UNKNOWN"


def test_hamiltonian_checker_rejects_a_forged_negative_decision(
    frontier_services: DomainTestServices,
) -> None:
    computed = frontier_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.hamiltonian_path.decide",
            input={
                "graph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["a", "b"], ["b", "c"]],
                }
            },
        )
    )
    input_payload = {
        "graph": {
            "vertices": ["a", "b", "c"],
            "edges": [["a", "b"], ["b", "c"]],
        }
    }
    forged = deepcopy(computed.output["result"])
    forged["decision"] = "DOES_NOT_EXIST"
    forged["path"] = []

    checked = frontier_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.hamiltonian_path.verify",
            mode=CapabilityMode.VERIFY,
            input={"input": input_payload, "candidate": forged},
        )
    )
    assert checked.execution.status is ExecutionStatus.COMPLETED
    assert checked.output["status"] == "REJECTED"
    assert checked.output["conclusion"] == "UNKNOWN"


def test_sparse_map_detail_is_explicitly_opt_in(
    frontier_services: DomainTestServices,
) -> None:
    computed = frontier_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=("polynomial.jacobian_syzygy.coefficients.materialize"),
            input={
                "polynomial": _polynomial([(1, (1, 1, 1))]),
                "max_degree": 0,
                "coefficient_map_detail": "SPARSE_ENTRIES",
            },
        )
    )
    entries = _result_payload(frontier_services, computed)["degree_maps"][0][
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
    frontier_services: DomainTestServices,
    factors: tuple[tuple[int, int, int], ...],
    expected_degree: int,
) -> None:
    computed = frontier_services.core.capabilities.invoke(
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
        _result_payload(frontier_services, computed)["first_syzygy_degree"]
        == expected_degree
    )
    verified = frontier_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=("polynomial.jacobian_syzygy.minimum_degree.verify"),
            mode=CapabilityMode.VERIFY,
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
