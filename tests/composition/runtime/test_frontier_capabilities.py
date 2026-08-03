"""Reproductions and attack tests for accepted frontier capabilities."""

from __future__ import annotations

import shutil
from copy import deepcopy
from typing import Any

import pytest

from jacobian.contracts.capabilities import CapabilityMode, CapabilityRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.runtime import CheckerAuthorityMode, create_runtime
from jacobian.runtime.model import JacobianRuntime


@pytest.fixture(scope="module")
def frontier_runtime(
    tmp_path_factory: pytest.TempPathFactory,
    authorized_portfolio_template,
) -> JacobianRuntime:
    root = tmp_path_factory.mktemp("frontier-capabilities")
    shutil.copytree(authorized_portfolio_template, root, dirs_exist_ok=True)
    return create_runtime(root, checker_authority=CheckerAuthorityMode.HYDRATE_EXISTING)


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
    runtime: JacobianRuntime,
    computed: Any,
) -> dict[str, Any]:
    return runtime.core.store.get(computed.output["result_uri"]).payload


def test_projective_arrangement_materializes_the_nine_line_flat_lattice(
    frontier_runtime: JacobianRuntime,
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
    result = frontier_runtime.core.capabilities.invoke(
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
    payload = _result_payload(frontier_runtime, result)
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
    verified = frontier_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=("geometry.projective_line_arrangement.flats.verify"),
            mode=CapabilityMode.VERIFY,
            input={"result_uri": result.output["result_uri"]},
        )
    )
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"


def test_projective_arrangement_rejects_projectively_duplicate_lines(
    frontier_runtime: JacobianRuntime,
) -> None:
    result = frontier_runtime.core.capabilities.invoke(
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
    frontier_runtime: JacobianRuntime,
) -> None:
    computed = frontier_runtime.core.capabilities.invoke(
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
    forged = deepcopy(_result_payload(frontier_runtime, computed))
    forged["normalized_lines"][0]["coefficients"]["coordinates"] = [
        "1",
        "1",
        "0",
    ]
    installation = frontier_runtime.portfolio.domain_bundles["projective_geometry"]
    forged_uri = frontier_runtime.core.artifacts.put(
        schema_uri=installation.result_schema_uris[
            "geometry.projective_line_arrangement.flats.materialize"
        ],
        semantics_uri=installation.semantics_uri,
        payload=forged,
        parents=(computed.output["input_uri"],),
        summary="schema-valid forged projective normalization",
    ).artifact_uri
    checked = frontier_runtime.core.capabilities.invoke(
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
    frontier_runtime: JacobianRuntime,
    graph: dict[str, object],
    decision: str,
) -> None:
    computed = frontier_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.hamiltonian_path.decide",
            input={"graph": graph},
        )
    )
    assert computed.execution.status is ExecutionStatus.COMPLETED
    assert _result_payload(frontier_runtime, computed)["decision"] == decision

    verified = frontier_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.hamiltonian_path.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": computed.output["result_uri"]},
        )
    )
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"


def test_graded_jacobian_syzygy_finds_and_verifies_the_first_kernel(
    frontier_runtime: JacobianRuntime,
) -> None:
    computed = frontier_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=("polynomial.jacobian_syzygy.minimum_degree.compute"),
            input={
                "polynomial": _polynomial([(1, (1, 1, 1))]),
                "max_degree": 3,
            },
        )
    )

    assert computed.execution.status is ExecutionStatus.COMPLETED
    result = _result_payload(frontier_runtime, computed)
    assert result["status"] == "FOUND"
    assert result["first_syzygy_degree"] == 1
    assert [(item["rank"], item["nullity"]) for item in result["degree_maps"]] == [
        (3, 0),
        (7, 2),
    ]
    assert result["degree_maps"][0]["rank_minor"]["determinant"] != _q(0)
    assert result["coefficient_map_detail"] == "CERTIFICATES"
    assert all(not item["sparse_entries"] for item in result["degree_maps"])

    verified = frontier_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=("polynomial.jacobian_syzygy.minimum_degree.verify"),
            mode=CapabilityMode.VERIFY,
            input={"result_uri": computed.output["result_uri"]},
        )
    )
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"


def test_syzygy_checker_rejects_schema_valid_forged_evidence(
    frontier_runtime: JacobianRuntime,
) -> None:
    computed = frontier_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=("polynomial.jacobian_syzygy.minimum_degree.compute"),
            input={
                "polynomial": _polynomial([(1, (1, 1, 1))]),
                "max_degree": 1,
            },
        )
    )
    installation = frontier_runtime.portfolio.domain_bundles["polynomial"]
    input_uri = computed.output["input_uri"]
    for name in (
        "map_digest",
        "rank_minor",
        "kernel_vector",
        "partial_derivative",
    ):
        forged = deepcopy(_result_payload(frontier_runtime, computed))
        if name == "map_digest":
            forged["degree_maps"][0]["matrix_digest"] = f"sha256:{'0' * 64}"
        elif name == "rank_minor":
            determinant = forged["degree_maps"][0]["rank_minor"]["determinant"]
            determinant["num"] = str(-int(determinant["num"]))
        elif name == "kernel_vector":
            forged["kernel_witness"]["coefficient_vector"][0] = _q(2)
        else:
            forged["partial_derivatives"][0]["polynomial"]["terms"][0][
                "coefficient"
            ] = _q(2)
        forged_uri = frontier_runtime.core.artifacts.put(
            schema_uri=installation.result_schema_uris[
                "polynomial.jacobian_syzygy.minimum_degree.compute"
            ],
            semantics_uri=installation.semantics_uri,
            payload=forged,
            parents=(input_uri,),
            summary=f"schema-valid adversarial graded-map result: {name}",
        ).artifact_uri

        checked = frontier_runtime.core.capabilities.invoke(
            CapabilityRequest(
                capability_id=("polynomial.jacobian_syzygy.minimum_degree.verify"),
                mode=CapabilityMode.VERIFY,
                input={"result_uri": forged_uri},
            )
        )
        assert checked.execution.status is ExecutionStatus.COMPLETED
        assert checked.output["status"] == "REJECTED"
        assert checked.output["conclusion"] == "UNKNOWN"


def test_hamiltonian_checker_rejects_a_forged_negative_decision(
    frontier_runtime: JacobianRuntime,
) -> None:
    computed = frontier_runtime.core.capabilities.invoke(
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
    forged = deepcopy(_result_payload(frontier_runtime, computed))
    forged["decision"] = "DOES_NOT_EXIST"
    forged["path"] = []
    installation = frontier_runtime.portfolio.domain_bundles["graph_optimization"]
    forged_uri = frontier_runtime.core.artifacts.put(
        schema_uri=installation.result_schema_uris["graph.hamiltonian_path.decide"],
        semantics_uri=installation.semantics_uri,
        payload=forged,
        parents=(computed.output["input_uri"],),
        summary="schema-valid forged negative Hamiltonian-path decision",
    ).artifact_uri

    checked = frontier_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.hamiltonian_path.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": forged_uri},
        )
    )
    assert checked.execution.status is ExecutionStatus.COMPLETED
    assert checked.output["status"] == "REJECTED"
    assert checked.output["conclusion"] == "UNKNOWN"


def test_sparse_map_detail_is_explicitly_opt_in(
    frontier_runtime: JacobianRuntime,
) -> None:
    computed = frontier_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=("polynomial.jacobian_syzygy.minimum_degree.compute"),
            input={
                "polynomial": _polynomial([(1, (1, 1, 1))]),
                "max_degree": 0,
                "coefficient_map_detail": "SPARSE_ENTRIES",
            },
        )
    )
    entries = _result_payload(frontier_runtime, computed)["degree_maps"][0][
        "sparse_entries"
    ]
    assert entries
    assert entries == sorted(entries, key=lambda item: (item["row"], item["column"]))


def test_nine_line_challenge_mdr_values_are_end_to_end_verified(
    frontier_runtime: JacobianRuntime,
) -> None:
    cases = (
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
    )
    for factors, expected_degree in cases:
        computed = frontier_runtime.core.capabilities.invoke(
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
            _result_payload(frontier_runtime, computed)["first_syzygy_degree"]
            == expected_degree
        )
        verified = frontier_runtime.core.capabilities.invoke(
            CapabilityRequest(
                capability_id=("polynomial.jacobian_syzygy.minimum_degree.verify"),
                mode=CapabilityMode.VERIFY,
                input={"result_uri": computed.output["result_uri"]},
            )
        )
        assert verified.execution.status is ExecutionStatus.COMPLETED
        assert verified.output["status"] == "VERIFIED"
