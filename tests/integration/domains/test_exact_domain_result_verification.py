from __future__ import annotations

from pathlib import Path

from tests.helpers.rationals import rational_payload as _q

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.exact_domain_checkers import install_exact_domain_verification
from jacobian.kernel import JacobianKernel


def _poly(*coefficients_ascending: int) -> dict[str, object]:
    return {
        "variables": ["x"],
        "polynomial": {
            "terms": [
                {"coefficient": _q(coefficient), "exponents": [exponent]}
                for exponent, coefficient in reversed(
                    tuple(enumerate(coefficients_ascending))
                )
                if coefficient
            ]
        },
    }


def _poly_xy(*terms: tuple[tuple[int, int], int]) -> dict[str, object]:
    return {
        "variables": ["x", "y"],
        "polynomial": {
            "terms": [
                {
                    "coefficient": _q(coefficient),
                    "exponents": list(exponents),
                }
                for exponents, coefficient in terms
            ]
        },
    }


def _install_verification(
    kernel: JacobianKernel, *, authorize: bool
) -> tuple[object, ...]:
    adapters, _ = install_exact_domain_verification(
        kernel.store,
        kernel.schemas,
        kernel.artifacts,
        kernel.verification,
        kernel.checkers,
        polynomial=kernel.domain_bundles["polynomial"],
        matrix=kernel.domain_bundles["matrix"],
        authorize=authorize,
    )
    for adapter in adapters:
        kernel.register_capability(adapter)
    return adapters


def _computed_gcd(kernel: JacobianKernel):
    return kernel.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.compute.gcd",
            input={
                "left": _poly(-1, 0, 1),
                "right": _poly(0, 1, 1),
            },
        )
    )


def test_public_seam_verifies_exact_producer_result(tmp_path: Path) -> None:
    kernel = JacobianKernel(tmp_path)
    adapters = _install_verification(kernel, authorize=True)
    runtime = adapters[0].descriptor.provider_runtime
    assert runtime is not None
    assert {
        component["provider"] for component in runtime.configuration["components"]
    } == {"jacobian.exact-domain-checker-source", "python-flint"}
    computed = _computed_gcd(kernel)

    verified = kernel.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.result.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": computed.output["result_uri"]},
        )
    )

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == "polynomial.compute.gcd"
    assert verified.output["result_uri"] == computed.output["result_uri"]
    assert verified.output["verification_record_uri"] is not None
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert len(verified.artifact_uris) == 4


def test_public_seam_rejects_validly_shaped_false_result(tmp_path: Path) -> None:
    kernel = JacobianKernel(tmp_path)
    _install_verification(kernel, authorize=True)
    computed = _computed_gcd(kernel)
    input_uri = computed.output["input_uri"]
    installed = kernel.domain_bundles["polynomial"]
    false_result = kernel.artifacts.put(
        schema_uri=installed.result_schema_uris["polynomial.compute.gcd"],
        semantics_uri=installed.semantics_uri,
        parents=(input_uri,),
        payload={
            "gcd": _poly(1),
            "bezout": {
                "left_multiplier": _poly(),
                "right_multiplier": _poly(),
            },
            "normalization": "MONIC",
        },
        summary="adversarial false GCD candidate",
    )

    rejected = kernel.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.result.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": false_result.artifact_uri},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
    assert rejected.assurance.level is CapabilityAssuranceLevel.COMPUTED


def test_public_seam_reports_valid_multivariate_result_as_unsupported(
    tmp_path: Path,
) -> None:
    kernel = JacobianKernel(tmp_path)
    _install_verification(kernel, authorize=True)
    computed = kernel.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.compute.resultant",
            input={
                "left": _poly_xy(((1, 0), 1), ((0, 1), 1)),
                "right": _poly_xy(((1, 0), 1), ((0, 0), 1)),
                "elimination_variable": "x",
            },
        )
    )

    checked = kernel.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.result.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": computed.output["result_uri"]},
        )
    )

    assert checked.execution.status is ExecutionStatus.COMPLETED
    assert checked.output["status"] == "UNSUPPORTED"
    assert checked.output["conclusion"] == "UNKNOWN"
    assert checked.output["witness_uri"] is None
    assert checked.output["verification_record_uri"] is None
    assert checked.assurance.level is CapabilityAssuranceLevel.COMPUTED


def test_induced_tree_result_is_domain_bound_and_independently_replayed(
    tmp_path: Path,
) -> None:
    kernel = JacobianKernel(tmp_path, install_references=True)
    computed = kernel.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.induced_tree.maximum.compute",
            input={
                "graph": {
                    "vertices": ["a", "b", "c", "d"],
                    "edges": [
                        ["a", "b"],
                        ["b", "c"],
                        ["c", "d"],
                        ["d", "a"],
                    ],
                },
                "resource_budget": {
                    "wall_seconds": 5,
                    "max_solver_calls": 33,
                    "max_order": 16,
                },
            },
        )
    )
    assert computed.output["optimum_value"] == 3
    result_uri = computed.artifact_uris[1]

    verified = kernel.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.induced_tree.maximum.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": result_uri},
        )
    )

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == "graph.induced_tree.maximum.compute"
    assert verified.output["verification_record_uri"] is not None
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.execution.detail == (
        "independent finite-subset exhaustive replay accepted "
        "graph.induced_tree.maximum.compute"
    )
    assert "FLINT" not in verified.execution.detail

    result_artifact = kernel.store.get(result_uri)
    false_payload = dict(result_artifact.payload)
    false_payload.update(
        {
            "optimum_value": 4,
            "incumbent_value": 4,
            "lower_bound": 4,
            "upper_bound": 4,
            "witness_vertices": ["a", "b", "c", "d"],
        }
    )
    false_result = kernel.artifacts.put(
        schema_uri=result_artifact.manifest.schema_uri,
        semantics_uri=result_artifact.manifest.semantics_uri,
        parents=result_artifact.manifest.parents,
        payload=false_payload,
        summary="adversarial false maximum induced-tree result",
    )
    rejected = kernel.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.induced_tree.maximum.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": false_result.artifact_uri},
        )
    )
    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None


def test_maximum_matching_result_uses_independent_tutte_berge_replay(
    tmp_path: Path,
) -> None:
    kernel = JacobianKernel(tmp_path, install_references=True)
    computed = kernel.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.maximum_matching.compute",
            input={
                "graph": {
                    "vertices": ["center", "x", "y", "z"],
                    "edges": [
                        ["center", "x"],
                        ["center", "y"],
                        ["center", "z"],
                    ],
                }
            },
        )
    )
    result_uri = computed.artifact_uris[1]

    verified = kernel.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.maximum_matching.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": result_uri},
        )
    )

    assert computed.capability_version == "2"
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == (
        "graph.invariant.maximum_matching.compute"
    )
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.execution.detail == (
        "independent Tutte-Berge barrier replay accepted "
        "graph.invariant.maximum_matching.compute"
    )
    runtime = next(
        descriptor.provider_runtime
        for descriptor in kernel.capabilities.catalog().capabilities
        if descriptor.capability_id == "graph.invariant.maximum_matching.verify"
    )
    assert runtime is not None
    assert runtime.provider == "jacobian.graph-exact-checkers"
    assert {
        component["provider"] for component in runtime.configuration["components"]
    } == {"jacobian.graph-exact-checker-source"}

    result_artifact = kernel.store.get(result_uri)
    false_result = kernel.artifacts.put(
        schema_uri=result_artifact.manifest.schema_uri,
        semantics_uri=result_artifact.manifest.semantics_uri,
        parents=result_artifact.manifest.parents,
        payload={
            "maximum_matching_cardinality": 0,
            "witness_edges": [],
            "certificate": {
                **result_artifact.payload["certificate"],
                "upper_bound": 0,
            },
        },
        summary="adversarial feasible but nonmaximum matching result",
    )
    rejected = kernel.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.maximum_matching.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": false_result.artifact_uri},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None


def test_operator_can_leave_exact_result_verification_unavailable(
    tmp_path: Path,
) -> None:
    kernel = JacobianKernel(tmp_path)

    adapters = _install_verification(kernel, authorize=False)

    assert adapters == ()
    assert {"polynomial.result.verify", "matrix.result.verify"}.isdisjoint(
        {
            descriptor.capability_id
            for descriptor in kernel.capabilities.catalog().capabilities
        }
    )
