from __future__ import annotations

from pathlib import Path

import pytest
from tests.support.state import copy_template

from jacobian.capability_errors import CapabilityError
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityInstallTier,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
    CapabilityRequest,
    CapabilityResult,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.graph_optimization.bundle import build_graph_optimization_bundle
from jacobian.domains.graph_optimization.invariant_bundle import (
    build_graph_invariant_bundle,
)
from jacobian.portfolio.builtin import build_builtin_portfolio_components
from jacobian.provider_measurements import _cold_install_spec
from jacobian.runtime import CheckerAuthorityMode, create_runtime


class UnavailableAdapter:
    descriptor = CapabilityDescriptor(
        capability_id="fixture.unavailable",
        version="1",
        title="Unavailable fixture",
        description="Exercise fail-closed provider registration.",
        provider="tests.unavailable",
        provider_runtime=CapabilityProviderRuntime(
            provider="tests.unavailable",
            availability=CapabilityProviderAvailability.UNAVAILABLE,
            platform="linux-x86_64",
            install_tier=CapabilityInstallTier.T2,
            license_id="MIT",
            diagnostic="The fixture executable is not installed.",
        ),
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )

    def prepare(self, request: CapabilityRequest) -> CapabilityRequest:
        return request

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        raise AssertionError(f"unavailable adapter invoked: {request.capability_id}")


def test_catalog_exposes_exact_runtime_identity_for_every_adapter(
    attached_complete_runtime,
) -> None:
    runtime = attached_complete_runtime
    descriptors = {
        item.capability_id: item
        for item in runtime.core.capabilities.catalog().capabilities
    }

    assert descriptors
    assert all(
        item.provider_runtime.availability is CapabilityProviderAvailability.AVAILABLE
        for item in descriptors.values()
    )
    assert all(
        item.provider_runtime.digest is not None for item in descriptors.values()
    )
    assert descriptors["graph.compute.properties"].provider_runtime.provider == (
        "jacobian.networkx"
    )
    assert (
        descriptors["polynomial.map.compute_jacobian"].provider_runtime.provider
        == "jacobian.sympy"
    )
    assert (
        descriptors["universal_algebra.search.countermodel"].provider_runtime.provider
        == "jacobian.z3"
    )
    built_in_bundles = build_builtin_portfolio_components()
    built_in_ids = {
        capability_id
        for bundle in built_in_bundles
        if bundle.provider_runtime.availability
        is CapabilityProviderAvailability.AVAILABLE
        for capability_id in bundle.capability_ids
    }
    assert built_in_ids <= descriptors.keys()
    assert {
        "matrix.integer.row_hermite_normal_form",
        "matrix.integer.smith_normal_form",
        "polynomial.rational.gcd.compute",
        "lean.proof.repair_once",
    }.isdisjoint(descriptors)


def test_unavailable_adapter_is_rejected_before_catalog_advertisement(
    attached_complete_runtime,
) -> None:
    runtime = attached_complete_runtime

    with pytest.raises(CapabilityError, match="is unavailable"):
        runtime.core.capabilities.register(UnavailableAdapter())

    assert "fixture.unavailable" not in {
        item.capability_id for item in runtime.core.capabilities.catalog().capabilities
    }


def test_graph_domain_runtime_identities_bind_every_executed_backend() -> None:
    optimization_components = (
        build_graph_optimization_bundle().provider_runtime.configuration["components"]
    )
    invariant_components = (
        build_graph_invariant_bundle().provider_runtime.configuration["components"]
    )

    assert {component["provider"] for component in optimization_components} == {
        "jacobian.networkx",
        "jacobian.sympy",
        "jacobian.z3",
    }
    assert {component["provider"] for component in invariant_components} == {
        "jacobian.networkx",
        "jacobian.sympy",
    }
    assert (
        build_graph_optimization_bundle().provider_runtime.digest_kind
        is CapabilityProviderDigestKind.COMPOSITE
    )
    assert (
        build_graph_invariant_bundle().provider_runtime.digest_kind
        is CapabilityProviderDigestKind.COMPOSITE
    )


def test_unhealthy_optional_lean_runtime_is_absent_from_catalog(
    tmp_path: Path,
    authorized_portfolio_template: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = copy_template(authorized_portfolio_template, tmp_path / "state")
    unavailable = CapabilityProviderRuntime(
        provider="jacobian.lean4",
        availability=CapabilityProviderAvailability.UNAVAILABLE,
        platform="linux-x86_64",
        install_tier=CapabilityInstallTier.T3,
        license_id="Apache-2.0",
        diagnostic="The pinned Lean runtime is unavailable.",
    )
    monkeypatch.setattr(
        "jacobian.portfolio.provider_resolution.lean_provider_runtime",
        lambda **_kwargs: unavailable,
    )

    runtime = create_runtime(
        state, checker_authority=CheckerAuthorityMode.HYDRATE_EXISTING
    )
    try:
        assert runtime.portfolio_resources.lean is None
        capability_ids = {
            item.capability_id
            for item in runtime.core.capabilities.catalog().capabilities
        }
        assert {
            "lean.check",
            "lean.declaration.dependencies",
            "lean.declaration.inspect",
            "lean.declaration.search",
            "lean.proof.axioms.inspect",
            "lean.proof_edit.validate",
        }.isdisjoint(capability_ids)
    finally:
        runtime.close()


def test_unhealthy_lean_frontend_is_absent_from_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unavailable = CapabilityProviderRuntime(
        provider="jacobian.lean4",
        availability=CapabilityProviderAvailability.UNAVAILABLE,
        platform="linux-x86_64",
        install_tier=CapabilityInstallTier.T3,
        license_id="Apache-2.0",
        diagnostic=("TOOLCHAIN_RESOLUTION: the pinned Lean executable is unavailable"),
    )
    monkeypatch.setattr(
        "jacobian.portfolio.provider_resolution.lean_frontend_provider_runtime",
        lambda: unavailable,
    )

    runtime = create_runtime(tmp_path, checker_authority=CheckerAuthorityMode.NONE)
    try:
        capability_ids = {
            item.capability_id
            for item in runtime.core.capabilities.catalog().capabilities
        }
        assert {"lean.statement.propose", "lean.statement.compare"}.isdisjoint(
            capability_ids
        )
    finally:
        runtime.close()


def test_failed_invocation_keeps_provider_identity_in_the_descriptor(
    attached_complete_runtime,
) -> None:
    runtime = attached_complete_runtime
    descriptor = next(
        item
        for item in runtime.core.capabilities.catalog().capabilities
        if item.capability_id == "polynomial.map.evaluate"
    )
    # Invalid input is intentional: provenance must also survive failed execution.
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=descriptor.capability_id,
            input={},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert descriptor.provider_runtime is not None
    assert descriptor.provider_runtime.provider == descriptor.provider
    assert descriptor.provider_runtime.digest is not None


def test_source_runtime_has_no_implicit_working_directory_install() -> None:
    runtime = CapabilityProviderRuntime(
        provider="tests.fixture",
        availability=CapabilityProviderAvailability.AVAILABLE,
        version="1",
        digest="sha256:" + "a" * 64,
        digest_kind=CapabilityProviderDigestKind.SOURCE_TREE,
        platform="any",
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT",
    )

    assert _cold_install_spec(runtime) is None
