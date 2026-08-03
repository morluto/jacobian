from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.capabilities import (
    CapabilityInstallTier,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
)
from jacobian.contracts.checkers import CheckerRegistration, EvidenceKind

_ARTIFACT_URI = "artifact://sha256/" + "a" * 64


def _python_distribution_runtime(
    *,
    configuration: dict[str, str] | None = None,
) -> CapabilityProviderRuntime:
    return CapabilityProviderRuntime(
        provider="tests.distribution",
        availability=CapabilityProviderAvailability.AVAILABLE,
        version="1.2.3",
        digest="sha256:" + "b" * 64,
        digest_kind=CapabilityProviderDigestKind.PYTHON_DISTRIBUTION_RECORD,
        platform="linux-x86_64",
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT",
        configuration=configuration
        or {
            "distribution": "tests-distribution",
            "import_name": "tests_distribution",
        },
    )


def _registration(runtime: CapabilityProviderRuntime) -> CheckerRegistration:
    return CheckerRegistration(
        checker_id="checker://sha256/" + "c" * 64,
        name="distribution-backed checker",
        entrypoint="jacobian_checkers.reject:check",
        executable_digest="sha256:" + "d" * 64,
        provider_runtime=runtime,
        evidence_kind=EvidenceKind.WITNESS,
        format_id="tests.distribution",
        format_version="1",
        claim_schema_uris=(_ARTIFACT_URI,),
        semantics_uris=(_ARTIFACT_URI,),
        candidate_schema_uris=(_ARTIFACT_URI,),
    )


def _source_tree_runtime(
    *,
    configuration: dict[str, str] | None = None,
) -> CapabilityProviderRuntime:
    return CapabilityProviderRuntime(
        provider="tests.source-tree",
        availability=CapabilityProviderAvailability.AVAILABLE,
        version="1",
        digest="sha256:" + "e" * 64,
        digest_kind=CapabilityProviderDigestKind.SOURCE_TREE,
        platform="any",
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT",
        configuration=(
            {"entrypoint": "jacobian_checkers.reject:check"}
            if configuration is None
            else configuration
        ),
    )


def test_checker_registration_accepts_bound_python_distribution_runtime() -> None:
    registration = _registration(_python_distribution_runtime())

    assert registration.provider_runtime is not None
    assert (
        registration.provider_runtime.digest_kind
        is CapabilityProviderDigestKind.PYTHON_DISTRIBUTION_RECORD
    )


def test_checker_registration_accepts_remeasurable_source_tree_runtime() -> None:
    registration = _registration(_source_tree_runtime())

    assert registration.provider_runtime is not None
    assert (
        registration.provider_runtime.digest_kind
        is CapabilityProviderDigestKind.SOURCE_TREE
    )


def test_checker_registration_rejects_source_tree_runtime_without_entrypoint() -> None:
    with pytest.raises(
        ValidationError, match="source runtime must name its entrypoint"
    ):
        _registration(_source_tree_runtime(configuration={}))


def test_checker_registration_rejects_source_tree_runtime_for_other_entrypoint() -> (
    None
):
    with pytest.raises(ValidationError, match="must bind the checker entrypoint"):
        _registration(
            _source_tree_runtime(
                configuration={"entrypoint": "jacobian_checkers.other:check"}
            )
        )


@pytest.mark.parametrize(
    "configuration",
    [
        {"import_name": "tests_distribution"},
        {"distribution": "tests-distribution"},
    ],
)
def test_checker_registration_rejects_incomplete_python_distribution_identity(
    configuration: dict[str, str],
) -> None:
    with pytest.raises(
        ValidationError,
        match="must name its distribution and import",
    ):
        _registration(_python_distribution_runtime(configuration=configuration))
