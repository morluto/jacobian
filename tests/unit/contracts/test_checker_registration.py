from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.checkers import (
    CheckerManifest,
    CheckerPythonDistribution,
    CheckerPythonRuntime,
    CheckerRegistration,
    CheckerSandboxPolicy,
    CheckerSourceModule,
    EvidenceKind,
)
from jacobian.contracts.operations import (
    ProviderAvailability,
    ProviderDigestKind,
    ProviderInstallTier,
    ProviderObservation,
)

_ARTIFACT_URI = "artifact://sha256/" + "a" * 64


def _worker_distributions() -> tuple[CheckerPythonDistribution, ...]:
    def distribution(name: str, version: str, digest: str) -> CheckerPythonDistribution:
        return CheckerPythonDistribution(
            distribution=name,
            version=version,
            file_count=1,
            files_digest=digest,
        )

    return (
        distribution("pydantic", "2.0.0", "sha256:" + "a" * 64),
        distribution("pydantic_core", "2.0.0", "sha256:" + "b" * 64),
        distribution("rfc8785", "0.1.0", "sha256:" + "c" * 64),
    )


def _python_distribution_runtime(
    *,
    configuration: dict[str, str] | None = None,
) -> ProviderObservation:
    return ProviderObservation(
        provider="tests.distribution",
        availability=ProviderAvailability.AVAILABLE,
        version="1.2.3",
        digest="sha256:" + "b" * 64,
        digest_kind=ProviderDigestKind.PYTHON_DISTRIBUTION_RECORD,
        platform="linux-x86_64",
        install_tier=ProviderInstallTier.T1,
        license_id="MIT",
        configuration=configuration
        or {
            "distribution": "tests-distribution",
            "import_name": "tests_distribution",
        },
    )


def _registration(runtime: ProviderObservation) -> CheckerRegistration:
    manifest = CheckerManifest(
        entrypoint="jacobian_checkers.reject:check",
        checker_source_modules=(
            CheckerSourceModule(
                module="jacobian_checkers.reject",
                source_digest="sha256:" + "d" * 64,
            ),
        ),
        worker_source_modules=(
            CheckerSourceModule(
                module="jacobian.checker_worker",
                source_digest="sha256:" + "d" * 64,
            ),
        ),
        python_distributions=_worker_distributions(),
        python_runtime=CheckerPythonRuntime(
            implementation="cpython",
            version="3.12.0",
            executable_digest="sha256:" + "f" * 64,
        ),
        provider_runtime=runtime,
        passive_contract_uris=(_ARTIFACT_URI,),
        sandbox=CheckerSandboxPolicy(
            max_wall_seconds=30,
            max_cpu_seconds=31,
            max_address_space_bytes=1024,
            max_stdout_bytes=1024,
            max_stderr_bytes=1024,
        ),
    )
    return CheckerRegistration(
        checker_id="checker://sha256/" + "c" * 64,
        name="distribution-backed checker",
        implementation=manifest,
        implementation_digest=manifest.implementation_digest(),
        evidence_kind=EvidenceKind.WITNESS,
        format_id="tests.distribution",
        format_version="1",
        claim_schema_uris=(_ARTIFACT_URI,),
        semantics_uris=(_ARTIFACT_URI,),
        candidate_schema_uris=(_ARTIFACT_URI,),
    )


def test_checker_registration_rejects_a_manifest_digest_mismatch() -> None:
    registration = _registration(_python_distribution_runtime())

    with pytest.raises(ValidationError, match="digest must match its manifest"):
        CheckerRegistration.model_validate(
            registration.model_dump(mode="json")
            | {"implementation_digest": "sha256:" + "0" * 64}
        )


def _source_tree_runtime(
    *,
    configuration: dict[str, str] | None = None,
) -> ProviderObservation:
    return ProviderObservation(
        provider="tests.source-tree",
        availability=ProviderAvailability.AVAILABLE,
        version="1",
        digest="sha256:" + "e" * 64,
        digest_kind=ProviderDigestKind.SOURCE_TREE,
        platform="any",
        install_tier=ProviderInstallTier.T1,
        license_id="MIT",
        configuration=(
            {"entrypoint": "jacobian_checkers.reject:check"}
            if configuration is None
            else configuration
        ),
    )


def test_checker_registration_accepts_bound_python_distribution_runtime() -> None:
    registration = _registration(_python_distribution_runtime())

    assert registration.implementation.provider_runtime is not None
    assert (
        registration.implementation.provider_runtime.digest_kind
        is ProviderDigestKind.PYTHON_DISTRIBUTION_RECORD
    )


def test_checker_registration_accepts_remeasurable_source_tree_runtime() -> None:
    registration = _registration(_source_tree_runtime())

    assert registration.implementation.provider_runtime is not None
    assert (
        registration.implementation.provider_runtime.digest_kind
        is ProviderDigestKind.SOURCE_TREE
    )


def test_checker_manifest_requires_exact_worker_python_distributions() -> None:
    manifest = _registration(_python_distribution_runtime()).implementation
    payload = manifest.model_dump(mode="json")
    payload["python_distributions"] = payload["python_distributions"][:-1]

    with pytest.raises(
        ValidationError,
        match="must bind worker Python distributions: rfc8785",
    ):
        CheckerManifest.model_validate(payload)


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
