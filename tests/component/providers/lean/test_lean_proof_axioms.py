from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from jacobian.artifacts import ArtifactService
from jacobian.contracts.capabilities import (
    CapabilityInstallTier,
    CapabilityMode,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
    CapabilityRequest,
)
from jacobian.contracts.lean import LeanEnvironment
from jacobian.lean_frontend import proof_axioms
from jacobian.lean_frontend.proof_axioms import install_lean_proof_axioms_capability
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository


def test_proof_hole_inspection_ignores_strings_and_identifiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = CapabilityProviderRuntime(
        provider="jacobian.lean4",
        availability=CapabilityProviderAvailability.AVAILABLE,
        version="4.31.0",
        digest="sha256:" + "a" * 64,
        digest_kind=CapabilityProviderDigestKind.EXECUTABLE,
        platform="test",
        install_tier=CapabilityInstallTier.T3,
        license_id="Apache-2.0",
        features=("CORE",),
    )
    proof = (
        'by let label : String := "'
        + ("sorry " * 65)
        + "\"; let admit' : Nat := 0; exact True.intro"
    )
    monkeypatch.setattr(
        proof_axioms,
        "_manifest_digest",
        lambda _environment: None,
    )
    monkeypatch.setattr(
        "jacobian_checkers.lean4._run_lean",
        lambda _source, *, environment_name: SimpleNamespace(
            stdout="'jacobian_theorem' does not depend on any axioms\n",
            stderr="",
            returncode=0,
        ),
    )

    with ArtifactRepository(tmp_path) as store:
        schemas = SchemaRegistry(store)
        artifacts = ArtifactService(store, schemas)
        adapter, _installation = install_lean_proof_axioms_capability(
            store,
            schemas,
            artifacts,
            {LeanEnvironment.CORE: SimpleNamespace(lean_commit="lean-test")},
            runtime,
        )
        result = adapter.invoke(
            CapabilityRequest(
                capability_id="lean.proof.axioms.inspect",
                mode=CapabilityMode.EXPLORE,
                input={
                    "environment": "CORE",
                    "statement": "True",
                    "proof": proof,
                },
            )
        )

    assert result.output["sorry_count"] == 0
    assert result.output["admit_count"] == 0


def test_proof_holes_are_counted_but_not_rejected_as_commands() -> None:
    proof_axioms._validate_source("True", "by sorry")
    assert proof_axioms._proof_hole_counts("by sorry admit") == (1, 1)


def test_proof_hole_count_is_bounded_before_artifact_validation() -> None:
    with pytest.raises(ValueError, match="more than 64"):
        proof_axioms._validate_source("True", "by " + " ".join(["sorry"] * 65))


@pytest.mark.parametrize("literal", (r"'\xAF'", r"'\uFACE'", r"'\''"))
def test_char_escapes_are_not_scanned_as_tokens(literal: str) -> None:
    assert proof_axioms._lean_source_tokens(literal) == ()


@pytest.mark.parametrize("literal", (r"'\xAF'", r"'\uFACE'", r"'\''"))
def test_tokens_after_char_escapes_are_scanned(literal: str) -> None:
    source = f"{literal} sorry admit"

    assert proof_axioms._lean_source_tokens(source) == ("sorry", "admit")
    assert proof_axioms._proof_hole_counts(source) == (1, 1)


def test_string_literal_escaped_quote_is_not_scanned_as_tokens() -> None:
    assert proof_axioms._lean_source_tokens(r'"\"" sorry') == ("sorry",)
