from __future__ import annotations

from jacobian.capability_discovery import discovery_relevance
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityInstallTier,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
)


def test_discovery_phrase_matching_respects_token_boundaries() -> None:
    runtime = CapabilityProviderRuntime(
        provider="tests",
        availability=CapabilityProviderAvailability.AVAILABLE,
        version="1",
        digest="sha256:" + "a" * 64,
        digest_kind=CapabilityProviderDigestKind.SOURCE_TREE,
        platform="any",
        install_tier=CapabilityInstallTier.T0,
        license_id="MIT",
    )
    descriptor = CapabilityDescriptor(
        capability_id="fixture.text.inspect",
        version="1",
        title="Inspect text",
        description="Inspect some paragraph of structured text.",
        provider="tests",
        provider_runtime=runtime,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )

    graph_score, graph_matches, *_ = discovery_relevance(descriptor, "graph")
    phrase_score, phrase_matches, *_ = discovery_relevance(
        descriptor,
        "paragraph of structured text",
    )

    assert graph_score == 0
    assert "phrase" not in graph_matches
    assert phrase_score >= 20
    assert "phrase" in phrase_matches
