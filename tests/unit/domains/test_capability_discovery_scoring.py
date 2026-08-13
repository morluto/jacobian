from __future__ import annotations

from jacobian.contracts.operations import (
    OperationDescriptor,
    ProviderAvailability,
    ProviderDigestKind,
    ProviderInstallTier,
    ProviderObservation,
)
from jacobian.operation_discovery import discovery_relevance


def test_discovery_phrase_matching_respects_token_boundaries() -> None:
    runtime = ProviderObservation(
        provider="tests",
        availability=ProviderAvailability.AVAILABLE,
        version="1",
        digest="sha256:" + "a" * 64,
        digest_kind=ProviderDigestKind.SOURCE_TREE,
        platform="any",
        install_tier=ProviderInstallTier.T0,
        license_id="MIT",
    )
    descriptor = OperationDescriptor(
        operation_id="fixture.text.inspect",
        version="1",
        title="Inspect text",
        description="Inspect some paragraph of structured text.",
        provider="tests",
        provider_runtime=runtime,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )

    graph_score = discovery_relevance(descriptor, "graph")
    phrase_score = discovery_relevance(
        descriptor,
        "paragraph of structured text",
    )

    assert graph_score == 0
    assert phrase_score >= 20
