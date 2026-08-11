"""Integration tests for deterministic structured-claim decomposition."""

from __future__ import annotations

import pytest

from jacobian.claim_decomposition_capabilities import reconstruct
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityRequest,
)
from jacobian.contracts.claim_decomposition import (
    ClaimDecompositionArtifact,
    LogicalClaimNode,
    LogicalConnective,
    StructuredClaimArtifact,
)
from jacobian.runtime.model import JacobianRuntime


def _atom(node_id: str, symbol: str | None = None) -> LogicalClaimNode:
    return LogicalClaimNode(
        node_id=node_id,
        connective=LogicalConnective.ATOM,
        atom={"symbol": symbol or node_id},
    )


def _store_claim(
    attached_complete_runtime: JacobianRuntime, root: LogicalClaimNode
) -> str:
    stored = attached_complete_runtime.core.artifacts.put(
        schema_uri=attached_complete_runtime.services.claim_decomposition.structured_claim_schema_uri,
        semantics_uri=attached_complete_runtime.services.claim_decomposition.semantics_uri,
        payload=StructuredClaimArtifact(root=root).model_dump(mode="json"),
    )
    return stored.artifact_uri


def _invoke(
    attached_complete_runtime: JacobianRuntime, capability_id: str, source_uri: str
):
    return attached_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=capability_id,
            input={"source_uri": source_uri},
        )
    )


def test_conjunction_split_preserves_order_grouping_and_reconstructs(
    attached_complete_runtime,
) -> None:
    nested = LogicalClaimNode(
        node_id="nested",
        connective=LogicalConnective.CONJUNCTION,
        children=(_atom("b"), _atom("c")),
    )
    root = LogicalClaimNode(
        node_id="root",
        connective=LogicalConnective.CONJUNCTION,
        children=(_atom("a"), nested),
        source_span=(0, 17),
    )
    source_uri = _store_claim(attached_complete_runtime, root)

    result = _invoke(attached_complete_runtime, "claim.conjunction.split", source_uri)

    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert [item["node"]["node_id"] for item in result.output["occurrences"]] == [
        "a",
        "nested",
    ]
    stored = attached_complete_runtime.core.store.get(
        result.output["decomposition_uri"]
    )
    record = ClaimDecompositionArtifact.model_validate(stored.payload)
    assert reconstruct(record) == root
    assert stored.manifest.parents == (source_uri,)
    assert (
        record.source_binding.object_digest
        == attached_complete_runtime.core.store.get(source_uri).manifest.object_digest
    )


def test_conjunction_split_preserves_duplicate_subtrees_as_occurrences(
    attached_complete_runtime,
) -> None:
    root = LogicalClaimNode(
        node_id="root",
        connective=LogicalConnective.CONJUNCTION,
        children=(_atom("left", "A"), _atom("right", "A")),
    )
    result = _invoke(
        attached_complete_runtime,
        "claim.conjunction.split",
        _store_claim(attached_complete_runtime, root),
    )
    occurrences = result.output["occurrences"]
    assert [item["position"] for item in occurrences] == [0, 1]
    assert [item["node"]["atom"] for item in occurrences] == [
        {"symbol": "A"},
        {"symbol": "A"},
    ]


def test_implication_obligations_are_directional_and_reconstruct(
    attached_complete_runtime,
) -> None:
    consequent = LogicalClaimNode(
        node_id="bc",
        connective=LogicalConnective.IMPLICATION,
        children=(_atom("b"), _atom("c")),
    )
    root = LogicalClaimNode(
        node_id="root",
        connective=LogicalConnective.IMPLICATION,
        children=(_atom("a"), consequent),
    )
    result = _invoke(
        attached_complete_runtime,
        "claim.implication.obligations",
        _store_claim(attached_complete_runtime, root),
    )
    assert [item["role"] for item in result.output["occurrences"]] == [
        "ASSUME_ANTECEDENT",
        "PROVE_CONSEQUENT_UNDER_ANTECEDENT",
    ]
    record = ClaimDecompositionArtifact.model_validate(
        attached_complete_runtime.core.store.get(
            result.output["decomposition_uri"]
        ).payload
    )
    assert reconstruct(record) == root


def test_reconstruction_rejects_tampered_ordered_child(
    attached_complete_runtime,
) -> None:
    root = LogicalClaimNode(
        node_id="root",
        connective=LogicalConnective.CONJUNCTION,
        children=(_atom("a"), _atom("b")),
    )
    result = _invoke(
        attached_complete_runtime,
        "claim.conjunction.split",
        _store_claim(attached_complete_runtime, root),
    )
    record = ClaimDecompositionArtifact.model_validate(
        attached_complete_runtime.core.store.get(
            result.output["decomposition_uri"]
        ).payload
    )
    tampered = record.model_copy(
        update={"occurrences": tuple(reversed(record.occurrences))}
    )
    with pytest.raises(ValueError, match="digest binding"):
        reconstruct(tampered)


@pytest.mark.parametrize(
    ("capability_id", "connective"),
    [
        ("claim.conjunction.split", LogicalConnective.ATOM),
        ("claim.conjunction.split", LogicalConnective.IMPLICATION),
        ("claim.implication.obligations", LogicalConnective.ATOM),
        ("claim.implication.obligations", LogicalConnective.CONJUNCTION),
    ],
)
def test_unsupported_top_level_connective_is_explicitly_rejected(
    attached_complete_runtime,
    capability_id: str,
    connective: LogicalConnective,
) -> None:
    children = (
        () if connective is LogicalConnective.ATOM else (_atom("left"), _atom("right"))
    )
    root = LogicalClaimNode(
        node_id="root",
        connective=connective,
        atom={"symbol": "A"} if connective is LogicalConnective.ATOM else None,
        children=children,
    )
    source_uri = _store_claim(attached_complete_runtime, root)
    result = _invoke(attached_complete_runtime, capability_id, source_uri)
    assert result.execution.status.value == "ERROR"
    assert result.diagnostics[0].code == "UNSUPPORTED_TOP_LEVEL_CONNECTIVE"
