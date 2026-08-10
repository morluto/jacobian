from __future__ import annotations

# Composition-lane admission category for architecture ratchets.
COMPOSITION_ADMISSION = "REFERENCE"


def test_isomorphic_graphs_share_one_canonical_object(
    attached_complete_runtime,
) -> None:
    reference = attached_complete_runtime.portfolio.references["graph_paths"]
    first = attached_complete_runtime.core.artifacts.put(
        schema_uri=reference.candidate_schema_uri,
        semantics_uri=reference.semantics_uri,
        payload={
            "vertices": ["a", "b", "c"],
            "arcs": [["a", "b"], ["b", "c"]],
        },
    )
    relabeled = attached_complete_runtime.core.artifacts.put(
        schema_uri=reference.candidate_schema_uri,
        semantics_uri=reference.semantics_uri,
        payload={
            "vertices": ["x", "z", "y"],
            "arcs": [["x", "z"], ["z", "y"]],
        },
    )

    first_result = attached_complete_runtime.services.structures.canonicalize(
        structure_uri=first.artifact_uri,
        plugin_id=reference.plugin_id,
        wall_seconds=30,
    )
    second_result = attached_complete_runtime.services.structures.canonicalize(
        structure_uri=relabeled.artifact_uri,
        plugin_id=reference.plugin_id,
        wall_seconds=30,
    )

    assert first_result.canonical_uri == second_result.canonical_uri
    assert first_result.canonical_key == second_result.canonical_key
    assert first_result.result.assurance.verification.value == "UNVERIFIED"


def test_nonisomorphic_graphs_have_distinct_canonical_keys(
    attached_complete_runtime,
) -> None:
    reference = attached_complete_runtime.portfolio.references["graph_paths"]
    path = attached_complete_runtime.core.artifacts.put(
        schema_uri=reference.candidate_schema_uri,
        semantics_uri=reference.semantics_uri,
        payload={
            "vertices": ["a", "b", "c"],
            "arcs": [["a", "b"], ["b", "c"]],
        },
    )
    cycle = attached_complete_runtime.core.artifacts.put(
        schema_uri=reference.candidate_schema_uri,
        semantics_uri=reference.semantics_uri,
        payload={
            "vertices": ["a", "b", "c"],
            "arcs": [["a", "b"], ["b", "c"], ["c", "a"]],
        },
    )

    path_result = attached_complete_runtime.services.structures.canonicalize(
        structure_uri=path.artifact_uri,
        plugin_id=reference.plugin_id,
        wall_seconds=30,
    )
    cycle_result = attached_complete_runtime.services.structures.canonicalize(
        structure_uri=cycle.artifact_uri,
        plugin_id=reference.plugin_id,
        wall_seconds=30,
    )

    assert path_result.canonical_key != cycle_result.canonical_key
