from __future__ import annotations

from tests.support.services import open_reference_services


def test_graph_canonicalizer_identifies_isomorphism_classes(tmp_path) -> None:
    with open_reference_services(tmp_path / "state", "graph_paths") as services:
        reference = services.references["graph_paths"]
        path = services.core.artifacts.put(
            schema_uri=reference.candidate_schema_uri,
            semantics_uri=reference.semantics_uri,
            payload={
                "vertices": ["a", "b", "c"],
                "arcs": [["a", "b"], ["b", "c"]],
            },
        )
        relabeled_path = services.core.artifacts.put(
            schema_uri=reference.candidate_schema_uri,
            semantics_uri=reference.semantics_uri,
            payload={
                "vertices": ["x", "z", "y"],
                "arcs": [["x", "z"], ["z", "y"]],
            },
        )
        cycle = services.core.artifacts.put(
            schema_uri=reference.candidate_schema_uri,
            semantics_uri=reference.semantics_uri,
            payload={
                "vertices": ["a", "b", "c"],
                "arcs": [["a", "b"], ["b", "c"], ["c", "a"]],
            },
        )

        path_result = services.application.structures.canonicalize(
            structure_uri=path.artifact_uri,
            plugin_id=reference.plugin_id,
            wall_seconds=30,
        )
        relabeled_result = services.application.structures.canonicalize(
            structure_uri=relabeled_path.artifact_uri,
            plugin_id=reference.plugin_id,
            wall_seconds=30,
        )
        cycle_result = services.application.structures.canonicalize(
            structure_uri=cycle.artifact_uri,
            plugin_id=reference.plugin_id,
            wall_seconds=30,
        )

        assert path_result.canonical_uri == relabeled_result.canonical_uri
        assert path_result.canonical_key == relabeled_result.canonical_key
        assert path_result.result.assurance.verification.value == "UNVERIFIED"
        assert path_result.canonical_key != cycle_result.canonical_key
