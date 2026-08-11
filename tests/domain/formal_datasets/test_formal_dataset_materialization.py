from __future__ import annotations

from pathlib import Path

import pytest

from jacobian.artifacts import ArtifactService
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.formal_datasets import build_formal_dataset_bundle
from jacobian.operation_installation import OperationInstaller
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository
from jacobian_checkers.lean4 import LEAN_VERSION, MATHLIB_COMMIT


def _adapter(tmp_path: Path):
    store = ArtifactRepository(tmp_path)
    schemas = SchemaRegistry(store)
    artifacts = ArtifactService(store, schemas)
    installation = OperationInstaller(store, schemas, artifacts).install(
        build_formal_dataset_bundle()
    )
    return installation.adapters[0]


def _output(adapter, result) -> dict[str, object]:
    stored = adapter.resources.artifacts.store.get(result.output["result_uri"])
    assert isinstance(stored.payload, dict)
    return stored.payload


def _environment() -> dict[str, object]:
    return {
        "lean_version": LEAN_VERSION,
        "project_source_url": "https://github.com/leanprover-community/mathlib4",
        "project_revision": "project-commit-123",
        "mathlib_revision": MATHLIB_COMMIT,
        "imports": ["Mathlib"],
        "namespace": "MiniF2F",
        "theorem_context": ["open Real"],
        "project_files": [
            {"path": "lean-toolchain", "digest": "sha256:" + "a" * 64},
            {"path": "lake-manifest.json", "digest": "sha256:" + "b" * 64},
        ],
    }


def _minif2f_request() -> dict[str, object]:
    return {
        "dataset_revision": "3a5dceb842b916345a4d7bb7dc4c1dbd4b98aa",
        "sample_id": "mathd_algebra_1",
        "source_url": (
            "https://huggingface.co/datasets/Tonic/MiniF2F/"
            "resolve/3a5dceb842b916345a4d7bb7dc4c1dbd4b98aa"
        ),
        "row": {
            "dataset_id": "MINIF2F",
            "name": "mathd_algebra_1",
            "split": "test",
            "header": "import Mathlib  \r\n",
            "formal_statement": (
                "theorem mathd_algebra_1 : (1 : Nat) = 1 := by  \r\n  rfl  "
            ),
            "goal": "(1 : Nat) = 1",
            "informal_statement": "One equals one.  ",
            "informal_proof": "This is reflexive.  ",
        },
        "environment": _environment(),
    }


def _proofnet_request() -> dict[str, object]:
    return {
        "dataset_revision": "proofnet-fixture-revision",
        "sample_id": "analysis_1",
        "source_url": "https://github.com/zhangir-azerbayev/ProofNet",
        "row": {
            "dataset_id": "PROOFNET",
            "name": "analysis_1",
            "split": "test",
            "header": "import Mathlib\n",
            "formal_statement": "theorem analysis_1 : True := by trivial",
            "informal_statement": "A fixture undergraduate theorem.",
            "informal_proof": "The fixture is immediate.",
        },
        "environment": _environment(),
    }


@pytest.mark.parametrize("payload_factory", [_minif2f_request, _proofnet_request])
def test_supported_row_materializes_deterministically(
    tmp_path: Path,
    payload_factory,
) -> None:
    adapter = _adapter(tmp_path)
    payload = payload_factory()

    first = adapter.invoke(
        CapabilityRequest(
            capability_id="dataset.formal.materialize",
            input=payload,
        )
    )
    second = adapter.invoke(
        CapabilityRequest(
            capability_id="dataset.formal.materialize",
            input=payload,
        )
    )

    first_output = _output(adapter, first)
    assert first.output == second.output
    assert first.output["result_uri"] == second.output["result_uri"]
    assert str(first_output["row_digest"]).startswith("sha256:")
    assert str(first_output["normalized_source_digest"]).startswith("sha256:")
    assert str(first_output["environment_digest"]).startswith("sha256:")
    assert first_output["execution_status"] == "NOT_EXECUTED"
    assert first_output["assurance"] == "UNVERIFIED"
    assert first.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert str(first_output["normalized_source"]).endswith("\n")


def test_materialization_preserves_environment_and_preprocessing(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)

    result = adapter.invoke(
        CapabilityRequest(
            capability_id="dataset.formal.materialize",
            input=_minif2f_request(),
        )
    )

    output = _output(adapter, result)
    assert output["normalized_source"] == (
        "import Mathlib  \ntheorem mathd_algebra_1 : (1 : Nat) = 1 := by  \n  rfl  \n"
    )
    assert output["environment"] == _environment()
    assert [item["operation"] for item in output["preprocessing"]] == [
        "NORMALIZE_NEWLINES",
        "TRIM_TRAILING_WHITESPACE",
        "ENSURE_FINAL_NEWLINE",
    ]
    assert [item["applied"] for item in output["preprocessing"]] == [
        True,
        False,
        True,
    ]
    assert [item["code"] for item in output["diagnostics"]] == [
        "EXECUTION_NOT_REQUESTED"
    ]
    assert output["diagnostic_baseline"] == {
        "lean_version": LEAN_VERSION,
        "mathlib_revision": MATHLIB_COMMIT,
    }


def test_preprocessing_reports_only_transformations_that_occurred(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    payload = _proofnet_request()
    assert isinstance(payload["row"], dict)
    payload["row"]["formal_statement"] += "\n"
    payload["row"]["informal_statement"] += "\n"
    payload["row"]["informal_proof"] += "\n"

    result = adapter.invoke(
        CapabilityRequest(
            capability_id="dataset.formal.materialize",
            input=payload,
        )
    )

    output = _output(adapter, result)
    assert [item["applied"] for item in output["preprocessing"]] == [
        False,
        False,
        False,
    ]


@pytest.mark.parametrize("field", ("informal_statement", "informal_proof"))
def test_empty_optional_text_is_preserved_without_a_reported_rewrite(
    tmp_path: Path,
    field: str,
) -> None:
    adapter = _adapter(tmp_path)
    payload = _minif2f_request()
    assert isinstance(payload["row"], dict)
    payload["row"]["informal_statement"] += "\n"
    payload["row"]["informal_proof"] += "\n"
    payload["row"][field] = ""
    payload["row"]["header"] = "import Mathlib\n"
    payload["row"]["formal_statement"] += "\n"

    result = adapter.invoke(
        CapabilityRequest(
            capability_id="dataset.formal.materialize",
            input=payload,
        )
    )

    output = _output(adapter, result)
    assert output[field] == ""
    assert [item["applied"] for item in output["preprocessing"]] == [
        True,
        False,
        False,
    ]


def test_incompatible_environment_has_explicit_diagnostics(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    payload = _proofnet_request()
    assert isinstance(payload["environment"], dict)
    payload["environment"] = {
        **payload["environment"],
        "lean_version": "3.51.1",
        "mathlib_revision": "untracked-mathlib-revision",
        "project_files": [],
    }

    result = adapter.invoke(
        CapabilityRequest(
            capability_id="dataset.formal.materialize",
            input=payload,
        )
    )

    output = _output(adapter, result)
    assert {item["code"] for item in output["diagnostics"]} == {
        "EXECUTION_NOT_REQUESTED",
        "LEAN_VERSION_NOT_PINNED_RUNTIME",
        "MATHLIB_REVISION_NOT_PINNED_RUNTIME",
        "PROJECT_FILES_UNDECLARED",
    }


def test_expected_row_digest_rejects_changed_content(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    payload = _minif2f_request()
    payload["expected_row_digest"] = "sha256:" + "0" * 64

    result = adapter.invoke(
        CapabilityRequest(
            capability_id="dataset.formal.materialize",
            input=payload,
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "FORMAL_DATASET_ROW_DIGEST_MISMATCH"
    assert result.artifact_uris == ()


def test_artifact_tampering_is_detected_by_store(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    result = adapter.invoke(
        CapabilityRequest(
            capability_id="dataset.formal.materialize",
            input=_proofnet_request(),
        )
    )
    output = _output(adapter, result)
    stored = adapter.resources.artifacts.store.get(result.output["result_uri"])

    assert stored.payload["row_digest"] == output["row_digest"]
    assert stored.payload["environment_digest"] == output["environment_digest"]


def test_materialization_preserves_split_and_leading_lines(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    payload = _proofnet_request()
    assert isinstance(payload["row"], dict)
    payload["row"]["header"] = "\nimport Mathlib\n"

    result = adapter.invoke(
        CapabilityRequest(
            capability_id="dataset.formal.materialize",
            input=payload,
        )
    )

    output = _output(adapter, result)
    assert output["split"] == "test"
    assert output["canonical_row"]["split"] == "test"
    assert str(output["normalized_source"]).startswith("\nimport Mathlib\n")
    stored = adapter.resources.artifacts.store.get(result.output["result_uri"])
    assert stored.payload["normalized_source"] == output["normalized_source"]


def test_materialization_preserves_trailing_spaces_inside_source(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    payload = _proofnet_request()
    assert isinstance(payload["row"], dict)
    payload["row"]["formal_statement"] = (
        'theorem analysis_1 : "line with spaces  \\n" = '
        '"line with spaces  \\n" := by rfl'
    )

    result = adapter.invoke(
        CapabilityRequest(
            capability_id="dataset.formal.materialize",
            input=payload,
        )
    )

    assert "spaces  \\n" in str(_output(adapter, result)["normalized_source"])


def test_model_backed_artifact_rejects_digest_tampering(tmp_path: Path) -> None:
    store = ArtifactRepository(tmp_path)
    schemas = SchemaRegistry(store)
    artifacts = ArtifactService(store, schemas)
    installation = OperationInstaller(store, schemas, artifacts).install(
        build_formal_dataset_bundle()
    )
    adapter = installation.adapters[0]
    result = adapter.invoke(
        CapabilityRequest(
            capability_id="dataset.formal.materialize",
            input=_proofnet_request(),
        )
    )
    replay = adapter.invoke(
        CapabilityRequest(
            capability_id="dataset.formal.materialize",
            input=_proofnet_request(),
        )
    )
    assert result.output == replay.output
    result_uri = result.output["result_uri"]
    stored = store.get(result_uri)
    assert stored.payload["split"] == "test"
    assert stored.payload["canonical_row"]["split"] == "test"
    tampered = dict(stored.payload)
    tampered["row_digest"] = "sha256:" + "0" * 64

    with pytest.raises(ValueError):
        artifacts.put(
            schema_uri=installation.result_schema_uris["dataset.formal.materialize"],
            semantics_uri=installation.semantics_uri,
            payload=tampered,
        )

    for field, replacement in (
        ("preprocessing", []),
        ("diagnostics", []),
        (
            "diagnostic_baseline",
            {"lean_version": "0.0.0", "mathlib_revision": "different"},
        ),
        ("dataset_revision", "       "),
        ("source_url", " "),
        ("source_url", "https://example.invalid/" + "x" * 2_000),
    ):
        malformed = dict(stored.payload)
        malformed[field] = replacement
        with pytest.raises(ValueError):
            artifacts.put(
                schema_uri=installation.result_schema_uris[
                    "dataset.formal.materialize"
                ],
                semantics_uri=installation.semantics_uri,
                payload=malformed,
            )
