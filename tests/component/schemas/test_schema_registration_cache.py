from __future__ import annotations

from pathlib import Path

import pytest

import jacobian.schema_compiler as schema_compiler
from jacobian.schema_registry import SchemaRegistry, SchemaRegistryError
from jacobian.storage.repository import ArtifactRepository


def test_existing_descriptor_reuses_cached_meta_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactRepository(tmp_path)
    first = SchemaRegistry(store)
    schema = {
        "title": "component schema cache probe",
        "type": "object",
        "properties": {"value": {"type": "integer"}},
    }

    calls = 0
    original = schema_compiler.check_draft202012_schema

    def count_validation(canonical_schema: bytes) -> None:
        nonlocal calls
        calls += 1
        original(canonical_schema)

    monkeypatch.setattr(
        schema_compiler,
        "check_draft202012_schema",
        count_validation,
    )
    uri = first.register(name="component.cache", version="1", schema=schema)

    second = SchemaRegistry(store)
    assert second.register(name="component.cache", version="1", schema=schema) == uri
    assert calls == 1
    store.close()


def test_existing_descriptor_written_without_schema_validation_is_rejected(
    tmp_path: Path,
) -> None:
    store = ArtifactRepository(tmp_path)
    schema = {"type": 17}
    uri = store.register_descriptor(
        kind="schema",
        name="component.unvalidated",
        version="1",
        definition=schema,
    )

    with pytest.raises(SchemaRegistryError, match="invalid Draft"):
        SchemaRegistry(store).register(
            name="component.unvalidated",
            version="1",
            schema=schema,
        )
    assert uri == store.descriptor_uri(
        kind="schema",
        name="component.unvalidated",
        version="1",
        definition=schema,
    )
    store.close()


def test_new_invalid_duplicate_definition_still_validates(
    tmp_path: Path,
) -> None:
    store = ArtifactRepository(tmp_path)
    registry = SchemaRegistry(store)
    registry.register(
        name="component.invalid-duplicate",
        version="1",
        schema={"type": "object"},
    )

    with pytest.raises(SchemaRegistryError, match="invalid Draft"):
        registry.register(
            name="component.invalid-duplicate",
            version="1",
            schema={"type": 17},
        )
    store.close()
