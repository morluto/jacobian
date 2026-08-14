from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Self

import pytest
from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema, model_validator

from jacobian.schema_registry import (
    SchemaRegistry,
    SchemaRegistryError,
    SchemaValidationError,
    model_schema,
)
from jacobian.storage.errors import StorageError
from jacobian.storage.repository import ArtifactRepository


class _CachedSchemaModel(BaseModel):
    value: int


class _EquivalentCachedSchemaModel(BaseModel):
    model_config = ConfigDict(title="_CachedSchemaModel")

    value: int


class _OrderedPair(BaseModel):
    first: int
    second: int

    @model_validator(mode="after")
    def require_order(self) -> Self:
        if self.first >= self.second:
            raise ValueError("pair must be ordered")
        return self


class _MalformedCustomizedSchema(BaseModel):
    value: int

    @classmethod
    def model_json_schema(cls, **_kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return {"type": 17}


_MalformedCustomizedSchema.__module__ = "jacobian.customized_test_model"


class _MalformedFieldExtraSchema(BaseModel):
    value: int = Field(json_schema_extra={"type": 17})


class _MalformedAnnotatedSchema(BaseModel):
    value: Annotated[int, WithJsonSchema({"type": 17})]


for _customized_model in (_MalformedFieldExtraSchema, _MalformedAnnotatedSchema):
    _customized_model.__module__ = "jacobian.customized_test_model"


def test_cached_model_schema_returns_independent_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = model_schema(_CachedSchemaModel)
    first["title"] = "mutated"

    def unexpected_regeneration() -> dict[str, object]:
        pytest.fail("cached schema was regenerated")

    monkeypatch.setattr(
        _CachedSchemaModel,
        "model_json_schema",
        unexpected_regeneration,
    )
    second = model_schema(_CachedSchemaModel)

    assert second["title"] == "_CachedSchemaModel"


def test_external_dynamic_reference_is_rejected(tmp_path: Path) -> None:
    registry = SchemaRegistry(ArtifactRepository(tmp_path))

    with pytest.raises(SchemaRegistryError):
        registry.register(
            name="external-dynamic-ref",
            version="1",
            schema={"$dynamicRef": "https://example.test/schema"},
        )


@pytest.mark.parametrize(
    "model",
    [_MalformedCustomizedSchema, _MalformedFieldExtraSchema, _MalformedAnnotatedSchema],
)
def test_customized_model_schema_is_validated_before_persistence(
    tmp_path: Path,
    model: type[BaseModel],
) -> None:
    store = ArtifactRepository(tmp_path)
    registry = SchemaRegistry(store)

    with pytest.raises(SchemaRegistryError, match="invalid Draft"):
        registry.register_model(
            name="customized-invalid-schema",
            version="1",
            model=model,
        )

    # Compute the rejected descriptor identity without passing the malformed
    # schema through the public, validating model-schema boundary again.
    schema = model.model_json_schema()
    uri = store.descriptor_uri(
        kind="schema",
        name="customized-invalid-schema",
        version="1",
        definition=schema,
    )
    with pytest.raises(StorageError):
        store.get_descriptor(uri, expected_kind="schema")


def test_schema_validator_cache_is_bound_to_canonical_schema(
    tmp_path: Path,
) -> None:
    registry = SchemaRegistry(ArtifactRepository(tmp_path))
    integer_schema = registry.register(
        name="cache-binding",
        version="1",
        schema={"type": "object", "properties": {"value": {"type": "integer"}}},
    )
    string_schema = registry.register(
        name="cache-binding",
        version="2",
        schema={"type": "object", "properties": {"value": {"type": "string"}}},
    )

    registry.validate(integer_schema, {"value": 1})
    with pytest.raises(SchemaValidationError):
        registry.validate(string_schema, {"value": 1})


def test_registered_schema_resolves_from_immutable_local_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactRepository(tmp_path)
    registry = SchemaRegistry(store)
    schema_uri = registry.register(
        name="local-cache",
        version="1",
        schema={"type": "object"},
    )

    first = registry.resolve(schema_uri)
    first["type"] = "array"

    def unexpected_store_read(*_args: object, **_kwargs: object) -> dict[str, object]:
        pytest.fail("registered schema was read from the store again")

    monkeypatch.setattr(store, "get_descriptor", unexpected_store_read)

    assert registry.resolve(schema_uri) == {"type": "object"}


@pytest.mark.parametrize("inside_transaction", [False, True])
def test_identical_registration_writes_one_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inside_transaction: bool,
) -> None:
    store = ArtifactRepository(tmp_path)
    registry = SchemaRegistry(store)
    calls = 0
    register_descriptor = store.register_descriptor

    def counting_register_descriptor(
        *,
        kind: str,
        name: str,
        version: str,
        definition: Any,
    ) -> str:
        nonlocal calls
        calls += 1
        return register_descriptor(
            kind=kind,
            name=name,
            version=version,
            definition=definition,
        )

    monkeypatch.setattr(store, "register_descriptor", counting_register_descriptor)

    def register_twice() -> None:
        for _ in range(2):
            registry.register(
                name="deduplicated",
                version="1",
                schema={"type": "object"},
            )

    if inside_transaction:
        with store.transaction():
            register_twice()
    else:
        register_twice()

    assert calls == 1


@pytest.mark.parametrize("model", [None, _CachedSchemaModel])
def test_rolled_back_schema_is_not_retained_in_local_cache(
    tmp_path: Path,
    model: type[BaseModel] | None,
) -> None:
    store = ArtifactRepository(tmp_path)
    registry = SchemaRegistry(store)
    schema_uri = ""

    with pytest.raises(RuntimeError, match="rollback"), store.transaction():
        schema_uri = (
            registry.register(
                name="rolled-back",
                version="1",
                schema={"type": "object"},
            )
            if model is None
            else registry.register_model(
                name="rolled-back",
                version="1",
                model=model,
            )
        )
        registry.resolve(schema_uri)
        raise RuntimeError("rollback")

    with pytest.raises(SchemaRegistryError, match="unregistered schema"):
        registry.resolve(schema_uri)

    schema_uri = (
        registry.register(
            name="rolled-back",
            version="1",
            schema={"type": "object"},
        )
        if model is None
        else registry.register_model(
            name="rolled-back",
            version="1",
            model=_EquivalentCachedSchemaModel,
        )
    )
    assert registry.resolve(schema_uri)["type"] == "object"


def test_model_backed_schema_applies_cross_field_contracts(
    tmp_path: Path,
) -> None:
    registry = SchemaRegistry(ArtifactRepository(tmp_path))
    schema_uri = registry.register_model(
        name="ordered-pair",
        version="1",
        model=_OrderedPair,
    )

    assert registry.validate(schema_uri, {"first": 1, "second": 2}) == {
        "first": 1,
        "second": 2,
    }
    with pytest.raises(SchemaValidationError, match="pair must be ordered"):
        registry.validate(schema_uri, {"first": 2, "second": 1})


@pytest.mark.parametrize("inside_transaction", [False, True])
def test_producer_only_model_registration_is_runtime_enforced(
    tmp_path: Path,
    inside_transaction: bool,
) -> None:
    store = ArtifactRepository(tmp_path)
    registry = SchemaRegistry(store)

    def register() -> str:
        return registry.register_model(
            name="producer-owned",
            version="1",
            model=_CachedSchemaModel,
            producer_only=True,
        )

    if inside_transaction:
        with store.transaction():
            schema_uri = register()
            assert registry.is_producer_only(schema_uri)
    else:
        schema_uri = register()

    assert registry.is_producer_only(schema_uri)


def test_existing_model_schema_reattaches_without_durable_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactRepository(tmp_path)
    installer = SchemaRegistry(store)
    schema_uri = installer.register_model(
        name="ordered-pair",
        version="1",
        model=_OrderedPair,
    )
    runtime_registry = SchemaRegistry(store)

    def unexpected_blob_write(_data: bytes) -> str:
        pytest.fail("reattaching a model rewrote durable schema content")

    monkeypatch.setattr(store._blobs, "write", unexpected_blob_write)

    assert (
        runtime_registry.register_model(
            name="ordered-pair",
            version="1",
            model=_OrderedPair,
        )
        == schema_uri
    )

    with pytest.raises(SchemaValidationError, match="pair must be ordered"):
        runtime_registry.validate(schema_uri, {"first": 2, "second": 1})


def test_failed_transactional_registration_does_not_leave_empty_pending_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactRepository(tmp_path)
    registry = SchemaRegistry(store)
    register_descriptor = store.register_descriptor

    def fail_first_registration(
        *,
        kind: str,
        name: str,
        version: str,
        definition: dict[str, Any],
    ) -> str:
        if name == "will-fail":
            raise RuntimeError("simulated descriptor failure")
        return register_descriptor(
            kind=kind,
            name=name,
            version=version,
            definition=definition,
        )

    monkeypatch.setattr(store, "register_descriptor", fail_first_registration)

    with (
        store.transaction(),
        pytest.raises(
            RuntimeError,
            match="simulated descriptor failure",
        ),
    ):
        registry.register(
            name="will-fail",
            version="1",
            schema={"type": "object"},
        )

    assert registry.register(
        name="still-usable",
        version="1",
        schema={"type": "object"},
    ).startswith("artifact://sha256/")


def test_transaction_cannot_replace_committed_model_contract(tmp_path: Path) -> None:
    store = ArtifactRepository(tmp_path)
    registry = SchemaRegistry(store)
    schema_uri = registry.register_model(
        name="shared-model-contract",
        version="1",
        model=_CachedSchemaModel,
    )

    with store.transaction():
        with pytest.raises(SchemaRegistryError, match="one schema URI"):
            registry.register_model(
                name="shared-model-contract",
                version="1",
                model=_EquivalentCachedSchemaModel,
            )
        assert registry._pending == {}

    assert registry._model_contracts[schema_uri] is _CachedSchemaModel


def test_transaction_can_reattach_the_same_model_contract(tmp_path: Path) -> None:
    store = ArtifactRepository(tmp_path)
    registry = SchemaRegistry(store)
    schema_uri = registry.register_model(
        name="same-model-contract",
        version="1",
        model=_CachedSchemaModel,
    )

    with store.transaction():
        assert (
            registry.register_model(
                name="same-model-contract",
                version="1",
                model=_CachedSchemaModel,
            )
            == schema_uri
        )

    assert registry._model_contracts[schema_uri] is _CachedSchemaModel


def test_transaction_cannot_bind_two_models_to_the_same_schema(tmp_path: Path) -> None:
    store = ArtifactRepository(tmp_path)
    registry = SchemaRegistry(store)

    with store.transaction():
        registry.register_model(
            name="intra-transaction-model-conflict",
            version="1",
            model=_CachedSchemaModel,
        )
        with pytest.raises(SchemaRegistryError, match="one schema URI"):
            registry.register_model(
                name="intra-transaction-model-conflict",
                version="1",
                model=_EquivalentCachedSchemaModel,
            )
