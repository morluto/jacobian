"""Operator-owned, content-addressed JSON Schema registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, cast

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from jacobian.canonical import canonicalize_json, loads_strict_json
from jacobian.schema_validation import check_draft202012_schema
from jacobian.storage.errors import StorageError
from jacobian.storage.repository import ArtifactRepository


class SchemaRegistryError(RuntimeError):
    """Schema registration or resolution failed."""


class SchemaValidationError(SchemaRegistryError):
    """A payload does not satisfy its registered schema."""

    def __init__(
        self,
        message: str,
        *,
        path: str,
        required_field: str | None = None,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.required_field = required_field


@dataclass(slots=True)
class _PendingRegistrations:
    schemas: dict[str, bytes] = field(default_factory=dict)
    registrations: dict[tuple[str, str, bytes], str] = field(default_factory=dict)
    model_contracts: dict[str, type[BaseModel]] = field(default_factory=dict)
    producer_only_schemas: set[str] = field(default_factory=set)


@lru_cache(maxsize=1024)
def _model_schema_bytes(model: type[BaseModel]) -> bytes:
    """Generate one canonical JSON Schema per Pydantic model and process."""

    return canonicalize_json(model.model_json_schema())


def model_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a fresh copy of a cached Pydantic model JSON Schema."""

    return cast(dict[str, Any], loads_strict_json(_model_schema_bytes(model)))


def _reject_external_references(value: Any) -> None:
    if isinstance(value, dict):
        for keyword in ("$ref", "$dynamicRef"):
            reference = value.get(keyword)
            if isinstance(reference, str) and not reference.startswith("#"):
                raise SchemaRegistryError(
                    "registered schemas cannot resolve external or network references"
                )
        for nested in value.values():
            _reject_external_references(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_external_references(nested)


@lru_cache(maxsize=1024)
def _validated_schema(canonical_schema: bytes) -> Draft202012Validator:
    """Validate and compile one exact schema definition per process.

    Runtime construction registers the same contract schemas repeatedly across
    isolated stores, especially in tests. The canonical bytes are the cache
    key, so a changed schema cannot reuse an older validation result or
    validator. The returned validator is read-only during validation.
    """

    normalized = loads_strict_json(canonical_schema)
    _reject_external_references(normalized)
    try:
        check_draft202012_schema(canonical_schema)
    except SchemaError as exc:
        raise SchemaRegistryError("invalid Draft 2020-12 JSON Schema") from exc
    return Draft202012Validator(normalized, format_checker=FormatChecker())


class SchemaRegistry:
    """Store and apply closed local JSON Schemas used by artifact contracts."""

    def __init__(self, store: ArtifactRepository) -> None:
        self.store = store
        self._model_contracts: dict[str, type[BaseModel]] = {}
        self._producer_only_schemas: set[str] = set()
        self._schema_bytes: dict[str, bytes] = {}
        self._registrations: dict[tuple[str, str, bytes], str] = {}
        self._pending: dict[int, _PendingRegistrations] = {}

    def register(
        self,
        *,
        name: str,
        version: str,
        schema: dict[str, Any],
    ) -> str:
        """Register a schema after rejecting unsupported external references."""

        self._reconcile_pending()
        canonical_schema = canonicalize_json(schema)
        return self._register_canonical(
            name=name,
            version=version,
            canonical_schema=canonical_schema,
        )

    def _register_canonical(
        self,
        *,
        name: str,
        version: str,
        canonical_schema: bytes,
    ) -> str:
        """Register an already canonical schema without serializing it again."""

        registration = (name, version, canonical_schema)
        transaction_identity = self.store.transaction_identity
        if transaction_identity is None:
            registrations = self._registrations
        else:
            pending = self._pending.get(transaction_identity)
            registrations = pending.registrations if pending is not None else {}
        cached_uri = registrations.get(registration)
        if cached_uri is not None:
            return cached_uri

        schema = cast(dict[str, Any], loads_strict_json(canonical_schema))
        # Descriptor identity is content-addressed, but an existing descriptor
        # may have been written through ArtifactRepository.register_descriptor()
        # without ever passing Draft 2020-12 validation.  Keep the validation
        # boundary here; _validated_schema is content-cached, so repeated
        # registrations still reuse the compiled validator.
        _reject_external_references(schema)
        _validated_schema(canonical_schema)
        schema_uri = self.store.register_descriptor(
            kind="schema",
            name=name,
            version=version,
            definition=schema,
        )
        if transaction_identity is None:
            self._registrations[registration] = schema_uri
            self._schema_bytes[schema_uri] = canonical_schema
        else:
            pending = self._pending.setdefault(
                transaction_identity,
                _PendingRegistrations(),
            )
            pending.registrations[registration] = schema_uri
            pending.schemas[schema_uri] = canonical_schema
        return schema_uri

    def register_model(
        self,
        *,
        name: str,
        version: str,
        model: type[BaseModel],
        producer_only: bool = False,
    ) -> str:
        """Register JSON shape plus the model's cross-field contract.

        JSON Schema carries the durable structural contract. The operator-owned
        model adds invariants JSON Schema cannot express, such as canonical
        ordering and digests derived from multiple fields. Runtime construction
        repeats this registration after every restart before accepting writes.
        """

        self._reconcile_pending()
        canonical_schema = _model_schema_bytes(model)
        schema = cast(dict[str, Any], loads_strict_json(canonical_schema))
        self._ensure_model_contract_available(
            self.store.descriptor_uri(
                kind="schema",
                name=name,
                version=version,
                definition=schema,
            ),
            model,
        )
        schema_uri = self._register_canonical(
            name=name,
            version=version,
            canonical_schema=canonical_schema,
        )
        self._bind_model_contract(schema_uri, model)
        if producer_only:
            transaction_identity = self.store.transaction_identity
            if transaction_identity is None:
                self._producer_only_schemas.add(schema_uri)
            else:
                self._pending[transaction_identity].producer_only_schemas.add(
                    schema_uri
                )
        return schema_uri

    def is_producer_only(self, schema_uri: str) -> bool:
        """Return whether generic artifact writes are forbidden for this schema."""

        self._reconcile_pending()
        transaction_identity = self.store.transaction_identity
        pending = (
            self._pending.get(transaction_identity)
            if transaction_identity is not None
            else None
        )
        return schema_uri in self._producer_only_schemas or (
            pending is not None and schema_uri in pending.producer_only_schemas
        )

    def _bind_model_contract(
        self,
        schema_uri: str,
        model: type[BaseModel],
    ) -> None:
        self._ensure_model_contract_available(schema_uri, model)
        transaction_identity = self.store.transaction_identity
        if transaction_identity is not None:
            model_contracts = self._pending[transaction_identity].model_contracts
        else:
            model_contracts = self._model_contracts
        model_contracts[schema_uri] = model

    def _ensure_model_contract_available(
        self,
        schema_uri: str,
        model: type[BaseModel],
    ) -> None:
        committed = self._model_contracts.get(schema_uri)
        if committed is not None and committed is not model:
            raise SchemaRegistryError(
                "one schema URI cannot use multiple model-backed contracts"
            )
        transaction_identity = self.store.transaction_identity
        if transaction_identity is None:
            return
        pending = self._pending.get(transaction_identity)
        registered = (
            pending.model_contracts.get(schema_uri) if pending is not None else None
        )
        if registered is not None and registered is not model:
            raise SchemaRegistryError(
                "one schema URI cannot use multiple model-backed contracts"
            )

    def resolve(self, schema_uri: str) -> dict[str, Any]:
        """Load a previously registered schema definition."""

        self._reconcile_pending()
        transaction_identity = self.store.transaction_identity
        if transaction_identity is None:
            cached_schema = self._schema_bytes.get(schema_uri)
            if cached_schema is not None:
                return cast(dict[str, Any], loads_strict_json(cached_schema))
        else:
            pending_schema = self._pending.get(
                transaction_identity,
                _PendingRegistrations(),
            ).schemas.get(schema_uri)
            if pending_schema is not None:
                return cast(dict[str, Any], loads_strict_json(pending_schema))
        try:
            descriptor = self.store.get_descriptor(
                schema_uri,
                expected_kind="schema",
            )
        except StorageError as exc:
            raise SchemaRegistryError(f"unregistered schema: {schema_uri}") from exc
        definition = descriptor.get("definition")
        if not isinstance(definition, dict):
            raise SchemaRegistryError("schema descriptor has no object definition")
        _reject_external_references(definition)
        canonical_schema = canonicalize_json(definition)
        _validated_schema(canonical_schema)
        if transaction_identity is None:
            self._schema_bytes[schema_uri] = canonical_schema
        return cast(dict[str, Any], loads_strict_json(canonical_schema))

    def _reconcile_pending(self) -> None:
        active_identity = self.store.transaction_identity
        for transaction_identity, pending in tuple(self._pending.items()):
            if transaction_identity == active_identity:
                continue
            if not pending.schemas:
                del self._pending[transaction_identity]
                continue
            witness_uri = next(iter(pending.schemas))
            try:
                self.store.get_descriptor(witness_uri, expected_kind="schema")
            except StorageError:
                del self._pending[transaction_identity]
                continue
            self._schema_bytes.update(pending.schemas)
            self._registrations.update(pending.registrations)
            self._model_contracts.update(pending.model_contracts)
            self._producer_only_schemas.update(pending.producer_only_schemas)
            del self._pending[transaction_identity]

    def validate(self, schema_uri: str, payload: Any) -> Any:
        """Validate and canonically normalize a payload."""

        normalized = loads_strict_json(canonicalize_json(payload))
        schema = self.resolve(schema_uri)
        validator = _validated_schema(canonicalize_json(schema))
        errors = sorted(
            validator.iter_errors(normalized),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
        if errors:
            first: JsonSchemaValidationError = errors[0]
            location = "/".join(str(part) for part in first.absolute_path) or "$"
            required_field = None
            if (
                first.validator == "required"
                and isinstance(first.validator_value, list)
                and isinstance(first.instance, dict)
            ):
                required_field = next(
                    (
                        field
                        for field in first.validator_value
                        if isinstance(field, str) and field not in first.instance
                    ),
                    None,
                )
            raise SchemaValidationError(
                f"{location}: {first.message}",
                path=location,
                required_field=required_field,
            )
        model = self._model_contracts.get(schema_uri)
        if model is not None:
            try:
                normalized = model.model_validate(normalized).model_dump(mode="json")
            except PydanticValidationError as exc:
                first_error = exc.errors(include_url=False, include_context=False)[0]
                raw_location = first_error.get("loc", ())
                location = "/".join(str(part) for part in raw_location) or "$"
                required_field = None
                if first_error.get("type") == "missing" and raw_location:
                    required_field = str(raw_location[-1])
                raise SchemaValidationError(
                    f"{location}: {first_error['msg']}",
                    path=location,
                    required_field=required_field,
                ) from exc
            normalized = loads_strict_json(canonicalize_json(normalized))
        return normalized
