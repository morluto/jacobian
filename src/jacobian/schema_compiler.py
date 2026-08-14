"""One process-wide compiler for model-backed and explicit JSON Schemas."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from hashlib import sha256
from typing import Any, cast

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel

from jacobian.canonical import canonicalize_json, loads_strict_json
from jacobian.schema_validation import check_draft202012_schema


class SchemaCompilationError(ValueError):
    """A schema cannot be compiled under Jacobian's closed local profile."""


class SchemaDialect(StrEnum):
    DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"


@dataclass(frozen=True, slots=True)
class CompiledSchema:
    """Canonical schema bytes and their reusable read-only validator."""

    canonical_schema: bytes
    digest: str
    validator: Draft202012Validator

    def definition(self) -> dict[str, Any]:
        return cast(dict[str, Any], loads_strict_json(self.canonical_schema))


def _reject_external_references(value: Any) -> None:
    if isinstance(value, dict):
        for keyword in ("$ref", "$dynamicRef"):
            reference = value.get(keyword)
            if isinstance(reference, str) and not reference.startswith("#"):
                raise SchemaCompilationError(
                    "schemas cannot resolve external or network references"
                )
        for nested in value.values():
            _reject_external_references(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_external_references(nested)


@lru_cache(maxsize=2048)
def _compile(
    model: type[BaseModel] | None,
    dialect: SchemaDialect,
    digest: str,
    canonical_schema: bytes,
) -> CompiledSchema:
    del model
    if dialect is not SchemaDialect.DRAFT_2020_12:
        raise SchemaCompilationError(f"unsupported schema dialect: {dialect}")
    measured_digest = "sha256:" + sha256(canonical_schema).hexdigest()
    if measured_digest != digest:
        raise SchemaCompilationError("schema digest does not match canonical bytes")
    normalized = loads_strict_json(canonical_schema)
    _reject_external_references(normalized)
    try:
        check_draft202012_schema(canonical_schema)
    except SchemaError as exc:
        raise SchemaCompilationError("invalid Draft 2020-12 JSON Schema") from exc
    return CompiledSchema(
        canonical_schema=canonical_schema,
        digest=digest,
        validator=Draft202012Validator(
            normalized,
            format_checker=FormatChecker(),
        ),
    )


@lru_cache(maxsize=2048)
def _compile_model(
    model: type[BaseModel],
    dialect: SchemaDialect,
) -> CompiledSchema:
    schema = model.model_json_schema()
    canonical_schema = canonicalize_json(schema)
    digest = "sha256:" + sha256(canonical_schema).hexdigest()
    return _compile(model, dialect, digest, canonical_schema)


class SchemaCompiler:
    """Compile exact schemas once by model identity, dialect, and digest."""

    def compile(
        self,
        schema: dict[str, Any],
        *,
        model: type[BaseModel] | None = None,
        dialect: SchemaDialect = SchemaDialect.DRAFT_2020_12,
    ) -> CompiledSchema:
        canonical_schema = canonicalize_json(schema)
        digest = "sha256:" + sha256(canonical_schema).hexdigest()
        return _compile(model, dialect, digest, canonical_schema)

    def compile_model(
        self,
        model: type[BaseModel],
        *,
        dialect: SchemaDialect = SchemaDialect.DRAFT_2020_12,
    ) -> CompiledSchema:
        return _compile_model(model, dialect)


SCHEMA_COMPILER = SchemaCompiler()


__all__ = [
    "SCHEMA_COMPILER",
    "CompiledSchema",
    "SchemaCompilationError",
    "SchemaCompiler",
    "SchemaDialect",
]
