from __future__ import annotations

import pytest
from jsonschema.exceptions import ValidationError
from pydantic import BaseModel

from jacobian.schema_compiler import SchemaCompilationError, SchemaCompiler


class Request(BaseModel):
    value: int


def test_model_compilation_reuses_exact_model_and_schema_identity() -> None:
    compiler = SchemaCompiler()

    first = compiler.compile_model(Request)
    second = compiler.compile_model(Request)

    assert first is second
    assert first.definition() == Request.model_json_schema()


def test_explicit_schema_compilation_reuses_canonical_digest() -> None:
    compiler = SchemaCompiler()

    first = compiler.compile(
        {"type": "object", "properties": {"x": {"type": "integer"}}}
    )
    second = compiler.compile(
        {"properties": {"x": {"type": "integer"}}, "type": "object"}
    )

    assert first is second


def test_compiler_rejects_external_references() -> None:
    with pytest.raises(SchemaCompilationError, match="external or network"):
        SchemaCompiler().compile({"$ref": "https://example.test/schema.json"})


def test_compiler_applies_format_validation_consistently() -> None:
    compiled = SchemaCompiler().compile({"type": "string", "format": "ipv4"})

    with pytest.raises(ValidationError):
        compiled.validator.validate("999.999.999.999")
