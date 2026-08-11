"""Shared wire contracts for domain operations."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from jacobian.contracts.common import ArtifactUri
from jacobian.contracts.results import ContractModel


class InlineOperationOutput[ResultT: ContractModel](ContractModel):
    """Inline typed mathematical value with backend provenance."""

    result: ResultT
    backend_version: str


class DurableOperationOutput[PreviewT: ContractModel](ContractModel):
    """Durable artifact carriers with an optional typed preview."""

    input_uri: ArtifactUri
    result_uri: ArtifactUri
    preview: PreviewT | None = None
    preview_complete: bool = False
    backend_version: str

    @model_validator(mode="after")
    def complete_preview_requires_a_value(self) -> Self:
        if self.preview_complete and self.preview is None:
            raise ValueError("a complete materialized preview requires a preview")
        return self
