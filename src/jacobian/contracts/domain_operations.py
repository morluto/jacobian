"""Shared wire contracts for domain operations."""

from __future__ import annotations

from jacobian.contracts.common import ArtifactUri
from jacobian.contracts.results import ContractModel


class ComputedOperationOutput[ResultT: ContractModel](ContractModel):
    """Inline typed computed result with backend provenance, no artifacts."""

    result: ResultT
    backend_version: str


class MaterializedOperationOutput[PreviewT: ContractModel](ContractModel):
    """Artifact-linked output with an optional typed preview of the result."""

    input_uri: ArtifactUri
    result_uri: ArtifactUri
    preview: PreviewT | None = None
    preview_complete: bool = False
    backend_version: str
