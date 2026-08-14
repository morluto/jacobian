"""Shared wire contracts for domain operations."""

from __future__ import annotations

from jacobian.contracts.results import ContractModel


class InlineOperationOutput[ResultT: ContractModel](ContractModel):
    """Inline typed mathematical value with backend provenance."""

    result: ResultT
    backend_version: str
