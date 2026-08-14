"""Static catalog entries for Jacobian's typed mathematical functions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jacobian.contracts.base import ContractModel
from jacobian.contracts.operations import OperationExample


@dataclass(frozen=True, slots=True)
class MathTool[RequestT: ContractModel, ResultT: ContractModel]:
    """One discoverable mathematical function and its public typed contract."""

    operation_id: str
    version: str
    title: str
    description: str
    request_type: type[RequestT]
    result_type: type[ResultT]
    run: Callable[[RequestT], ResultT]
    tags: tuple[str, ...] = ()
    examples: tuple[OperationExample, ...] = ()

    def __post_init__(self) -> None:
        if not self.operation_id.strip() or not self.version.strip():
            raise ValueError("math tools require an ID and version")
        if not self.title.strip() or not self.description.strip():
            raise ValueError("math tools require title and description")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("math tool tags must be unique")
        if any(not tag.strip() for tag in self.tags):
            raise ValueError("math tool tags must not be empty")
        if len({example.name for example in self.examples}) != len(self.examples):
            raise ValueError("math tool example names must be unique")


type MathTools = tuple[MathTool[Any, Any], ...]


__all__ = ["MathTool", "MathTools"]
