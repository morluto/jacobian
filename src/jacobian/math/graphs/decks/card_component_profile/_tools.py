"""No catalog tools: card component profile is a native deterministic projection."""

from typing import Any

from jacobian.catalog.models import MathTool

TOOLS: tuple[MathTool[Any, Any], ...] = ()

__all__ = ["TOOLS"]
