"""Exact logic operations."""

from jacobian.catalog.models import MathTools
from jacobian.math.logic._operations import LOGIC_OPERATIONS
from jacobian.math.logic._unsat_core import SMT_UNSAT_CORE_OPERATION

__all__ = ["TOOLS"]

TOOLS: MathTools = (*LOGIC_OPERATIONS, SMT_UNSAT_CORE_OPERATION)
