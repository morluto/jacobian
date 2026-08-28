"""Supported exact truncated formal-power-series API.

Native callables take canonical series values and semantic scalar parameters.
They delegate to owner-local operation entrypoints that admit once before
running the direct kernels; wire request models remain an implementation detail
of the catalog and MCP projection.
"""

from jacobian.math.polynomials.series._models import TruncatedSeries
from jacobian.math.polynomials.series.operations import (
    add,
    compose,
    derivative,
    divide,
    from_polynomial,
    identity_check,
    integral_zero_constant,
    inverse,
    multiply,
    power,
    reversion,
    scalar_multiply,
    subtract,
    to_polynomial,
    truncate,
)

__all__ = [
    "TruncatedSeries",
    "add",
    "compose",
    "derivative",
    "divide",
    "from_polynomial",
    "identity_check",
    "integral_zero_constant",
    "inverse",
    "multiply",
    "power",
    "reversion",
    "scalar_multiply",
    "subtract",
    "to_polynomial",
    "truncate",
]
